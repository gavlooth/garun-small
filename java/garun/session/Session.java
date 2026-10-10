package garun.session;

import ai.djl.ndarray.NDManager;
import clojure.lang.ExceptionInfo;
import clojure.lang.IFn;
import clojure.lang.Keyword;
import clojure.lang.PersistentArrayMap;
import clojure.lang.Util;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import org.duckdb.DuckDBConnection;

/** Owns native resources; only this context and exclusive shutdown use its root connection. */
public final class Session implements Context, AutoCloseable {
    private final String identifier;
    private final String schema;
    private final DuckDBConnection connection;
    private final NDManager manager;
    private final Defaults defaults;

    // Shutdown excludes all admitted operations, including reentrant calls from those operations.
    private final ReentrantReadWriteLock lifecycleLock = new ReentrantReadWriteLock(true);
    // Replacement removes, closes and installs a named resource as one observable registry change.
    private final ReentrantReadWriteLock registryLock = new ReentrantReadWriteLock(true);
    // Named and anonymous resources have one canonical registry entry each.
    private final ConcurrentHashMap<Object, Resource> registry = new ConcurrentHashMap<>();
    // Every open worker remains tracked until its own native cleanup completes.
    private final Set<Worker> workers = ConcurrentHashMap.newKeySet();
    // Registration order is unique even when several contexts register concurrently.
    private final AtomicLong nextResourceIdentifier = new AtomicLong();
    // Generated table identifiers never collide within this session's private schema.
    private final AtomicLong nextTableIdentifier = new AtomicLong();
    private volatile boolean closed;

    private Session(String identifier, DuckDBConnection connection, NDManager manager, Defaults defaults) {
        this.identifier = identifier;
        this.schema = "garun_" + identifier;
        this.connection = connection;
        this.manager = manager;
        this.defaults = defaults;
    }

    public static Session open(DuckDBConnection connection, NDManager manager, Defaults defaults) {
        Session session = new Session(UUID.randomUUID().toString().replace("-", ""),
                connection, manager, defaults);
        try (Statement statement = connection.createStatement()) {
            // Only this generated, hexadecimal identifier enters the DDL string.
            statement.execute("CREATE SCHEMA \"" + session.schema + "\"");
            return session;
        } catch (Throwable failure) {
            failure = closeAfter(failure, manager);
            failure = closeAfter(failure, connection);
            throw Util.sneakyThrow(failure);
        }
    }

    public String identifier() {
        return identifier;
    }

    public String schema() {
        return schema;
    }

    public Defaults defaults() {
        return defaults;
    }

    public boolean isClosed() {
        return closed;
    }

    @Override
    public DuckDBConnection connection() {
        return connection;
    }

    @Override
    public NDManager manager() {
        return manager;
    }

    @Override
    public Session session() {
        return this;
    }

    @Override
    public Object call(IFn operation) {
        beginOperation();
        try {
            return operation.invoke();
        } finally {
            endOperation();
        }
    }

    // Workers admit the session before taking their own lifetime lock, without allocating callbacks.
    void beginOperation() {
        lifecycleLock.readLock().lock();
        try {
            ensureOpen();
        } catch (Throwable failure) {
            lifecycleLock.readLock().unlock();
            throw failure;
        }
    }

    void endOperation() {
        lifecycleLock.readLock().unlock();
    }

    private void ensureOpen() {
        if (closed) {
            throw error("closed", "Session is closed", Keyword.intern(null, "identifier"), identifier);
        }
    }

    public String nextTableName() {
        lifecycleLock.readLock().lock();
        try {
            ensureOpen();
            return "t_" + nextTableIdentifier.incrementAndGet();
        } finally {
            lifecycleLock.readLock().unlock();
        }
    }

    /** Register a value, closing any replacement on the calling context's connection. */
    public Resource register(Object name, Kind kind, Object value, boolean owned, boolean replace,
            Context context) {
        lifecycleLock.readLock().lock();
        try {
            ensureOpen();
            if (name != null && !(name instanceof Keyword)) {
                throw error("invalid-name", "Resource names must be keywords",
                        Keyword.intern(null, "name"), name);
            }
            if (kind == null) {
                throw error("invalid-kind", "A resource kind is required");
            }
            if (value == null) {
                throw error("invalid-resource", "A resource value cannot be null");
            }
            if (name == null || !replace) {
                registryLock.writeLock().lock();
                try {
                    if (name != null && registry.containsKey(name)) {
                        throw error("duplicate-name", "Resource name is already registered",
                                Keyword.intern(null, "name"), name);
                    }
                    return install(name, kind, value, owned);
                } finally {
                    registryLock.writeLock().unlock();
                }
            }
            return replace(name, kind, value, owned, context);
        } finally {
            lifecycleLock.readLock().unlock();
        }
    }

    private Resource install(Object name, Kind kind, Object value, boolean owned) {
        long identifier = nextResourceIdentifier.incrementAndGet();
        Resource resource = new Resource(name, identifier, kind, value, owned);
        registry.put(name == null ? Long.valueOf(identifier) : name, resource);
        return resource;
    }

    private Resource replace(Object name, Kind kind, Object value, boolean owned, Context context) {
        // Acquire the resource before the registry: an in-flight callback may resolve other names.
        // Retry only when a concurrent registry mutation changed the candidate before admission.
        for (;;) {
            Resource previous = namedResource(name);
            if (previous != null) {
                previous.lockForRelease();
            }
            registryLock.writeLock().lock();
            try {
                if (registry.get(name) != previous) {
                    continue;
                }
                if (previous != null) {
                    registry.remove(name);
                    previous.close(context);
                }
                return install(name, kind, value, owned);
            } finally {
                registryLock.writeLock().unlock();
                if (previous != null) {
                    previous.unlockForRelease();
                }
            }
        }
    }

    private Resource namedResource(Object name) {
        registryLock.readLock().lock();
        try {
            return registry.get(name);
        } finally {
            registryLock.readLock().unlock();
        }
    }

    public Resource resolve(Object name, Kind expected) {
        lifecycleLock.readLock().lock();
        try {
            ensureOpen();
            Resource resource = name == null ? null : namedResource(name);
            if (resource == null) {
                throw unknownName(name);
            }
            if (resource.kind != expected) {
                throw error("wrong-kind", "Resource has a different kind",
                        Keyword.intern(null, "name"), name,
                        Keyword.intern(null, "kind"), kindKeyword(resource.kind),
                        Keyword.intern(null, "expected"), kindKeyword(expected));
            }
            return resource;
        } finally {
            lifecycleLock.readLock().unlock();
        }
    }

    public Resource lookup(Object value) {
        lifecycleLock.readLock().lock();
        try {
            ensureOpen();
            registryLock.readLock().lock();
            try {
                return lookupRegistered(value);
            } finally {
                registryLock.readLock().unlock();
            }
        } finally {
            lifecycleLock.readLock().unlock();
        }
    }

    private Resource lookupRegistered(Object value) {
        for (Resource resource : registry.values()) {
            if (resource.value == value) {
                return resource;
            }
        }
        return null;
    }

    /** Release a registered value, dropping owned tables on the releasing context's connection. */
    public void release(Object nameOrValue, Context context) {
        lifecycleLock.readLock().lock();
        try {
            ensureOpen();
            for (;;) {
                registryLock.readLock().lock();
                Resource resource;
                try {
                    resource = nameOrValue instanceof Keyword
                            ? registry.get(nameOrValue) : lookupRegistered(nameOrValue);
                } finally {
                    registryLock.readLock().unlock();
                }
                if (resource == null) {
                    throw unknownName(nameOrValue);
                }
                resource.lockForRelease();
                registryLock.writeLock().lock();
                try {
                    Object registryKey = resource.name == null ? Long.valueOf(resource.identifier) : resource.name;
                    if (!registry.remove(registryKey, resource)) {
                        continue;
                    }
                    resource.close(context);
                    return;
                } finally {
                    registryLock.writeLock().unlock();
                    resource.unlockForRelease();
                }
            }
        } finally {
            lifecycleLock.readLock().unlock();
        }
    }

    public List<Resource> resources() {
        registryLock.readLock().lock();
        try {
            List<Resource> snapshot = new ArrayList<>(registry.values());
            snapshot.sort(Comparator.comparingLong(resource -> resource.identifier));
            return snapshot;
        } finally {
            registryLock.readLock().unlock();
        }
    }

    public int workerCount() {
        return workers.size();
    }

    /** Duplicate the parent context's connection, so workers can safely create other workers. */
    public Worker worker(Context parent) {
        lifecycleLock.readLock().lock();
        try {
            ensureOpen();
            DuckDBConnection workerConnection = parent.connection().duplicate();
            NDManager workerManager;
            try {
                workerManager = manager.newSubManager();
            } catch (Throwable failure) {
                throw Util.sneakyThrow(closeAfter(failure, workerConnection));
            }
            Worker worker = new Worker(this, workerConnection, workerManager);
            workers.add(worker);
            return worker;
        } catch (java.sql.SQLException failure) {
            throw Util.sneakyThrow(failure);
        } finally {
            lifecycleLock.readLock().unlock();
        }
    }

    void removeWorker(Worker worker) {
        workers.remove(worker);
    }

    @Override
    public void close() {
        if (lifecycleLock.getReadHoldCount() != 0) {
            throw error("close-inside-operation", "Cannot close a session inside an operation",
                    Keyword.intern(null, "identifier"), identifier);
        }
        lifecycleLock.writeLock().lock();
        try {
            if (closed) {
                return;
            }
            closed = true;
            Throwable failure = null;
            for (Worker worker : workers) {
                failure = closeAfter(failure, worker);
            }
            List<Resource> snapshot = resources();
            registryLock.writeLock().lock();
            try {
                registry.clear();
            } finally {
                registryLock.writeLock().unlock();
            }
            for (int index = snapshot.size() - 1; index >= 0; index--) {
                failure = closeAfter(failure, snapshot.get(index));
            }
            try (Statement statement = connection.createStatement()) {
                statement.execute("DROP SCHEMA \"" + schema + "\" CASCADE");
            } catch (Throwable schemaFailure) {
                failure = appendFailure(failure, schemaFailure);
            }
            failure = closeAfter(failure, manager);
            failure = closeAfter(failure, connection);
            rethrow(failure);
        } finally {
            lifecycleLock.writeLock().unlock();
        }
    }

    static Keyword kindKeyword(Kind kind) {
        return kind == null ? null : Keyword.intern(null, kind.name().toLowerCase(java.util.Locale.ROOT));
    }

    static ExceptionInfo error(String code, String message, Object... context) {
        Object[] entries = new Object[context.length + 2];
        entries[0] = Keyword.intern(null, "error");
        entries[1] = Keyword.intern("session", code);
        System.arraycopy(context, 0, entries, 2, context.length);
        return new ExceptionInfo(message, new PersistentArrayMap(entries));
    }

    private static ExceptionInfo unknownName(Object nameOrValue) {
        return error("unknown-name", "Resource is not registered",
                Keyword.intern(null, "name"), nameOrValue);
    }

    static Throwable appendFailure(Throwable firstFailure, Throwable laterFailure) {
        if (firstFailure == null) {
            return laterFailure;
        }
        if (firstFailure != laterFailure) {
            firstFailure.addSuppressed(laterFailure);
        }
        return firstFailure;
    }

    static Throwable closeAfter(Throwable failure, AutoCloseable resource) {
        try {
            resource.close();
        } catch (Throwable closeFailure) {
            return appendFailure(failure, closeFailure);
        }
        return failure;
    }

    static void rethrow(Throwable failure) {
        if (failure != null) {
            throw Util.sneakyThrow(failure);
        }
    }
}
