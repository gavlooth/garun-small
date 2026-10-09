package garun.session;

import ai.djl.ndarray.NDManager;
import clojure.lang.IFn;
import clojure.lang.Keyword;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import org.duckdb.DuckDBConnection;

/** A session-owned, per-thread duplicated connection and tensor sub-manager. */
public final class Worker implements Context, AutoCloseable {
    private final Session session;
    private final DuckDBConnection connection;
    private final NDManager manager;

    // Worker cleanup must wait for its admitted operations without closing sibling contexts.
    private final ReentrantReadWriteLock lifecycleLock = new ReentrantReadWriteLock(true);
    private volatile boolean closed;

    Worker(Session session, DuckDBConnection connection, NDManager manager) {
        this.session = session;
        this.connection = connection;
        this.manager = manager;
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
        return session;
    }

    public boolean isClosed() {
        return closed;
    }

    @Override
    public Object call(IFn operation) {
        // Session-before-worker ordering also lets session shutdown close idle workers safely.
        session.beginOperation();
        lifecycleLock.readLock().lock();
        try {
            if (closed) {
                throw Session.error("closed", "Worker is closed",
                        Keyword.intern(null, "worker"), true);
            }
            return session.call(operation);
        } finally {
            lifecycleLock.readLock().unlock();
            session.endOperation();
        }
    }

    @Override
    public void close() {
        if (lifecycleLock.getReadHoldCount() != 0) {
            throw Session.error("close-inside-operation", "Cannot close a worker inside its operation",
                    Keyword.intern(null, "worker"), true);
        }
        lifecycleLock.writeLock().lock();
        try {
            if (closed) {
                return;
            }
            closed = true;
            Throwable failure = Session.closeAfter(null, manager);
            failure = Session.closeAfter(failure, connection);
            session.removeWorker(this);
            Session.rethrow(failure);
        } finally {
            lifecycleLock.writeLock().unlock();
        }
    }
}
