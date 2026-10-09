package garun.session;

import clojure.lang.IFn;
import clojure.lang.Keyword;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/** A registered value: concurrent readers, exclusive mutation and exclusive release. */
public final class Resource implements AutoCloseable {
    public final Object name;
    public final long id;
    public final Kind kind;
    public final Object value;
    public final boolean owned;

    // A value cannot be mutated or closed while any shared operation is using it.
    private final ReentrantReadWriteLock resourceLock = new ReentrantReadWriteLock(true);
    private volatile boolean released;

    Resource(Object name, long identifier, Kind kind, Object value, boolean owned) {
        this.name = name;
        this.id = identifier;
        this.kind = kind;
        this.value = value;
        this.owned = owned;
    }

    public Object shared(IFn operation) {
        resourceLock.readLock().lock();
        try {
            ensureAvailable();
            return operation.invoke(value);
        } finally {
            resourceLock.readLock().unlock();
        }
    }

    public Object exclusive(IFn operation) {
        resourceLock.writeLock().lock();
        try {
            ensureAvailable();
            return operation.invoke(value);
        } finally {
            resourceLock.writeLock().unlock();
        }
    }

    private void ensureAvailable() {
        if (released) {
            throw Session.error("released", "Resource has been released",
                    Keyword.intern(null, "name"), name,
                    Keyword.intern(null, "kind"), Session.kindKeyword(kind));
        }
    }

    public boolean isReleased() {
        return released;
    }

    // Registry mutation acquires the resource first, so callbacks may safely resolve other names.
    void lockForRelease() {
        if (resourceLock.getReadHoldCount() != 0 && !resourceLock.isWriteLockedByCurrentThread()) {
            throw Session.error("close-inside-operation", "Cannot release a resource inside its shared operation",
                    Keyword.intern(null, "name"), name);
        }
        resourceLock.writeLock().lock();
    }

    void unlockForRelease() {
        resourceLock.writeLock().unlock();
    }

    /** Session shutdown closes owned tables on the root connection under its lifecycle write lock. */
    @Override
    public void close() {
        close(null);
    }

    /** Release owned tables using the calling context; other natives close normally. */
    public void close(Context context) {
        lockForRelease();
        try {
            if (released) {
                return;
            }
            released = true;
            if (owned && context != null && value instanceof MaterializedTable) {
                ((MaterializedTable) value).drop(context.connection());
            } else if (owned && value instanceof AutoCloseable) {
                Session.rethrow(Session.closeAfter(null, (AutoCloseable) value));
            }
        } finally {
            unlockForRelease();
        }
    }
}
