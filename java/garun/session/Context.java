package garun.session;

import ai.djl.ndarray.NDManager;
import clojure.lang.IFn;
import org.duckdb.DuckDBConnection;

/** One thread's connection and tensor owner within a session. */
public interface Context {
    DuckDBConnection connection();
    NDManager manager();
    Session session();

    /** Invoke a zero-argument operation while session shutdown is excluded. */
    Object call(IFn operation);
}
