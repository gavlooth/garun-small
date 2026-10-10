(ns garun.context
  "One thread's connection and tensor owner within a session.
  Context operations exclude session cleanup until their zero-argument callback
  returns; closed contexts reject admission with :session/closed.")

(set! *warn-on-reflection* true)

(defprotocol Context
  (connection ^org.duckdb.DuckDBConnection [context] "Return this context's borrowed connection.")
  (manager ^ai.djl.ndarray.NDManager [context] "Return this context's borrowed tensor manager.")
  (owner [context] "Return the owning Session.")
  (call [context operation] "Run a zero-argument function under the session lifecycle read lock."))
