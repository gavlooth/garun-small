(ns garun.session
  "Session-owned DuckDB connections, tensor managers and named native resources.

  Close a Session with with-open: it closes workers, then owned resources in
  reverse registration order, drops its private schema, closes the root manager,
  and finally closes its connection. Borrowed values are never closed. Tensors
  returned through a context belong to that context's manager.

  Use one thread per context and one Worker per additional thread. Workers own
  duplicated connections and tensor sub-managers; predictions may share a
  registered network while training takes its exclusive resource lock.

  Final atom and lock fields own the synchronous native-lifetime invariants.
  Reentrant read/write locks wait for blocking JDBC/DJL calls before cleanup;
  registry replacement excludes observers until old-value cleanup completes.
  HoneySQL extensions express schema DDL absent from its built-in clauses;
  only generated schema identifiers enter these lifecycle statements."
  (:require [clojure.core.protocols :as protocols]
            [clojure.string :as string]
            [garun.context :as context]
            [garun.tensor :as tensor]
            [honey.sql :as sql])
  (:import [ai.djl Device]
           [ai.djl.engine Engine]
           [ai.djl.ndarray NDManager]
           [ai.djl.ndarray.types DataType]
           [clojure.lang Atom]
           [java.io Writer]
           [java.lang AutoCloseable]
           [java.sql Connection DriverManager]
           [java.util UUID]
           [java.util.concurrent.locks Lock ReentrantReadWriteLock]
           [org.duckdb DuckDBConnection]))

(set! *warn-on-reflection* true)

(declare begin-operation! end-operation! close-session! close-worker! close-resource! describe)

(deftype Session [identifier schema ^DuckDBConnection connection ^NDManager manager defaults
                  ;; Shutdown waits for all admitted operations, including reentrant calls.
                  ^ReentrantReadWriteLock lifecycle-lock
                  ;; Replacement closes and installs a name as one observable registry change.
                  ^ReentrantReadWriteLock registry-lock
                  ;; Every named or anonymous resource has one canonical registry entry.
                  ^Atom registry
                  ;; Workers remain tracked until their native cleanup finishes.
                  ^Atom workers
                  ;; Registration identifiers uniquely order concurrent resource admissions.
                  ^Atom next-resource-identifier
                  ;; Table identifiers never collide within this generated schema.
                  ^Atom next-table-identifier
                  ;; A closed session never admits another operation or repeats cleanup.
                  ^Atom closed]
  context/Context
  (connection [_] connection)
  (manager [_] manager)
  (owner [current-session] current-session)
  (call [current-session operation]
    (begin-operation! current-session)
    (try
      (operation)
      (finally (end-operation! current-session))))
  AutoCloseable
  (close [current-session] (close-session! current-session))
  protocols/Datafiable
  (datafy [current-session] (describe current-session)))

(deftype Worker [^Session parent ^DuckDBConnection connection ^NDManager manager
                 ;; Worker cleanup waits for its operations without closing sibling contexts.
                 ^ReentrantReadWriteLock lifecycle-lock
                 ;; Native cleanup runs once, even when session and worker close race.
                 ^Atom closed]
  context/Context
  (connection [_] connection)
  (manager [_] manager)
  (owner [_] parent)
  (call [_ operation]
    ;; Session-before-worker admission lets shutdown safely close idle workers.
    (begin-operation! parent)
    (.lock (.readLock lifecycle-lock))
    (try
      (when @closed
        (throw (ex-info "Worker is closed" {:error :session/closed :worker true})))
      (operation)
      (finally
        (.unlock (.readLock lifecycle-lock))
        (end-operation! parent))))
  AutoCloseable
  (close [current-worker] (close-worker! current-worker))
  protocols/Datafiable
  (datafy [current-worker] (describe current-worker)))

(deftype Resource [name identifier kind value owned?
                   ;; Mutation and release wait until every shared operation has finished.
                   ^ReentrantReadWriteLock resource-lock
                   ;; A released resource rejects all operations and repeats no cleanup.
                   ^Atom released]
  clojure.lang.ILookup
  (valAt [resource key]
    (.valAt ^clojure.lang.ILookup resource key nil))
  (valAt [_ key not-found]
    (case key
      :name name
      :identifier identifier
      :kind kind
      :value value
      :owned? owned?
      :released? @released
      not-found))
  AutoCloseable
  (close [resource] (close-resource! resource nil))
  protocols/Datafiable
  (datafy [resource] (describe resource)))

(defn- invalid! [error message value]
  (throw (ex-info message {:error error :value value})))

(defn- options! [options allowed]
  (when-not (map? options)
    (invalid! :session/invalid-options "Session options must be a map" options))
  (when-let [unknown (seq (remove allowed (keys options)))]
    (invalid! :session/unknown-option "Unknown session option" (vec unknown))))

(defn- append-failure! [^Throwable failure ^Throwable later-failure]
  (when (and failure (not (identical? failure later-failure)))
    (.addSuppressed failure later-failure))
  (or failure later-failure))

(defn- close-after! [failure ^AutoCloseable resource]
  (try
    (.close resource)
    failure
    (catch Throwable close-failure
      (append-failure! failure close-failure))))

(defn- begin-operation! [^Session current-session]
  (let [^ReentrantReadWriteLock lifecycle-lock (.-lifecycle-lock current-session)]
    (.lock (.readLock lifecycle-lock))
    (when @(.-closed current-session)
      (.unlock (.readLock lifecycle-lock))
      (throw (ex-info "Session is closed"
                      {:error :session/closed :identifier (.-identifier current-session)})))))

(defn- end-operation! [^Session current-session]
  (.unlock (.readLock ^ReentrantReadWriteLock (.-lifecycle-lock current-session))))

(defmulti describe-value
  "Return plain, non-owning data about a resource value, dispatched by kind."
  (fn [kind _value] kind))

(defmethod describe-value :default [_kind value]
  {:class (.getName ^Class (class value))})

(defmulti release-value!
  "Release an owned value. Type-specific cleanup may use the releasing context's
  connection; nil context denotes session shutdown under its lifecycle write lock."
  (fn [_kind value _context] (class value)))

(defmethod release-value! :default [_kind value _context]
  (when (instance? AutoCloseable value)
    (.close ^AutoCloseable value)))

(defmulti release-scope
  "Run operation, which releases value through context, inside any locks that the
  value's users take before its own Resource lock. Values whose users lock another
  resource first (trainers lock their network) extend this to keep one lock order."
  (fn [_kind value _context _operation] (class value)))

(defmethod release-scope :default [_kind _value _context operation]
  (operation))

(defn- lock-for-release! [^Resource resource]
  (let [^ReentrantReadWriteLock resource-lock (.-resource-lock resource)]
    (when (and (pos? (.getReadHoldCount resource-lock))
               (not (.isWriteLockedByCurrentThread resource-lock)))
      (throw (ex-info "Cannot release a resource inside its shared operation"
                      {:error :session/close-inside-operation :name (.-name resource)})))
    (.lock (.writeLock resource-lock))))

(defn- close-resource! [^Resource resource releasing-context]
  (lock-for-release! resource)
  (try
    (when-not @(.-released resource)
      (reset! (.-released resource) true)
      (when (.-owned? resource)
        (release-value! (.-kind resource) (.-value resource) releasing-context)))
    (finally (.unlock (.writeLock ^ReentrantReadWriteLock (.-resource-lock resource))))))

(defn- resource-call [^Resource resource shared? operation]
  (let [^ReentrantReadWriteLock resource-lock (.-resource-lock resource)
        ^Lock operation-lock (if shared? (.readLock resource-lock) (.writeLock resource-lock))]
    (.lock operation-lock)
    (try
      (when @(.-released resource)
        (throw (ex-info "Resource has been released"
                        {:error :session/released :name (.-name resource) :kind (.-kind resource)})))
      (operation (.-value resource))
      (finally (.unlock operation-lock)))))

(defn shared-resource
  "Run an operation under a registered Resource's shared lock, rejecting release."
  [resource operation]
  (resource-call resource true operation))

(defn exclusive-resource
  "Run an operation under a registered Resource's exclusive lock, rejecting release."
  [resource operation]
  (resource-call resource false operation))

(defn resources
  "Snapshot a session's registry in registration order, including after close."
  [current-context]
  (let [^Session current-session (context/owner current-context)
        ^ReentrantReadWriteLock registry-lock (.-registry-lock current-session)]
    (.lock (.readLock registry-lock))
    (try
      (->> @(.-registry current-session) vals (sort-by #(.-identifier ^Resource %)) vec)
      (finally (.unlock (.readLock registry-lock))))))

(sql/register-clause! :garun/create-schema
                      (fn [_ schema] [(str "CREATE SCHEMA " (sql/format-entity schema))])
                      nil)

(sql/register-clause! :garun/drop-schema
                      (fn [_ schema] [(str "DROP SCHEMA " (sql/format-entity schema) " CASCADE")])
                      nil)

(defn- schema-ddl! [^Connection connection operation schema]
  (let [[statement-sql] (sql/format {operation (keyword schema)} {:quoted true})]
    (with-open [statement (.createStatement connection)]
      (.execute statement ^String statement-sql))))

(defn- close-worker! [^Worker current-worker]
  (let [^ReentrantReadWriteLock lifecycle-lock (.-lifecycle-lock current-worker)]
    (when (pos? (.getReadHoldCount lifecycle-lock))
      (throw (ex-info "Cannot close a worker inside its operation"
                      {:error :session/close-inside-operation :worker true})))
    (.lock (.writeLock lifecycle-lock))
    (try
      (when-not @(.-closed current-worker)
        (reset! (.-closed current-worker) true)
        (let [failure (-> nil
                          (close-after! (.-manager current-worker))
                          (close-after! (.-connection current-worker)))]
          (swap! (.-workers ^Session (.-parent current-worker)) disj current-worker)
          (when failure (throw failure))))
      (finally (.unlock (.writeLock lifecycle-lock))))))

(defn- close-session! [^Session current-session]
  (let [^ReentrantReadWriteLock lifecycle-lock (.-lifecycle-lock current-session)
        ^ReentrantReadWriteLock registry-lock (.-registry-lock current-session)]
    (when (pos? (.getReadHoldCount lifecycle-lock))
      (throw (ex-info "Cannot close a session inside an operation"
                      {:error :session/close-inside-operation :identifier (.-identifier current-session)})))
    (.lock (.writeLock lifecycle-lock))
    (try
      (when-not @(.-closed current-session)
        (reset! (.-closed current-session) true)
        (let [worker-failure (reduce close-after! nil @(.-workers current-session))
              snapshot (resources current-session)
              _ (do
                  (.lock (.writeLock registry-lock))
                  (try
                    (reset! (.-registry current-session) {})
                    (finally (.unlock (.writeLock registry-lock)))))
              resource-failure (reduce close-after! worker-failure (rseq snapshot))
              schema-failure (try
                               (schema-ddl! (.-connection current-session) :garun/drop-schema
                                            (.-schema current-session))
                               resource-failure
                               (catch Throwable failure (append-failure! resource-failure failure)))
              failure (-> schema-failure
                          (close-after! (.-manager current-session))
                          (close-after! (.-connection current-session)))]
          (when failure (throw failure))))
      (finally (.unlock (.writeLock lifecycle-lock))))))

(defn open
  "Open an owned Session; close it with with-open.
  Closed options: :database (:memory or a nonempty path), :device (:cpu),
  :dtype (:float32), :engine (DJL's default engine name), and :seed (11)."
  (^Session [] (open {}))
  (^Session [options]
   (options! options #{:database :device :dtype :engine :seed})
   (let [database (get options :database :memory)
         seed (get options :seed 11)
         engine-name (:engine options)]
     (when-not (or (= :memory database)
                   (and (string? database) (not (empty? database))))
       (invalid! :session/invalid-database "Database must be :memory or a nonempty path" database))
     (when-not (and (integer? seed) (<= Long/MIN_VALUE seed Long/MAX_VALUE))
       (invalid! :session/invalid-seed "Session seed must be an int64 integer" seed))
     (when (and (contains? options :engine)
                (not (and (string? engine-name) (not (string/blank? engine-name)))))
       (invalid! :session/invalid-engine "Session :engine must be a DJL engine name" engine-name))
     (let [^Engine engine (if engine-name (Engine/getEngine engine-name) (Engine/getInstance))
           resolved-engine (.getEngineName engine)
           resolved-device (tensor/device (get options :device :cpu) engine)
           resolved-data-type (tensor/dtype (get options :dtype :float32) engine)
           native-defaults {:device resolved-device :dtype resolved-data-type :engine resolved-engine
                            :seed (long seed) :database (if (= :memory database) ":memory" database)}
           ^DuckDBConnection connection (DriverManager/getConnection
                                         (str "jdbc:duckdb:" (when-not (= :memory database) database)))
           manager (try
                     (NDManager/newBaseManager resolved-device resolved-engine)
                     (catch Throwable failure
                       (throw (close-after! failure connection))))
           identifier (string/replace (str (UUID/randomUUID)) "-" "")
           schema (str "garun_" identifier)
           current-session (Session. identifier schema connection manager native-defaults
                                     (ReentrantReadWriteLock. true) (ReentrantReadWriteLock. true)
                                     (atom {}) (atom #{}) (atom 0) (atom 0) (atom false))]
       (try
         (schema-ddl! connection :garun/create-schema schema)
         current-session
         (catch Throwable failure
           (throw (-> failure (close-after! manager) (close-after! connection)))))))))

(defn context?
  "Whether value implements the session Context contract."
  [value]
  (satisfies? context/Context value))

(defn worker
  "Create a session-owned per-thread Worker by duplicating this context's connection.
  A Worker can create another Worker without using the session's root connection.
  Close it with with-open when its thread finishes; the session also closes it."
  ^Worker [parent-context]
  (context/call
   parent-context
   (fn []
     (let [^Session current-session (context/owner parent-context)
           connection (.duplicate (context/connection parent-context))
           manager (try
                     (.newSubManager ^NDManager (.-manager current-session))
                     (catch Throwable failure (throw (close-after! failure connection))))
           current-worker (Worker. current-session connection manager (ReentrantReadWriteLock. true) (atom false))]
       (swap! (.-workers current-session) conj current-worker)
       current-worker))))

(defn defaults
  "Return immutable native defaults shared by context's session; owns no resources."
  [current-context]
  (select-keys (.-defaults ^Session (context/owner current-context)) [:device :dtype :engine :seed]))

(defn identifier
  "Return the owning session's generated identifier, including after close."
  [current-context]
  (.-identifier ^Session (context/owner current-context)))

(defn schema
  "Return the owning session's generated schema name, including after close."
  [current-context]
  (.-schema ^Session (context/owner current-context)))

(defn next-table-name!
  "Allocate a unique generated table identifier while the context remains open."
  [current-context]
  (context/call current-context
                #(str "t_" (swap! (.-next-table-identifier ^Session (context/owner current-context)) unchecked-inc))))

(defn closed?
  "Whether a Session or Worker has closed; a Worker also reflects session close."
  [current-context]
  (let [^Session current-session (context/owner current-context)]
    (or @(.-closed current-session)
        (and (instance? Worker current-context) @(.-closed ^Worker current-context)))))

(defn worker-count
  "Return the number of workers still awaiting native cleanup."
  [current-context]
  (count @(.-workers ^Session (context/owner current-context))))

(defn- resource-kind [kind]
  (or (#{:network :model :table :trainer} kind)
      (invalid! :session/invalid-kind "Resource kind must be :network, :model, :table, or :trainer" kind)))

(defn- named-resource [^Session current-session resource-name]
  (let [^ReentrantReadWriteLock registry-lock (.-registry-lock current-session)]
    (.lock (.readLock registry-lock))
    (try
      (get @(.-registry current-session) resource-name)
      (finally (.unlock (.readLock registry-lock))))))

(defn- install-resource! [^Session current-session resource-name kind value owned?]
  (let [identifier (swap! (.-next-resource-identifier current-session) unchecked-inc)
        resource (Resource. resource-name identifier kind value owned? (ReentrantReadWriteLock. true) (atom false))]
    (swap! (.-registry current-session) assoc (or resource-name identifier) resource)
    resource))

(defn- replace-resource! [^Session current-session resource-name kind value owned? releasing-context]
  ;; Resource-before-registry ordering lets in-flight callbacks resolve other names.
  ;; Retry only when a concurrent registry mutation changed the admission candidate.
  (let [^ReentrantReadWriteLock registry-lock (.-registry-lock current-session)]
    (loop []
      (let [^Resource previous (named-resource current-session resource-name)
            attempt (fn []
                      (when previous (lock-for-release! previous))
                      (try
                        (.lock (.writeLock registry-lock))
                        (try
                          (if-not (identical? previous (get @(.-registry current-session) resource-name))
                            ::retry
                            (do
                              (when previous
                                (swap! (.-registry current-session) dissoc resource-name)
                                (close-resource! previous releasing-context))
                              (install-resource! current-session resource-name kind value owned?)))
                          (finally (.unlock (.writeLock registry-lock))))
                        (finally
                          (when previous (.unlock (.writeLock ^ReentrantReadWriteLock (.-resource-lock previous)))))))
            result (if previous
                     (release-scope (.-kind previous) (.-value previous) releasing-context attempt)
                     (attempt))]
        (if (= ::retry result) (recur) result)))))

(defn register!
  "Register value in context's session and return it.
  :kind is one of :network, :model, :table, or :trainer. Optional :name is a keyword;
  :owned? defaults to true and :replace? to false. Owned values close on
  release/session close; borrowed ones do not. Replacement releases the previous
  value before registering the new one; owned tables drop on context's connection."
  [current-context value options]
  (options! options #{:kind :name :owned? :replace?})
  (let [kind (resource-kind (:kind options))
        resource-name (:name options)
        owned? (get options :owned? true)
        replace? (get options :replace? false)]
    (when-not (or (nil? resource-name) (keyword? resource-name))
      (invalid! :session/invalid-name "Resource names must be keywords" resource-name))
    (when-not (boolean? owned?)
      (invalid! :session/invalid-options ":owned? must be boolean" owned?))
    (when-not (boolean? replace?)
      (invalid! :session/invalid-options ":replace? must be boolean" replace?))
    (context/call
     current-context
     (fn []
       (when (nil? value)
         (throw (ex-info "A resource value cannot be null" {:error :session/invalid-resource})))
       (let [^Session current-session (context/owner current-context)
             ^ReentrantReadWriteLock registry-lock (.-registry-lock current-session)]
         (if (and resource-name replace?)
           (replace-resource! current-session resource-name kind value owned? current-context)
           (do
             (.lock (.writeLock registry-lock))
             (try
               (when (and resource-name (contains? @(.-registry current-session) resource-name))
                 (throw (ex-info "Resource name is already registered"
                                 {:error :session/duplicate-name :name resource-name})))
               (install-resource! current-session resource-name kind value owned?)
               (finally (.unlock (.writeLock registry-lock)))))))))
    value))

(defn- lookup-resource [registry value]
  (some (fn [^Resource resource] (when (identical? (.-value resource) value) resource)) (vals registry)))

(defn resource
  "Resolve a keyword name or identity-look up an object in context's registry.
  Unregistered objects return nil and remain borrowed; names must match kind."
  ^Resource [current-context value kind]
  (resource-kind kind)
  (context/call
   current-context
   (fn []
     (let [^Session current-session (context/owner current-context)
           ^ReentrantReadWriteLock registry-lock (.-registry-lock current-session)]
       (.lock (.readLock registry-lock))
       (try
         (let [registry @(.-registry current-session)
               ^Resource registered (if (keyword? value) (get registry value) (lookup-resource registry value))]
           (when (and (keyword? value) (nil? registered))
             (throw (ex-info "Resource is not registered" {:error :session/unknown-name :name value})))
           (when (and (keyword? value) (not= kind (.-kind registered)))
             (throw (ex-info "Resource has a different kind"
                             {:error :session/wrong-kind :name value :kind (.-kind registered) :expected kind})))
           registered)
         (finally (.unlock (.readLock registry-lock))))))))

(defn value
  "Resolve a keyword resource name; object values pass through unchanged.
  Does not transfer ownership; a named resource must match kind."
  [current-context object-or-name kind]
  (if (keyword? object-or-name)
    (.-value ^Resource (resource current-context object-or-name kind))
    object-or-name))

(defn shared
  "Invoke operation on a borrowed value under context's lifecycle read lock.
  Registered values additionally use their shared resource lock; callers must
  synchronize unregistered borrowed values themselves."
  [current-context object-or-name kind operation]
  (context/call current-context
                #(if-let [registered (resource current-context object-or-name kind)]
                   (shared-resource registered operation)
                   (operation object-or-name))))

(defn exclusive
  "Invoke operation on a borrowed value under context's lifecycle read lock.
  Registered values additionally use their exclusive mutation lock; callers must
  synchronize unregistered borrowed values themselves."
  [current-context object-or-name kind operation]
  (context/call current-context
                #(if-let [registered (resource current-context object-or-name kind)]
                   (exclusive-resource registered operation)
                   (operation object-or-name))))

(defn release!
  "Unregister and release a keyword name or registered object in context's session.
  Closes owned AutoCloseable values; owned tables drop on context's connection.
  Borrowed values remain caller-owned."
  [current-context object-or-name]
  (context/call
   current-context
   (fn []
     (let [^Session current-session (context/owner current-context)
           ^ReentrantReadWriteLock registry-lock (.-registry-lock current-session)]
       (loop []
         (let [^Resource registered (do
                                      (.lock (.readLock registry-lock))
                                      (try
                                        (let [registry @(.-registry current-session)]
                                          (if (keyword? object-or-name)
                                            (get registry object-or-name)
                                            (lookup-resource registry object-or-name)))
                                        (finally (.unlock (.readLock registry-lock)))))
               _ (when-not registered
                   (throw (ex-info "Resource is not registered"
                                   {:error :session/unknown-name :name object-or-name})))
               result (release-scope
                       (.-kind registered) (.-value registered) current-context
                       (fn []
                         (lock-for-release! registered)
                         (try
                           (.lock (.writeLock registry-lock))
                           (try
                             (let [registry-key (or (.-name registered) (.-identifier registered))]
                               (if-not (identical? registered (get @(.-registry current-session) registry-key))
                                 ::retry
                                 (do
                                   (swap! (.-registry current-session) dissoc registry-key)
                                   (close-resource! registered current-context))))
                             (finally (.unlock (.writeLock registry-lock))))
                           (finally (.unlock (.writeLock ^ReentrantReadWriteLock (.-resource-lock registered)))))))]
           (when (= ::retry result) (recur)))))))
  nil)

(defn- device-data [^Device device]
  (if (.isGpu device) [:gpu (.getDeviceId device)] :cpu))

(defn describe
  "Inspect Session, Worker or Resource as plain data without transferring ownership.
  Inspection remains available after closing a context or releasing a resource."
  [value]
  (cond
    (instance? Session value)
    (let [^Session current-session value
          native-defaults (.-defaults current-session)]
      {:identifier (identifier current-session)
       :status (if (closed? current-session) :closed :open)
       :database (:database native-defaults)
       :device (device-data (:device native-defaults))
       :dtype (keyword (string/lower-case (.name ^DataType (:dtype native-defaults))))
       :engine (:engine native-defaults)
       :seed (:seed native-defaults)
       :schema (schema current-session)
       :workers (worker-count current-session)
       :resources (mapv describe (resources current-session))})

    (instance? Worker value)
    {:session-identifier (identifier value)
     :status (if (closed? value) :closed :open)}

    (instance? Resource value)
    (let [^Resource registered value]
      (merge {:identifier (.-identifier registered)
              :name (.-name registered)
              :kind (.-kind registered)
              :owned? (.-owned? registered)
              :released? @(.-released registered)}
             (describe-value (.-kind registered) (.-value registered))))

    :else
    (invalid! :session/invalid-description "Expected Session, Worker, or Resource" value)))

(defmethod print-method Session [value ^Writer writer]
  (.write writer "#garun/session ")
  (print-method (describe value) writer))

(defmethod print-method Worker [value ^Writer writer]
  (.write writer "#garun/worker ")
  (print-method (describe value) writer))

(defmethod print-method Resource [value ^Writer writer]
  (.write writer "#garun/resource ")
  (print-method (describe value) writer))
