(ns garun.session
  "Session-owned DuckDB connections, tensor managers and named native resources.

  Close a Session with with-open: it closes workers, then owned resources in
  reverse registration order, drops its private schema, closes the root manager,
  and finally closes its connection. Borrowed values are never closed. Tensors
  returned through a context belong to that context's manager.

  Use one thread per context and one Worker per additional thread. Workers own
  duplicated connections and tensor sub-managers; predictions may share a
  registered network while training takes its exclusive resource lock.

  Every path takes locks in one order: the session lifecycle lock, then the
  Resource a value was registered :after, then the value's own Resource, then
  the registry lock. A resource registered :after another is always locked in
  the same mode as that other one first, so users of a trainer and users of its
  network cannot wait on each other in opposite orders. Releasing a resource
  first releases everything registered after it.

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
           [java.lang.ref WeakReference]
           [java.sql Connection DriverManager]
           [java.util UUID]
           [java.util.concurrent.locks Lock ReentrantReadWriteLock]
           [org.duckdb DuckDBConnection]))

(set! *warn-on-reflection* true)

(declare admitted close-session! close-worker! close-resource! describe)

(defmacro ^:private with-lock
  "Run body holding lock; the lock is released when body returns or throws."
  [lock & body]
  `(let [^Lock lock# ~lock]
     (.lock lock#)
     (try
       ~@body
       (finally (.unlock lock#)))))

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
                  ^Atom closed
                  ;; Weak references to owned values this session released, so that reusing
                  ;; one is reported instead of being run as a caller's unregistered object.
                  ^Atom released-values]
  context/Context
  (connection [_] connection)
  (manager [_] manager)
  (owner [current-session] current-session)
  (call [current-session operation] (admitted current-session operation))
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
    (admitted parent
              (fn []
                (with-lock (.readLock lifecycle-lock)
                  (when @closed
                    (throw (ex-info "Worker is closed" {:error :session/closed :worker true})))
                  (operation)))))
  AutoCloseable
  (close [current-worker] (close-worker! current-worker))
  protocols/Datafiable
  (datafy [current-worker] (describe current-worker)))

(deftype Resource [name identifier kind value owned?
                   ;; The Resource this one was registered :after, or nil. Its lock is taken first.
                   after
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
      :after after
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

(defn- admitted
  "Run operation while the session is open; closing the session waits for it."
  [^Session current-session operation]
  (with-lock (.readLock ^ReentrantReadWriteLock (.-lifecycle-lock current-session))
    (when @(.-closed current-session)
      (throw (ex-info "Session is closed"
                      {:error :session/closed :identifier (.-identifier current-session)})))
    (operation)))

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

(def ^:private exclusive-lock-refusals
  {:exclusive [:session/exclusive-inside-shared
               "Cannot take exclusive access to a resource inside its shared operation"]
   :release [:session/close-inside-operation
             "Cannot release a resource inside its shared operation"]})

(defn- holding-own-lock
  "Run operation holding resource's own lock: shared for :shared, exclusive for
  :exclusive and :release. A thread inside a shared operation on resource is
  refused instead of waiting forever for its own shared lock to be released."
  [^Resource resource mode operation]
  (let [^ReentrantReadWriteLock resource-lock (.-resource-lock resource)]
    (if (= :shared mode)
      (with-lock (.readLock resource-lock) (operation))
      (do
        (when (and (pos? (.getReadHoldCount resource-lock))
                   (not (.isWriteLockedByCurrentThread resource-lock)))
          (let [[error message] (get exclusive-lock-refusals mode)]
            (throw (ex-info message {:error error :name (.-name resource)}))))
        (with-lock (.writeLock resource-lock) (operation))))))

(defn- holding
  "Run operation holding, in the same mode, the lock of every resource this one
  was registered after, the earliest registration outermost, then its own lock."
  [^Resource resource mode operation]
  (if-let [dependency (.-after resource)]
    (holding dependency mode #(holding-own-lock resource mode operation))
    (holding-own-lock resource mode operation)))

(defn- close-resource! [^Resource resource releasing-context]
  (holding-own-lock
   resource :release
   (fn []
     (when-not @(.-released resource)
       (reset! (.-released resource) true)
       (when (.-owned? resource)
         (release-value! (.-kind resource) (.-value resource) releasing-context))))))

(defn- released-failure [^Resource resource]
  (ex-info "Resource has been released"
           {:error :session/released :name (.-name resource) :kind (.-kind resource)}))

(defn- resource-call [^Resource resource mode operation]
  (holding resource mode
           (fn []
             (when @(.-released resource)
               (throw (released-failure resource)))
             (operation (.-value resource)))))

(defn shared-resource
  "Run an operation under a registered Resource's shared lock, rejecting release.
  The resource it was registered :after is locked shared first."
  [resource operation]
  (resource-call resource :shared operation))

(defn exclusive-resource
  "Run an operation under a registered Resource's exclusive lock, rejecting release.
  The resource it was registered :after is locked exclusively first. Calling this
  inside a shared operation on the same resource throws
  :session/exclusive-inside-shared."
  [resource operation]
  (resource-call resource :exclusive operation))

(defn resources
  "Snapshot a session's registry in registration order, including after close."
  [current-context]
  (let [^Session current-session (context/owner current-context)]
    (with-lock (.readLock ^ReentrantReadWriteLock (.-registry-lock current-session))
      (->> @(.-registry current-session) vals (sort-by #(.-identifier ^Resource %)) vec))))

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
    (with-lock (.writeLock lifecycle-lock)
      (when-not @(.-closed current-worker)
        (reset! (.-closed current-worker) true)
        (let [failure (-> nil
                          (close-after! (.-manager current-worker))
                          (close-after! (.-connection current-worker)))]
          (swap! (.-workers ^Session (.-parent current-worker)) disj current-worker)
          (when failure (throw failure)))))))

(defn- close-session! [^Session current-session]
  (let [^ReentrantReadWriteLock lifecycle-lock (.-lifecycle-lock current-session)]
    (when (pos? (.getReadHoldCount lifecycle-lock))
      (throw (ex-info "Cannot close a session inside an operation"
                      {:error :session/close-inside-operation :identifier (.-identifier current-session)})))
    (with-lock (.writeLock lifecycle-lock)
      (when-not @(.-closed current-session)
        (reset! (.-closed current-session) true)
        (let [worker-failure (reduce close-after! nil @(.-workers current-session))
              snapshot (resources current-session)
              _ (with-lock (.writeLock ^ReentrantReadWriteLock (.-registry-lock current-session))
                  (reset! (.-registry current-session) {}))
              resource-failure (reduce close-after! worker-failure (rseq snapshot))
              schema-failure (try
                               (schema-ddl! (.-connection current-session) :garun/drop-schema
                                            (.-schema current-session))
                               resource-failure
                               (catch Throwable failure (append-failure! resource-failure failure)))
              failure (-> schema-failure
                          (close-after! (.-manager current-session))
                          (close-after! (.-connection current-session)))]
          (when failure (throw failure)))))))

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
                                     (atom {}) (atom #{}) (atom 0) (atom 0) (atom false) (atom []))]
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

(defn- registry-key [^Resource resource]
  (or (.-name resource) (.-identifier resource)))

(defn- released-by-session? [^Session current-session object]
  (boolean (some #(identical? object (.get ^WeakReference %)) @(.-released-values current-session))))

(defn- registered
  "Return the Resource registered under a keyword name or holding this exact
  object, or nil for an object this session never registered: its caller owns
  and synchronizes it. Unknown names, owned objects this session has released,
  and a kind other than the non-nil expected kind throw."
  ^Resource [^Session current-session object-or-name kind]
  (let [named? (keyword? object-or-name)
        ^Resource found (with-lock (.readLock ^ReentrantReadWriteLock (.-registry-lock current-session))
                          (let [registry @(.-registry current-session)]
                            (if named?
                              (get registry object-or-name)
                              (some (fn [^Resource resource]
                                      (when (identical? (.-value resource) object-or-name) resource))
                                    (vals registry)))))]
    (cond
      (and (nil? found) named?)
      (throw (ex-info "Resource is not registered" {:error :session/unknown-name :name object-or-name}))

      (and (nil? found) (released-by-session? current-session object-or-name))
      (throw (ex-info "Resource has been released" {:error :session/released :kind kind}))

      (nil? found) nil

      (and kind (not= kind (.-kind found)))
      (throw (ex-info "Resource has a different kind"
                      {:error :session/wrong-kind :name (.-name found) :kind (.-kind found) :expected kind}))

      :else found)))

(defn- remember-released!
  "Record the owned values of resources leaving the registry; the caller holds
  the registry's exclusive lock. References whose value was collected are dropped."
  [^Session current-session resources]
  (swap! (.-released-values current-session)
         (fn [references]
           (into (filterv #(some? (.get ^WeakReference %)) references)
                 (comp (filter #(.-owned? ^Resource %))
                       (map #(WeakReference. (.-value ^Resource %))))
                 resources))))

(defn- registered-after
  "List, in registration order, the resources registered directly after resource."
  [^Session current-session ^Resource resource]
  (filterv #(identical? resource (.-after ^Resource %)) (resources current-session)))

(defn- holding-registered-after
  "Run (operation locked) with every resource registered after resource, directly
  or through another, exclusively locked. The caller already holds resource's
  exclusive lock, which stops new registrations after it, so each listing is
  complete once its own dependency is locked. locked starts with resource and
  lists every other resource later than the one it was registered after."
  [^Session current-session ^Resource resource operation]
  (letfn [(hold [pending locked]
            (if-let [dependent (first pending)]
              (holding-own-lock dependent :release
                                #(hold (into (vec (rest pending))
                                             (registered-after current-session dependent))
                                       (conj locked dependent)))
              (operation locked)))]
    (hold (registered-after current-session resource) [resource])))

(defn- holding-new-dependency
  "Run operation holding the shared lock of the resource a new registration
  goes after, so that resource cannot be released before the registration is in
  the registry. nil means the new registration goes after nothing."
  [^Resource after operation]
  (if after
    (holding after :shared operation)
    (operation)))

(defn- install-resource!
  "Add a new registry entry. The caller holds the registry's exclusive lock and,
  when after is given, after's shared lock."
  [^Session current-session resource-name kind value owned? ^Resource after]
  (when (and after @(.-released after))
    (throw (released-failure after)))
  (let [identifier (swap! (.-next-resource-identifier current-session) unchecked-inc)
        resource (Resource. resource-name identifier kind value owned? after
                            (ReentrantReadWriteLock. true) (atom false))]
    (swap! (.-registry current-session) assoc (or resource-name identifier) resource)
    resource))

;; Returned instead of a result when the registry changed between looking a
;; resource up and locking it, so the caller looks it up again.
(def ^:private look-up-again (Object.))

(defn- remove-and-release!
  "Remove resource and everything registered after it from the registry, release
  them (later registrations first), then return (afterwards) while the registry
  is still exclusively locked. new-dependency is the Resource a replacement goes
  after, or nil. Returns look-up-again when resource is no longer the registered
  entry. A failed release leaves the entries removed and propagates without
  calling afterwards."
  [^Session current-session ^Resource resource releasing-context ^Resource new-dependency afterwards]
  (holding
   resource :release
   (fn []
     (holding-registered-after
      current-session resource
      (fn [locked]
        (holding-new-dependency
         new-dependency
         (fn []
           (with-lock (.writeLock ^ReentrantReadWriteLock (.-registry-lock current-session))
             (if-not (identical? resource (get @(.-registry current-session) (registry-key resource)))
               look-up-again
               (do
                 (swap! (.-registry current-session) #(apply dissoc % (map registry-key locked)))
                 (remember-released! current-session locked)
                 (when-let [failure (reduce (fn [failure dependent]
                                              (try
                                                (close-resource! dependent releasing-context)
                                                failure
                                                (catch Throwable release-failure
                                                  (append-failure! failure release-failure))))
                                            nil (rseq locked))]
                   (throw failure))
                 (afterwards)))))))))))

(defn- replace-resource!
  "Release whatever is registered under resource-name and register value in its
  place as one registry change; retries when another thread changed that entry."
  [^Session current-session resource-name kind value owned? ^Resource after releasing-context]
  (let [install #(install-resource! current-session resource-name kind value owned? after)]
    (loop []
      (let [^Resource previous (with-lock (.readLock ^ReentrantReadWriteLock (.-registry-lock current-session))
                                 (get @(.-registry current-session) resource-name))
            result (if previous
                     (remove-and-release! current-session previous releasing-context after install)
                     (holding-new-dependency
                      after
                      (fn []
                        (with-lock (.writeLock ^ReentrantReadWriteLock (.-registry-lock current-session))
                          (if (contains? @(.-registry current-session) resource-name)
                            look-up-again
                            (install))))))]
        (if (identical? look-up-again result)
          (recur)
          result)))))

(defn register!
  "Register value in context's session and return it.
  :kind is one of :network, :model, :table, or :trainer. Optional :name is a keyword;
  :owned? defaults to true and :replace? to false. Owned values close on
  release/session close; borrowed ones do not. Replacement releases the previous
  value before registering the new one; owned tables drop on context's connection.

  Optional :after names a registered value (keyword or object) that every user of
  the new value needs too, as a trainer needs its network. Operations on the new
  value then lock that one first in the same mode, and releasing or replacing that
  one releases the new value first."
  [current-context value options]
  (options! options #{:kind :name :owned? :replace? :after})
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
    (when (and (contains? options :after) (nil? (:after options)))
      (invalid! :session/invalid-options ":after must be a registered name or object" nil))
    (context/call
     current-context
     (fn []
       (when (nil? value)
         (throw (ex-info "A resource value cannot be null" {:error :session/invalid-resource})))
       (let [^Session current-session (context/owner current-context)
             after (when (contains? options :after)
                     (or (registered current-session (:after options) nil)
                         (throw (ex-info ":after must be registered in this session"
                                         {:error :session/unregistered}))))]
         (if (and resource-name replace?)
           (replace-resource! current-session resource-name kind value owned? after current-context)
           (holding-new-dependency
            after
            (fn []
              (with-lock (.writeLock ^ReentrantReadWriteLock (.-registry-lock current-session))
                (when (and resource-name (contains? @(.-registry current-session) resource-name))
                  (throw (ex-info "Resource name is already registered"
                                  {:error :session/duplicate-name :name resource-name})))
                (install-resource! current-session resource-name kind value owned? after))))))))
    value))

(defn resource
  "Return the Resource registered under a keyword name or holding this exact
  object, checking its kind. Objects this session never registered return nil
  and remain their caller's. Unknown names throw :session/unknown-name; owned
  objects this session has released throw :session/released."
  ^Resource [current-context value kind]
  (resource-kind kind)
  (context/call current-context
                #(registered (context/owner current-context) value kind)))

(defn value
  "Resolve a keyword resource name, checking its kind; objects pass through
  unchanged unless this session has released them. Does not transfer ownership."
  [current-context object-or-name kind]
  (if-let [found (resource current-context object-or-name kind)]
    (.-value found)
    object-or-name))

(defn- registered-call
  "Run operation on a value under context's lifecycle lock and, when the value
  is registered, its resource locks in mode. A keyword name whose resource was
  replaced between lookup and locking is looked up again. Objects this session
  never registered run unlocked: their caller synchronizes them."
  [current-context object-or-name kind mode operation]
  (resource-kind kind)
  (context/call
   current-context
   (fn []
     (let [^Session current-session (context/owner current-context)]
       (loop []
         (let [^Resource found (registered current-session object-or-name kind)
               result (if found
                        (holding found mode
                                 #(if @(.-released found)
                                    look-up-again
                                    (operation (.-value found))))
                        (operation object-or-name))]
           (cond
             (not (identical? look-up-again result)) result
             (keyword? object-or-name) (recur)
             :else (throw (released-failure found)))))))))

(defn shared
  "Invoke operation on a value under context's lifecycle lock. Registered values
  (keyword name or object) additionally hold their shared resource lock; callers
  must synchronize objects they never registered themselves. Owned objects this
  session has released throw :session/released."
  [current-context object-or-name kind operation]
  (registered-call current-context object-or-name kind :shared operation))

(defn exclusive
  "Invoke operation on a value under context's lifecycle lock. Registered values
  (keyword name or object) additionally hold their exclusive resource lock;
  callers must synchronize objects they never registered themselves. Owned
  objects this session has released throw :session/released; calling this
  inside a shared operation on the same value throws
  :session/exclusive-inside-shared."
  [current-context object-or-name kind operation]
  (registered-call current-context object-or-name kind :exclusive operation))

(defn release!
  "Unregister and release a keyword name or registered object in context's session,
  after releasing everything registered :after it. Closes owned AutoCloseable
  values; owned tables drop on context's connection. Borrowed values remain
  caller-owned."
  [current-context object-or-name]
  (context/call
   current-context
   (fn []
     (let [^Session current-session (context/owner current-context)]
       (loop []
         (let [found (or (registered current-session object-or-name nil)
                         (throw (ex-info "Resource is not registered"
                                         {:error :session/unknown-name :name object-or-name})))]
           (when (identical? look-up-again
                             (remove-and-release! current-session found current-context nil (constantly nil)))
             (recur)))))))
  nil)

(defn- device-data [^Device device]
  (if (.isGpu device) [:gpu (.getDeviceId device)] :cpu))

(defn describe
  "Inspect Session, Worker or Resource as plain data without transferring ownership.
  Inspection remains available after closing a context or releasing a resource.
  A resource registered :after another reports that one's name or identifier."
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
    (let [^Resource found value]
      (merge (cond-> {:identifier (.-identifier found)
                      :name (.-name found)
                      :kind (.-kind found)
                      :owned? (.-owned? found)
                      :released? @(.-released found)}
               (.-after found) (assoc :after (registry-key (.-after found))))
             (describe-value (.-kind found) (.-value found))))

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
