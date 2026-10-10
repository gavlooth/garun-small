(ns garun.session
  "Session-owned DuckDB connections, tensor managers and named native resources.

  Close a Session with with-open: it closes workers, then owned resources in
  reverse registration order, drops its private schema, closes the root manager,
  and finally closes its connection. Borrowed values are never closed. Tensors
  returned through a context belong to that context's manager.

  Use one thread per context and one Worker per additional thread. Workers own
  duplicated connections and tensor sub-managers; predictions may share a
  registered network while training takes its exclusive resource lock.

  The small Java leaf owns lifecycle, registry and resource-lock invariants that
  span blocking JDBC/DJL calls. Reentrant read/write locks exclude cleanup without
  mutable Clojure state or a coordination loop. Generated schema/table identifiers
  are the only values used in the leaf's fixed lifecycle DDL."
  (:require [clojure.core.protocols :as protocols]
            [clojure.string :as string]
            [garun.tensor :as tensor])
  (:import [ai.djl Device]
           [ai.djl.engine Engine]
           [ai.djl.ndarray NDManager]
           [ai.djl.ndarray.types DataType]
           [garun.session Context Defaults Kind MaterializedTable Resource Session Worker]
           [java.io Writer]
           [java.lang AutoCloseable]
           [java.sql DriverManager]
           [org.duckdb DuckDBConnection]))

(set! *warn-on-reflection* true)

(defn- invalid! [error message value]
  (throw (ex-info message {:error error :value value})))

(defn- options! [options allowed]
  (when-not (map? options)
    (invalid! :session/invalid-options "Session options must be a map" options))
  (when-let [unknown (seq (remove allowed (keys options)))]
    (invalid! :session/unknown-option "Unknown session option" (vec unknown))))

(defn- close-after-failure! [^AutoCloseable resource ^Throwable failure]
  (try
    (.close resource)
    (catch Throwable close-failure
      (when-not (identical? failure close-failure)
        (.addSuppressed failure close-failure)))))

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
           database-label (if (= :memory database) ":memory" database)
           native-defaults (Defaults. resolved-device resolved-data-type resolved-engine (long seed) database-label)
           ^DuckDBConnection connection (DriverManager/getConnection
                                         (str "jdbc:duckdb:" (when-not (= :memory database) database)))
           manager (try
                     (NDManager/newBaseManager resolved-device resolved-engine)
                     (catch Throwable failure
                       (close-after-failure! connection failure)
                       (throw failure)))]
       ;; Session/open assumes ownership and closes both natives if schema creation fails.
       (Session/open connection manager native-defaults)))))

(defn context?
  "Whether value is a Session or Worker implementing the native Context contract."
  [value]
  (instance? Context value))

(defn worker
  "Create a session-owned per-thread Worker by duplicating this context's connection.
  A Worker can create another Worker without using the session's root connection.
  Close it with with-open when its thread finishes; the session also closes it."
  ^Worker [^Context context]
  (.call context #(.worker (.session context) context)))

(defn defaults
  "Return immutable native defaults shared by context's session; owns no resources."
  [^Context context]
  (let [^Defaults native-defaults (.defaults (.session context))]
    {:device (.-device native-defaults)
     :dtype (.-dataType native-defaults)
     :engine (.-engine native-defaults)
     :seed (.-seed native-defaults)}))

(def ^:private kinds
  {:network Kind/NETWORK :model Kind/MODEL :table Kind/TABLE :trainer Kind/TRAINER})

(defn- resource-kind [kind]
  (or (get kinds kind)
      (invalid! :session/invalid-kind "Resource kind must be :network, :model, :table, or :trainer" kind)))

(defn register!
  "Register value in context's session and return it.
  :kind is one of :network, :model, :table, or :trainer. Optional :name is a keyword;
  :owned? defaults to true and :replace? to false. Owned values close on
  release/session close; borrowed ones do not. Replacement releases the previous
  value before registering the new one; owned tables drop on context's connection."
  [^Context context value options]
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
    (.call context #(.register (.session context) resource-name kind value owned? replace? context))
    value))

(defn resource
  "Resolve a keyword name or identity-look up an object in context's registry.
  Unregistered objects return nil and remain borrowed; names must match kind."
  ^Resource [^Context context value kind]
  (let [native-kind (resource-kind kind)]
    (.call context
           #(if (keyword? value)
              (.resolve (.session context) value native-kind)
              (.lookup (.session context) value)))))

(defn value
  "Resolve a keyword resource name; object values pass through unchanged.
  Does not transfer ownership; a named resource must match kind."
  [^Context context object-or-name kind]
  (if (keyword? object-or-name)
    (.-value ^Resource (resource context object-or-name kind))
    object-or-name))

(defn shared
  "Invoke operation on a borrowed value under context's lifecycle read lock.
  Registered values additionally use their shared resource lock; callers must
  synchronize unregistered borrowed values themselves."
  [^Context context object-or-name kind operation]
  (.call context
         #(if-let [^Resource registered (resource context object-or-name kind)]
            (.shared registered operation)
            (operation object-or-name))))

(defn exclusive
  "Invoke operation on a borrowed value under context's lifecycle read lock.
  Registered values additionally use their exclusive mutation lock; callers must
  synchronize unregistered borrowed values themselves."
  [^Context context object-or-name kind operation]
  (.call context
         #(if-let [^Resource registered (resource context object-or-name kind)]
            (.exclusive registered operation)
            (operation object-or-name))))

(defn release!
  "Unregister and release a keyword name or registered object in context's session.
  Closes owned AutoCloseable values; owned tables drop on context's connection.
  Borrowed values remain caller-owned."
  [^Context context object-or-name]
  (.call context #(.release (.session context) object-or-name context))
  nil)

(defmulti describe-value
  "Return plain, non-owning data about a resource value, dispatched by kind."
  (fn [kind _value] kind))

(defmethod describe-value :default [_kind value]
  {:class (.getName ^Class (class value))})

(defmethod describe-value :table [_kind ^MaterializedTable table]
  {:sql-name (str (.-schema table) "." (.-table table))
   :columns (.-columns table)
   :rows (.-rows table)})

(defn- device-data [^Device device]
  (if (.isGpu device) [:gpu (.getDeviceId device)] :cpu))

(defn describe
  "Inspect Session, Worker or Resource as plain data without transferring ownership.
  Inspection remains available after closing a context or releasing a resource."
  [value]
  (cond
    (instance? Session value)
    (let [^Session current-session value
          ^Defaults native-defaults (.defaults current-session)]
      {:identifier (.identifier current-session)
       :status (if (.isClosed current-session) :closed :open)
       :database (.-database native-defaults)
       :device (device-data (.-device native-defaults))
       :dtype (keyword (string/lower-case (.name ^DataType (.-dataType native-defaults))))
       :engine (.-engine native-defaults)
       :seed (.-seed native-defaults)
       :schema (.schema current-session)
       :workers (.workerCount current-session)
       :resources (mapv describe (.resources current-session))})

    (instance? Worker value)
    (let [^Worker current-worker value
          ^Session current-session (.session current-worker)]
      {:session-identifier (.identifier current-session)
       :status (if (or (.isClosed current-worker) (.isClosed current-session)) :closed :open)})

    (instance? Resource value)
    (let [^Resource current-resource value
          kind (keyword (string/lower-case (.name (.-kind current-resource))))]
      (merge {:identifier (.-identifier current-resource)
              :name (.-name current-resource)
              :kind kind
              :owned? (.-owned current-resource)
              :released? (.isReleased current-resource)}
             (describe-value kind (.-value current-resource))))

    :else
    (invalid! :session/invalid-description "Expected Session, Worker, or Resource" value)))

(extend-protocol protocols/Datafiable
  Session
  (datafy [value] (describe value))

  Worker
  (datafy [value] (describe value))

  Resource
  (datafy [value] (describe value)))

(defmethod print-method Session [value ^Writer writer]
  (.write writer "#garun/session ")
  (print-method (describe value) writer))

(defmethod print-method Worker [value ^Writer writer]
  (.write writer "#garun/worker ")
  (print-method (describe value) writer))

(defmethod print-method Resource [value ^Writer writer]
  (.write writer "#garun/resource ")
  (print-method (describe value) writer))
