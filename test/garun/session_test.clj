(ns garun.session-test
  "Behavioral boundaries for session ownership and the Java concurrency leaf."
  (:require [clojure.datafy :as datafy]
            [clojure.string :as string]
            [clojure.test :refer [deftest is]]
            [garun.session :as session])
  (:import [ai.djl Device]
           [ai.djl.ndarray NDManager]
           [ai.djl.ndarray.types DataType]
           [garun.session MaterializedTable Resource Session]
           [java.lang AutoCloseable]
           [java.nio.file Files]
           [java.nio.file.attribute FileAttribute]
           [java.sql Connection DriverManager]
           [java.util.concurrent CompletableFuture ConcurrentLinkedQueue CountDownLatch TimeUnit TimeoutException]
           [java.util.function Supplier]))

(set! *warn-on-reflection* true)

(defn- error-code [operation]
  (try
    (operation)
    nil
    (catch clojure.lang.ExceptionInfo failure
      (:error (ex-data failure)))))

(defn- recorded-resource [^ConcurrentLinkedQueue recording label]
  (reify AutoCloseable
    (close [_] (.add recording label))))

(defn- execute! [^Connection connection ^String statement-text]
  (with-open [statement (.createStatement connection)]
    (.execute statement statement-text)))

(defn- scalar [^Connection connection ^String query-text]
  (with-open [statement (.createStatement connection)
              result (.executeQuery statement query-text)]
    (when (.next result)
      (.getLong result (int 1)))))

(defn- materialized-table! ^MaterializedTable [^Session current-session]
  (let [table-name (.nextTableName current-session)
        schema-name (.schema current-session)]
    (execute! (.connection current-session)
              (str "CREATE TABLE \"" schema-name "\".\"" table-name "\" AS SELECT 7::BIGINT AS number"))
    (MaterializedTable. current-session schema-name table-name [{:name "number" :type :int64}] (long 1))))

(deftest session-defaults-and-closed-options
  (with-open [current-session (session/open)]
    (let [{:keys [device engine seed] data-type :dtype :as defaults} (session/defaults current-session)]
      (is (session/context? current-session))
      (is (not (session/context? (Object.))))
      (is (= #{:device :dtype :engine :seed} (set (keys defaults))))
      (is (= (Device/cpu) device))
      (is (= DataType/FLOAT32 data-type))
      (is (= 11 seed))
      (is (= (.getEngineName (.getEngine (.manager current-session))) engine))
      (is (= ":memory" (:database (session/describe current-session))))
      (is (= {:class "java.lang.Object"} (session/describe-value :unregistered (Object.))))
      (is (identical? current-session (.session current-session)))
      (is (.isOpen (.manager current-session)))
      (is (not (.isClosed (.connection current-session))))))
  (with-open [current-session (session/open {:device :cpu :dtype :float64 :engine "PyTorch" :seed 37})]
    (is (= {:device (Device/cpu) :dtype DataType/FLOAT64 :engine "PyTorch" :seed 37}
           (session/defaults current-session))))
  (is (= :session/unknown-option (error-code #(session/open {:unknown true}))))
  (is (= :session/invalid-options (error-code #(session/open []))))
  (doseq [seed [nil 1.5 "11" (inc (bigint Long/MAX_VALUE))]]
    (is (= :session/invalid-seed (error-code #(session/open {:seed seed})))))
  (doseq [database [nil "" 42 :elsewhere]]
    (is (= :session/invalid-database (error-code #(session/open {:database database}))))))

(deftest inspectable-contexts-and-resource-data
  (let [current-session (session/open)]
    (try
      (with-open [current-worker (session/worker current-session)]
        (let [table (materialized-table! current-session)
              _ (session/register! current-session table {:kind :table :name :sample})
              resource (session/resource current-session :sample :table)
              session-description (session/describe current-session)
              worker-description (session/describe current-worker)
              resource-description (session/describe resource)]
          (is (= #{:id :status :database :device :dtype :engine :seed :schema :workers :resources}
                 (set (keys session-description))))
          (is (= :open (:status session-description)))
          (is (= 1 (:workers session-description)))
          (is (= :cpu (:device session-description)))
          (is (= :float32 (:dtype session-description)))
          (is (= [resource-description] (:resources session-description)))
          (is (= {:session-id (.id current-session) :status :open} worker-description))
          (is (= #{:id :name :kind :owned? :released? :sql-name :columns :rows}
                 (set (keys resource-description))))
          (is (= {:name :sample :kind :table :owned? true :released? false
                  :sql-name (str (.schema current-session) "." (.-table table))
                  :columns [{:name "number" :type :int64}] :rows 1}
                 (dissoc resource-description :id)))
          (doseq [[value description tag] [[current-session session-description "#garun/session "]
                                           [current-worker worker-description "#garun/worker "]
                                           [resource resource-description "#garun/resource "]]]
            (is (map? (datafy/datafy value)))
            (is (= description (datafy/datafy value)))
            (is (string/starts-with? (with-out-str (prn value)) (str tag "{"))))
          (session/release! current-session :sample)
          (is (:released? (session/describe resource)))))
      (.close current-session)
      (is (= :closed (:status (session/describe current-session))))
      (is (= [] (:resources (datafy/datafy current-session))))
      (finally
        (.close current-session)))))

(deftest named-anonymous-and-borrowed-resources
  (with-open [current-session (session/open)]
    (let [recording (ConcurrentLinkedQueue.)
          first-value (recorded-resource recording :first)
          replacement (recorded-resource recording :replacement)
          anonymous (recorded-resource recording :anonymous)
          borrowed (recorded-resource recording :borrowed)
          unregistered (Object.)]
      (is (identical? first-value
                      (session/register! current-session first-value {:kind :network :name :network})))
      (let [^Resource resource (session/resource current-session :network :network)]
        (is (identical? first-value (.-value resource)))
        (is (identical? resource (session/resource current-session first-value :network)))
        (is (identical? first-value (session/value current-session :network :network)))
        (is (identical? first-value (session/value current-session first-value :network)))
        (is (nil? (session/resource current-session unregistered :network)))
        (is (identical? unregistered (session/value current-session unregistered :network)))
        (is (= :session/duplicate-name
               (error-code #(session/register! current-session replacement {:kind :network :name :network}))))
        (is (.isEmpty recording))
        (is (= :session/wrong-kind (error-code #(session/resource current-session :network :model))))
        (is (= :session/unknown-name (error-code #(session/value current-session :missing :network))))
        (is (identical? replacement
                        (session/register! current-session replacement
                                           {:kind :network :name :network :replace? true})))
        (is (= [:first] (vec recording)))
        (is (.isReleased resource))
        (is (= :session/released (error-code #(.shared resource identity))))
        (is (= :session/released (error-code #(.exclusive resource identity)))))
      (session/register! current-session anonymous {:kind :network})
      (session/register! current-session borrowed {:kind :network :name :borrowed :owned? false})
      (let [^Resource anonymous-resource (session/resource current-session anonymous :network)
            resources (.resources current-session)]
        (is (nil? (.-name anonymous-resource)))
        (is (= 3 (count resources)))
        (is (= (sort (map #(.-id ^Resource %) resources))
               (map #(.-id ^Resource %) resources))))
      (is (nil? (session/release! current-session :network)))
      (is (= [:first :replacement] (vec recording)))
      (is (= :session/unknown-name (error-code #(session/resource current-session :network :network))))
      (is (= :session/unknown-name (error-code #(session/release! current-session unregistered))))
      (is (nil? (session/release! current-session anonymous)))
      (is (= [:first :replacement :anonymous] (vec recording)))
      (.close current-session)
      (is (= [:first :replacement :anonymous] (vec recording))))))

(deftest trainer-handles-use-the-session-resource-contract
  (with-open [current-session (session/open)]
    (let [recording (ConcurrentLinkedQueue.)
          trainer (recorded-resource recording :trainer)]
      (is (identical? trainer
                      (session/register! current-session trainer {:kind :trainer :name :training})))
      (is (identical? trainer (.-value ^Resource (session/resource current-session :training :trainer))))
      (is (identical? trainer (session/value current-session :training :trainer)))
      (is (identical? trainer (session/shared current-session :training :trainer identity)))
      (is (identical? trainer (session/exclusive current-session :training :trainer identity)))
      (is (= :session/wrong-kind (error-code #(session/resource current-session :training :network))))
      (session/release! current-session :training)
      (is (= [:trainer] (vec recording)))
      (is (= :session/unknown-name (error-code #(session/value current-session :training :trainer)))))))

(deftest owned-resources-close-in-reverse-registration-order
  (let [recording (ConcurrentLinkedQueue.)
        current-session (session/open)]
    (try
      (doseq [label [:first :second :third]]
        (session/register! current-session (recorded-resource recording label)
                           {:kind :network :name label}))
      (session/register! current-session (recorded-resource recording :borrowed)
                         {:kind :network :name :borrowed :owned? false})
      (.close current-session)
      (is (= [:third :second :first] (vec recording)))
      (.close current-session)
      (is (= [:third :second :first] (vec recording)))
      (finally
        (.close current-session)))))

(deftest cleanup-aggregates-failures-and-finishes-native-close
  (with-open [current-session (session/open)]
    (let [recording (ConcurrentLinkedQueue.)
          first-failure (Exception. "first registered")
          last-failure (Exception. "last registered")]
      (doseq [[label failure] [[:first first-failure] [:last last-failure]]]
        (session/register! current-session
                           (reify AutoCloseable
                             (close [_]
                               (.add recording label)
                               (throw failure)))
                           {:kind :network :name label}))
      (let [failure (try
                      (.close current-session)
                      nil
                      (catch Throwable failure failure))]
        (is (identical? last-failure failure))
        (is (= [first-failure] (vec (.getSuppressed ^Throwable failure))))
        (is (= [:last :first] (vec recording)))
        (is (.isClosed current-session))
        (is (not (.isOpen (.manager current-session))))
        (is (.isClosed (.connection current-session)))
        (is (nil? (.close current-session)))))))

(deftest lifecycle-rejects-closed-calls-and-reentrant-close
  (let [current-session (session/open)]
    (try
      (is (= :session/close-inside-operation
             (error-code #(.call current-session (fn [] (.close current-session))))))
      (is (= :open (:status (session/describe current-session))))
      (is (= :nested (.call current-session (fn [] (.call current-session (constantly :nested))))))
      (.close current-session)
      (is (= :session/closed (error-code #(.call current-session (constantly :unexpected)))))
      (is (= :session/closed (error-code #(session/worker current-session))))
      (is (= :session/closed
             (error-code #(session/register! current-session (Object.) {:kind :network :name :closed}))))
      (is (= :session/closed (error-code #(session/shared current-session (Object.) :network identity))))
      (is (= :session/closed (error-code #(session/exclusive current-session (Object.) :network identity))))
      (is (nil? (.close current-session)))
      (finally
        (.close current-session)))))

(deftest workers-own-connections-and-sub-managers
  (let [current-session (session/open)
        current-worker (session/worker current-session)
        ^NDManager worker-manager (.manager current-worker)
        worker-connection (.connection current-worker)]
    (try
      (is (session/context? current-worker))
      (is (identical? current-session (.session current-worker)))
      (is (not (identical? (.connection current-session) worker-connection)))
      (is (not (identical? (.manager current-session) worker-manager)))
      (is (identical? (.manager current-session) (.getParentManager worker-manager)))
      (is (= (session/defaults current-session) (session/defaults current-worker)))
      (is (= 1 (.workerCount current-session)))
      (let [table (materialized-table! current-session)
            query-text (str "SELECT number FROM \"" (.-schema table) "\".\"" (.-table table) "\"")]
        (is (= 7 (.call current-worker #(scalar worker-connection query-text)))))
      (.close current-worker)
      (is (.isClosed current-worker))
      (is (.isClosed worker-connection))
      (is (not (.isOpen worker-manager)))
      (is (.isOpen (.manager current-session)))
      (is (zero? (.workerCount current-session)))
      (is (= :closed (:status (datafy/datafy current-worker))))
      (is (= :session/closed (error-code #(.call current-worker (constantly :unexpected)))))
      (is (nil? (.close current-worker)))
      (let [open-worker (session/worker current-session)]
        (.close current-session)
        (is (.isClosed open-worker))
        (is (not (.isOpen (.manager open-worker))))
        (is (.isClosed (.connection open-worker)))
        (is (zero? (.workerCount current-session)))
        (is (= :session/closed (error-code #(.call open-worker (constantly :unexpected))))))
      (finally
        (.close current-worker)
        (.close current-session)))))

(deftest exclusive-sections-block-concurrent-shared-sections
  (with-open [current-session (session/open)
              current-worker (session/worker current-session)]
    (session/register! current-session (Object.) {:kind :network :name :shared-network})
    ;; Latches expose lock admission deterministically; timed futures bound failures, not work scheduling.
    (let [exclusive-entered (CountDownLatch. (int 1))
          release-exclusive (CountDownLatch. (int 1))
          shared-attempted (CountDownLatch. (int 1))
          shared-entered (CountDownLatch. (int 1))
          exclusive-result (CompletableFuture/supplyAsync
                            (reify Supplier
                              (get [_]
                                (session/exclusive
                                 current-session :shared-network :network
                                 (fn [_value]
                                   (.countDown exclusive-entered)
                                   (when-not (.await release-exclusive (long 5) TimeUnit/SECONDS)
                                     (throw (ex-info "Exclusive release timed out" {:error :test/timeout})))
                                   :exclusive)))))
          shared-result (CompletableFuture/supplyAsync
                         (reify Supplier
                           (get [_]
                             (when-not (.await exclusive-entered (long 5) TimeUnit/SECONDS)
                               (throw (ex-info "Exclusive admission timed out" {:error :test/timeout})))
                             (.countDown shared-attempted)
                             (session/shared
                              current-worker :shared-network :network
                              (fn [_value]
                                (.countDown shared-entered)
                                :shared)))))]
      (try
        (is (.await exclusive-entered (long 5) TimeUnit/SECONDS))
        (is (.await shared-attempted (long 5) TimeUnit/SECONDS))
        (is (thrown? TimeoutException (.get shared-result (long 100) TimeUnit/MILLISECONDS)))
        (is (= 1 (.getCount shared-entered)))
        (.countDown release-exclusive)
        (is (= :exclusive (.get exclusive-result (long 5) TimeUnit/SECONDS)))
        (is (= :shared (.get shared-result (long 5) TimeUnit/SECONDS)))
        (is (zero? (.getCount shared-entered)))
        (finally
          (.countDown release-exclusive)
          (.get exclusive-result (long 5) TimeUnit/SECONDS)
          (.get shared-result (long 5) TimeUnit/SECONDS))))))

(deftest file-session-drops-owned-schema-on-close
  (let [directory (Files/createTempDirectory "garun-session-" (make-array FileAttribute 0))
        database-path (.resolve directory "session.duckdb")
        database (str database-path)]
    (try
      (let [schema-name (with-open [current-session (session/open {:database database})]
                          (is (= database (:database (session/describe current-session))))
                          (materialized-table! current-session)
                          (.schema current-session))]
        (with-open [connection (DriverManager/getConnection (str "jdbc:duckdb:" database))
                    statement (.prepareStatement connection
                                                 "SELECT count(*) FROM information_schema.schemata WHERE schema_name = ?")]
          (.setString statement (int 1) schema-name)
          (with-open [result (.executeQuery statement)]
            (is (.next result))
            (is (zero? (.getLong result (int 1)))))))
      (finally
        (Files/deleteIfExists (.resolve directory "session.duckdb.wal"))
        (Files/deleteIfExists database-path)
        (Files/deleteIfExists directory)))))
