(ns garun.tensor-test
  (:require [clojure.test :refer [deftest is]]
            [garun.context :as context]
            [garun.session :as session]
            [garun.table :as table]
            [garun.tensor :as tensor])
  (:import [ai.djl Device]
           [ai.djl.engine Engine]
           [ai.djl.ndarray NDArray NDList NDManager]
           [ai.djl.ndarray.types DataType Shape]
           [garun.session Session]))

(set! *warn-on-reflection* true)

(defn- error-code [operation]
  (try
    (operation)
    nil
    (catch clojure.lang.ExceptionInfo error
      (:error (ex-data error)))))

(deftest devices-and-concrete-shapes
  (let [^Engine engine (Engine/getInstance)
        unavailable (.getGpuCount engine)
        cpu (Device/cpu)
        dimensions (Shape. (long-array [2 3 4]))]
    (is (= cpu (tensor/device :cpu)))
    (is (identical? cpu (tensor/device cpu)))
    (is (= (.defaultDevice engine) (tensor/device :auto)))
    (is (= :tensor/unavailable-device
           (error-code #(tensor/device [:gpu unavailable]))))
    (is (= :tensor/unavailable-device
           (error-code #(tensor/device (Device/gpu unavailable)))))
    (is (= :tensor/unavailable-device
           (error-code #(session/open {:device [:gpu unavailable]}))))
    (when (zero? unavailable)
      (is (= :tensor/unavailable-device
             (error-code #(tensor/device [:gpu 0])))))
    (is (= :tensor/invalid-device (error-code #(tensor/device [:gpu -1]))))
    (is (= :tensor/invalid-device (error-code #(tensor/device :gpu))))
    (is (identical? dimensions (tensor/shape dimensions)))
    (is (= [2 3 4] (vec (.getShape (tensor/shape [2 3 4])))))
    (is (= [] (vec (.getShape (tensor/shape [])))))
    (is (= [0 3] (vec (.getShape (tensor/shape [0 3])))))
    (doseq [invalid [[-1 2] [1.5 2] [Long/MAX_VALUE 2] [9223372036854775808N]]]
      (is (= :tensor/invalid-shape (error-code #(tensor/shape invalid)))))))

(deftest primitive-arrays-preserve-dtypes-and-exact-int64-values
  (with-open [current-session (session/open)]
    (let [large [9007199254740993 Long/MAX_VALUE Long/MIN_VALUE -9007199254740993]
          ^NDArray primitive (tensor/array current-session (long-array large) {:shape [2 2]})
          ^NDArray nested (tensor/array current-session [(subvec large 0 2) (subvec large 2)] {})]
      (is (= DataType/INT64 (.getDataType primitive)))
      (is (= DataType/INT64 (.getDataType nested)))
      (is (= large (vec (.toLongArray primitive))))
      (is (= large (vec (.toLongArray nested))))
      (is (= [2 2] (vec (.getShape (.getShape primitive))))))
    (doseq [[input expected] [[(float-array [1.25 -2.5]) DataType/FLOAT32]
                              [(double-array [1.25 -2.5]) DataType/FLOAT64]
                              [(int-array [1 -2]) DataType/INT32]
                              [(byte-array [Byte/MIN_VALUE Byte/MAX_VALUE]) DataType/INT8]
                              [(boolean-array [true false]) DataType/BOOLEAN]]]
      (let [^NDArray value (tensor/array current-session input {})]
        (is (= expected (.getDataType value)))
        (is (= [2] (vec (.getShape (.getShape value)))))))
    (let [expected [Short/MIN_VALUE -1 0 Short/MAX_VALUE]
          ^NDArray shorts (tensor/array current-session (short-array expected)
                                        {:shape [2 2] :dtype :int32})
          ^NDArray nested-shorts (tensor/array current-session [(short-array [Short/MIN_VALUE -1])
                                                                (short-array [0 Short/MAX_VALUE])]
                                               {:dtype :int64})
          ^NDArray bytes (tensor/array current-session (byte-array [Byte/MIN_VALUE -1 0 Byte/MAX_VALUE]))
          ^NDArray booleans (tensor/array current-session (boolean-array [true false false true]))]
      (is (= DataType/INT32 (.getDataType shorts)))
      (is (= DataType/INT64 (.getDataType nested-shorts)))
      (is (= expected (vec (.toIntArray shorts)) (vec (.toLongArray nested-shorts))))
      (is (= [2 2] (vec (.getShape (.getShape shorts)))))
      (is (identical? (context/manager ^Session current-session) (.getManager shorts)))
      (is (= [Byte/MIN_VALUE -1 0 Byte/MAX_VALUE] (vec (.toByteArray bytes))))
      (is (= [true false false true] (vec (.toBooleanArray booleans)))))
    (let [^NDArray value (tensor/array current-session [1 2 3] {:dtype :float32})]
      (is (= [1.0 2.0 3.0] (vec (.toFloatArray value)))))
    (is (= DataType/BOOLEAN (tensor/dtype :bool)))
    (is (= DataType/BFLOAT16 (tensor/dtype :bfloat16)))
    (is (= :tensor/invalid-dtype (error-code #(tensor/dtype :made-up))))))

(deftest unsupported-pytorch-dtypes-fail-before-native-creation-or-casting
  (with-open [^Session current-session (session/open {:engine "PyTorch"})]
    (let [^Engine engine (.getEngine (context/manager current-session))
          shorts (short-array [Short/MIN_VALUE -1 0 Short/MAX_VALUE])
          ^NDArray input (tensor/array current-session [1 2])
          ^NDArray boolean-shorts (tensor/array current-session shorts {:dtype :bool})
          ^NDArray floating-shorts (tensor/array current-session shorts {:dtype :float64})]
      (is (= :tensor/unsupported-dtype (error-code #(tensor/array current-session shorts))))
      (is (= :tensor/unsupported-dtype
             (error-code #(tensor/array current-session [shorts shorts]))))
      (doseq [unsupported [:int16 :bfloat16 :uint16 :uint32 :uint64 :unknown]]
        (is (= :tensor/unsupported-dtype
               (error-code #(tensor/dtype unsupported engine))))
        (is (= :tensor/unsupported-dtype
               (error-code #(tensor/array current-session shorts {:dtype unsupported}))))
        (is (= :tensor/unsupported-dtype
               (error-code #(tensor/array current-session input {:dtype unsupported})))))
      (is (= [true true false true] (vec (.toBooleanArray boolean-shorts))))
      (is (= [-32768.0 -1.0 0.0 32767.0] (vec (.toDoubleArray floating-shorts))))
      (is (not (.isReleased input)))
      (is (identical? (context/manager current-session) (.getManager input)))
      (is (= DataType/INT32 (tensor/dtype :int32 engine)))
      (is (= DataType/BOOLEAN (tensor/dtype :bool "PyTorch")))
      (is (= :tensor/unsupported-dtype
             (error-code #(tensor/array current-session [1] {:dtype :string}))))
      (is (= :tensor/unsupported-dtype
             (error-code #(tensor/array current-session ["value"] {:dtype :int64}))))
      (let [^NDArray strings (tensor/array current-session ["value"])]
        (is (= DataType/STRING (.getDataType strings)))
        (is (= ["value"] (vec (.toStringArray strings))))))))

(deftest multidimensional-host-values-and-cardinality-validation
  (with-open [current-session (session/open)]
    (let [java-matrix (into-array (class (int-array 0)) [(int-array [1 2]) (int-array [3 4])])
          ^NDArray matrix (tensor/array current-session java-matrix {})
          ^NDArray cube (tensor/array current-session [[[1 2] [3 4]] [[5 6] [7 8]]] {})
          ^NDArray scalar (tensor/array current-session 9007199254740993 {})]
      (is (= DataType/INT32 (.getDataType matrix)))
      (is (= [2 2] (vec (.getShape (.getShape matrix)))))
      (is (= [1 2 3 4] (vec (.toIntArray matrix))))
      (is (= [2 2 2] (vec (.getShape (.getShape cube)))))
      (is (= [1 2 3 4 5 6 7 8] (vec (.toLongArray cube))))
      (is (= [] (vec (.getShape (.getShape scalar)))))
      (is (= [9007199254740993] (vec (.toLongArray scalar)))))
    (doseq [input [(float-array [1 2 3]) [1 2 3]]]
      (is (= :tensor/shape-mismatch
             (error-code #(tensor/array current-session input {:shape [2 2]})))))
    (is (= :tensor/shape-mismatch
           (error-code #(tensor/array current-session [[1 2] [3 4]] {:shape [4]}))))
    (is (= :tensor/ragged-data
           (error-code #(tensor/array current-session [[1 2] [3]] {}))))
    (is (= :tensor/ragged-data
           (error-code #(tensor/array current-session [[1 2] 3] {}))))
    (is (= :tensor/invalid-data
           (error-code #(tensor/array current-session [9223372036854775808N] {}))))
    (is (= :tensor/unknown-option
           (error-code #(tensor/array current-session [1] {:unused true}))))))

(deftest context-owned-results-follow-session-lifetime
  (let [arrays (with-open [^Session current-session (session/open)]
                 (let [^NDArray array (tensor/array current-session [1 2])
                       ^NDList inputs (tensor/ndlist current-session {:data [3 4]})
                       ^NDArray listed (.singletonOrThrow inputs)]
                   (is (identical? (context/manager current-session) (.getManager array)))
                   (is (identical? (context/manager current-session) (.getManager listed)))
                   [array listed]))]
    (doseq [^NDArray array arrays]
      (is (.isReleased array)))))

(deftest native-array-copy-ownership-and-manager-lifetimes
  (with-open [^Session current-session (session/open)
              ^NDManager owner (.newSubManager (context/manager current-session))]
    (let [^NDArray input (tensor/array owner (long-array [9007199254740993 7]) {})
          ^NDManager target (.newSubManager (context/manager current-session))
          ^NDArray copy (tensor/array target input {})]
      (is (not (identical? input copy)))
      (is (identical? owner (.getManager input)))
      (is (identical? target (.getManager copy)))
      (.close target)
      (is (.isReleased copy))
      (is (not (.isReleased input)))
      (is (= [9007199254740993 7] (vec (.toLongArray input))))
      (is (= :tensor/closed-manager
             (error-code #(tensor/array target [1] {}))))
      (is (= :tensor/closed-manager
             (error-code #(tensor/ndlist target input))))))
  (with-open [^Session current-session (session/open)
              ^NDManager target (.newSubManager (context/manager current-session))]
    (let [^NDArray copy (with-open [source (.newSubManager (context/manager current-session))]
                          (tensor/array target (tensor/array source (long-array [9007199254740993]) {}) {}))]
      (is (= [9007199254740993] (vec (.toLongArray copy)))))
    (let [^NDArray closed (tensor/array target [1] {})]
      (.close closed)
      (is (= :tensor/closed-array (error-code #(tensor/array target closed {})))))
    (with-open [source (.newSubManager (context/manager current-session))]
      (let [^NDArray original (tensor/array source (int-array [1 2 3 4]) {})
            ^NDArray copy (tensor/array target original {:shape [2 2] :dtype :float64})]
        (is (= DataType/INT32 (.getDataType original)))
        (is (= [4] (vec (.getShape (.getShape original)))))
        (is (= DataType/FLOAT64 (.getDataType copy)))
        (is (= [2 2] (vec (.getShape (.getShape copy)))))
        (is (= [1.0 2.0 3.0 4.0] (vec (.toDoubleArray copy))))))))

(deftest ndlist-descriptors-and-native-inputs
  (with-open [^Session current-session (session/open)
              source (.newSubManager (context/manager current-session))
              target (.newSubManager (context/manager current-session))]
    (let [^NDArray input (tensor/array source (long-array [9007199254740993 8]) {})
          native (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray [input]))]
      (with-open [copied (tensor/ndlist target native)]
        (let [^NDArray value (.singletonOrThrow copied)]
          (is (not (identical? native copied)))
          (is (not (identical? input value)))
          (is (= [9007199254740993 8] (vec (.toLongArray value))))
          (.set value (long-array [2 3]))
          (is (= [9007199254740993 8] (vec (.toLongArray input))))))
      (is (not (.isReleased input)))
      (with-open [inputs (tensor/ndlist target [input {:data [[true false] [false true]] :dtype :bool}])]
        (is (= 2 (.size inputs)))
        (is (= [9007199254740993 8] (vec (.toLongArray ^NDArray (.get inputs (int 0))))))
        (is (= [true false false true] (vec (.toBooleanArray ^NDArray (.get inputs (int 1)))))))
      (is (identical? source (.getManager input))))
    (with-open [single (tensor/ndlist target [[1 2] [3 4]])]
      (is (= 1 (.size single)))
      (is (= [2 2] (vec (.getShape (.getShape (.singletonOrThrow single)))))))
    (with-open [described (tensor/ndlist target {:data (float-array [1 2 3 4]) :shape [2 2]})]
      (is (= DataType/FLOAT32 (.getDataType (.singletonOrThrow described)))))
    (is (= :tensor/invalid-data (error-code #(tensor/ndlist target {:shape [2]}))))
    (is (= :tensor/unknown-option (error-code #(tensor/ndlist target {:data [1] :typo true}))))))

(deftest aligned-table-matrices-use-one-source-realization
  ;; Trusted SQL deliberately changes both values and ordering per execution.
  ;; A materialized CTE defines one row relation within each query. Independent
  ;; feature/label queries cannot satisfy the returned row-wise relationships.
  (with-open [current-session (session/open)]
    (let [source (table/from
                  [:duckdb {:query (str "WITH samples AS MATERIALIZED ("
                                        "SELECT CAST(floor(random() * 100000) AS INTEGER) AS x FROM range(128)) "
                                        "SELECT x, 2 * x + 1 AS label, -x AS negative "
                                        "FROM samples ORDER BY random()")}])]
      (dotimes [_ 4]
        (let [[features labels repeated] (table/matrices current-session source
                                                         [[:negative :x] [:label :x] [:x :x]])
              feature-values (partition 2 (:data features))
              label-values (partition 2 (:data labels))
              duplicate-values (partition 2 (:data repeated))]
          (is (= [128 128 128] (mapv :rows [features labels repeated])))
          (is (= [2 2 2] (mapv :cols [features labels repeated])))
          (is (every? true? (map (fn [[negative feature] [label label-feature] [first-copy second-copy]]
                                   (and (= (- feature) negative) (= (inc (* 2 feature)) label)
                                        (= feature label-feature first-copy second-copy)))
                                 feature-values label-values duplicate-values))))))
    (is (= :table/null-value
           (error-code #(table/matrices current-session (table/from {:x [nil] :y [1]}) [[:x] [:y]]))))
    (is (= :table/non-numeric
           (error-code #(table/matrices current-session (table/from {:x [1] :y ["text"]}) [[:x] [:y]]))))
    (is (= :tensor/invalid-columns
           (error-code #(table/matrices current-session (table/from {:x [1]}) [[]]))))))
