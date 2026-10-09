(ns garun.tensor-test
  (:require [clojure.test :refer [deftest is]]
            [garun.table :as table]
            [garun.tensor :as tensor])
  (:import [ai.djl Device]
           [ai.djl.engine Engine]
           [ai.djl.ndarray NDArray NDList NDManager]
           [ai.djl.ndarray.types DataType Shape]))

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
           (error-code #(tensor/manager {:device [:gpu unavailable]}))))
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
  (with-open [manager (tensor/manager)]
    (let [large [9007199254740993 Long/MAX_VALUE Long/MIN_VALUE -9007199254740993]
          ^NDArray primitive (tensor/array manager (long-array large) {:shape [2 2]})
          ^NDArray nested (tensor/array manager [(subvec large 0 2) (subvec large 2)] {})]
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
      (let [^NDArray value (tensor/array manager input {})]
        (is (= expected (.getDataType value)))
        (is (= [2] (vec (.getShape (.getShape value)))))))
    (let [expected [Short/MIN_VALUE -1 0 Short/MAX_VALUE]
          ^NDArray shorts (tensor/array manager (short-array expected)
                                        {:shape [2 2] :dtype :int32})
          ^NDArray nested-shorts (tensor/array manager [(short-array [Short/MIN_VALUE -1])
                                                        (short-array [0 Short/MAX_VALUE])]
                                               {:dtype :int64})
          ^NDArray bytes (tensor/array manager (byte-array [Byte/MIN_VALUE -1 0 Byte/MAX_VALUE]))
          ^NDArray booleans (tensor/array manager (boolean-array [true false false true]))]
      (is (= DataType/INT32 (.getDataType shorts)))
      (is (= DataType/INT64 (.getDataType nested-shorts)))
      (is (= expected (vec (.toIntArray shorts)) (vec (.toLongArray nested-shorts))))
      (is (= [2 2] (vec (.getShape (.getShape shorts)))))
      (is (identical? manager (.getManager shorts)))
      (is (= [Byte/MIN_VALUE -1 0 Byte/MAX_VALUE] (vec (.toByteArray bytes))))
      (is (= [true false false true] (vec (.toBooleanArray booleans)))))
    (let [^NDArray value (tensor/array manager [1 2 3] {:dtype :float32})]
      (is (= [1.0 2.0 3.0] (vec (.toFloatArray value)))))
    (is (= DataType/BOOLEAN (tensor/dtype :bool)))
    (is (= DataType/BFLOAT16 (tensor/dtype :bfloat16)))
    (is (= :tensor/invalid-dtype (error-code #(tensor/dtype :made-up))))))

(deftest unsupported-pytorch-dtypes-fail-before-native-creation-or-casting
  (with-open [manager (tensor/manager {:engine "PyTorch"})]
    (let [^Engine engine (.getEngine manager)
          shorts (short-array [Short/MIN_VALUE -1 0 Short/MAX_VALUE])
          ^NDArray input (tensor/array manager [1 2])
          ^NDArray boolean-shorts (tensor/array manager shorts {:dtype :bool})
          ^NDArray floating-shorts (tensor/array manager shorts {:dtype :float64})]
      (is (= :tensor/unsupported-dtype (error-code #(tensor/array manager shorts))))
      (is (= :tensor/unsupported-dtype
             (error-code #(tensor/array manager [shorts shorts]))))
      (doseq [unsupported [:int16 :bfloat16 :uint16 :uint32 :uint64 :unknown]]
        (is (= :tensor/unsupported-dtype
               (error-code #(tensor/dtype unsupported engine))))
        (is (= :tensor/unsupported-dtype
               (error-code #(tensor/array manager shorts {:dtype unsupported}))))
        (is (= :tensor/unsupported-dtype
               (error-code #(tensor/array manager input {:dtype unsupported})))))
      (is (= [true true false true] (vec (.toBooleanArray boolean-shorts))))
      (is (= [-32768.0 -1.0 0.0 32767.0] (vec (.toDoubleArray floating-shorts))))
      (is (not (.isReleased input)))
      (is (identical? manager (.getManager input)))
      (is (= DataType/INT32 (tensor/dtype :int32 engine)))
      (is (= DataType/BOOLEAN (tensor/dtype :bool "PyTorch")))
      (is (= :tensor/unsupported-dtype
             (error-code #(tensor/array manager [1] {:dtype :string}))))
      (is (= :tensor/unsupported-dtype
             (error-code #(tensor/array manager ["value"] {:dtype :int64}))))
      (let [^NDArray strings (tensor/array manager ["value"])]
        (is (= DataType/STRING (.getDataType strings)))
        (is (= ["value"] (vec (.toStringArray strings))))))))

(deftest multidimensional-host-values-and-cardinality-validation
  (with-open [manager (tensor/manager)]
    (let [java-matrix (into-array (class (int-array 0)) [(int-array [1 2]) (int-array [3 4])])
          ^NDArray matrix (tensor/array manager java-matrix {})
          ^NDArray cube (tensor/array manager [[[1 2] [3 4]] [[5 6] [7 8]]] {})
          ^NDArray scalar (tensor/array manager 9007199254740993 {})]
      (is (= DataType/INT32 (.getDataType matrix)))
      (is (= [2 2] (vec (.getShape (.getShape matrix)))))
      (is (= [1 2 3 4] (vec (.toIntArray matrix))))
      (is (= [2 2 2] (vec (.getShape (.getShape cube)))))
      (is (= [1 2 3 4 5 6 7 8] (vec (.toLongArray cube))))
      (is (= [] (vec (.getShape (.getShape scalar)))))
      (is (= [9007199254740993] (vec (.toLongArray scalar)))))
    (doseq [input [(float-array [1 2 3]) [1 2 3]]]
      (is (= :tensor/shape-mismatch
             (error-code #(tensor/array manager input {:shape [2 2]})))))
    (is (= :tensor/shape-mismatch
           (error-code #(tensor/array manager [[1 2] [3 4]] {:shape [4]}))))
    (is (= :tensor/ragged-data
           (error-code #(tensor/array manager [[1 2] [3]] {}))))
    (is (= :tensor/ragged-data
           (error-code #(tensor/array manager [[1 2] 3] {}))))
    (is (= :tensor/invalid-data
           (error-code #(tensor/array manager [9223372036854775808N] {}))))
    (is (= :tensor/unknown-option
           (error-code #(tensor/array manager [1] {:unused true}))))))

(deftest native-array-copy-ownership-and-manager-lifetimes
  (with-open [owner (tensor/manager)]
    (let [^NDArray input (tensor/array owner (long-array [9007199254740993 7]) {})
          ^NDManager target (tensor/manager)
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
  (with-open [target (tensor/manager)]
    (let [^NDArray copy (with-open [source (tensor/manager)]
                          (tensor/array target (tensor/array source (long-array [9007199254740993]) {}) {}))]
      (is (= [9007199254740993] (vec (.toLongArray copy)))))
    (let [^NDArray closed (tensor/array target [1] {})]
      (.close closed)
      (is (= :tensor/closed-array (error-code #(tensor/array target closed {})))))
    (with-open [source (tensor/manager)]
      (let [^NDArray original (tensor/array source (int-array [1 2 3 4]) {})
            ^NDArray copy (tensor/array target original {:shape [2 2] :dtype :float64})]
        (is (= DataType/INT32 (.getDataType original)))
        (is (= [4] (vec (.getShape (.getShape original)))))
        (is (= DataType/FLOAT64 (.getDataType copy)))
        (is (= [2 2] (vec (.getShape (.getShape copy)))))
        (is (= [1.0 2.0 3.0 4.0] (vec (.toDoubleArray copy))))))))

(deftest ndlist-descriptors-and-native-inputs
  (with-open [source (tensor/manager)
              target (tensor/manager)]
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
  (with-open [db (table/open)]
    (let [source (table/from
                  [:duckdb {:query (str "WITH samples AS MATERIALIZED ("
                                        "SELECT CAST(floor(random() * 100000) AS INTEGER) AS x FROM range(128)) "
                                        "SELECT x, 2 * x + 1 AS label, -x AS negative "
                                        "FROM samples ORDER BY random()")}])]
      (dotimes [_ 4]
        (let [[features labels repeated] (tensor/table-matrices db source
                                                                [[:negative :x] [:label :x] [:x :x]])
              xs (partition 2 (:data features))
              ys (partition 2 (:data labels))
              duplicates (partition 2 (:data repeated))]
          (is (= [128 128 128] (mapv :rows [features labels repeated])))
          (is (= [2 2 2] (mapv :cols [features labels repeated])))
          (is (every? true? (map (fn [[negative x] [label label-x] [a b]]
                                   (and (= (- x) negative) (= (inc (* 2 x)) label)
                                        (= x label-x a b)))
                                 xs ys duplicates))))))
    (is (= :table/null-value
           (error-code #(tensor/table-matrices db (table/from {:x [nil] :y [1]}) [[:x] [:y]]))))
    (is (= :table/non-numeric
           (error-code #(tensor/table-matrices db (table/from {:x [1] :y ["text"]}) [[:x] [:y]]))))
    (is (= :tensor/invalid-columns
           (error-code #(tensor/table-matrices db (table/from {:x [1]}) [[]]))))))
