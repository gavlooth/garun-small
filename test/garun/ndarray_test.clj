(ns garun.ndarray-test
  (:require [clojure.string :as string]
            [clojure.test :refer [deftest is testing]]
            [garun.blocks :as blocks]
            [garun.context :as context]
            [garun.ndarray :as nd]
            [garun.session :as session]
            [garun.tensor :as tensor])
  (:import [ai.djl Device]
           [ai.djl.ndarray NDArray NDList NDManager]
           [ai.djl.ndarray.index NDIndex]
           [ai.djl.ndarray.types DataType]
           [ai.djl.nn Block]
           [ai.djl.training GradientCollector]
           [java.io StringWriter]))

(set! *warn-on-reflection* true)

(defn- near-vector? [expected actual]
  (and (= (count expected) (count actual))
       (every? true? (map #(< (Math/abs (- (double %1) (double %2))) 1.0e-5)
                          expected actual))))

(deftest indexing-accepts-djl-strings-indices-and-integer-coordinates
  (with-open [current-session (session/open)]
    (let [array (tensor/array current-session [[1 2 3] [4 5 6]])
          index (NDIndex. ":, -1" (object-array 0))]
      (doseq [selection [(nd/get array ":, -1") (nd/get array index)]]
        (is (= [3 6] (nd/to-vector selection)))
        (is (= [2] (nd/shape selection))))
      (is (= [4 5 6] (nd/to-vector (nd/get array 1))))
      (doseq [selection [(nd/get array [1 2])
                         (nd/get array (long-array [1 2]))
                         (nd/get array 1 2)]]
        (is (= [6] (nd/to-vector selection)))
        (is (= [] (nd/shape selection))))
      (is (= [1 2 3 4 5 6] (nd/to-vector array))))))

(deftest arithmetic-dispatches-number-and-ndarray-overloads
  (with-open [current-session (session/open)]
    (let [array (tensor/array current-session [2 4] {:dtype :float32})
          other (tensor/array current-session [3 2] {:dtype :float32})]
      (doseq [[operation scalar-expected array-expected]
              [[nd/add [4 6] [5 6]]
               [nd/sub [0 2] [-1 2]]
               [nd/mul [4 8] [6 8]]
               [nd/div [1 2] [(/ 2.0 3.0) 2]]
               [nd/pow [4 16] [8 16]]
               [nd/maximum [2 4] [3 4]]
               [nd/minimum [2 2] [2 2]]]]
        (is (near-vector? scalar-expected (nd/to-vector (operation array 2))))
        (is (near-vector? array-expected (nd/to-vector (operation array other)))))
      (is (= [-2.0 -4.0] (nd/to-vector (nd/neg array))))
      (is (= [2.0 4.0] (nd/to-vector array))))))

(deftest unary-math-calls-djl-methods
  (with-open [current-session (session/open)]
    (let [signed (tensor/array current-session [-1.2 0.0 2.3] {:dtype :float32})
          positive (tensor/array current-session [1.0 4.0] {:dtype :float32})
          angles (tensor/array current-session [0.0 1.0] {:dtype :float32})]
      (doseq [[operation array expected]
              [[nd/abs signed [1.2 0 2.3]]
               [nd/sign signed [-1 0 1]]
               [nd/floor signed [-2 0 2]]
               [nd/ceil signed [-1 0 3]]
               [nd/round signed [-1 0 2]]
               [nd/square positive [1 16]]
               [nd/sqrt positive [1 2]]
               [nd/exp angles [1 (Math/exp 1)]]
               [nd/log positive [0 (Math/log 4)]]
               [nd/sin angles [0 (Math/sin 1)]]
               [nd/cos angles [1 (Math/cos 1)]]
               [nd/tanh angles [0 (Math/tanh 1)]]]]
        (is (near-vector? expected (nd/to-vector (operation array))))))))

(deftest reductions-support-vectors-int-arrays-and-kept-dimensions
  (with-open [current-session (session/open)]
    (let [array (tensor/array current-session [[1 2 3] [4 5 6]] {:dtype :float32})]
      (doseq [[operation all-expected row-expected column-expected]
              [[nd/sum [21] [6 15] [5 7 9]]
               [nd/mean [3.5] [2 5] [2.5 3.5 4.5]]
               [nd/max [6] [3 6] [4 5 6]]
               [nd/min [1] [1 4] [1 2 3]]
               [nd/prod [720] [6 120] [4 10 18]]]]
        (testing (str operation)
          (is (near-vector? all-expected (nd/to-vector (operation array))))
          (is (near-vector? all-expected (nd/to-vector (operation array nil))))
          (is (near-vector? row-expected (nd/to-vector (operation array [1]))))
          (is (= [2] (nd/shape (operation array [1] false))))
          (is (near-vector? column-expected (nd/to-vector (operation array (int-array [0])))))
          (let [kept (operation array (int-array [1]) true)
                all-kept (operation array nil true)]
            (is (= [2 1] (nd/shape kept)))
            (is (near-vector? row-expected (nd/to-vector kept)))
            (is (= [1 1] (nd/shape all-kept)))
            (is (near-vector? all-expected (nd/to-vector all-kept))))))
      (is (= [21.0] (nd/to-vector (nd/sum array [0 1]))))
      (is (= [2 1] (nd/shape (nd/sum array [-1] true)))))))

(deftest extrema-indices-follow-djl-single-axis-semantics
  (with-open [current-session (session/open)]
    (let [array (tensor/array current-session [[1 9 3] [8 2 6]])]
      (is (= [1] (nd/to-vector (nd/argmax array))))
      (is (= [0] (nd/to-vector (nd/argmin array))))
      (is (= [1 0] (nd/to-vector (nd/argmax array 1))))
      (is (= [0 1] (nd/to-vector (nd/argmin array [1]))))
      (is (= [1 0 1] (nd/to-vector (nd/argmax array (int-array [0])))))
      (is (= [0 1 0] (nd/to-vector (nd/argmin array 0))))
      (is (= [2 1] (nd/shape (nd/argmax array [-1] true))))
      (is (= [1 3] (nd/shape (nd/argmin array (int-array [0]) true))))
      (is (= [1 1] (nd/shape (nd/argmax array nil true))))
      (is (= [1] (nd/to-vector (nd/argmax array nil true))))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"exactly one axis"
                            (nd/argmax array [0 1]))))))

(deftest shape-transforms-concatenate-and-stack-arrays
  (with-open [current-session (session/open)]
    (let [array (tensor/array current-session [[1 2 3] [4 5 6]])
          row (tensor/array current-session [7 8 9])
          expanded (nd/expand-dims array 1)]
      (is (= [2 3] (nd/shape array)))
      (is (= [3 2] (nd/shape (nd/reshape array [3 -1]))))
      (is (= [1 2 3 4 5 6] (nd/to-vector (nd/reshape array [3 -1]))))
      (doseq [transposed [(nd/transpose array) (nd/transpose array [1 0])]]
        (is (= [3 2] (nd/shape transposed)))
        (is (= [1 4 2 5 3 6] (nd/to-vector transposed))))
      (is (= [2 1 3] (nd/shape expanded)))
      (doseq [squeezed [(nd/squeeze expanded) (nd/squeeze expanded 1)
                        (nd/squeeze expanded (int-array [1]))]]
        (is (= [2 3] (nd/shape squeezed)))
        (is (= [1 2 3 4 5 6] (nd/to-vector squeezed))))
      (is (= [6] (nd/shape (nd/flatten array))))
      (is (= [1 2 3 4 5 6] (nd/to-vector (nd/flatten array))))
      (is (= [4 3] (nd/shape (nd/concat [array array]))))
      (is (= [2 6] (nd/shape (nd/concat [array array] 1))))
      (is (= [1 2 3 1 2 3 4 5 6 4 5 6] (nd/to-vector (nd/concat [array array] 1))))
      (is (= [2 3] (nd/shape (nd/stack [row row]))))
      (is (= [3 2] (nd/shape (nd/stack [row row] 1))))
      (is (= [7 7 8 8 9 9] (nd/to-vector (nd/stack [row row] 1)))))))

(deftest linear-algebra-multiplies-and-dots-native-arrays
  (with-open [current-session (session/open)]
    (let [array (tensor/array current-session [[1 2] [3 4]] {:dtype :float32})
          other (tensor/array current-session [[5 6] [7 8]] {:dtype :float32})
          left (tensor/array current-session [1 2] {:dtype :float32})
          right (tensor/array current-session [3 4] {:dtype :float32})]
      (is (= [2 2] (nd/shape (nd/matmul array other))))
      (is (= [19.0 22.0 43.0 50.0] (nd/to-vector (nd/matmul array other))))
      (is (= [11.0] (nd/to-vector (nd/dot left right)))))))

(deftest comparisons-and-softmax-retain-native-semantics
  (with-open [current-session (session/open)]
    (let [array (tensor/array current-session [1 2 3] {:dtype :float32})
          other (tensor/array current-session [1 3 2] {:dtype :float32})
          logits (tensor/array current-session [[0 0] [0 0]] {:dtype :float32})]
      (doseq [[operation scalar-expected array-expected]
              [[nd/eq [0 1 0] [1 0 0]]
               [nd/gt [0 0 1] [0 0 1]]
               [nd/gte [0 1 1] [1 0 1]]
               [nd/lt [1 0 0] [0 1 0]]
               [nd/lte [1 1 0] [1 1 0]]]]
        (is (= scalar-expected (nd/to-vector (operation array 2))))
        (is (= array-expected (nd/to-vector (operation array other)))))
      (is (= [2 2] (nd/shape (nd/softmax logits 1))))
      (is (near-vector? [0.5 0.5 0.5 0.5] (nd/to-vector (nd/softmax logits 1))))
      (is (near-vector? (repeat 4 (- (Math/log 2))) (nd/to-vector (nd/log-softmax logits -1)))))))

(deftest conversion-preserves-dtypes-exact-integers-and-list-identity
  (with-open [current-session (session/open)]
    (doseq [[data expected] [[(float-array [1.5 -2.5]) [1.5 -2.5]]
                             [(double-array [1.5 -2.5]) [1.5 -2.5]]
                             [(int-array [1 -2]) [1 -2]]
                             [(long-array [9007199254740993 Long/MAX_VALUE])
                              [9007199254740993 Long/MAX_VALUE]]
                             [(byte-array [-128 127]) [-128 127]]
                             [(boolean-array [true false]) [1 0]]]]
      (is (= expected (nd/to-vector (tensor/array current-session data)))))
    (let [^NDArray array (tensor/array current-session [1.5 -2.75] {:dtype :float32})
          ^NDArray converted (nd/to-type array :int32)
          ^NDArray copied (nd/duplicate array)
          ^NDArray same-type-copy (nd/to-type array :float32 true)
          ^NDList arrays (nd/ndlist [array])]
      (is (= DataType/INT32 (.getDataType converted)))
      (is (= [1 -2] (nd/to-vector converted)))
      (is (identical? array (nd/to-type array :float32)))
      (is (not (identical? array same-type-copy)))
      (is (= [1.5 -2.75] (nd/to-vector copied) (nd/to-vector same-type-copy)))
      (is (not (identical? array copied)))
      (is (identical? array (nd/head arrays)))
      (is (identical? array (nd/singleton arrays)))
      (.set copied (float-array [7 8]))
      (is (= [1.5 -2.75] (nd/to-vector array)))
      (is (= [7.0 8.0] (nd/to-vector copied)))
      (is (= [] (vec (nd/ndlist))))
      (is (thrown? IndexOutOfBoundsException (nd/singleton (nd/ndlist [array copied]))))
      (is (thrown? IndexOutOfBoundsException (nd/singleton (nd/ndlist))))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not implemented"
                            (nd/to-type array :int16))))))

(deftest creation-uses-context-or-manager-ownership-and-dtype-options
  (with-open [current-session (session/open {:engine "PyTorch" :device :cpu})
              ^NDManager child (.newSubManager (context/manager current-session))]
    (is (= "PyTorch" (.getEngineName (.getEngine child))))
    (doseq [owner [current-session child]
            [operation options expected dtype]
            [[nd/zeros {:dtype :int32} [0 0 0 0 0 0] DataType/INT32]
             [nd/ones {:dtype :float64} [1 1 1 1 1 1] DataType/FLOAT64]
             [nd/full {:value 7 :dtype :int64} [7 7 7 7 7 7] DataType/INT64]
             [nd/arange {} [0 1 2 3 4 5] DataType/INT32]
             [nd/random-normal {:mean 7 :standard-deviation 0 :dtype :float64}
              [7 7 7 7 7 7] DataType/FLOAT64]
             [nd/random-uniform {:low 7 :high 7 :dtype :float32}
              [7 7 7 7 7 7] DataType/FLOAT32]]]
      (let [^NDArray result (operation owner [2 3] options)]
        (is (= [2 3] (nd/shape result)))
        (is (near-vector? expected (nd/to-vector result)))
        (is (= dtype (.getDataType result)))
        (is (= (Device/cpu) (.getDevice result)))
        (is (identical? (if (identical? owner child) child (context/manager current-session))
                        (.getManager result)))))
    (doseq [[operation expected] [[nd/zeros [0 0]] [nd/ones [1 1]] [nd/full [0 0]]]]
      (let [^NDArray result (operation current-session [2])]
        (is (= DataType/FLOAT32 (.getDataType result)))
        (is (near-vector? expected (nd/to-vector result)))))
    (is (= [3 5 7 9] (nd/to-vector (nd/arange child [2 2] {:start 3 :step 2 :dtype :int64}))))
    (is (near-vector? [1.5 1.0 0.5 0.0]
                      (nd/to-vector (nd/arange child [2 2]
                                               {:start 1.5 :step -0.5 :dtype :float32}))))
    (is (= [] (nd/shape (nd/zeros child []))))
    (is (= [0.0] (nd/to-vector (nd/zeros child []))))
    (is (= [0 2] (nd/shape (nd/arange child [0 2]))))
    (is (= [] (nd/to-vector (nd/arange child [0 2]))))
    (let [normal (nd/to-vector (nd/random-normal child [32]))
          uniform (nd/to-vector (nd/random-uniform current-session [32] {:low -2 :high 3}))]
      (is (= 32 (count normal) (count uniform)))
      (is (every? #(Double/isFinite (double %)) normal))
      (is (< 1 (count (distinct normal))))
      (is (every? #(and (<= -2 %) (< % 3)) uniform))
      (is (< 1 (count (distinct uniform)))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not implemented"
                          (nd/zeros child [2] {:dtype :int16})))))

(deftest like-creation-retains-native-shape-dtype-device-and-input
  (with-open [current-session (session/open {:engine "PyTorch" :device :cpu})]
    (let [^NDArray array (tensor/array current-session [[1 2] [3 4]] {:dtype :int64})]
      (doseq [[result expected] [[(nd/zeros-like array) [0 0 0 0]]
                                 [(nd/ones-like array) [1 1 1 1]]
                                 [(nd/full-like array 7) [7 7 7 7]]]]
        (is (= expected (nd/to-vector result)))
        (is (= [2 2] (nd/shape result)))
        (is (= DataType/INT64 (.getDataType ^NDArray result)))
        (is (= (.getDevice array) (.getDevice ^NDArray result)))
        (is (identical? (.getManager array) (.getManager ^NDArray result)))
        (is (not (identical? array result))))
      (is (= [1 2 3 4] (nd/to-vector array))))))

(deftest selection-and-logical-masks-check-native-values
  (with-open [current-session (session/open {:engine "PyTorch" :device :cpu})
              ^NDManager mask-manager (.newSubManager (context/manager current-session))]
    (let [array (tensor/array current-session [[1 3 2] [6 4 5]] {:dtype :float32})
          condition (tensor/array mask-manager [[true false true] [false true false]])
          suffix-mask (tensor/array current-session [true false true])
          indices (tensor/array current-session [[2 0] [1 2]])
          classes (tensor/array current-session [0 2])
          nonfinite (tensor/array current-session [Double/NaN Double/POSITIVE_INFINITY
                                                   Double/NEGATIVE_INFINITY 1]
                                  {:dtype :float32})
          ^NDArray selected (nd/where condition array (nd/neg array))]
      (is (= [2.0 3.0 2.0 5.0 4.0 5.0] (nd/to-vector (nd/clip array 2 5))))
      (is (= [1.0 -3.0 2.0 -6.0 4.0 -5.0] (nd/to-vector selected)))
      (is (identical? (context/manager current-session) (.getManager selected)))
      (is (= (Device/cpu) (.getDevice selected)))
      (is (= [1.0 2.0 4.0] (nd/to-vector (nd/boolean-mask array condition))))
      (is (= [3] (nd/shape (nd/boolean-mask array condition))))
      (is (= [1.0 2.0 6.0 5.0] (nd/to-vector (nd/boolean-mask array suffix-mask 1))))
      (is (= [2 2] (nd/shape (nd/boolean-mask array suffix-mask 1))))
      (is (= [2.0 1.0 4.0 5.0] (nd/to-vector (nd/gather array indices 1))))
      (is (= [2 2] (nd/shape (nd/gather array indices 1))))
      (doseq [encoded [(nd/one-hot classes 3) (nd/one-hot classes 3 :int32)]]
        (is (= [2 3] (nd/shape encoded)))
        (is (near-vector? [1 0 0 0 0 1] (nd/to-vector encoded))))
      (is (= DataType/INT32 (.getDataType (nd/one-hot classes 3 :int32))))
      (is (= [0 1 0 1 0 1] (nd/to-vector (nd/logical-not condition))))
      (is (= [0 0 0 0 0 0]
             (nd/to-vector (nd/logical-and condition (nd/logical-not condition)))))
      (is (= [1 1 1 1 1 1]
             (nd/to-vector (nd/logical-or condition (nd/logical-not condition)))))
      (is (= [1 0 0 0] (nd/to-vector (nd/is-nan nonfinite))))
      (is (= [0 1 1 0] (nd/to-vector (nd/is-infinite nonfinite))))
      (is (= [1.0 3.0 2.0 6.0 4.0 5.0] (nd/to-vector array))))))

(deftest ordering-cumulative-sums-and-flips-check-values
  (with-open [current-session (session/open {:engine "PyTorch" :device :cpu})]
    (let [^NDArray array (tensor/array current-session [[1 3 2] [6 4 5]] {:dtype :float32})]
      (is (= [1.0 2.0 3.0 4.0 5.0 6.0] (nd/to-vector (nd/sort array))))
      (is (= [1.0 3.0 2.0 6.0 4.0 5.0] (nd/to-vector (nd/sort array 0))))
      (is (= [0 2 1 1 2 0] (nd/to-vector (nd/arg-sort array))))
      (is (= [0 0 0 1 1 1] (nd/to-vector (nd/arg-sort array 0))))
      (is (= [1 2 0 0 2 1] (nd/to-vector (nd/arg-sort array 1 false))))
      (let [^NDList largest (nd/top-k array 2 1)
            ^NDList smallest (nd/top-k array 1 1 false true)
            ^NDList unsorted (nd/top-k array 2 1 true false)]
        (is (= [[3.0 2.0 6.0 5.0] [1 2 0 2]] (mapv nd/to-vector largest)))
        (is (= [[1.0 4.0] [0 1]] (mapv nd/to-vector smallest)))
        (is (= [[2.0 3.0] [5.0 6.0]]
               (mapv #(vec (clojure.core/sort %))
                     (partition 2 (nd/to-vector (nd/head unsorted))))))
        (is (= [2 2] (nd/shape (nd/head largest))))
        (is (every? #(identical? (.getManager array) (.getManager ^NDArray %)) largest)))
      (is (= [1.0 3.0 2.0 7.0 7.0 7.0] (nd/to-vector (nd/cum-sum array))))
      (is (= [1.0 4.0 6.0 6.0 10.0 15.0] (nd/to-vector (nd/cum-sum array 1))))
      (is (= [2.0 3.0 1.0 5.0 4.0 6.0] (nd/to-vector (nd/flip array 1))))
      (is (= [5.0 4.0 6.0 2.0 3.0 1.0] (nd/to-vector (nd/flip array [0 1]))))
      (is (= [6.0 4.0 5.0 1.0 3.0 2.0] (nd/to-vector (nd/flip array (int-array [0])))))
      (is (= [1.0 3.0 2.0 6.0 4.0 5.0] (nd/to-vector array))))))

(deftest additional-math-norms-and-batching-check-values
  (with-open [current-session (session/open {:engine "PyTorch" :device :cpu})]
    (let [positive (tensor/array current-session [1 10 100] {:dtype :float32})
          powers (tensor/array current-session [1 2 8] {:dtype :float32})
          angles (tensor/array current-session [0 1] {:dtype :float32})
          other-angles (tensor/array current-session [1 1] {:dtype :float32})
          array (tensor/array current-session [[3 4] [0 5]] {:dtype :float32})
          batch (tensor/array current-session [[[1 2] [3 4]] [[2 0] [0 2]]] {:dtype :float32})
          column (tensor/array current-session [[1] [2]] {:dtype :float32})]
      (is (near-vector? [0 1 2] (nd/to-vector (nd/log10 positive))))
      (is (near-vector? [0 1 3] (nd/to-vector (nd/log2 powers))))
      (is (near-vector? [0 0.8427007929497149] (nd/to-vector (nd/erf angles))))
      (is (near-vector? [0 (/ Math/PI 4)] (nd/to-vector (nd/atan2 angles other-angles))))
      (is (near-vector? [(Math/sqrt 50)] (nd/to-vector (nd/norm array))))
      (is (near-vector? [5 5] (nd/to-vector (nd/norm array [1]))))
      (is (= [2 1] (nd/shape (nd/norm array (int-array [1]) true))))
      (is (near-vector? [5 5] (nd/to-vector (nd/norm array (int-array [1]) true))))
      (is (= [1 1] (nd/shape (nd/norm array nil true))))
      (is (near-vector? [(Math/sqrt 50)] (nd/to-vector (nd/norm array nil true))))
      (is (near-vector? [7 5] (nd/to-vector (nd/norm array 1 [1] false))))
      (is (near-vector? [12] (nd/to-vector (nd/norm array 1 nil false))))
      (is (= [2 2 2] (nd/shape (nd/batch-matmul batch batch))))
      (is (= [7.0 10.0 15.0 22.0 4.0 0.0 0.0 4.0]
             (nd/to-vector (nd/batch-matmul batch batch))))
      (is (= [2 3] (nd/shape (nd/broadcast column [2 3]))))
      (is (= [1.0 1.0 1.0 2.0 2.0 2.0] (nd/to-vector (nd/broadcast column [2 3])))))))

(deftest repeating-elements-and-tiling-blocks-remain-distinct
  (with-open [current-session (session/open {:engine "PyTorch" :device :cpu})]
    (let [array (tensor/array current-session [[1 2] [3 4]])
          row (tensor/array current-session [1 2])]
      (is (= [1 1 2 2] (nd/to-vector (nd/repeat row 2))))
      (is (= [1 2 1 2] (nd/to-vector (nd/tile row 2))))
      (doseq [counts [[1 2] (long-array [1 2])]]
        (is (= [1 1 2 2 3 3 4 4] (nd/to-vector (nd/repeat array counts))))
        (is (= [1 2 1 2 3 4 3 4] (nd/to-vector (nd/tile array counts)))))
      (is (= [1 1 2 2 3 3 4 4] (nd/to-vector (nd/repeat array 1 2))))
      (is (= [4 4] (nd/shape (nd/repeat array 2))))
      (is (= [1 1 2 2 1 1 2 2 3 3 4 4 3 3 4 4] (nd/to-vector (nd/repeat array 2))))
      (is (= [4 4] (nd/shape (nd/tile array 2))))
      (is (= [1 2 1 2 3 4 3 4 1 2 1 2 3 4 3 4] (nd/to-vector (nd/tile array 2))))
      (is (= [1 2 3 4] (nd/to-vector array))))))

(deftest gradient-control-and-device-copies-retain-native-values
  (with-open [current-session (session/open {:engine "PyTorch" :device :cpu})]
    (let [^NDArray array (tensor/array current-session [2 3] {:dtype :float32})
          ^NDArray tracking (nd/requires-gradient! array true)
          ^NDArray detached (nd/stop-gradient array)
          ^NDArray copied (nd/to-device array :cpu true)]
      (is (identical? array tracking))
      (is (.hasGradient array))
      (is (= [2.0 3.0] (nd/to-vector tracking) (nd/to-vector detached) (nd/to-vector copied)))
      (is (not (.hasGradient detached)))
      (is (not (identical? array detached)))
      (is (identical? array (nd/to-device array :cpu)))
      (is (identical? array (nd/to-device array (Device/cpu))))
      (is (not (identical? array copied)))
      (is (= (Device/cpu) (.getDevice copied)))
      (is (identical? (.getManager array) (.getManager copied)))
      (is (identical? (.getManager array) (.getManager detached)))
      (with-open [^GradientCollector collector (.newGradientCollector (.getEngine (.getManager array)))]
        (let [objective (nd/sum (nd/add (nd/square array) (nd/mul detached 10)))]
          (is (= [63.0] (nd/to-vector objective)))
          (.backward collector objective)))
      (let [^NDArray result (nd/gradient array)]
        (is (= [4.0 6.0] (nd/to-vector result)))
        (is (identical? (.getManager array) (.getManager result))))
      (is (identical? array (nd/requires-gradient! array false)))
      (is (not (.hasGradient array)))
      (is (= [2.0 3.0] (nd/to-vector array))))))

(deftest generated-wrappers-name-the-djl-method-in-their-docstrings
  (doseq [[function method] [[#'nd/neg "neg"] [#'nd/abs "abs"] [#'nd/square "square"]
                             [#'nd/sqrt "sqrt"] [#'nd/exp "exp"] [#'nd/log "log"]
                             [#'nd/sin "sin"] [#'nd/cos "cos"] [#'nd/tanh "tanh"]
                             [#'nd/sign "sign"] [#'nd/floor "floor"] [#'nd/ceil "ceil"]
                             [#'nd/round "round"] [#'nd/flatten "flatten"] [#'nd/duplicate "duplicate"]
                             [#'nd/add "add"] [#'nd/sum "sum"] [#'nd/argmax "argMax"]
                             [#'nd/zeros-like "zerosLike"] [#'nd/ones-like "onesLike"]
                             [#'nd/log10 "log10"] [#'nd/log2 "log2"] [#'nd/erf "erf"]
                             [#'nd/logical-and "logicalAnd"] [#'nd/logical-or "logicalOr"]
                             [#'nd/logical-not "logicalNot"] [#'nd/is-nan "isNaN"]
                             [#'nd/is-infinite "isInfinite"] [#'nd/atan2 "atan2"]
                             [#'nd/batch-matmul "batchMatMul"] [#'nd/stop-gradient "stopGradient"]
                             [#'nd/gradient "getGradient"]]]
    (is (string/includes? (:doc (meta function)) (str "NDArray." method "(")))))

(deftest unhinted-lambda-blocks-compile-without-reflection-warnings
  (let [errors (StringWriter.)
        ^Block block (binding [*ns* (the-ns 'garun.ndarray-test)
                               *warn-on-reflection* true
                               *err* errors]
                       (blocks/compile-block
                        (eval '[:lambda {:singleton? true}
                                (fn [array] (nd/get array ":, -1"))])))
        operation (binding [*ns* (the-ns 'garun.ndarray-test)
                            *warn-on-reflection* true
                            *err* errors]
                    (eval '(fn [array other]
                             (.toFloatArray (nd/square (nd/add array other))))))
        new-operations (binding [*ns* (the-ns 'garun.ndarray-test)
                                 *warn-on-reflection* true
                                 *err* errors]
                         (eval '(fn [owner array]
                                  [(vec (.toFloatArray (nd/log10 (nd/clip array 1 100))))
                                   (.size (nd/top-k array 1 0))
                                   (.getDataType (nd/zeros owner [2]))
                                   (.getDevice (nd/to-device (nd/sort array) :cpu))])))]
    (is (instance? Block block))
    (is (not (string/includes? (str errors) "Reflection warning")) (str errors))
    (with-open [current-session (session/open {:engine "PyTorch" :device :cpu})]
      (let [array (tensor/array current-session [1 2] {:dtype :float32})
            [logs count dtype device] (new-operations current-session array)]
        (is (= [9.0 16.0] (vec (operation array 2))))
        (is (near-vector? [0 (Math/log10 2)] logs))
        (is (= 2 count))
        (is (= DataType/FLOAT32 dtype))
        (is (= (Device/cpu) device))))))
