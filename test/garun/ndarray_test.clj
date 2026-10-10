(ns garun.ndarray-test
  (:require [clojure.string :as string]
            [clojure.test :refer [deftest is testing]]
            [garun.blocks :as blocks]
            [garun.ndarray :as nd]
            [garun.session :as session]
            [garun.tensor :as tensor])
  (:import [ai.djl.ndarray NDArray NDList]
           [ai.djl.ndarray.index NDIndex]
           [ai.djl.ndarray.types DataType]
           [ai.djl.nn Block]
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

(deftest generated-wrappers-name-the-djl-method-in-their-docstrings
  (doseq [[function method] [[#'nd/neg "neg"] [#'nd/abs "abs"] [#'nd/square "square"]
                             [#'nd/sqrt "sqrt"] [#'nd/exp "exp"] [#'nd/log "log"]
                             [#'nd/sin "sin"] [#'nd/cos "cos"] [#'nd/tanh "tanh"]
                             [#'nd/sign "sign"] [#'nd/floor "floor"] [#'nd/ceil "ceil"]
                             [#'nd/round "round"] [#'nd/flatten "flatten"] [#'nd/duplicate "duplicate"]
                             [#'nd/add "add"] [#'nd/sum "sum"] [#'nd/argmax "argMax"]]]
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
                             (.toFloatArray (nd/square (nd/add array other))))))]
    (is (instance? Block block))
    (is (not (string/includes? (str errors) "Reflection warning")) (str errors))
    (with-open [current-session (session/open)]
      (is (= [9.0 16.0]
             (vec (operation (tensor/array current-session [1 2] {:dtype :float32}) 2)))))))
