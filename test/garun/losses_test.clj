(ns garun.losses-test
  (:require [clojure.test :refer [deftest is testing]]
            [garun.tensor :as tensor]
            [garun.training.losses :as losses])
  (:import [ai.djl.ndarray NDArray NDList NDManager]
           [ai.djl.ndarray.types DataType]
           [ai.djl.training GradientCollector]
           [ai.djl.training.loss Loss]))

(set! *warn-on-reflection* true)

(defn- close? [expected actual]
  (<= (Math/abs (- (double expected) (double actual)))
      (* 1.0e-6 (max 1.0 (Math/abs (double expected))))))

(defn- values [^NDArray array]
  (vec (.toDoubleArray (.toType array DataType/FLOAT64 false))))

(defn- ndlist [arrays]
  (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray arrays)))

(defn- evaluate-with-gradient!
  [^NDManager manager ^Loss loss labels ^NDArray predictions]
  (.setRequiresGradient predictions true)
  (with-open [^GradientCollector collector (.newGradientCollector (.getEngine manager))]
    (let [^NDArray result (.evaluate loss (ndlist labels) (ndlist [predictions]))
          result-values (values result)
          result-shape (vec (.getShape (.getShape result)))]
      (.backward collector (.sum result))
      {:values result-values
       :shape result-shape
       :gradient (values (.getGradient predictions))})))

(deftest probability-bce-interior-values-and-both-term-gradients
  (with-open [^NDManager manager (tensor/manager {:device :cpu})]
    (let [targets [0.0 1.0 0.3 0.7]
          probabilities [0.2 0.8 0.4 0.6]
          weight 2.5
          expected (* (/ weight 4)
                      (reduce + (map (fn [y p]
                                       (- (+ (* y (Math/log p))
                                             (* (- 1 y) (Math/log (- 1 p))))))
                                     targets probabilities)))
          expected-gradient (mapv (fn [y p]
                                    (* (/ weight 4)
                                       (- (/ (- 1 y) (+ (- 1 p) 1.0e-12))
                                          (/ y (+ p 1.0e-12)))))
                                  targets probabilities)
          ^NDArray labels (tensor/array manager targets {:dtype :float32})
          ^NDArray input (tensor/array manager probabilities {:dtype :float32 :shape [2 2]})
          result (evaluate-with-gradient! manager (losses/binary-cross-entropy "probability" weight true)
                                          [labels] input)]
      (is (close? expected (first (:values result))))
      (is (pos? (first (:values result))))
      (is (= [] (:shape result)))
      (is (every? true? (map close? expected-gradient (:gradient result)))))))

(deftest probability-bce-endpoints-remain-finite-and-corrective
  (with-open [^NDManager manager (tensor/manager {:device :cpu})]
    (doseq [dtype [:float32 :float64]]
      (testing (name dtype)
        (let [^NDArray labels (tensor/array manager [0.0 1.0 1.0 0.0] {:dtype dtype})
              ^NDArray input (tensor/array manager [0.0 1.0 0.0 1.0] {:dtype dtype})
              result (evaluate-with-gradient! manager (losses/binary-cross-entropy "endpoints" 1 true)
                                              [labels] input)
              expected (/ (Math/log (/ (+ 1 1.0e-12) 1.0e-12)) 2)
              gradient [0.25 -0.25 -2.5e11 2.5e11]]
          (is (close? expected (first (:values result))))
          (is (every? #(Double/isFinite (double %)) (concat (:values result) (:gradient result))))
          (is (every? true? (map close? gradient (:gradient result)))))
        (let [^NDArray labels (tensor/array manager [0.0 1.0] {:dtype dtype})
              ^NDArray input (tensor/array manager [0.0 1.0] {:dtype dtype})
              result (evaluate-with-gradient! manager (losses/binary-cross-entropy "correct-endpoints" 1 true)
                                              [labels] input)]
          (is (<= 0.0 (first (:values result)) 1.0e-10))
          (is (every? true? (map close? [0.5 -0.5] (:gradient result)))))))))

(deftest raw-logit-bce-keeps-stable-extreme-loss-and-gradients
  (with-open [^NDManager manager (tensor/manager {:device :cpu})]
    (let [^NDArray labels (tensor/array manager [0.0 1.0 1.0 0.0] {:dtype :float32})
          ^NDArray input (tensor/array manager [-1000.0 1000.0 -1000.0 1000.0] {:dtype :float32})
          result (evaluate-with-gradient! manager (losses/binary-cross-entropy "logits" 2 false)
                                          [labels] input)]
      (is (close? 1000.0 (first (:values result))))
      (is (every? #(Double/isFinite (double %)) (concat (:values result) (:gradient result))))
      (is (every? true? (map close? [0.0 0.0 -0.5 0.5] (:gradient result)))))))

(deftest masked-sparse-ce-values-gradients-and-invalid-padding
  (with-open [^NDManager manager (tensor/manager {:device :cpu})]
    (let [log-three (Math/log 3)
          ^NDArray labels (tensor/array manager [[0 0 -100]
                                                 [Long/MAX_VALUE -1 200]
                                                 [0 1 0]] {:dtype :int64})
          ^NDArray lengths (tensor/array manager [2 0 3] {:dtype :int64})
          ^NDArray input (tensor/array manager [[[log-three 0.0] [0.0 log-three] [1000.0 -1000.0]]
                                                [[Double/NaN Double/NaN] [1000.0 -1000.0] [0.0 0.0]]
                                                [[0.0 0.0] [0.0 0.0] [0.0 0.0]]]
                                       {:dtype :float32})
          result (evaluate-with-gradient! manager (losses/masked-cross-entropy "sparse" 2 -1 true true)
                                          [labels lengths] input)
          expected [(* (/ 2.0 3) (+ (Math/log (/ 4.0 3)) (Math/log 4)))
                    0.0 (* 2 (Math/log 2))]
          expected-gradient [(/ -1.0 6) (/ 1.0 6) -0.5 0.5 0.0 0.0
                             0.0 0.0 0.0 0.0 0.0 0.0
                             (/ -1.0 3) (/ 1.0 3) (/ 1.0 3) (/ -1.0 3) (/ -1.0 3) (/ 1.0 3)]]
      (is (= [3 1] (:shape result)))
      (is (every? true? (map close? expected (:values result))))
      (is (every? true? (map close? expected-gradient (:gradient result))))
      (is (every? zero? (subvec (:gradient result) 4 12))))))

(deftest masked-ce-does-not-normalize-by-valid-length
  (with-open [^NDManager manager (tensor/manager {:device :cpu})]
    (let [^NDArray labels (tensor/array manager [[0 -1 -1 -1]] {:dtype :int64})
          ^NDArray lengths (tensor/array manager [1] {:dtype :int64})
          ^NDArray input (tensor/array manager [[[0.0 0.0] [0.0 0.0] [0.0 0.0] [0.0 0.0]]]
                                       {:dtype :float32})
          result (evaluate-with-gradient! manager (losses/masked-cross-entropy "width" 1 -1 true true)
                                          [labels lengths] input)]
      (is (close? (/ (Math/log 2) 4) (first (:values result))))
      (is (every? true? (map close? [-0.125 0.125 0.0 0.0 0.0 0.0 0.0 0.0]
                             (:gradient result)))))))

(deftest masked-dense-ce-supports-nondefault-class-axis-and-soft-labels
  (with-open [^NDManager manager (tensor/manager {:device :cpu})]
    (let [log-three (Math/log 3)
          ^NDArray labels (tensor/array manager [[[0.25 1.0 Double/NaN]
                                                  [0.75 0.0 Double/NaN]]]
                                        {:dtype :float32})
          ^NDArray lengths (tensor/array manager [2] {:dtype :int64})
          ^NDArray input (tensor/array manager [[[log-three 0.0 1000.0]
                                                 [0.0 log-three -1000.0]]]
                                       {:dtype :float32})
          result (evaluate-with-gradient! manager (losses/masked-cross-entropy "dense" 1.5 1 false true)
                                          [labels lengths] input)
          expected (* 0.5 (+ (* 0.25 (Math/log (/ 4.0 3)))
                             (* 0.75 (Math/log 4)) (Math/log 4)))]
      (is (= [1 1] (:shape result)))
      (is (close? expected (first (:values result))))
      (is (every? true? (map close? [0.25 -0.375 0.0 -0.25 0.375 0.0] (:gradient result)))))))

(deftest masked-sparse-ce-supports-class-axis-before-sequence
  (with-open [^NDManager manager (tensor/manager {:device :cpu})]
    (let [^NDArray labels (tensor/array manager [[0 99]] {:dtype :int64})
          ^NDArray lengths (tensor/array manager [1] {:dtype :int64})
          ^NDArray input (tensor/array manager [[[(Math/log 2) 0.0]
                                                 [0.0 (Math/log 3)]
                                                 [0.0 (Math/log 2)]]]
                                       {:dtype :float32})
          result (evaluate-with-gradient! manager (losses/masked-cross-entropy "class-first" 1 1 true true)
                                          [labels lengths] input)]
      (is (= [1 1] (:shape result)))
      (is (close? (/ (Math/log 2) 2) (first (:values result))))
      (is (every? true? (map close? [-0.25 0.0 0.125 0.0 0.125 0.0] (:gradient result)))))))

(deftest masked-ce-consumes-log-probabilities-without-renormalization
  (with-open [^NDManager manager (tensor/manager {:device :cpu})]
    (testing "Sparse labels and default class axis"
      (let [^NDArray labels (tensor/array manager [[0 1 -1]] {:dtype :int64})
            ^NDArray lengths (tensor/array manager [2] {:dtype :int64})
            ^NDArray input (tensor/array manager [[[(Math/log 0.75) (Math/log 0.25)]
                                                   [(Math/log 0.4) (Math/log 0.6)]
                                                   [Double/NEGATIVE_INFINITY Double/NaN]]]
                                         {:dtype :float32})
            result (evaluate-with-gradient! manager (losses/masked-cross-entropy "sparse-logp" 1 -1 true false)
                                            [labels lengths] input)]
        (is (close? (/ (- (+ (Math/log 0.75) (Math/log 0.6))) 3) (first (:values result))))
        (is (every? true? (map close? [(/ -1.0 3) 0.0 0.0 (/ -1.0 3) 0.0 0.0]
                               (:gradient result))))))
    (testing "Dense labels and a negative, nondefault class axis"
      (let [^NDArray labels (tensor/array manager [[[0.25 1.0 Double/NaN]
                                                    [0.75 0.0 Double/NaN]]]
                                          {:dtype :float32})
            ^NDArray lengths (tensor/array manager [2] {:dtype :int64})
            ^NDArray input (tensor/array manager [[[(Math/log 0.75) (Math/log 0.25) Double/NaN]
                                                   [(Math/log 0.25) (Math/log 0.75) Double/NEGATIVE_INFINITY]]]
                                         {:dtype :float32})
            result (evaluate-with-gradient! manager (losses/masked-cross-entropy "dense-logp" 1.5 -2 false false)
                                            [labels lengths] input)
            expected (* -0.5 (+ (* 0.25 (Math/log 0.75))
                                (* 0.75 (Math/log 0.25)) (Math/log 0.25)))]
        (is (close? expected (first (:values result))))
        (is (every? true? (map close? [-0.125 -0.5 0.0 -0.375 0.0 0.0] (:gradient result))))))))

(deftest masked-sparse-ce-still-rejects-invalid-unpadded-class-indices
  (with-open [^NDManager manager (tensor/manager {:device :cpu})]
    (doseq [invalid [-1 2]]
      (let [^NDArray labels (tensor/array manager [[invalid]] {:dtype :int64})
            ^NDArray lengths (tensor/array manager [1] {:dtype :int64})
            ^NDArray input (tensor/array manager [[[0.0 0.0]]] {:dtype :float32})
            ^Loss loss (losses/masked-cross-entropy "invalid-class" 1 -1 true true)]
        (is (thrown? Exception (.evaluate loss (ndlist [labels lengths]) (ndlist [input]))))))))
