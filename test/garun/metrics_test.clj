(ns garun.metrics-test
  "Examples and edge cases for prediction-first, sklearn-style vector metrics."
  (:require [clojure.test :refer [deftest is testing]]
            [garun.metrics :as metrics]))

(set! *warn-on-reflection* true)

(defn- near? [expected actual]
  (< (Math/abs (- (double expected) (double actual))) 1e-12))

(defn- error-kind [f]
  (try
    (f)
    nil
    (catch clojure.lang.ExceptionInfo error
      (:error (ex-data error)))))

(deftest accuracy-is-the-fraction-of-matching-labels
  (doseq [[description predictions targets expected]
          [["Three of four correct" [1 0 1 1] [1 0 0 1] 0.75]
           ["Perfect nonnumeric labels" [:a :b :a] [:a :b :a] 1.0]
           ["Every prediction wrong" [0 1] [1 0] 0.0]]]
    (testing description
      (is (near? expected (metrics/accuracy predictions targets))))))

(deftest regression-metrics-follow-their-error-formulas
  ;; SSres = 1.5, absolute error sum = 2, SStot = 29.1875.
  (doseq [[description metric predictions targets expected]
          [["Mean squared error" metrics/mse [2.5 0.0 2.0 8.0] [3.0 -0.5 2.0 7.0] 0.375]
           ["Mean absolute error" metrics/mae [2.5 0.0 2.0 8.0] [3.0 -0.5 2.0 7.0] 0.5]
           ["R² from residual and target variance" metrics/r2
            [2.5 0.0 2.0 8.0] [3.0 -0.5 2.0 7.0] 0.9486081370449679]
           ["Perfect squared error" metrics/mse [1 2 3] [1 2 3] 0.0]
           ["Perfect absolute error" metrics/mae [1 2 3] [1 2 3] 0.0]
           ["Perfect R²" metrics/r2 [1 2 3] [1 2 3] 1.0]
           ["R² can be negative" metrics/r2 [3 2 1] [1 2 3] -3.0]
           ["Perfect constant targets" metrics/r2 [2 2 2] [2 2 2] 1.0]
           ["Imperfect constant targets" metrics/r2 [2 2 3] [2 2 2] 0.0]]]
    (testing description
      (is (near? expected (metric predictions targets)))))
  (testing "One observation has undefined R²"
    (is (Double/isNaN (double (metrics/r2 [2] [2])))))
  (testing "Both predictions and targets must contain finite numeric values"
    (doseq [metric [metrics/mse metrics/mae metrics/r2]
            invalid [[nil 1] ["1" 1] [Double/NaN 1] [Double/POSITIVE_INFINITY 1]]]
      (is (= :metrics/non-numeric (error-kind #(metric invalid [0 1]))))
      (is (= :metrics/non-numeric (error-kind #(metric [0 1] invalid)))))))

(deftest confusion-matrix-uses-true-rows-and-predicted-columns
  (let [predictions [0 2 2 1 0 0]
        targets [0 1 2 1 2 0]]
    (doseq [[classes expected]
            [[3 [[2 0 0] [0 1 1] [1 0 1]]]
             [[0 1 2] [[2 0 0] [0 1 1] [1 0 1]]]
             [[2 1 0] [[1 0 1] [1 1 0] [0 0 2]]]]]
      (testing (str "Requested class order: " classes)
        (is (= expected (metrics/confusion-matrix predictions targets classes))))))
  (doseq [[description predictions targets classes expected]
          [["Rows are true labels, columns are predicted labels"
            [:b :b :b] [:a :a :b] [:a :b] [[0 2] [0 1]]]
           ["Unselected true or predicted labels omit the entire sample"
            [0 1 0] [0 0 1] [0] [[1]]]
           ["Perfect predictions give a diagonal matrix, including absent classes"
            [0 1 0] [0 1 0] 3 [[2 0 0] [0 1 0] [0 0 0]]]]]
    (testing description
      (is (= expected (metrics/confusion-matrix predictions targets classes)))))
  (testing "Classes must be a positive count or a distinct nonempty label vector"
    (doseq [classes [0 -1 [] [0 0] nil]]
      (is (= :metrics/invalid-classes
             (error-kind #(metrics/confusion-matrix [0] [0] classes)))))))

(deftest binary-classification-counts-the-selected-positive-class
  (let [predictions [1 1 1 0 0]
        targets [1 0 0 1 0]]
    ;; Positive 1: TP = 1, FP = 2, FN = 1; positive 0 swaps FP and FN.
    (doseq [[description options expected-precision expected-recall expected-f1]
            [["Default positive class" {} (/ 1.0 3) 0.5 0.4]
             ["Explicit positive class zero" {:positive 0} 0.5 (/ 1.0 3) 0.4]
             ["Macro average of both classes" {:average :macro} (/ 5.0 12) (/ 5.0 12) 0.4]]
            [metric expected] [[metrics/precision expected-precision]
                               [metrics/recall expected-recall]
                               [metrics/f1 expected-f1]]]
      (testing description
        (is (near? expected (metric predictions targets options))))))
  (testing "An explicit positive label need not be numeric"
    (is (near? 0.5 (metrics/precision [:yes :yes] [:yes :no] {:positive :yes}))))
  (doseq [metric [metrics/precision metrics/recall metrics/f1]]
    (doseq [[description predictions targets expected]
            [["Perfect mixed labels" [0 1 1 0] [0 1 1 0] 1.0]
             ["Only positives, all correct" [1 1] [1 1] 1.0]
             ["Positive class absent: zero division gives zero" [0 0] [0 0] 0.0]
             ["All false negatives" [0 0] [1 1] 0.0]
             ["All false positives" [1 1] [0 0] 0.0]
             ["No true positives" [0 1] [1 0] 0.0]]]
      (testing description
        (is (near? expected (metric predictions targets)))))
    (testing "Binary averaging rejects multiclass data or an unknown positive class"
      (is (= :metrics/invalid-classes (error-kind #(metric [0 1 2] [0 1 2]))))
      (is (= :metrics/invalid-classes
             (error-kind #(metric [0 1] [0 1] {:positive 2})))))))

(deftest macro-classification-averages-scores-over-all-observed-classes
  ;; Per-class precision = [2/3, 1, 1/2], recall = [1, 1/2, 1/2],
  ;; F1 = [4/5, 2/3, 1/2]. Macro F1 averages F1, not precision/recall.
  (doseq [[description predictions targets expected-precision expected-recall expected-f1]
          [["Multiclass scores" [0 2 2 1 0 0] [0 1 2 1 2 0]
            (/ 13.0 18) (/ 2.0 3) (/ 59.0 90)]
           ["Classes appearing in only predictions or truth still contribute"
            [0 0] [0 1] 0.25 0.5 (/ 1.0 3)]]
          [metric expected] [[metrics/precision expected-precision]
                             [metrics/recall expected-recall]
                             [metrics/f1 expected-f1]]]
    (testing description
      (is (near? expected (metric predictions targets {:average :macro})))))
  (doseq [metric [metrics/precision metrics/recall metrics/f1]
          [description predictions targets expected]
          [["Perfect multiclass" [0 1 2] [0 1 2] 1.0]
           ["Only one observed class" [0 0] [0 0] 1.0]
           ["Every class wrong" [1 2 0] [0 1 2] 0.0]
           ["Nonnumeric labels, including nil" [nil :other] [nil :other] 1.0]]]
    (testing description
      (is (near? expected (metric predictions targets {:average :macro}))))))

(deftest log-loss-averages-negative-log-true-class-probabilities
  (doseq [[description arguments expected]
          [["Binary positive-class probabilities"
            [[0.8 0.25] [1 0]] 0.25541281188299536]
           ["Equivalent two-column probabilities"
            [[[0.2 0.8] [0.75 0.25]] [1 0]] 0.25541281188299536]
           ["Multiclass: -(log(0.7) + log(0.6))/2"
            [[[0.7 0.2 0.1] [0.1 0.3 0.6]] [0 2]] 0.4337502838523616]
           ["Default upper clipping for perfect probabilities"
            [[1.0 0.0] [1 0]] (- (Math/log (- 1.0 1e-15)))]
           ["Default lower clipping for impossible probabilities"
            [[0.0 1.0] [1 0]] (- (Math/log 1e-15))]
           ["Explicit epsilon clips binary probabilities"
            [[0.0 1.0] [1 0] {:epsilon 0.1}] (- (Math/log 0.1))]
           ["Explicit epsilon clips multiclass probabilities"
            [[[1.0 0.0] [0.0 1.0]] [0 1] {:epsilon 0.1}] (- (Math/log 0.9))]]]
    (testing description
      (is (near? expected (apply metrics/log-loss arguments))))))

(deftest log-loss-validates-probabilities-columns-and-clipping
  (testing "Probabilities must be finite numbers in [0, 1]"
    (doseq [probabilities [[-0.1] [1.1] [nil] [Double/NaN] [[0.5 "0.5"]]]]
      (is (= :metrics/invalid-probability
             (error-kind #(metrics/log-loss probabilities [0]))))))
  (testing "Probability rows must be vectors of equal width, at least two"
    (doseq [probabilities [[[0.5 0.5] [0.1 0.2 0.7]] [[0.5 0.5] 0.5] [[1.0] [1.0]]]]
      (is (= :metrics/shape-mismatch
             (error-kind #(metrics/log-loss probabilities [0 1]))))))
  (testing "Targets must be integer column indices"
    (doseq [targets [[-1] [2] [0.5] [nil]]]
      (is (= :metrics/invalid-classes
             (error-kind #(metrics/log-loss [0.5] targets))))))
  (testing "Epsilon must be finite and strictly between zero and one half"
    (doseq [epsilon [0.0 -0.1 0.5 1.0 nil Double/NaN]]
      (is (= :metrics/invalid-options
             (error-kind #(metrics/log-loss [0.5] [0] {:epsilon epsilon})))))))

(deftest every-metric-requires-nonempty-equally-sized-vectors
  (doseq [metric [metrics/accuracy metrics/mse metrics/mae metrics/r2
                  metrics/precision metrics/recall metrics/f1 metrics/log-loss
                  #(metrics/confusion-matrix %1 %2 [0 1])]
          [predictions targets] [[[] []] [[0] []] [[] [0]] [[0 1] [0]] [[0] [0 1]]
                                 ['(0) [0]] [[0] '(0)]]]
    (testing (str "Invalid prediction/target shapes: " predictions " / " targets)
      (is (= :metrics/shape-mismatch (error-kind #(metric predictions targets)))))))

(deftest metric-options-reject-unknown-keys-and-unsupported-values
  (doseq [metric [metrics/precision metrics/recall metrics/f1]
          [predictions targets options] [[[0 1] [0 1] {:averge :macro}]
                                         [[0 1] [0 1] {:average :micro}]
                                         [[0] [0] nil]]]
    (testing (str "Invalid classification options: " options)
      (is (= :metrics/invalid-options
             (error-kind #(metric predictions targets options))))))
  (doseq [options [{:eps 0.01} nil]]
    (testing (str "Invalid log-loss options: " options)
      (is (= :metrics/invalid-options
             (error-kind #(metrics/log-loss [0.5] [0] options)))))))
