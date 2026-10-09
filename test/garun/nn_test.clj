(ns garun.nn-test
  (:require [clojure.test :refer [deftest is testing]]
            [garun.nn :as nn]
            [garun.table :as table])
  (:import [ai.djl Model]
           [ai.djl.nn Block]
           [ai.djl.ndarray.types Shape]
           [java.nio.file Files]
           [java.nio.file.attribute FileAttribute]))

(set! *warn-on-reflection* true)

(defn- squared-error [prediction actual]
  (let [difference (- prediction actual)]
    (* difference difference)))

(defn- mse [predictions truth]
  (/ (reduce + (map squared-error predictions truth)) (count truth)))

(defn- accuracy [predictions truth]
  (/ (count (filter true? (map = predictions truth))) (double (count truth))))

(defn- error-key [operation]
  (try
    (operation)
    nil
    (catch clojure.lang.ExceptionInfo error
      (:error (ex-data error)))))

(defn- regression-table []
  (let [x (mapv #(/ % 20.0) (range -20 21))]
    (table/from {:x x :y (mapv #(+ 1 (* 2 %)) x)})))

(deftest linear-regression-and-validation
  (with-open [db (table/open)
              net (nn/build [:linear 1] {:input-shape [1] :seed 11})]
    (let [source (regression-table)
          before (nn/predict net db source {:x [:x]})
          {:keys [history]} (nn/train! net db source
                                       {:x [:x] :y :y :epochs 180 :batch 16
                                        :optim [:adam {:lr 0.05}] :validation source})
          predictions (nn/predict net db source {:x [:x]})
          truth (mapv :y (table/rows db source))]
      (is (= 41 (count before)))
      (is (= 180 (count history)))
      (is (< (mse predictions truth) 0.05))
      (is (< (:loss (last history)) (:loss (first history))))
      (is (< (:validation-loss (last history)) 0.05)))))

(deftest binary-classification-uses-logits-and-sigmoid-probabilities
  (let [x (vec (concat (range -20 -1) (range 2 21)))
        truth (mapv #(if (pos? %) 1 0) x)
        source (table/from {:x (mapv #(/ % 10.0) x) :label truth})]
    (with-open [db (table/open)
                net (nn/build [:linear 1] {:input-shape [1] :seed 13})]
      (nn/train! net db source
                 {:x [:x] :y :label :loss :bce :epochs 90 :batch 16 :optim [:adam {:lr 0.08}]})
      (is (>= (accuracy (nn/predict net db source {:x [:x] :output :class}) truth) 0.9))
      (is (every? #(<= 0 % 1) (nn/predict net db source {:x [:x] :output :probability}))))))

(deftest three-class-classification-uses-sparse-labels-and-softmax
  (let [centres [[-2.0 -2.0] [2.0 -2.0] [0.0 2.0]]
        points (vec (for [[label [a b]] (map-indexed vector centres)
                          offset (range -6 6)]
                      {:a (+ a (* 0.03 offset)) :b (+ b (* 0.02 offset)) :label label}))
        source (table/from {:a (mapv :a points) :b (mapv :b points) :label (mapv :label points)})
        truth (mapv :label points)]
    (with-open [db (table/open)
                net (nn/build [:linear 3] {:input-shape [2] :seed 17})]
      (nn/train! net db source
                 {:x [:a :b] :y :label :loss :ce :epochs 70 :batch 12 :optim [:adam {:lr 0.05}]})
      (is (>= (accuracy (nn/predict net db source {:x [:a :b] :output :class}) truth) 0.9))
      (is (every? #(= 3 (count %)) (nn/predict net db source {:x [:a :b]})))
      (is (every? #(< (Math/abs (- 1.0 (reduce + %))) 1.0e-12)
                  (nn/predict net db source {:x [:a :b] :output :probability}))))))

(def ^:private composite-spec
  [:sequential
   [:skip {:combine :add} [:linear 2] [:tanh]]
   [:parallel {:combine :add} [:linear 2] [:identity]]
   [:skip {:combine :concat} [:linear 2]]
   [:parallel {:combine :concat} [:linear 2] [:linear 1]]
   [:linear 1]])

(defn- network-output-shape [net input-shape]
  (let [^Model model (:model net)
        ^Block block (.getBlock model)
        shapes (.getOutputShapes block (into-array Shape [(Shape. (long-array input-shape))]))
        ^Shape output-shape (first shapes)]
    (vec (.getShape output-shape))))

(defn- saved-network-predictions! [net db source features]
  (let [dir (.toFile (Files/createTempDirectory "garun-nn-test-" (make-array FileAttribute 0)))]
    (try
      (nn/save! net dir)
      (with-open [restored (nn/load dir)]
        {:spec (:spec restored)
         :predictions (nn/predict restored db source {:x features})})
      (finally
        (doseq [^java.io.File file (reverse (file-seq dir))]
          (.delete file))))))

(deftest fnblock-gradients-shapes-and-child-serialization
  (with-open [db (table/open)
              net (nn/build composite-spec {:input-shape [2] :seed 19})]
    (let [x (mapv #(/ % 10.0) (range -10 11))
          source (table/from {:a x :b (mapv #(* 0.5 %) x) :y (mapv #(+ 1 (* 2 %)) x)})
          {:keys [history]} (nn/train! net db source {:x [:a :b] :y :y :epochs 80 :batch 21 :optim [:adam {:lr 0.025}]})
          predictions (nn/predict net db source {:x [:a :b]})]
      (is (= [7 1] (network-output-shape net [7 2])))
      (is (= 21 (count predictions)))
      (is (every? double? predictions))
      (is (< (:loss (last history)) (:loss (first history))))
      (is (= {:spec composite-spec :predictions predictions}
             (saved-network-predictions! net db source [:a :b]))))))

(deftest direct-combination-output-shapes
  (with-open [db (table/open)]
    (doseq [[spec width] [[[:parallel {:combine :add} [:linear 2] [:linear 2]] 2]
                          [[:parallel {:combine :concat} [:linear 2] [:linear 3]] 5]
                          [[:skip {:combine :add} [:linear 2]] 2]
                          [[:skip {:combine :concat} [:linear 3]] 5]]]
      (with-open [net (nn/build spec {:input-shape [2] :seed 4})]
        (is (= [width width]
               (mapv count (nn/predict net db (table/from {:a [1 2] :b [3 4]}) {:x [:a :b]}))))))))

(deftest normalization-dropout-flatten-and-activations
  (with-open [db (table/open)
              net (nn/build [:sequential [:linear 4] [:batchnorm] [:layernorm]
                             [:dropout {:p 0.2}] [:flatten] [:activation :relu]
                             [:tanh] [:sigmoid] [:gelu] [:identity] [:linear 1]]
                            {:input-shape [1] :seed 21})]
    (let [source (regression-table)
          {:keys [history]} (nn/train! net db source {:x [:x] :y [:y] :epochs 3 :batch 16})]
      (is (= 3 (count history)))
      (is (every? #(Double/isFinite (double (:loss %))) history))
      (is (= 41 (count (nn/predict net db source {:x [:x]})))))))

(deftest optimizer-and-mae-options
  (with-open [db (table/open)]
    (doseq [optim [[:sgd {:lr 0.05 :momentum 0.5}] [:adamw {:lr 0.05 :weight-decay 0.01}]]]
      (with-open [net (nn/build [:linear {:units 1 :bias? false}] {:input-shape [1]})]
        (let [history (:history (nn/train! net db (regression-table)
                                           {:x [:x] :y :y :loss :mae :epochs 3 :batch 16 :shuffle? false :optim optim}))]
          (is (= 3 (count history)))
          (is (every? #(Double/isFinite (double (:loss %))) history)))))))

(deftest multi-output-label-matrices
  (with-open [db (table/open)
              net (nn/build [:linear 2] {:input-shape [1] :seed 23})]
    (let [x (mapv #(/ % 10.0) (range -10 11))
          ya (mapv #(+ 1 (* 2 %)) x)
          yb (mapv #(- (* 3 %) 1) x)
          source (table/from {:x x :ya ya :yb yb})
          _ (nn/train! net db source
                       {:x [:x] :y [:ya :yb] :epochs 140 :batch 21 :optim [:adam {:lr 0.08}]})
          predictions (nn/predict net db source {:x [:x]})]
      (is (< (mse (mapv first predictions) ya) 0.05))
      (is (< (mse (mapv second predictions) yb) 0.05)))))

(defn- train-seeded-network! [db source]
  (with-open [net (nn/build [:sequential [:linear 3] [:tanh] [:dropout {:p 0.2}] [:linear 1]]
                            {:input-shape [1] :seed 29})]
    (nn/train! net db source {:x [:x] :y :y :epochs 5 :batch 13 :seed 31})))

(deftest seeds-reproduce-initialization-shuffling-and-dropout
  (with-open [db (table/open)]
    (let [source (regression-table)]
      (is (= (train-seeded-network! db source) (train-seeded-network! db source))))))

(deftest scalar-predictions-use-zero-threshold-and-stable-sigmoid
  (let [source (table/from {:x [-1000.0 -1.0 0.0 1.0 1000.0]})
        exponential (Math/exp -1.0)]
    (with-open [db (table/open)
                net (nn/build [:identity] {:input-shape [1]})]
      (is (= [-1000.0 -1.0 0.0 1.0 1000.0]
             (nn/predict net db source {:x [:x]})))
      (is (= [0 0 0 1 1] (nn/predict net db source {:x [:x] :output :class})))
      (is (= [0.0 (/ exponential (+ 1.0 exponential)) 0.5
              (/ 1.0 (+ 1.0 exponential)) 1.0]
             (nn/predict net db source {:x [:x] :output :probability}))))))

(deftest vector-predictions-use-first-argmax-and-stable-softmax
  (let [source (table/from {:a [0.0 10.0 -1000.0 1000.0]
                            :b [0.0 10.0 1000.0 -1000.0]})]
    (with-open [db (table/open)
                net (nn/build [:identity] {:input-shape [2]})]
      (is (= [[0.0 0.0] [10.0 10.0] [-1000.0 1000.0] [1000.0 -1000.0]]
             (nn/predict net db source {:x [:a :b]})))
      (is (= [0 0 1 0] (nn/predict net db source {:x [:a :b] :output :class})))
      (is (= [[0.5 0.5] [0.5 0.5] [0.0 1.0] [1.0 0.0]]
             (nn/predict net db source {:x [:a :b] :output :probability}))))))

(deftest block-argument-validation-is-consistent
  (testing "option-free blocks reject arguments instead of silently ignoring them"
    (doseq [tag [:relu :tanh :sigmoid :gelu :identity :flatten :batchnorm :layernorm]]
      (is (= :nn/invalid-spec
             (error-key #(nn/build [tag {}] {:input-shape [1]})))
          (str tag " rejects options"))))
  (testing "parameterized blocks report missing or surplus arguments"
    (doseq [[spec error] [[[:activation] :nn/invalid-spec]
                          [[:activation :relu :tanh] :nn/invalid-spec]
                          [[:linear 1 2] :nn/invalid-linear]
                          [[:dropout {} {}] :nn/invalid-dropout]
                          [[:parallel {:combine :add}] :nn/invalid-combination]
                          [[:skip {:combine :add}] :nn/invalid-combination]
                          [[:skip {:combine :multiply} [:identity]] :nn/invalid-combination]]]
      (is (= error (error-key #(nn/build spec {:input-shape [1]}))) (pr-str spec)))))

(deftest invalid-dsl-and-options
  (doseq [[spec error] [[[:unknown] :nn/unknown-tag]
                        [[:linear {:units 2 :typo true}] :nn/unknown-option]
                        [[:linear 0] :nn/invalid-linear]
                        [[:linear 1.5] :nn/invalid-linear]
                        [[:linear {:units 1 :bias? :yes}] :nn/invalid-linear]
                        [[:dropout {:p 1}] :nn/invalid-dropout]
                        [[:parallel {:combine :multiply} [:linear 1]] :nn/invalid-combination]
                        [[:parallel {:combine :add :oops 1} [:linear 1]] :nn/unknown-option]
                        [[:activation :unknown] :nn/invalid-activation]]]
    (is (= error (error-key #(nn/build spec {:input-shape [1]})))))
  (is (= :nn/unknown-option (error-key #(nn/build [:linear 1] {:input-shape [1] :oops 1}))))
  (is (= :nn/invalid-input-shape (error-key #(nn/build [:linear 1] {:input-shape [0]}))))
  (with-open [db (table/open)
              net (nn/build [:linear 1] {:input-shape [1]})]
    (let [source (regression-table)]
      (is (= :nn/unknown-option (error-key #(nn/train! net db source {:x [:x] :y :y :oops true}))))
      (is (= :nn/unknown-option (error-key #(nn/predict net db source {:x [:x] :oops true}))))
      (is (= :nn/invalid-output (error-key #(nn/predict net db source {:x [:x] :output :unknown}))))
      (is (= :nn/unknown-option (error-key #(nn/train! net db source {:x [:x] :y :y :optim [:adam {:lr 0.01 :oops 1}]}))))
      (is (= :nn/invalid-features (error-key #(nn/predict net db source {:x [:x :y]})))))))
