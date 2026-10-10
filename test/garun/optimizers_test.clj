(ns garun.optimizers-test
  (:require [clojure.test :refer [deftest is testing]]
            [garun.context :as context]
            [garun.session :as session]
            [garun.tensor :as tensor]
            [garun.training :as training])
  (:import [ai.djl.ndarray NDArray NDManager NDScope]
           [ai.djl.ndarray.types DataType]
           [ai.djl.training GradientCollector]
           [ai.djl.training.optimizer Optimizer]
           [ai.djl.training.tracker ParameterTracker]
           [garun.session Session]))

(set! *warn-on-reflection* true)

(defn- near? [expected actual]
  (< (Math/abs (- (double expected) (double actual))) 1.0e-5))

(defn- reference-step
  "Scalar reference equations independent of NDArray execution and optimizer state."
  [tag options update [weight history] gradient]
  (let [rescaled (* gradient (get options :rescale-grad 1))
        clipped (if-let [bound (:clip-grad options)]
                  (max (- bound) (min bound rescaled))
                  rescaled)
        decay (get options :weight-decay (if (= tag :adamw) 0.01 0))
        gradient-with-decay (+ clipped (if (= tag :adamw) 0 (* decay weight)))
        learning-rate (get options :lr 0.01)
        momentum (get options :momentum (case tag
                                          :nag 0.9
                                          :rmsprop (if (:centered? options) 0.9 0)
                                          0))
        beta1 (get options :beta1 0.9)
        beta2 (get options :beta2 0.999)
        rho (get options :rho 0.9)
        epsilon (get options :epsilon 1.0e-8)
        old-mean (get history :mean 0)
        old-variance (get history :variance 0)
        old-velocity (get history :velocity 0)
        old-delta (get history :delta 0)]
    (case tag
      (:sgd :nag)
      (let [velocity (+ (* momentum old-velocity) (* learning-rate gradient-with-decay))
            step (if (= tag :nag) (+ (* momentum velocity) (* learning-rate gradient-with-decay)) velocity)]
        [(- weight step) {:velocity velocity}])
      (:adam :adamw)
      (let [mean (+ (* beta1 old-mean) (* (- 1 beta1) gradient-with-decay))
            variance (+ (* beta2 old-variance) (* (- 1 beta2) gradient-with-decay gradient-with-decay))
            corrected-rate (/ (* learning-rate (Math/sqrt (- 1 (Math/pow beta2 update))))
                              (- 1 (Math/pow beta1 update)))
            decayed-weight (if (= tag :adamw) (* weight (- 1 (* learning-rate decay))) weight)
            step (/ (* corrected-rate mean) (+ (Math/sqrt variance) epsilon))]
        [(- decayed-weight step) {:mean mean :variance variance}])
      :rmsprop
      (let [variance (+ (* rho old-variance) (* (- 1 rho) gradient-with-decay gradient-with-decay))
            mean (if (:centered? options) (+ (* rho old-mean) (* (- 1 rho) gradient-with-decay)) 0)
            denominator (Math/sqrt (+ (- variance (* mean mean)) epsilon))
            velocity (+ (* momentum old-velocity) (/ (* learning-rate gradient-with-decay) denominator))]
        [(- weight velocity) {:mean mean :variance variance :velocity velocity}])
      :adagrad
      (let [variance (+ old-variance (* gradient-with-decay gradient-with-decay))
            step (/ (* learning-rate gradient-with-decay) (+ (Math/sqrt variance) epsilon))]
        [(- weight step) {:variance variance}])
      :adadelta
      (let [variance (+ (* rho old-variance) (* (- 1 rho) gradient-with-decay gradient-with-decay))
            step (* gradient-with-decay (Math/sqrt (/ (+ old-delta epsilon) (+ variance epsilon))))
            delta (+ (* rho old-delta) (* (- 1 rho) step step))]
        [(- weight step) {:variance variance :delta delta}]))))

(def ^:private cases
  [[:sgd {:lr 0.1 :momentum 0}]
   [:sgd {:lr 0.1 :momentum 0.5}]
   [:nag {:lr 0.1 :momentum 0}]
   [:nag {:lr 0.1 :momentum 0.5}]
   [:adam {:lr 0.1 :beta1 0.5 :beta2 0.75}]
   [:adamw {:lr 0.1 :beta1 0.5 :beta2 0.75}]
   [:rmsprop {:lr 0.1 :rho 0.5 :momentum 0 :centered? false}]
   [:rmsprop {:lr 0.1 :rho 0.5 :momentum 0.25 :centered? false}]
   [:rmsprop {:lr 0.1 :rho 0.5 :momentum 0.25 :centered? true}]
   [:adagrad {:lr 0.1}]
   [:adadelta {:rho 0.5}]])

(deftest every-family-matches-two-step-multidimensional-reference
  (doseq [[tag family-options] cases
          controls [{} {:rescale-grad 0.5 :clip-grad 1.2 :weight-decay 0.2}]]
    (testing (pr-str [tag family-options controls])
      (with-open [^Session current-session (session/open {:device :cpu})]
        (let [options (merge family-options {:epsilon 0.01} controls)
              ^Optimizer optimizer (training/optimizer [tag (if (#{:sgd :nag} tag)
                                                              (dissoc options :epsilon) options)])
              ^NDArray weight (tensor/array current-session [[1.0 -2.0] [0.5 3.0]] {:dtype :float32})
              gradients [[2.0 -4.0 0.25 -0.5] [-1.0 3.0 0.75 -2.0]]]
          (reduce (fn [expected [update values]]
                    (with-open [^NDManager batch (.newSubManager (context/manager current-session))]
                      (let [^NDArray gradient (tensor/array batch values {:shape [2 2] :dtype :float32})
                            next-values (mapv #(reference-step tag options update %1 %2) expected values)]
                        (.update optimizer "matrix" weight gradient)
                        (is (= values (mapv double (.toFloatArray gradient))) "Input gradient is not mutated")
                        (is (every? true? (map near? (mapv first next-values) (.toFloatArray weight))))
                        (is (= [2 2] (vec (.getShape (.getShape weight)))))
                        next-values)))
                  (mapv #(vector % {}) [1.0 -2.0 0.5 3.0])
                  (map vector [1 2] gradients)))))))

(deftest nag-has-exact-lookahead-not-ordinary-momentum
  (with-open [current-session (session/open {:device :cpu})]
    (let [^Optimizer optimizer (training/optimizer [:nag {:lr 0.1 :momentum 0.5}])
          ^NDArray weight (tensor/array current-session [1.0] {:dtype :float32})
          ^NDArray gradient (tensor/array current-session [2.0] {:dtype :float32})]
      (.update optimizer "nag" weight gradient)
      (is (near? 0.7 (first (.toFloatArray weight))))
      (.update optimizer "nag" weight gradient)
      (is (near? 0.35 (first (.toFloatArray weight)))))))

(deftest clipping-is-symmetric-and-precedes-l2-decay
  (with-open [current-session (session/open {:device :cpu})]
    (let [^Optimizer optimizer (training/optimizer [:sgd {:lr 0.1 :rescale-grad 0.5
                                                          :clip-grad 1 :weight-decay 0.2}])
          ^NDArray weight (tensor/array current-session [2.0 -2.0] {:dtype :float32})
          ^NDArray gradient (tensor/array current-session [10.0 -10.0] {:dtype :float32})]
      (.update optimizer "clipped" weight gradient)
      (is (every? true? (map near? [1.86 -1.86] (.toFloatArray weight))))
      (is (= [10.0 -10.0] (mapv double (.toFloatArray gradient)))))))

(deftest adamw-decay-is-decoupled-from-moments
  (with-open [current-session (session/open {:device :cpu})]
    (let [^NDArray coupled (tensor/array current-session [2.0] {:dtype :float32})
          ^NDArray decoupled (tensor/array current-session [2.0] {:dtype :float32})
          ^NDArray gradient (tensor/array current-session [0.0] {:dtype :float32})
          ^Optimizer adam (training/optimizer [:adam {:lr 0.1 :weight-decay 0.2 :beta1 0 :beta2 0}])
          ^Optimizer adamw (training/optimizer [:adamw {:lr 0.1 :weight-decay 0.2 :beta1 0 :beta2 0}])]
      (.update adam "coupled" coupled gradient)
      (.update adamw "decoupled" decoupled gradient)
      (is (near? 1.9 (first (.toFloatArray coupled))))
      (is (near? 1.96 (first (.toFloatArray decoupled))))
      (.update adam "coupled" coupled gradient)
      (.update adamw "decoupled" decoupled gradient)
      (is (near? 1.8 (first (.toFloatArray coupled))))
      (is (near? 1.9208 (first (.toFloatArray decoupled)))))))

(deftest default-learning-rate-and-adamw-decay-remain-compatible
  (with-open [current-session (session/open {:device :cpu})]
    (doseq [[spec expected] [[[:sgd {}] 1.99] [[:adam {}] 1.99] [[:adamw {}] 1.9898]]]
      (let [^Optimizer optimizer (training/optimizer spec)
            ^NDArray weight (tensor/array current-session [2.0] {:dtype :float32})
            ^NDArray gradient (tensor/array current-session [1.0] {:dtype :float32})]
        (.update optimizer "default" weight gradient)
        (is (near? expected (first (.toFloatArray weight))))))))

(deftest schedules-observe-native-counter-and-begin-update-boundaries
  (with-open [current-session (session/open {:device :cpu})]
    (let [^Optimizer scheduled (training/optimizer [:sgd {:lr [:multi-factor {:base 0.2 :steps [2 5] :factor 0.5}]
                                                          :begin-update 1}])
          ^NDArray weight (tensor/array current-session [1.0] {:dtype :float32})
          ^NDArray gradient (tensor/array current-session [1.0] {:dtype :float32})]
      (.update scheduled "schedule" weight gradient)
      (is (near? 0.8 (first (.toFloatArray weight))))
      (.update scheduled "schedule" weight gradient)
      (is (near? 0.7 (first (.toFloatArray weight))))))
  (with-open [current-session (session/open {:device :cpu})]
    (let [rate (reify ParameterTracker
                 (getNewValue [_ parameter update]
                   (float (* (if (= parameter "left") 0.01 0.02) update))))
          ^Optimizer optimizer (training/optimizer [:sgd {:lr rate :begin-update 2}])
          ^NDArray left (tensor/array current-session [1.0] {:dtype :float32})
          ^NDArray right (tensor/array current-session [1.0] {:dtype :float32})
          ^NDArray gradient (tensor/array current-session [1.0] {:dtype :float32})]
      (.update optimizer "left" left gradient)
      (.update optimizer "right" right gradient)
      (.update optimizer "left" left gradient)
      ;; DJL updateCount returns the maximum parameter update count, rather than a global increment.
      (.update optimizer "right" right gradient)
      (is (near? 0.93 (first (.toFloatArray left))))
      (is (near? 0.86 (first (.toFloatArray right)))))))

(deftest begin-update-also-controls-adam-bias-correction
  (with-open [current-session (session/open {:device :cpu})]
    (let [options {:lr 0.1 :beta1 0.5 :beta2 0.75 :epsilon 0.01 :begin-update 3}
          ^Optimizer optimizer (training/optimizer [:adam options])
          ^NDArray weight (tensor/array current-session [1.0] {:dtype :float32})
          ^NDArray gradient (tensor/array current-session [2.0] {:dtype :float32})
          expected (reductions (fn [state update] (reference-step :adam options update state 2.0))
                               [1.0 {}] [4 5])]
      (doseq [value (rest expected)]
        (.update optimizer "adam" weight gradient)
        (is (near? (first value) (first (.toFloatArray weight))))))))

(deftest every-learning-rate-family-consumes-parameter-schedules
  (doseq [[tag options] (remove #(= :adadelta (first %)) cases)]
    (with-open [current-session (session/open {:device :cpu})]
      (let [rate (reify ParameterTracker
                   (getNewValue [_ parameter update]
                     (is (= "scheduled-family" parameter))
                     (float (* 0.1 update))))
            scheduled-options (assoc options :lr rate :begin-update 1)
            ^Optimizer optimizer (training/optimizer [tag scheduled-options])
            ^NDArray weight (tensor/array current-session [1.0] {:dtype :float32})
            ^NDArray gradient (tensor/array current-session [2.0] {:dtype :float32})]
        (reduce (fn [expected update]
                  (let [next-value (reference-step tag (assoc options :lr (* 0.1 update)) update expected 2.0)]
                    (.update optimizer "scheduled-family" weight gradient)
                    (is (near? (first next-value) (first (.toFloatArray weight))) (name tag))
                    next-value))
                [1.0 {}] [2 3])))))

(defn- managed-history [^NDManager manager ^NDArray weight]
  ;; NDArray equals compares contents, so identify state by UID rather than tensor equality.
  (into {} (comp (remove #(identical? weight %))
                 (map (fn [^NDArray array] [(.getUid array) array])))
        (.getManagedArrays manager)))

(deftest state-is-reused-device-native-and-owned-by-parameter-manager
  (doseq [[tag options state-count] [[:nag {:lr 0.1 :momentum 0.5} 1]
                                     [:adam {:lr 0.1} 2]
                                     [:adamw {:lr 0.1} 2]
                                     [:rmsprop {:lr 0.1 :centered? true} 3]
                                     [:adagrad {:lr 0.1} 1]
                                     [:adadelta {} 2]]]
    (testing (name tag)
      (let [state-arrays
            (with-open [^Session current-session (session/open {:device :cpu})
                        ^NDManager owner (.newSubManager (context/manager current-session))]
              (let [^Optimizer optimizer (training/optimizer [tag options])
                    ^NDArray weight (tensor/array owner [[1.0 2.0] [3.0 4.0]] {:dtype :float64})]
                (with-open [^NDManager batch (.newSubManager owner)
                            ^NDScope scope (NDScope.)]
                  (.suppressNotUsedWarning scope)
                  (.update optimizer "owned" weight
                           (tensor/array batch [[2.0 -1.0] [0.5 3.0]] {:dtype :float64})))
                (let [history (managed-history owner weight)]
                  (is (= state-count (count history)))
                  (doseq [^NDArray array (vals history)]
                    (is (identical? owner (.getManager array)))
                    (is (= (.getDevice weight) (.getDevice array)))
                    (is (= DataType/FLOAT64 (.getDataType array)))
                    (is (= (.getShape weight) (.getShape array))))
                  (with-open [^NDManager next-batch (.newSubManager owner)]
                    (.update optimizer "owned" weight
                             (tensor/array next-batch [[-1.0 0.5] [2.0 -3.0]] {:dtype :float64})))
                  (let [next-history (managed-history owner weight)]
                    (is (= (set (keys history)) (set (keys next-history)))
                        "Second update leaves no temporaries")
                    (is (every? (fn [[identifier array]] (identical? array (get next-history identifier))) history)
                        "Second update reuses the same state arrays"))
                  (is (= DataType/FLOAT64 (.getDataType weight)))
                  (vals history))))]
        (doseq [^NDArray array state-arrays]
          (is (thrown? IllegalStateException (.toDoubleArray array))
              "Closing parameter manager closes optimizer state"))))))

(deftest parameters-have-independent-history-and-closed-manager-state-is-not-reused
  (let [^Optimizer optimizer (training/optimizer [:nag {:lr 0.1 :momentum 0.5}])]
    (with-open [^Session current-session (session/open {:device :cpu})]
      (with-open [^NDManager manager (.newSubManager (context/manager current-session))]
        (let [^NDArray first-weight (tensor/array manager [1.0] {:dtype :float32})
              ^NDArray second-weight (tensor/array manager [1.0] {:dtype :float32})
              ^NDArray gradient (tensor/array manager [2.0] {:dtype :float32})]
          (.update optimizer "first" first-weight gradient)
          (.update optimizer "first" first-weight gradient)
          (.update optimizer "second" second-weight gradient)
          (is (near? 0.35 (first (.toFloatArray first-weight))))
          (is (near? 0.7 (first (.toFloatArray second-weight))))))
      (with-open [^NDManager replacement (.newSubManager (context/manager current-session))]
        (let [^NDArray weight (tensor/array replacement [1.0] {:dtype :float32})
              ^NDArray gradient (tensor/array replacement [2.0] {:dtype :float32})]
          (.update optimizer "first" weight gradient)
          (is (near? 0.7 (first (.toFloatArray weight)))))))))

(deftest updates-preserve-autograd-parameters-and-clear-attached-gradients
  (with-open [^Session current-session (session/open {:device :cpu})]
    (let [^NDManager manager (context/manager current-session)
          ^Optimizer optimizer (training/optimizer [:sgd {:lr 0.1}])
          ^NDArray weight (tensor/array current-session [1.0] {:dtype :float32})]
      (.setRequiresGradient weight true)
      (doseq [expected [0.8 0.64]]
        (with-open [^GradientCollector collector (.newGradientCollector (.getEngine manager))
                    ^NDArray squared (.square weight)
                    ^NDArray objective (.sum squared)]
          (.backward collector objective))
        (with-open [^NDArray attached-gradient (.getGradient weight)
                    ^NDArray independent-gradient (.duplicate attached-gradient)]
          (.update optimizer "autograd" weight independent-gradient)
          (is (pos? (first (.toFloatArray independent-gradient)))))
        (is (near? expected (first (.toFloatArray weight))))
        (is (.hasGradient weight))
        (with-open [^NDArray gradient (.getGradient weight)]
          (is (near? 0 (first (.toFloatArray gradient)))))))))

(deftest update-applies-state-across-steps-without-caller-hints
  (with-open [current-session (session/open)]
    (let [optimizer (training/optimizer [:sgd {:lr 0.1 :momentum 0.5}])
          weight (tensor/array current-session [1.0 2.0] {:dtype :float32})
          gradient (tensor/array current-session [1.0 1.0] {:dtype :float32})
          warnings (java.io.StringWriter.)]
      (binding [*err* (java.io.PrintWriter. warnings)
                *warn-on-reflection* true
                *ns* (the-ns 'garun.optimizers-test)]
        (eval '(fn [optimizer weight gradient]
                 (training/update! optimizer "w" weight gradient))))
      (is (= "" (str warnings)))
      (is (identical? weight (training/update! optimizer "w" weight gradient)))
      (is (every? true? (map near? [0.9 1.9] (.toFloatArray ^NDArray weight))))
      ;; Momentum state is keyed by identifier: the second step is 0.1 + 0.5 * 0.1.
      (training/update! optimizer "w" weight gradient)
      (is (every? true? (map near? [0.75 1.75] (.toFloatArray ^NDArray weight))))
      (is (= :nn/invalid-optimizer
             (:error (ex-data (try (training/update! [:sgd {}] "w" weight gradient)
                                   (catch clojure.lang.ExceptionInfo error error)))))))))
