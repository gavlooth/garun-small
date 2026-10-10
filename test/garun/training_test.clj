(ns garun.training-test
  (:require [clojure.test :refer [deftest is]]
            [garun.context :as context]
            [garun.session :as session]
            [garun.tensor :as tensor]
            [garun.training :as training])
  (:import [ai.djl Device Model]
           [ai.djl.ndarray NDArray NDList NDManager]
           [ai.djl.ndarray.types DataType Shape]
           [ai.djl.nn Parameter$Type]
           [ai.djl.nn.core Linear]
           [ai.djl.training DefaultTrainingConfig EasyTrain GradientCollector Trainer TrainingConfig TrainingResult]
           [ai.djl.training.dataset ArrayDataset ArrayDataset$Builder]
           [ai.djl.training.initializer Initializer]
           [ai.djl.training.listener TrainingListener$Defaults]
           [ai.djl.training.loss Loss]
           [ai.djl.training.optimizer Optimizer]
           [ai.djl.training.tracker ParameterTracker Tracker]
           [garun.session Session]
           [java.util.concurrent ExecutorService Executors]))

(set! *warn-on-reflection* true)

(defn- near? [expected actual]
  (< (Math/abs (- (double expected) (double actual))) 1.0e-5))

(defn- error-key [operation]
  (try
    (operation)
    nil
    (catch clojure.lang.ExceptionInfo error
      (:error (ex-data error)))))

(defn- loss-values! [owner spec labels predictions]
  (let [inputs (fn [values]
                 (if (or (map? values) (and (vector? values) (every? map? values)))
                   values
                   {:data values :dtype :float32}))
        ^Loss component (training/loss spec)
        ^NDArray result (.evaluate component (tensor/ndlist owner (inputs labels))
                                   (tensor/ndlist owner (inputs predictions)))]
    (vec (.toFloatArray result))))

(deftest established-loss-semantics-on-multidimensional-values
  (with-open [current-session (session/open {:device :cpu})]
    (let [labels [[[1.0 2.0] [3.0 4.0]]]
          predictions [[[2.0 0.0] [3.0 8.0]]]]
      (is (near? 5.25 (first (loss-values! current-session :mse labels predictions))))
      (is (near? 1.75 (first (loss-values! current-session :mae labels predictions))))
      (is (near? 10.5 (first (loss-values! current-session [:mse {:weight 2}] labels predictions))))
      (is (near? 1.75 (first (loss-values! current-session :quantile labels predictions))))))
  (with-open [current-session (session/open {:device :cpu})]
    (is (near? (Math/log 2) (first (loss-values! current-session :bce [0.0 1.0] [0.0 0.0]))))
    (is (near? (Math/log 2) (first (loss-values! current-session [:bce {:from-sigmoid? true}]
                                                 [0.0 1.0] [0.5 0.5]))))
    (is (near? (Math/log 2) (first (loss-values! current-session :ce
                                                 {:data [0 1] :dtype :int64}
                                                 [[0.0 0.0] [0.0 0.0]]))))
    (is (near? (Math/log 2) (first (loss-values! current-session [:ce {:sparse? false}]
                                                 [[1.0 0.0] [0.0 1.0]]
                                                 [[0.0 0.0] [0.0 0.0]]))))))

(deftest quantile-hinge-and-composite-boundaries
  (with-open [current-session (session/open {:device :cpu})]
    (doseq [[quantile expected] [[0 4.0] [0.25 3.5] [0.5 3.0] [1 2.0]]]
      (is (near? expected (first (loss-values! current-session [:quantile {:quantile quantile}]
                                               [2.0 0.0] [0.0 4.0])))))
    (is (near? 0.0 (first (loss-values! current-session :hinge [1.0 -1.0] [1.0 -1.0]))))
    (is (near? 1.5 (first (loss-values! current-session :hinge [1.0 -1.0] [0.0 1.0]))))
    (is (near? 4.0 (first (loss-values! current-session [:composite {:losses [:mse :mae]}]
                                        [0.0 0.0] [1.0 2.0]))))))

(deftest masked-cross-entropy-ignores-padding-and-zero-length-sequences
  (with-open [current-session (session/open {:device :cpu})]
    (let [labels [{:data [[0 1] [0 1]] :dtype :int64}
                  {:data [1 0] :dtype :int64}]
          logits [[[0.0 0.0] [100.0 -100.0]] [[0.0 0.0] [100.0 -100.0]]]
          values (loss-values! current-session :masked-ce labels logits)]
      ;; DJL averages over padded sequence width, rather than valid lengths.
      (is (= 2 (count values)))
      (is (near? (/ (Math/log 2) 2) (first values)))
      (is (near? 0 (second values))))))

(deftest native-regularization-values-and-gradients
  (with-open [^Session current-session (session/open {:device :cpu})]
    (let [^NDManager manager (context/manager current-session)
          ^NDArray parameters (tensor/array current-session [-2.0 3.0] {:dtype :float32})
          parameter-list (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray [parameters]))
          predictions (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray [(.zeros manager (Shape. (long-array [2])))]))
          labels (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray [(.zeros manager (Shape. (long-array [2])))]))
          ^Loss combined (training/loss [:composite {:losses [:mse [:elastic-net {:parameters parameter-list
                                                                                  :lambda1 0.5 :lambda2 0.25}]]}])]
      (is (near? 2.5 (first (loss-values! current-session [:l1-decay {:parameters parameter-list :lambda 0.5}]
                                          [0.0] [0.0]))))
      (is (near? 3.25 (first (loss-values! current-session [:l2-decay {:parameters parameter-list :lambda 0.25}]
                                           [0.0] [0.0]))))
      (.setRequiresGradient parameters true)
      (with-open [^GradientCollector collector (.newGradientCollector (.getEngine manager))]
        (let [^NDArray value (.evaluate combined labels predictions)]
          (is (near? 5.75 (aget ^floats (.toFloatArray value) 0)))
          (.backward collector value)))
      (is (every? true? (map near? [-1.5 2.0] (vec (.toFloatArray (.getGradient parameters)))))))))

(deftest schedules-follow-update-boundaries
  (doseq [[spec updates expected]
          [[[:factor {:base 1 :factor 0.5 :max-updates 2}] [0 1 2 3] [1 0.5 0.25 0.25]]
           [[:multi-factor {:base 1 :steps [2 4] :factor 0.5}] [0 2 3 4 5] [1 1 0.5 0.5 0.25]]
           [[:cosine {:base 1 :final 0 :max-updates 4}] [0 2 4 5] [1 0.5 0 0]]
           [[:linear {:base 1 :slope -0.25 :max-updates 4}] [0 2 4 5] [1 0.5 0 0]]
           [[:polynomial {:base 1 :end 0 :steps 4 :power 2}] [0 2 4 5] [1 0.25 0 0]]
           [[:warm-up {:tracker 1 :steps 2 :begin 0 :mode :linear}] [0 1 2 3] [0 0.5 1 1]]
           [[:warm-up {:tracker 1 :steps 2 :begin 0.1 :mode :constant}] [0 1 2] [0.1 0.1 1]]]]
    (let [^Tracker schedule (training/tracker spec)]
      (is (every? true? (map near? expected (mapv #(.getNewValue schedule (int %)) updates)))
          (pr-str spec)))))

(defn- train-step! [^Trainer trainer ^NDList input ^NDList labels]
  (with-open [^GradientCollector collector (.newGradientCollector trainer)]
    (.backward collector (.evaluate (.getLoss trainer) labels (.forward trainer input))))
  (.step trainer))

(defn- optimizer-trajectory! [optimizer-spec steps]
  (with-open [^Model model (Model/newInstance "training-test" (Device/cpu) "PyTorch")]
    (.setBlock model (-> (Linear/builder) (.setUnits (long 1)) (.optBias false) .build))
    (with-open [^Trainer trainer (.newTrainer model ^TrainingConfig
                                              (training/config {:device :cpu :optim optimizer-spec :initializer :zeros}))]
      (.initialize trainer (into-array Shape [(Shape. (long-array [2 2 1]))]))
      (let [^NDManager manager (.getManager trainer)
            input (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray [(tensor/array manager [[[1.0] [1.0]] [[1.0] [1.0]]]
                                                                                           {:dtype :float32})]))
            labels (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray [(tensor/array manager [[[2.0] [2.0]] [[2.0] [2.0]]]
                                                                                            {:dtype :float32})]))]
        (mapv (fn [_]
                (train-step! trainer input labels)
                (let [^NDArray output (.singletonOrThrow (.evaluate trainer input))]
                  (double (aget ^floats (.toFloatArray output) 0))))
              (range steps))))))

(deftest all-optimizer-families-update-real-multidimensional-models
  (doseq [optimizer-spec [[:sgd {:lr 0.05 :momentum 0.5}]
                          [:nag {:lr 0.05 :momentum 0.5}]
                          [:adam {:lr 0.05 :beta1 0.8 :beta2 0.95}]
                          [:adamw {:lr 0.05 :weight-decay 0.01}]
                          [:rmsprop {:lr 0.05 :rho 0.9 :momentum 0.2 :centered? true}]
                          [:adagrad {:lr 0.1 :epsilon 1.0e-6}]
                          [:adadelta {:rho 0.9 :epsilon 0.001}]]]
    (let [trajectory (optimizer-trajectory! optimizer-spec 12)
          first-error (Math/pow (- 2 (first trajectory)) 2)
          last-error (Math/pow (- 2 (last trajectory)) 2)]
      (is (every? #(Double/isFinite (double %)) trajectory) (pr-str optimizer-spec))
      (is (< last-error first-error) (pr-str optimizer-spec))
      (is (< last-error 4.0) (pr-str optimizer-spec)))))

(deftest schedules-and-gradient-controls-change-actual-updates
  (let [fixed (optimizer-trajectory! [:sgd {:lr 0.1}] 2)
        decayed (optimizer-trajectory! [:sgd {:lr [:factor {:base 0.1 :factor 0.5}]}] 2)
        clipped (optimizer-trajectory! [:sgd {:lr 0.1 :clip-grad 1}] 1)
        rescaled (optimizer-trajectory! [:sgd {:lr 0.1 :rescale-grad 0.5}] 1)
        native-rate (reify ParameterTracker
                      (getNewValue [_ _parameter _update] (float 0.05)))
        native (optimizer-trajectory! (-> (Optimizer/sgd) (.setLearningRateTracker native-rate) .build) 1)]
    (is (every? true? (map near? [0.4 0.72] fixed)))
    (is (every? true? (map near? [0.2 0.29] decayed)))
    (is (near? 0.1 (first clipped)))
    (is (near? 0.2 (first rescaled)))
    (is (near? 0.2 (first native)))))

(deftest initializers-generate-real-values-with-native-escape
  (with-open [^Session current-session (session/open {:device :cpu})]
    (doseq [[spec expected] [[:zeros 0] [:ones 1] [[:constant {:value -2}] -2]
                             [(training/initializer [:constant {:value 3}]) 3]]]
      (let [^Initializer component (training/initializer spec)
            ^NDArray values (.initialize component (context/manager current-session) (Shape. (long-array [2 3])) DataType/FLOAT32)]
        (is (= [2 3] (vec (.getShape (.getShape values)))))
        (is (every? #(near? expected %) (.toFloatArray values)))))
    (doseq [[spec bound] [[[:uniform {:scale 0.2}] 0.2]
                          [[:truncated-normal {:sigma 0.1}] 0.2]
                          [[:xavier {:random :uniform :factor :in :magnitude 3}] 0.5]]]
      (let [^Initializer component (training/initializer spec)
            ^NDArray values (.initialize component (context/manager current-session) (Shape. (long-array [24 12])) DataType/FLOAT32)
            numbers (vec (.toFloatArray values))]
        (is (every? #(<= (Math/abs (double %)) (+ bound 1.0e-6)) numbers))
        (is (some pos? numbers))
        (is (some neg? numbers))))))

(deftest native-config-evaluators-listeners-and-executor-train-a-real-dataset
  (with-open [current-session (session/open {:device :cpu})
              ^Model model (Model/newInstance "config-test" (Device/cpu) "PyTorch")]
    (let [^ExecutorService executor (Executors/newSingleThreadExecutor)
          ^NDArray input (tensor/array current-session [[1.0] [1.0] [1.0] [1.0]] {:dtype :float32})
          ^NDArray labels (tensor/array current-session [[2.0] [2.0] [2.0] [2.0]] {:dtype :float32})
          ^ArrayDataset$Builder dataset-builder (doto (ArrayDataset$Builder.)
                                                  (.setData (into-array NDArray [input]))
                                                  (.optLabels (into-array NDArray [labels]))
                                                  (.setSampling (int 4) false))
          ^ArrayDataset dataset (.build dataset-builder)
          ^Loss evaluator (training/loss [:mae {:name "absolute-error"}])
          ^DefaultTrainingConfig native-config (training/config {:devices [:cpu] :loss (training/loss :mse)
                                                                 :optim [:sgd {:lr 0.1}]
                                                                 :initializer :zeros :parameter-type Parameter$Type/WEIGHT
                                                                 :evaluators [evaluator]
                                                                 :listeners (vec (TrainingListener$Defaults/basic))
                                                                 :executor executor})]
      (try
        (.setBlock model (-> (Linear/builder) (.setUnits (long 1)) (.optBias false) .build))
        (.prepare dataset)
        (with-open [^Trainer trainer (.newTrainer model ^TrainingConfig (training/config native-config))]
          (.initialize trainer (into-array Shape [(Shape. (long-array [4 1]))]))
          (EasyTrain/fit trainer (int 8) dataset dataset)
          (let [^TrainingResult result (.getTrainingResult trainer)
                ^NDArray output (.singletonOrThrow (.evaluate trainer (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray [input]))))]
            (is (= 8 (.getEpoch result)))
            (is (< (.getTrainLoss result) 0.3))
            (is (< (.getValidateLoss result) 0.3))
            (is (< (.getTrainEvaluation result "absolute-error") 0.5))
            (is (< (.getValidateEvaluation result "absolute-error") 0.5))
            (is (> (aget ^floats (.toFloatArray output) 0) 1.5))))
        (finally
          (.shutdownNow executor))))))

(deftest training-options-fail-loudly
  (doseq [spec [:adam [:bogus {}] [:adam {:lr 0}] [:sgd {:momentum 1}] [:adamw {:weight-decay -1}]]]
    (is (= :nn/invalid-optimizer (error-key #(training/optimizer spec)))))
  (is (= :nn/unknown-option (error-key #(training/optimizer [:adadelta {:lr 0.1}]))))
  (is (= :nn/unknown-option (error-key #(training/optimizer [:adam {:typo 1}]))))
  (is (= :nn/invalid-options (error-key #(training/optimizer [:adam nil]))))
  (doseq [spec [:bogus [:quantile {:quantile -0.1}] [:hinge {:margin 1.5}]
                [:composite {:losses []}] [:l2-decay {}]]]
    (is (= :nn/invalid-loss (error-key #(training/loss spec)))))
  (doseq [spec [[:multi-factor {:base 1 :steps [2]}]
                [:multi-factor {:base 1 :steps [2 2]}] [:cosine {:base 1 :max-updates 0}]
                [:polynomial {:base 1 :end 0 :steps -1}]]]
    (is (= :nn/invalid-tracker (error-key #(training/tracker spec)))))
  (is (= :nn/invalid-initializer (error-key #(training/initializer [:xavier {:factor :bogus}]))))
  (is (= :nn/unknown-option (error-key #(training/config {:oops 1}))))
  (doseq [options [{:device :cpu :devices [:cpu]} {:devices []} {:parameter-type :bias}
                   {:listeners [:logging]} {:evaluators [:mae]} {:executor :ignored}]]
    (is (= :nn/invalid-training-config (error-key #(training/config options))))))
