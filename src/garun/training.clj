(ns garun.training
  "DJL training components from closed, readable option maps or native instances.
  Components retain DJL's tensor shape semantics. Regularization losses retain
  caller-owned parameter NDLists; keep their manager alive throughout training."
  (:require [garun.tensor :as tensor]
            [garun.training.losses :as losses]
            [garun.training.optimizers :as optimizers])
  (:import [ai.djl Device]
           [ai.djl.ndarray NDArray NDList]
           [ai.djl.nn Parameter$Type]
           [ai.djl.training DefaultTrainingConfig TrainingConfig]
           [ai.djl.training.evaluator Evaluator]
           [ai.djl.training.initializer ConstantInitializer Initializer NormalInitializer
            TruncatedNormalInitializer UniformInitializer XavierInitializer
            XavierInitializer$FactorType XavierInitializer$RandomType]
           [ai.djl.training.listener TrainingListener TrainingListener$Defaults]
           [ai.djl.training.loss Loss SimpleCompositeLoss]
           [ai.djl.training.optimizer Optimizer Optimizer$OptimizerBuilder]
           [ai.djl.training.tracker Tracker ParameterTracker LinearTracker
            PolynomialDecayTracker WarmUpTracker$Mode]
           [java.util Collection]
           [java.util.concurrent ExecutorService]))

(set! *warn-on-reflection* true)

(defn- invalid [error message data]
  (throw (ex-info message (assoc data :error error))))

(defn- options [opts allowed context]
  (when-not (map? opts)
    (invalid :nn/invalid-options "Options must be a map" {:context context :options opts}))
  (when-let [unknown (seq (remove allowed (keys opts)))]
    (invalid :nn/unknown-option "Unknown neural network option"
             {:context context :keys (vec unknown)}))
  opts)

(defn- component-spec [spec error context]
  (cond
    (keyword? spec) [spec {}]
    (and (vector? spec) (= 2 (count spec))) spec
    :else (invalid error "Component must be a keyword or [name options]"
                   {:context context :spec spec})))

(defn- finite? [x]
  (and (number? x) (Double/isFinite (double x))
       (Float/isFinite (float x))))

(defn- nonnegative? [x]
  (and (finite? x) (<= 0 x)))

(defn- positive? [x]
  (and (finite? x) (pos? x)))

(defn- positive-int? [x]
  (and (integer? x) (pos? x) (<= x Integer/MAX_VALUE)))

(defn- probability? [x]
  (and (finite? x) (<= 0 x) (< x 1)))

(defn- validate-values [opts predicates error context]
  (doseq [[key predicate] predicates
          :when (contains? opts key)]
    (when-not (predicate (get opts key))
      (invalid error "Invalid component option" {:context context :key key :value (get opts key)})))
  opts)

(defn tracker
  "Return a native Tracker, a fixed nonnegative number, or [family options].
  Families: :fixed {:value}, :factor {:base :factor} with optional :min or
  :max-updates, :multi-factor {:base :steps} with at least two increasing steps
  and optional :factor,
  :cosine {:base :max-updates} with optional :final,
  :linear {:base :slope :max-updates}, :polynomial {:base :end :steps} with
  optional :power, :warm-up {:tracker :steps} with optional :begin/:mode.
  Updates are DJL optimizer updates, not epochs. Native Trackers expose uncommon
  schedules without approximation."
  [spec]
  (cond
    (instance? Tracker spec) spec
    (number? spec)
    (do
      (when-not (nonnegative? spec)
        (invalid :nn/invalid-tracker "Fixed tracker must be finite and nonnegative" {:tracker spec}))
      (Tracker/fixed (float spec)))
    :else
    (let [[tag raw] (component-spec spec :nn/invalid-tracker :tracker)
          allowed (case tag
                    :fixed #{:value}
                    :factor #{:base :factor :min :max-updates}
                    :multi-factor #{:base :steps :factor}
                    :cosine #{:base :final :max-updates}
                    :linear #{:base :slope :max-updates}
                    :polynomial #{:base :end :steps :power}
                    :warm-up #{:tracker :steps :begin :mode}
                    (invalid :nn/invalid-tracker "Unknown tracker" {:tracker spec}))
          opts (options raw allowed :tracker)
          _ (validate-values opts {:base nonnegative? :value nonnegative? :factor positive?
                                   :min nonnegative? :final nonnegative? :end nonnegative?
                                   :power positive? :slope #(and (finite? %) (not (zero? %))) :max-updates positive-int?
                                   :begin nonnegative?}
                             :nn/invalid-tracker :tracker)
          required (case tag
                     :fixed [:value]
                     :factor [:base :factor]
                     :multi-factor [:base :steps]
                     :cosine [:base :max-updates]
                     :linear [:base :slope :max-updates]
                     :polynomial [:base :end :steps]
                     :warm-up [:tracker :steps])]
      (when-not (every? #(contains? opts %) required)
        (invalid :nn/invalid-tracker "Missing tracker options" {:tracker spec :required required}))
      (case tag
        :fixed (tracker (:value opts))
        :factor (-> (cond-> (-> (Tracker/factor)
                                (.setBaseValue (float (:base opts)))
                                (.setFactor (float (:factor opts))))
                      (contains? opts :min) (.optMinValue (float (:min opts)))
                      (contains? opts :max-updates) (.optMaxUpdates (int (:max-updates opts))))
                    .build)
        :multi-factor
        (let [steps (:steps opts)]
          (when-not (and (vector? steps) (<= 2 (count steps)) (every? positive-int? steps)
                         (apply < (cons 0 steps)))
            (invalid :nn/invalid-tracker "Multi-factor needs at least two strictly increasing positive steps" {:tracker spec}))
          (-> (Tracker/multiFactor) (.setBaseValue (float (:base opts)))
              (.setSteps (int-array steps)) (.optFactor (float (get opts :factor 0.1))) .build))
        :cosine (-> (Tracker/cosine) (.setBaseValue (float (:base opts)))
                    (.optFinalValue (float (get opts :final 0)))
                    (.setMaxUpdates (int (:max-updates opts))) .build)
        :linear (-> (LinearTracker/builder) (.setBaseValue (float (:base opts)))
                    (.optSlope (float (:slope opts)))
                    (.optMaxUpdates (int (:max-updates opts))) .build)
        :polynomial
        (do
          (when-not (positive-int? (:steps opts))
            (invalid :nn/invalid-tracker "Decay steps must be positive" {:tracker spec}))
          (-> (PolynomialDecayTracker/builder) (.setBaseValue (float (:base opts)))
              (.setEndLearningRate (float (:end opts))) (.setDecaySteps (int (:steps opts)))
              (.optPower (float (get opts :power 1))) .build))
        :warm-up
        (let [mode (case (get opts :mode :linear)
                     :linear WarmUpTracker$Mode/LINEAR
                     :constant WarmUpTracker$Mode/CONSTANT
                     (invalid :nn/invalid-tracker "Unknown warm-up mode" {:tracker spec}))]
          (when-not (positive-int? (:steps opts))
            (invalid :nn/invalid-tracker "Warm-up steps must be positive" {:tracker spec}))
          (-> (Tracker/warmUp) (.setMainTracker (tracker (:tracker opts)))
              (.optWarmUpSteps (int (:steps opts))) (.optWarmUpBeginValue (float (get opts :begin 0)))
              (.optWarmUpMode mode) .build))))))

(defn loss
  "Return a native Loss or keyword/[family options]. :mse/:mae weight defaults
  to 1. :bce consumes raw logits unless :from-sigmoid? true. :ce uses sparse
  labels and raw logits by default; :from-logits? false expects log probabilities.
  :masked-ce takes labels [class-indices valid-lengths] and predictions [logits],
  and returns per-sequence values (DJL semantics). :quantile is DJL's twice-pinball
  loss, so quantile .5 equals MAE. :composite sums :losses on the same inputs.
  :l1-decay/:l2-decay/:elastic-net use caller-owned :parameters NDList."
  ^Loss [spec]
  (if (instance? Loss spec)
    spec
    (let [[tag raw] (component-spec spec :nn/invalid-loss :loss)
          allowed (case tag
                    (:mse :mae) #{:name :weight}
                    :bce #{:name :weight :from-sigmoid?}
                    (:ce :masked-ce) #{:name :weight :axis :sparse? :from-logits?}
                    :quantile #{:name :quantile}
                    :hinge #{:name :margin :weight}
                    :composite #{:name :losses}
                    (:l1-decay :l2-decay) #{:name :lambda :parameters}
                    :elastic-net #{:name :lambda1 :lambda2 :parameters}
                    (invalid :nn/invalid-loss "Unknown loss" {:loss spec}))
          opts (options raw allowed :loss)
          _ (validate-values opts {:name #(and (string? %) (seq %)) :weight nonnegative?
                                   :from-sigmoid? boolean? :sparse? boolean? :from-logits? boolean?
                                   :axis #(and (integer? %) (<= Integer/MIN_VALUE % Integer/MAX_VALUE))
                                   :margin #(and (integer? %) (<= 0 % Integer/MAX_VALUE))
                                   :quantile #(and (finite? %) (<= 0 % 1))
                                   :lambda nonnegative? :lambda1 nonnegative? :lambda2 nonnegative?}
                             :nn/invalid-loss :loss)
          name (get opts :name (clojure.core/name tag))
          weight (float (get opts :weight 1))]
      (when (#{:l1-decay :l2-decay :elastic-net} tag)
        (when-not (and (instance? NDList (:parameters opts)) (seq (:parameters opts)))
          (invalid :nn/invalid-loss "Regularization needs a nonempty parameter NDList" {:loss spec})))
      (case tag
        :mse (Loss/l2Loss name weight)
        :mae (Loss/l1Loss name weight)
        :bce (losses/binary-cross-entropy name weight (get opts :from-sigmoid? false))
        :ce (Loss/softmaxCrossEntropyLoss name weight (int (get opts :axis -1))
                                          (get opts :sparse? true) (get opts :from-logits? true))
        :masked-ce (losses/masked-cross-entropy name weight (int (get opts :axis -1))
                                                (get opts :sparse? true) (get opts :from-logits? true))
        :quantile (Loss/quantileL1Loss name (float (get opts :quantile 0.5)))
        :hinge (Loss/hingeLoss name (int (get opts :margin 1)) weight)
        :composite
        (let [components (:losses opts)]
          (when-not (and (sequential? components) (seq components))
            (invalid :nn/invalid-loss "Composite loss requires nonempty :losses" {:loss spec}))
          (reduce (fn [^SimpleCompositeLoss composite component]
                    (.addLoss composite (loss component)))
                  (SimpleCompositeLoss. name) components))
        :l1-decay (Loss/l1WeightedDecay name (float (get opts :lambda 1)) (:parameters opts))
        :l2-decay (Loss/l2WeightedDecay name (float (get opts :lambda 1)) (:parameters opts))
        :elastic-net (Loss/elasticNetWeightedDecay name (float (get opts :lambda1 1))
                                                   (float (get opts :lambda2 1)) (:parameters opts))))))

(defn- optimizer-builder [tag opts rate]
  (case tag
    :sgd (-> (Optimizer/sgd) (.setLearningRateTracker rate)
             (.optMomentum (float (get opts :momentum 0))))
    :nag (-> (Optimizer/nag) (.setLearningRateTracker rate)
             (.setMomentum (float (get opts :momentum 0.9))))
    :adam (cond-> (-> (Optimizer/adam) (.optLearningRateTracker rate))
            (contains? opts :beta1) (.optBeta1 (float (:beta1 opts)))
            (contains? opts :beta2) (.optBeta2 (float (:beta2 opts)))
            (contains? opts :epsilon) (.optEpsilon (float (:epsilon opts))))
    :adamw (cond-> (-> (Optimizer/adamW) (.optLearningRateTracker rate))
             (contains? opts :beta1) (.optBeta1 (float (:beta1 opts)))
             (contains? opts :beta2) (.optBeta2 (float (:beta2 opts)))
             (contains? opts :epsilon) (.optEpsilon (float (:epsilon opts))))
    :rmsprop (cond-> (-> (Optimizer/rmsprop) (.optLearningRateTracker rate))
               (contains? opts :rho) (.optRho (float (:rho opts)))
               (contains? opts :momentum) (.optMomentum (float (:momentum opts)))
               (contains? opts :epsilon) (.optEpsilon (float (:epsilon opts)))
               (contains? opts :centered?) (.optCentered (:centered? opts)))
    :adagrad (cond-> (-> (Optimizer/adagrad) (.optLearningRateTracker rate))
               (contains? opts :epsilon) (.optEpsilon (float (:epsilon opts))))
    :adadelta (cond-> (Optimizer/adadelta)
                (contains? opts :rho) (.optRho (float (:rho opts)))
                (contains? opts :epsilon) (.optEpsilon (float (:epsilon opts))))))

(defn optimizer
  "Return a native Optimizer or [family options]. Families are :sgd, :nag,
  :adam, :adamw, :rmsprop, :adagrad, :adadelta. :lr defaults to .01 and accepts
  a positive number, Tracker spec, or native ParameterTracker. Adadelta has no
  learning-rate API and rejects :lr. Common options: :weight-decay, :clip-grad,
  :rescale-grad, :begin-update. Family options: :momentum, :beta1/:beta2,
  :rho, :epsilon, :centered? as supported by the corresponding DJL optimizer."
  ^Optimizer [spec]
  (if (instance? Optimizer spec)
    spec
    (let [_ (when-not (and (vector? spec) (= 2 (count spec)))
              (invalid :nn/invalid-optimizer "Optimizer must be [name options]" {:optim spec}))
          [tag raw] spec
          family (case tag
                   (:sgd :nag) #{:lr :momentum}
                   (:adam :adamw) #{:lr :beta1 :beta2 :epsilon}
                   :rmsprop #{:lr :rho :momentum :epsilon :centered?}
                   :adagrad #{:lr :epsilon}
                   :adadelta #{:rho :epsilon}
                   (invalid :nn/invalid-optimizer "Unknown optimizer" {:optim spec}))
          opts (options raw (into family #{:weight-decay :clip-grad :rescale-grad :begin-update}) :optimizer)
          _ (validate-values opts {:momentum probability? :beta1 probability? :beta2 probability?
                                   :rho probability? :epsilon positive? :centered? boolean?
                                   :weight-decay nonnegative? :clip-grad positive? :rescale-grad positive?
                                   :begin-update #(and (integer? %) (<= 0 % Integer/MAX_VALUE))}
                             :nn/invalid-optimizer :optimizer)
          lr (get opts :lr 0.01)
          _ (when (and (not= tag :adadelta)
                       (not (or (positive? lr) (instance? ParameterTracker lr)
                                (and (vector? lr) (= 2 (count lr))))))
              (invalid :nn/invalid-optimizer "Invalid optimizer learning rate" {:optim spec}))
          rate (when (not= tag :adadelta)
                 (if (instance? ParameterTracker lr) lr (tracker lr)))
          ^Optimizer$OptimizerBuilder builder (optimizer-builder tag opts rate)]
      (.optWeightDecays builder (float (get opts :weight-decay (if (= tag :adamw) 0.01 0))))
      (when (contains? opts :clip-grad) (.optClipGrad builder (float (:clip-grad opts))))
      (when (contains? opts :rescale-grad) (.setRescaleGrad builder (float (:rescale-grad opts))))
      (when (contains? opts :begin-update) (.optBeginNumUpdate builder (int (:begin-update opts))))
      (optimizers/build tag opts rate builder))))

(defn update!
  "Apply a stateful Optimizer (from optimizer) to one parameter in place and
  return weight. parameter-identifier keys the optimizer's per-parameter state
  and update count, so reuse the same identifier and Optimizer across steps.
  Runs device-natively on the weight's device; gradient is not closed."
  ^NDArray [optimizer parameter-identifier weight gradient]
  (when-not (instance? Optimizer optimizer)
    (invalid :nn/invalid-optimizer
             "update! needs a stateful Optimizer instance; build it once with optimizer"
             {:optimizer optimizer}))
  (.update ^Optimizer optimizer ^String parameter-identifier ^NDArray weight ^NDArray gradient)
  weight)

(defn initializer
  "Return a native Initializer, :zeros/:ones/:xavier, or [family options].
  :constant {:value}; :normal/:truncated-normal {:sigma}; :uniform {:scale};
  :xavier {:random :uniform|:gaussian, :factor :avg|:in|:out, :magnitude}."
  ^Initializer [spec]
  (if (instance? Initializer spec)
    spec
    (let [[tag raw] (component-spec spec :nn/invalid-initializer :initializer)
          allowed (case tag
                    (:zeros :ones) #{}
                    :constant #{:value}
                    (:normal :truncated-normal) #{:sigma}
                    :uniform #{:scale}
                    :xavier #{:random :factor :magnitude}
                    (invalid :nn/invalid-initializer "Unknown initializer" {:initializer spec}))
          opts (options raw allowed :initializer)
          _ (validate-values opts {:value finite? :sigma positive? :scale positive? :magnitude positive?}
                             :nn/invalid-initializer :initializer)]
      (case tag
        :zeros Initializer/ZEROS
        :ones Initializer/ONES
        :constant (do
                    (when-not (contains? opts :value)
                      (invalid :nn/invalid-initializer "Constant initializer requires :value" {:initializer spec}))
                    (ConstantInitializer. (float (:value opts))))
        :normal (NormalInitializer. (float (get opts :sigma 0.01)))
        :truncated-normal (TruncatedNormalInitializer. (float (get opts :sigma 0.01)))
        :uniform (UniformInitializer. (float (get opts :scale 0.07)))
        :xavier (XavierInitializer.
                 (case (get opts :random :uniform)
                   :uniform XavierInitializer$RandomType/UNIFORM
                   :gaussian XavierInitializer$RandomType/GAUSSIAN
                   (invalid :nn/invalid-initializer "Unknown Xavier random type" {:initializer spec}))
                 (case (get opts :factor :avg)
                   :avg XavierInitializer$FactorType/AVG
                   :in XavierInitializer$FactorType/IN
                   :out XavierInitializer$FactorType/OUT
                   (invalid :nn/invalid-initializer "Unknown Xavier factor type" {:initializer spec}))
                 (float (get opts :magnitude 3)))))))

(defn config
  "Return native TrainingConfig unchanged, or build a real DefaultTrainingConfig.
  Keys: :loss (default :mse), :optim (default Adam lr .01), :device or :devices,
  :listeners (default DJL basic evaluation/epoch listeners), :evaluators (native),
  :initializer with :parameter-type (default :weight), :executor (caller-owned
  ExecutorService). Explicit listeners replace defaults; [] disables them.
  Without explicit devices DJL selects its default devices. No executor is
  created implicitly. Native components and configs are not EDN serializable."
  ^TrainingConfig [spec]
  (if (instance? TrainingConfig spec)
    spec
    (let [opts (options spec #{:loss :optim :device :devices :listeners :evaluators
                               :initializer :parameter-type :executor} :training-config)
          _ (when (and (contains? opts :device) (contains? opts :devices))
              (invalid :nn/invalid-training-config "Choose :device or :devices, not both" {:options opts}))
          _ (when (and (contains? opts :parameter-type) (not (contains? opts :initializer)))
              (invalid :nn/invalid-training-config ":parameter-type requires :initializer" {:options opts}))
          devices (cond
                    (contains? opts :devices) (:devices opts)
                    (contains? opts :device) [(:device opts)]
                    :else nil)
          listeners (get opts :listeners (vec (TrainingListener$Defaults/basic)))
          evaluators (get opts :evaluators [])
          ^DefaultTrainingConfig result (-> (DefaultTrainingConfig. (loss (get opts :loss :mse)))
                                            (.optOptimizer (optimizer (get opts :optim [:adam {:lr 0.01}]))))]
      (when (and (contains? opts :devices) (not (and (sequential? devices) (seq devices))))
        (invalid :nn/invalid-training-config "Devices must be a nonempty sequence" {:options opts}))
      (when-not (and (sequential? listeners) (every? #(instance? TrainingListener %) listeners)
                     (sequential? evaluators) (every? #(instance? Evaluator %) evaluators))
        (invalid :nn/invalid-training-config "Listeners/evaluators must be sequences of native DJL instances" {:options opts}))
      (when devices (.optDevices result (into-array Device (map tensor/device devices))))
      (.addTrainingListeners result (into-array TrainingListener listeners))
      (.addEvaluators result ^Collection evaluators)
      (when (contains? opts :initializer)
        (let [parameter-type (get opts :parameter-type :weight)
              resolved (if (instance? Parameter$Type parameter-type)
                         parameter-type
                         (case parameter-type
                           :weight Parameter$Type/WEIGHT :bias Parameter$Type/BIAS
                           :gamma Parameter$Type/GAMMA :beta Parameter$Type/BETA
                           :running-mean Parameter$Type/RUNNING_MEAN
                           :running-var Parameter$Type/RUNNING_VAR :other Parameter$Type/OTHER
                           (invalid :nn/invalid-training-config "Unknown parameter type" {:parameter-type parameter-type})))]
          (.optInitializer result ^Initializer (initializer (:initializer opts)) ^Parameter$Type resolved)))
      (when (contains? opts :executor)
        (when-not (instance? ExecutorService (:executor opts))
          (invalid :nn/invalid-training-config "Executor must be a native ExecutorService" {:executor (:executor opts)}))
        (.optExecutorService result ^ExecutorService (:executor opts)))
      result)))
