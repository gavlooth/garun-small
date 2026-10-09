(ns garun.nn
  "CPU neural networks expressed as Garun Hiccup and backed by DJL PyTorch.
  Use build → train! → predict or save!; load restores a saved network.
  A Network owns native resources: close it with with-open."
  (:refer-clojure :exclude [load])
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [garun.table :as table])
  (:import [ai.djl Device Model]
           [ai.djl.engine Engine]
           [ai.djl.inference Predictor]
           [ai.djl.ndarray NDArray NDArrays NDList NDManager]
           [ai.djl.ndarray.types DataType Shape]
           [ai.djl.nn Activation Block Blocks Parameter$Type SequentialBlock]
           [ai.djl.nn.core Linear]
           [ai.djl.nn.norm BatchNorm Dropout LayerNorm]
           [ai.djl.training DefaultTrainingConfig GradientCollector Trainer]
           [ai.djl.training.initializer XavierInitializer]
           [ai.djl.training.loss Loss]
           [ai.djl.training.optimizer AdamW$Builder Optimizer]
           [ai.djl.training.tracker Tracker]
           [ai.djl.translate NoopTranslator]
           [garun.djl FnBlock]
           [java.util List Random]))

(set! *warn-on-reflection* true)

(defrecord Network [spec ^Model model input-shape]
  java.lang.AutoCloseable
  (close [_] (.close model)))

(defn- invalid [error message data]
  (throw (ex-info message (assoc data :error error))))

(defn- options [opts allowed context]
  (when-not (map? opts)
    (invalid :nn/invalid-options "Options must be a map" {:context context :options opts}))
  (when-let [unknown (seq (remove allowed (keys opts)))]
    (invalid :nn/unknown-option "Unknown neural network option" {:context context :keys (vec unknown)}))
  opts)

(defn- positive-int? [x]
  (and (integer? x) (pos? x) (<= x Integer/MAX_VALUE)))

(defn- finite-number? [x]
  (and (number? x) (Double/isFinite (double x))))

(defn- activation [tag]
  (case tag
    :relu (Activation/reluBlock)
    :tanh (Activation/tanhBlock)
    :sigmoid (Activation/sigmoidBlock)
    :gelu (Activation/geluBlock)
    :identity (Blocks/identityBlock)
    (invalid :nn/invalid-activation "Unknown activation" {:activation tag})))

(declare compile-block!)

(defn- sequential! [children]
  (.addAll (SequentialBlock.) ^java.util.Collection (mapv compile-block! children)))

(defn- combine [mode]
  (fn [^List outputs ^NDList _input]
    (let [arrays (mapv (fn [^NDList output] (.singletonOrThrow output)) outputs)
          result (case mode
                   :concat (NDArrays/concat (NDList. ^java.util.Collection arrays) (int -1))
                   :add (reduce (fn [^NDArray a ^NDArray b] (.add a b)) arrays))]
      (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray [result])))))

(defn- compile-linear! [args spec]
  (let [arg (first args)
        opts (if (map? arg) (options arg #{:units :bias?} :linear) {:units arg})
        {:keys [units bias?] :or {bias? true}} opts]
    (when-not (and (= 1 (count args)) (positive-int? units) (boolean? bias?))
      (invalid :nn/invalid-linear "Linear requires positive integer units and boolean bias?" {:spec spec}))
    (-> (Linear/builder) (.setUnits (long units)) (.optBias (boolean bias?)) .build)))

(defn- compile-activation! [args spec]
  (when-not (= 1 (count args))
    (invalid :nn/invalid-spec "Activation requires one name" {:spec spec}))
  (activation (first args)))

(defn- compile-dropout! [args spec]
  (let [{:keys [p] :or {p 0.5}} (options (first args) #{:p} :dropout)]
    (when-not (and (= 1 (count args)) (finite-number? p) (<= 0 p) (< p 1))
      (invalid :nn/invalid-dropout "Dropout p must be in [0, 1)" {:spec spec}))
    (-> (Dropout/builder) (.optRate (float p)) .build)))

(defn- combination-options [args spec]
  (let [opts (options (first args) #{:combine} (first spec))
        mode (:combine opts)
        children (rest args)]
    (when-not (and (#{:concat :add} mode) (seq children))
      (invalid :nn/invalid-combination "Combination requires :add or :concat and children" {:spec spec}))
    {:mode mode :children children}))

(defn- compile-parallel! [args spec]
  (let [{:keys [mode children]} (combination-options args spec)]
    (FnBlock. "parallel" ^List (mapv compile-block! children) (combine mode))))

(defn- compile-skip! [args spec]
  (let [{:keys [mode children]} (combination-options args spec)]
    (FnBlock. "skip" ^List [(sequential! children) (Blocks/identityBlock)] (combine mode))))

(def ^:private block-compilers
  {:sequential {:compile (fn [children _spec] (sequential! children))}
   :linear     {:compile compile-linear!}
   :activation {:compile compile-activation!}
   :dropout    {:compile compile-dropout!}
   :parallel   {:compile compile-parallel!}
   :skip       {:compile compile-skip!}
   :relu       {:no-args? true :compile (fn [_args _spec] (Activation/reluBlock))}
   :tanh       {:no-args? true :compile (fn [_args _spec] (Activation/tanhBlock))}
   :sigmoid    {:no-args? true :compile (fn [_args _spec] (Activation/sigmoidBlock))}
   :gelu       {:no-args? true :compile (fn [_args _spec] (Activation/geluBlock))}
   :identity   {:no-args? true :compile (fn [_args _spec] (Blocks/identityBlock))}
   :flatten    {:no-args? true :compile (fn [_args _spec] (Blocks/batchFlattenBlock))}
   :batchnorm  {:no-args? true :compile (fn [_args _spec] (.build (BatchNorm/builder)))}
   :layernorm  {:no-args? true :compile (fn [_args _spec]
                                          (-> (LayerNorm/builder) (.axis (int-array [1])) .build))}})

(defn- compile-block! [spec]
  (when-not (and (vector? spec) (keyword? (first spec)))
    (invalid :nn/invalid-spec "A network block must be a Hiccup vector" {:spec spec}))
  (let [[tag & args] spec
        {:keys [no-args?] compiler :compile} (get block-compilers tag)]
    (when-not compiler
      (invalid :nn/unknown-tag "Unknown neural network block" {:tag tag}))
    (when (and no-args? (seq args))
      (invalid :nn/invalid-spec "This block accepts no options or children" {:spec spec}))
    (compiler args spec)))

(defn- initialize-network! [spec input-shape seed]
  (let [^Block block (compile-block! spec)
        ^Engine engine (Engine/getInstance)
        _ (.setRandomSeed engine (int seed))
        ^Model model (Model/newInstance "garun" (Device/cpu))]
    (try
      (.setBlock model block)
      (.setInitializer block (XavierInitializer.) Parameter$Type/WEIGHT)
      (.initialize block (.getNDManager model) DataType/FLOAT32
                   (into-array Shape [(Shape. (long-array (into [1] input-shape)))]))
      (->Network spec model input-shape)
      (catch Throwable error
        (.close model)
        (throw error)))))

(defn build
  "Build and initialize a CPU Network from Hiccup and {:input-shape [features]}.
  Close it with with-open. Throws ExceptionInfo for invalid specs or options."
  ^Network [spec opts]
  (let [{:keys [input-shape seed] :or {seed 11}} (options opts #{:input-shape :seed} :build)]
    (when-not (and (vector? input-shape) (= 1 (count input-shape)) (every? positive-int? input-shape))
      (invalid :nn/invalid-input-shape "Input shape must be [positive-feature-count]" {:input-shape input-shape}))
    (when-not (integer? seed)
      (invalid :nn/invalid-seed "Seed must be an integer" {:seed seed}))
    (initialize-network! spec input-shape seed)))

(defn- optimizer! [spec]
  (when-not (and (vector? spec) (= 2 (count spec)))
    (invalid :nn/invalid-optimizer "Optimizer must be [name options]" {:optim spec}))
  (let [[tag opts] spec
        allowed (case tag :sgd #{:lr :momentum} :adam #{:lr} :adamw #{:lr :weight-decay}
                      (invalid :nn/invalid-optimizer "Unknown optimizer" {:optim spec}))
        {:keys [lr momentum weight-decay] :or {lr 0.01 momentum 0.0 weight-decay 0.01}}
        (options opts allowed :optimizer)]
    (when-not (and (finite-number? lr) (pos? lr)
                   (finite-number? momentum) (<= 0 momentum) (< momentum 1)
                   (finite-number? weight-decay) (<= 0 weight-decay))
      (invalid :nn/invalid-optimizer "Invalid optimizer numeric option" {:optim spec}))
    (case tag
      :sgd (-> (Optimizer/sgd) (.setLearningRateTracker (Tracker/fixed (float lr)))
               (.optMomentum (float momentum)) .build)
      :adam (-> (Optimizer/adam) (.optLearningRateTracker (Tracker/fixed (float lr))) .build)
      :adamw (let [^AdamW$Builder builder (-> (Optimizer/adamW)
                                              (.optLearningRateTracker (Tracker/fixed (float lr)))
                                              (.optWeightDecays (float weight-decay)))]
               (.build builder)))))

(defn- loss-function [loss]
  (case loss
    :mse (Loss/l2Loss "mse" (float 1.0))
    :mae (Loss/l1Loss "mae" (float 1.0))
    ;; DJL's boolean is fromSigmoid, so false means raw logits.
    :bce (Loss/sigmoidBinaryCrossEntropyLoss "bce" (float 1.0) false)
    :ce (Loss/softmaxCrossEntropyLoss "ce" (float 1.0) (int -1) true true)
    (invalid :nn/invalid-loss "Unknown loss" {:loss loss})))

(defn- feature-columns [net x]
  (when-not (and (vector? x) (seq x) (every? keyword? x) (= (count x) (first (:input-shape net))))
    (invalid :nn/invalid-features "Features must match the network input shape" {:x x :input-shape (:input-shape net)}))
  x)

(defn- dataset! [net db source x y loss]
  (let [y-cols (if (keyword? y) [y] y)
        _ (when-not (and (vector? y-cols) (seq y-cols) (every? keyword? y-cols))
            (invalid :nn/invalid-labels "Labels must be a keyword or vector of keywords" {:y y}))
        features (table/matrix db source x)
        labels (table/matrix db source y-cols)
        ^Model model (:model net)
        ^Block block (.getBlock model)
        ^Shape output (first (.getOutputShapes block (into-array Shape [(Shape. (long-array [1 (count x)]))])))
        width (.getLastDimension output)
        ^floats label-data (:data labels)]
    (when-not (pos? (:rows features))
      (invalid :nn/empty-data "Training requires at least one row" {}))
    (when-not (= (:rows features) (:rows labels))
      (invalid :nn/invalid-labels "Feature and label row counts differ" {}))
    (when-not (if (= loss :ce) (= 1 (:cols labels)) (= width (:cols labels)))
      (invalid :nn/invalid-labels "Label width does not match output/loss" {:label-width (:cols labels) :output-width width :loss loss}))
    (when (and (= loss :ce)
               (not-every? #(and (<= 0 %) (< % width) (== (double %) (Math/floor (double %)))) label-data))
      (invalid :nn/invalid-labels "CE labels must be integral class indices within the output width" {:output-width width}))
    (when (and (= loss :bce) (not-every? #(<= 0 % 1) label-data))
      (invalid :nn/invalid-labels "BCE labels must be in [0, 1]" {}))
    {:features features :labels labels}))

(defn- matrix-row [^floats values cols row]
  (let [offset (* row cols)]
    (map #(aget values (int (+ offset %))) (range cols))))

(defn- batch-array! [^NDManager manager matrix indices sparse?]
  (let [{:keys [cols data]} matrix
        ^floats values data
        selected (mapcat #(matrix-row values cols %) indices)]
    (if sparse?
      (.create manager (long-array (map long selected)) (Shape. (long-array [(count indices)])))
      (.create manager (float-array selected) (Shape. (long-array [(count indices) cols]))))))

(defn- train-one-batch! [^Trainer trainer ^Loss loss ^NDList input ^NDList labels]
  (let [scalar (with-open [^GradientCollector collector (.newGradientCollector trainer)]
                 (let [^NDArray value (.evaluate loss labels (.forward trainer input))
                       scalar (double (aget ^floats (.toFloatArray value) 0))]
                   (.backward collector value)
                   scalar))]
    ;; DJL requires the gradient collector to close before the optimizer steps.
    (.step trainer)
    scalar))

(defn- batch-loss! [^Trainer trainer ^Loss loss dataset indices sparse? training?]
  (with-open [^NDManager manager (.newSubManager (.getManager trainer))]
    (let [input (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray [(batch-array! manager (:features dataset) indices false)]))
          labels (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray [(batch-array! manager (:labels dataset) indices sparse?)]))]
      (if training?
        (train-one-batch! trainer loss input labels)
        (let [^NDArray value (.evaluate loss labels (.evaluate trainer input))]
          (double (aget ^floats (.toFloatArray value) 0)))))))

(defn- epoch-loss! [trainer loss dataset indices batch sparse? training?]
  (/ (reduce (fn [total rows]
               (+ total (* (count rows) (batch-loss! trainer loss dataset rows sparse? training?))))
             0.0 (partition-all batch indices))
     (count indices)))

(defn- shuffle-indices! [indices ^Random random]
  ;; Fisher-Yates over a persistent vector: RNG is the only mutable host boundary.
  (reduce (fn [order i]
            (let [j (.nextInt random (int (inc i)))]
              (assoc order i (nth order j) j (nth order i))))
          indices (range (dec (count indices)) 0 -1)))

(defn- training-options [net opts]
  (let [{:keys [x y loss epochs batch shuffle? seed optim validation]
         :or {loss :mse epochs 10 batch 32 shuffle? true seed 0 optim [:adam {:lr 0.01}]}}
        (options opts #{:x :y :loss :epochs :batch :shuffle? :seed :optim :validation} :train)]
    (feature-columns net x)
    (when-not (and (positive-int? epochs) (positive-int? batch) (boolean? shuffle?) (integer? seed))
      (invalid :nn/invalid-training-options
               "Epochs/batch must be positive integers; shuffle? boolean; seed integer"
               {:options opts}))
    {:x x :y y :loss loss :epochs epochs :batch batch :shuffle? shuffle?
     :seed seed :optim optim :validation validation}))

(defn- prepare-training! [net db source opts]
  (let [{:keys [x y loss epochs batch shuffle? seed optim validation]} (training-options net opts)
        ^Loss loss-fn (loss-function loss)
        dataset (dataset! net db source x y loss)
        validation-data (when validation (dataset! net db validation x y loss))
        indices (vec (range (get-in dataset [:features :rows])))
        validation-indices (when validation-data
                             (vec (range (get-in validation-data [:features :rows]))))
        config (-> (DefaultTrainingConfig. loss-fn)
                   (.optDevices (into-array Device [(Device/cpu)]))
                   (.optOptimizer (optimizer! optim)))]
    {:loss-fn loss-fn :config config :dataset dataset :indices indices
     :validation-data validation-data :validation-indices validation-indices
     :epochs epochs :batch batch :shuffle? shuffle? :sparse? (= loss :ce)
     :seed seed :random (Random. (long seed))}))

(defn- run-epoch! [trainer training epoch]
  (let [{:keys [loss-fn dataset indices validation-data validation-indices
                batch sparse? shuffle? random]} training
        order (if shuffle? (shuffle-indices! indices random) indices)
        train-loss (epoch-loss! trainer loss-fn dataset order batch sparse? true)]
    (cond-> {:epoch epoch :loss train-loss}
      validation-data
      (assoc :validation-loss
             (epoch-loss! trainer loss-fn validation-data validation-indices batch sparse? false)))))

(defn train!
  "Train net from source's DuckDB columns selected by :x and :y in opts.
  Return {:history [per-epoch loss maps]}, with validation loss when requested.
  BCE/CE consume logits; each call uses a fresh optimizer.
  Throws ExceptionInfo for invalid options, labels or empty data."
  [net db source opts]
  (let [{:keys [config seed epochs] :as training} (prepare-training! net db source opts)
        ^Model model (:model net)]
    (.setRandomSeed (Engine/getInstance) (int seed))
    (with-open [^Trainer trainer (.newTrainer model ^DefaultTrainingConfig config)]
      (.initialize trainer (into-array Shape [(Shape. (long-array (into [1] (:input-shape net))))]))
      {:history (mapv #(run-epoch! trainer training %) (range 1 (inc epochs)))})))

(defn- argmax [values]
  (let [maximum (reduce max values)]
    ;; Keep the first maximum, including when two logits tie.
    (first (keep-indexed (fn [index value] (when (= value maximum) index)) values))))

(defn- sigmoid [value]
  ;; The negative branch avoids overflowing exp for large negative logits.
  (if (neg? value)
    (let [exponential (Math/exp value)]
      (/ exponential (+ 1.0 exponential)))
    (/ 1.0 (+ 1.0 (Math/exp (- value))))))

(defn- softmax [values]
  (let [maximum (reduce max values)
        exponentials (mapv #(Math/exp (- % maximum)) values)
        total (reduce + exponentials)]
    (mapv #(/ % total) exponentials)))

(defn- output-value [values output]
  (let [single? (= 1 (count values))
        value (first values)]
    (cond
      (= output :raw) (if single? value values)
      (and (= output :class) single?) (if (pos? value) 1 0)
      (= output :class) (argmax values)
      single? (sigmoid value)
      :else (softmax values))))

(defn- output-rows [^NDArray array output]
  (let [width (.getLastDimension (.getShape array))
        rows (partition (int width) (.toFloatArray array))]
    (mapv #(output-value (mapv double %) output) rows)))

(defn predict
  "Predict net from source's columns selected by :x in opts, in table order.
  :raw returns logits; :class applies argmax/zero threshold; :probability
  applies softmax/sigmoid. Single-output networks return a flat vector.
  Throws ExceptionInfo for invalid options or feature columns."
  [net db source opts]
  (let [{:keys [x output] :or {output :raw}} (options opts #{:x :output} :predict)
        _ (feature-columns net x)
        _ (when-not (#{:raw :class :probability} output)
            (invalid :nn/invalid-output "Unknown prediction output" {:output output}))
        {:keys [rows cols data]} (table/matrix db source x)
        ^Model model (:model net)]
    (if (zero? rows)
      []
      (with-open [^NDManager manager (.newSubManager (.getNDManager model))
                  ^Predictor predictor (.newPredictor model (NoopTranslator. nil))
                  ^NDList result (.predict predictor
                                           (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray [(.create manager ^floats data (Shape. (long-array [rows cols])))])))]
        (output-rows (.singletonOrThrow result) output)))))

(defn save!
  "Save net's spec.edn and DJL parameters to dir, returning nil.
  Paths are trusted caller input. Filesystem or DJL failures propagate."
  [net dir]
  (let [file (io/file dir)
        ^Model model (:model net)]
    (.mkdirs file)
    (.save model (.toPath file) "garun")
    (spit (io/file file "spec.edn") (pr-str (select-keys net [:spec :input-shape])))
    nil))

(defn load
  "Load a saved Network from dir; close it with with-open.
  Throws ExceptionInfo for an invalid saved spec; read/DJL failures propagate."
  ^Network [dir]
  (let [file (io/file dir)
        {:keys [spec input-shape]} (options (edn/read-string (slurp (io/file file "spec.edn"))) #{:spec :input-shape} :saved-spec)
        net (build spec {:input-shape input-shape})
        ^Model model (:model net)]
    (try
      (.load model (.toPath file) "garun")
      net
      (catch Throwable error
        (.close model)
        (throw error)))))
