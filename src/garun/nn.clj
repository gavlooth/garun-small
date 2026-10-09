(ns garun.nn
  "DJL networks with Hiccup blocks, native datasets and translators, and DuckDB tables.
  A Network owns its Model; close it with with-open. Native Model APIs remain available
  through :model. Caller datasets, managers and input tensors remain caller-owned."
  (:refer-clojure :exclude [load])
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [garun.blocks :as blocks]
            [garun.tensor :as tensor]
            [garun.training :as training])
  (:import [ai.djl Device Model]
           [ai.djl.engine Engine]
           [ai.djl.inference Predictor]
           [ai.djl.ndarray NDArray NDList NDManager]
           [ai.djl.ndarray.types DataType Shape]
           [ai.djl.nn Block Parameter Parameter$Type]
           [ai.djl.training EasyTrain GradientCollector Trainer TrainingConfig]
           [ai.djl.training.dataset Dataset]
           [ai.djl.training.initializer Initializer XavierInitializer]
           [ai.djl.training.loss Loss]
           [ai.djl.translate Translator TranslatorContext]
           [java.util Random]))

(set! *warn-on-reflection* true)

(defrecord Network [spec ^Model model input-shape input-shapes dtype device engine]
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

(defn- initialization-shapes [opts]
  (let [{:keys [input-shape input-shapes]} opts
        per-sample? (contains? opts :input-shape)
        full? (contains? opts :input-shapes)]
    (when (= per-sample? full?)
      (invalid :nn/invalid-input-shape
               "Provide exactly one of :input-shape (per sample) or :input-shapes (full DJL shapes)"
               (select-keys opts [:input-shape :input-shapes])))
    (when (and per-sample?
               (not (and (vector? input-shape)
                         (every? #(and (integer? %) (pos? %) (<= % Long/MAX_VALUE)) input-shape))))
      (invalid :nn/invalid-input-shape "Input shape must contain positive integer dimensions"
               {:input-shape input-shape}))
    (when (and full? (not (and (vector? input-shapes) (seq input-shapes))))
      (invalid :nn/invalid-input-shape "Input shapes must be a nonempty vector of full shapes"
               {:input-shapes input-shapes}))
    (let [shapes (if per-sample? [(into [1] input-shape)] input-shapes)]
      (mapv (fn [dims] (vec (.getShape ^Shape (tensor/shape dims))))
            shapes))))

(defn build
  "Initialize a Network. :input-shape contains arbitrary per-sample dimensions
  (batch 1 is prepended); :input-shapes contains full concrete DJL shapes,
  including scalar and zero-sized tensors.
  These options are mutually exclusive. :dtype defaults to :float32, :device
  to :cpu, :engine to DJL's default, and :seed to 11. Native Blocks are accepted."
  ^Network [spec opts]
  (let [{:keys [seed device dtype engine] :or {seed 11 device :cpu dtype :float32}}
        (options opts #{:input-shape :input-shapes :seed :device :dtype :engine} :build)
        shapes (initialization-shapes opts)
        _ (when (and (contains? opts :engine) (not (string? engine)))
            (invalid :nn/invalid-engine ":engine must be a DJL engine name" {:engine engine}))
        ^Engine backend (if engine (Engine/getEngine engine) (Engine/getInstance))
        ^DataType data-type (tensor/dtype dtype backend)
        ^Device model-device (tensor/device device backend)
        ^Block block (blocks/compile-block spec)]
    (when-not (integer? seed)
      (invalid :nn/invalid-seed "Seed must be an integer" {:seed seed}))
    (.setRandomSeed backend (int seed))
    (let [^Model model (Model/newInstance "garun" model-device (.getEngineName backend))]
      (try
        (.setBlock model block)
        (.setDataType model data-type)
        (.setInitializer block (XavierInitializer.) Parameter$Type/WEIGHT)
        (.initialize block (.getNDManager model) data-type
                     (into-array Shape (map tensor/shape shapes)))
        (->Network spec model
                   (when (= 1 (count shapes)) (vec (rest (first shapes))))
                   shapes data-type model-device (.getEngineName backend))
        (catch Throwable error
          (.close model)
          (throw error))))))

(defn- feature-columns [net x]
  (when-not (and (= 1 (count (:input-shapes net))) (= 1 (count (:input-shape net))))
    (invalid :nn/non-tabular-input
             "Table operations require one rank-2 input; use native datasets/tensors for other shapes"
             {:input-shapes (:input-shapes net)}))
  (when-not (= DataType/FLOAT32 (:dtype net))
    (invalid :nn/non-tabular-dtype "Table operations require a float32 network"
             {:dtype (:dtype net)}))
  (when-not (and (vector? x) (seq x) (every? keyword? x) (= (count x) (first (:input-shape net))))
    (invalid :nn/invalid-features "Features must match the network input shape" {:x x :input-shape (:input-shape net)}))
  x)

(defn- dataset! [net db source x y loss]
  (let [y-cols (if (keyword? y) [y] y)
        _ (when-not (and (vector? y-cols) (seq y-cols) (every? keyword? y-cols))
            (invalid :nn/invalid-labels "Labels must be a keyword or vector of keywords" {:y y}))
        [features labels] (tensor/table-matrices db source [x y-cols])
        ^Model model (:model net)
        ^Block block (.getBlock model)
        outputs (.getOutputShapes block (into-array Shape (map tensor/shape (:input-shapes net))))
        _ (when-not (and (= 1 (count outputs)) (= 2 (.dimension ^Shape (first outputs))))
            (invalid :nn/non-tabular-output "Table training requires one rank-2 output"
                     {:output-shapes (mapv #(vec (.getShape ^Shape %)) outputs)}))
        ^Shape output (first outputs)
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
        ^Loss loss-fn (training/loss loss)
        dataset (dataset! net db source x y loss)
        validation-data (when validation (dataset! net db validation x y loss))
        indices (vec (range (get-in dataset [:features :rows])))
        validation-indices (when validation-data
                             (vec (range (get-in validation-data [:features :rows]))))
        config (training/config {:loss loss-fn :optim optim :device (:device net)})]
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

(def ^:private native-training-keys
  #{:loss :optim :device :devices :listeners :evaluators :initializer :parameter-type :executor})

(defn- native-training-config [net opts]
  (when (and (contains? opts :config) (some #(contains? opts %) native-training-keys))
    (invalid :nn/invalid-training-options
             "A native :config cannot be combined with training configuration options"
             {:options opts}))
  (when (and (contains? opts :config) (not (instance? TrainingConfig (:config opts))))
    (invalid :nn/invalid-training-options ":config must be a native TrainingConfig" {:config (:config opts)}))
  (let [^TrainingConfig config (if (contains? opts :config)
                                 (:config opts)
                                 (training/config
                                  (cond-> (select-keys opts native-training-keys)
                                    (not (contains? opts :devices))
                                    (assoc :device (:device net)))))
        devices (vec (.getDevices config))]
    (when-not (some #(= (:device net) %) devices)
      (invalid :nn/device-mismatch
               "Training devices must include the network device"
               {:model-device (:device net) :training-devices devices}))
    (when (and (contains? opts :device)
               (not= (:device net) (tensor/device (:device opts) (:engine net))))
      (invalid :nn/device-mismatch "Training :device must match the network device"
               {:model-device (:device net) :device (:device opts)}))
    config))

(defn train!
  "Three arguments train a native DJL Dataset using EasyTrain.fit and return
  TrainingResult. :epochs defaults to 10, :validation is an optional Dataset,
  :config accepts a native TrainingConfig, or pass garun.training/config options.
  Dataset.prepare is called; datasets and their source arrays are never closed.
  TrainingResult metrics depend on configured listeners (defaults supply them).

  Four arguments train DuckDB columns selected by :x/:y and return
  {:history [per-epoch loss maps]}, with optional validation loss.
  Tables require one rank-2 float32 input. BCE/CE consume logits."
  ([net dataset opts]
   (let [{:keys [epochs validation seed] :or {epochs 10 seed 0}}
         (options opts (into native-training-keys #{:config :epochs :validation :seed}) :train-native)
         ^Model model (:model net)
         ^TrainingConfig config (native-training-config net opts)]
     (when-not (and (instance? Dataset dataset)
                    (or (nil? validation) (instance? Dataset validation))
                    (positive-int? epochs) (integer? seed))
       (invalid :nn/invalid-training-options
                "Native training requires Dataset(s), positive epochs and an integer seed"
                {:dataset dataset :options opts}))
     (.setRandomSeed (.getEngine (.getNDManager model)) (int seed))
     (.prepare ^Dataset dataset)
     (when validation (.prepare ^Dataset validation))
     (with-open [^Trainer trainer (.newTrainer model config)]
       (.initialize trainer (into-array Shape (map tensor/shape (:input-shapes net))))
       (EasyTrain/fit trainer (int epochs) dataset validation)
       (.getTrainingResult trainer))))
  ([net db source opts]
   (let [{:keys [config seed epochs] :as training} (prepare-training! net db source opts)
         ^Model model (:model net)]
     (.setRandomSeed (.getEngine (.getNDManager model)) (int seed))
     (with-open [^Trainer trainer (.newTrainer model ^TrainingConfig config)]
       (.initialize trainer (into-array Shape (map tensor/shape (:input-shapes net))))
       {:history (mapv #(run-epoch! trainer training %) (range 1 (inc epochs)))}))))

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
  (when-not (= 2 (.dimension (.getShape array)))
    (invalid :nn/non-tabular-output "Table predictions require a rank-2 output"
             {:shape (vec (.getShape (.getShape array)))}))
  (let [width (.getLastDimension (.getShape array))
        rows (partition (int width) (.toFloatArray array))]
    (mapv #(output-value (mapv double %) output) rows)))

(defn predict
  "Three arguments perform native tensor inference. opts requires :manager, a
  caller-owned NDManager; inputs may be an NDList, tensors or tensor descriptors.
  Return an independent NDList owned by that manager, valid after the predictor
  and Network close. Inputs are copied, never closed, detached or stolen.

  Four arguments predict from DuckDB columns :x in table order. :raw returns
  logits; :class applies argmax/zero threshold; :probability softmax/sigmoid.
  Single-output tables return a flat vector; input/output must be rank 2."
  ([net inputs opts]
   (let [{:keys [manager]} (options opts #{:manager} :predict-native)
         ^Model model (:model net)]
     (when-not (instance? NDManager manager)
       (invalid :nn/invalid-manager "Native prediction requires a caller-owned :manager"
                {:manager manager}))
     (with-open [^NDManager input-manager (.newSubManager (.getNDManager model))]
       (let [input (tensor/ndlist input-manager inputs)
             translator (reify Translator
                          (getBatchifier [_] nil)
                          (processInput [_ _ value] value)
                          (processOutput [_ _ output]
                            ;; Copy before DJL closes its per-prediction context.
                            (tensor/ndlist manager output)))]
         (with-open [^Predictor predictor (.newPredictor model translator ^Device (:device net))]
           (.predict predictor input))))))
  ([net db source opts]
   (let [{:keys [x output] :or {output :raw}} (options opts #{:x :output} :predict)
         _ (feature-columns net x)
         _ (when-not (#{:raw :class :probability} output)
             (invalid :nn/invalid-output "Unknown prediction output" {:output output}))
         [{:keys [rows cols data]}] (tensor/table-matrices db source [x])
         ^Model model (:model net)]
     (if (zero? rows)
       []
       (with-open [^NDManager manager (.newSubManager (.getNDManager model))
                   ^NDList result (predict net {:data data :shape [rows cols] :dtype :float32}
                                           {:manager manager})]
         (when-not (= 1 (.size result))
           (invalid :nn/non-tabular-output "Table predictions require a single output tensor"
                    {:outputs (.size result)}))
         (output-rows (.singletonOrThrow result) output))))))

(defn- independent-output! [value ^NDManager manager]
  (cond
    (instance? NDArray value)
    (let [^NDArray copy (.duplicate ^NDArray value)]
      (.attach copy manager)
      copy)

    (instance? NDList value)
    (NDList. ^java.util.Collection (mapv #(independent-output! % manager) value))

    :else value))

(defn predict-with
  "Run a native Translator pipeline, returning its typed output and closing the
  internal Predictor. Direct NDArray/NDList outputs are independent detached
  copies: caller MUST close them. Other outputs follow the translator's resource
  contract; nested tensor resources must be managed by the translator itself.
  Native translator callbacks govern their own input ownership."
  [net translator input]
  (when-not (instance? Translator translator)
    (invalid :nn/invalid-translator "predict-with requires a native Translator"
             {:translator translator}))
  (let [^Translator native translator
        wrapper (reify Translator
                  (getBatchifier [_] (.getBatchifier native))
                  (prepare [_ context] (.prepare native context))
                  (processInput [_ context value] (.processInput native context value))
                  (batchProcessInput [_ context values] (.batchProcessInput native context values))
                  (processOutput [_ context output]
                    (independent-output! (.processOutput native context output)
                                         (.getPredictorManager ^TranslatorContext context)))
                  (batchProcessOutput [_ context output]
                    (mapv #(independent-output! % (.getPredictorManager ^TranslatorContext context))
                          (.batchProcessOutput native context output))))
        ^Model model (:model net)]
    (with-open [^Predictor predictor (.newPredictor model wrapper ^Device (:device net))]
      (let [result (.predict predictor input)]
        (cond
          (instance? NDArray result) (.detach ^NDArray result)
          (instance? NDList result) (.detach ^NDList result))
        result))))

(defn save!
  "Save spec.edn and DJL parameters, returning nil. Plain Hiccup is persisted.
  Native/custom blocks require explicit opts {:spec reconstruction-spec}; if
  that spec is not plain reconstructible EDN, metadata records that load must
  receive :spec. Missing reconstruction fails BEFORE any files are written.
  Paths are trusted caller input; filesystem/DJL failures propagate."
  ([net dir] (save! net dir {}))
  ([net dir opts]
   (options opts #{:spec} :save)
   (let [spec (if (contains? opts :spec) (:spec opts) (:spec net))
         serializable? (blocks/serializable-spec? spec)
         _ (when (and (not serializable?) (not (contains? opts :spec)))
             (invalid :nn/nonserializable-spec
                      "Native/custom blocks require an explicit reconstruction :spec when saving"
                      {}))
         _ (when (nil? spec)
             (invalid :nn/nonserializable-spec "Reconstruction :spec cannot be nil" {}))
         ^Device device (:device net)
         ^DataType dtype (:dtype net)
         metadata (cond-> {:input-shapes (:input-shapes net)
                           :dtype (.name dtype) :engine (:engine net)
                           :device (if (.isGpu device) [:gpu (.getDeviceId device)] :cpu)}
                    serializable? (assoc :spec spec)
                    (not serializable?) (assoc :requires-spec? true))
         file (io/file dir)
         ^Model model (:model net)]
     (.mkdirs file)
     (.save model (.toPath file) "garun")
     (spit (io/file file "spec.edn") (pr-str metadata))
     nil)))

(defn- cast-parameters! [^Model model ^DataType dtype]
  ;; Block.cast is unimplemented in DJL's base blocks. Reinitialize each loaded
  ;; parameter from its converted array using the public Initializer contract.
  (doseq [^Parameter parameter (.values (.getParameters (.getBlock model)))
          :let [^NDArray array (.getArray parameter)]
          :when (not= dtype (.getDataType array))]
    (let [converted (.toType array dtype true)
          initializer (.getInitializer parameter)]
      (try
        (.setInitializer parameter
                         (reify Initializer (initialize [_ _ _ _] converted)))
        (.close parameter)
        (.initialize parameter (.getNDManager model) dtype)
        (finally (.setInitializer parameter initializer)))))
  (.setDataType model dtype))

(defn load
  "Restore a saved Network. opts :spec overrides reconstruction (required when
  saved metadata says :requires-spec?). :device/:dtype/:engine override metadata;
  use :device :cpu to restore a GPU save on CPU. Saved dtype is restored unless
  explicitly overridden. Close the returned Network with with-open."
  ([dir] (load dir {}))
  ([dir opts]
   (options opts #{:spec :device :dtype :engine} :load)
   (let [file (io/file dir)
         metadata (options (edn/read-string (slurp (io/file file "spec.edn")))
                           #{:spec :input-shapes :dtype :engine :device :requires-spec?} :saved-spec)
         spec (if (contains? opts :spec) (:spec opts) (:spec metadata))
         _ (when (nil? spec)
             (invalid :nn/missing-reconstruction-spec
                      "Saved custom network requires :spec on load" {}))
         build-opts (merge (select-keys metadata [:input-shapes :engine :device])
                           {:dtype (DataType/valueOf ^String (:dtype metadata))}
                           (select-keys opts [:dtype :engine :device]))
         net (build spec build-opts)
         ^Model model (:model net)]
     (try
       (.load model (.toPath file) "garun")
       (when (contains? opts :dtype)
         (cast-parameters! model (:dtype net)))
       net
       (catch Throwable error
         (.close model)
         (throw error))))))
