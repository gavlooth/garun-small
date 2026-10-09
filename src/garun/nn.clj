(ns garun.nn
  "Own session-registered DJL networks, inference, persistence, and training.

  trainer/epoch!/validate/restore-best! expose the native training lifecycle with
  immutable epoch state; train! composes them with observation and stopping.
  :loss/:forward/:metrics functions extend the same loop without new plumbing.
  garun.nn.examples prepares table inputs and aligns labels/predictions/metrics.

  Native prediction results belong to the context manager unless :manager is
  supplied. Inference shares the network lock; training uses its exclusive lock.
  Training operations acquire network before trainer to keep one lock order.
  Native datasets remain borrowed; table datasets and best-parameter snapshots
  belong to the session-owned trainer."
  (:refer-clojure :exclude [load])
  (:require [clojure.core.protocols :as protocols]
            [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as string]
            [garun.blocks :as blocks]
            [garun.nn.examples :as examples]
            [garun.session :as session]
            [garun.table :as table]
            [garun.tensor :as tensor]
            [garun.training :as training])
  (:import [ai.djl Device Model]
           [ai.djl.engine Engine]
           [ai.djl.inference Predictor]
           [ai.djl.ndarray NDArray NDList NDManager NDResource]
           [ai.djl.ndarray.types DataType Shape]
           [ai.djl.nn Block Parameter Parameter$Type]
           [ai.djl.ndarray.index NDIndex]
           [ai.djl.training GradientCollector Trainer TrainingConfig]
           [ai.djl.training.dataset Batch Dataset]
           [ai.djl.training.listener TrainingListener TrainingListener$BatchData]
           [ai.djl.training.initializer Initializer XavierInitializer]
           [ai.djl.training.loss Loss]
           [ai.djl.translate Translator TranslatorContext]
           [garun.session Context Resource]
           [java.io Writer]
           [java.util.function Consumer]))

(set! *warn-on-reflection* true)

(defrecord Network [spec ^Model model input-shape input-shapes dtype device engine]
  java.lang.AutoCloseable
  (close [_] (.close model)))

(defmethod session/describe-value :network [_ network]
  (let [^Model model (:model network)
        ^Device device (:device network)
        ^DataType data-type (:dtype network)]
    {:spec (:spec network)
     :input-shapes (:input-shapes network)
     :dtype (keyword (string/lower-case (.name data-type)))
     :device (if (.isGpu device) [:gpu (.getDeviceId device)] :cpu)
     :engine (:engine network)
     :parameters (reduce + 0 (map (fn [^Parameter parameter]
                                    (.size (.getShape parameter)))
                                  (.values (.getParameters (.getBlock model)))))}))

(extend-type Network
  protocols/Datafiable
  (datafy [network] (session/describe-value :network network)))

(defmethod print-method Network [network ^Writer writer]
  (.write writer "#garun/network ")
  (print-method (session/describe-value :network network) writer))

(defn- invalid [error message data]
  (throw (ex-info message (assoc data :error error))))

(defn- validate-options [options allowed context]
  (when-not (map? options)
    (invalid :nn/invalid-options "Options must be a map" {:context context :options options}))
  (when-let [unknown (seq (remove allowed (keys options)))]
    (invalid :nn/unknown-option "Unknown neural network option" {:context context :keys (vec unknown)}))
  options)

(defn- positive-int? [value]
  (and (integer? value) (pos? value) (<= value Integer/MAX_VALUE)))

(defn- initialization-shapes [options]
  (let [{:keys [input-shape input-shapes]} options
        per-sample? (contains? options :input-shape)
        full? (contains? options :input-shapes)]
    (when (= per-sample? full?)
      (invalid :nn/invalid-input-shape
               "Provide exactly one of :input-shape (per sample) or :input-shapes (full DJL shapes)"
               (select-keys options [:input-shape :input-shapes])))
    (when (and per-sample?
               (not (and (vector? input-shape)
                         (every? #(and (integer? %) (pos? %) (<= % Long/MAX_VALUE)) input-shape))))
      (invalid :nn/invalid-input-shape "Input shape must contain positive integer dimensions"
               {:input-shape input-shape}))
    (when (and full? (not (and (vector? input-shapes) (seq input-shapes))))
      (invalid :nn/invalid-input-shape "Input shapes must be a nonempty vector of full shapes"
               {:input-shapes input-shapes}))
    (let [shapes (if per-sample? [(into [1] input-shape)] input-shapes)]
      (mapv (fn [dimensions] (vec (.getShape ^Shape (tensor/shape dimensions))))
            shapes))))

(defn- initialize-network! [spec options defaults]
  (let [{:keys [seed device engine] data-type-specification :dtype} (merge defaults options)
        shapes (initialization-shapes options)
        _ (when-not (string? engine)
            (invalid :nn/invalid-engine ":engine must be a DJL engine name" {:engine engine}))
        ^Engine backend (Engine/getEngine engine)
        ^DataType data-type (tensor/dtype data-type-specification backend)
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

(defn build
  "Initialize and register a session-owned Network through context.
  :input-shape contains per-sample dimensions (batch 1 is prepended);
  :input-shapes contains full DJL shapes, including scalars and empty tensors.
  Exactly one shape option is required. :seed/:device/:dtype/:engine default
  to the session defaults. Native Blocks are accepted. Optional :name names the
  network; :replace? replaces and closes an existing resource with that name."
  ^Network [^Context context spec options]
  (.call context
         (fn []
           (validate-options options
                             #{:input-shape :input-shapes :seed :device :dtype :engine :name :replace?}
                             :build)
           (let [network (initialize-network! spec options (session/defaults context))]
             (try
               (session/register! context network
                                  (merge {:kind :network} (select-keys options [:name :replace?])))
               (catch Throwable error
                 (.close ^Network network)
                 (throw error)))))))

(defn- resolve-loss [specification]
  (if (fn? specification)
    (proxy [Loss] ["garun-function-loss"]
      (evaluate [labels predictions]
        (let [value (specification labels predictions)]
          (when-not (instance? NDArray value)
            (invalid :nn/invalid-loss "Loss functions must return an NDArray" {:value value}))
          value)))
    (training/loss specification)))

(defn- early-stopping-options [options]
  (let [options (merge {:patience 5 :min-delta 0.0 :restore-best? false}
                       (validate-options options #{:patience :min-delta :restore-best?} :early-stopping))]
    (when-not (and (positive-int? (:patience options))
                   (number? (:min-delta options))
                   (Double/isFinite (double (:min-delta options)))
                   (<= 0 (:min-delta options)) (boolean? (:restore-best? options)))
      (invalid :nn/invalid-early-stopping "Invalid early-stopping settings" {:early-stopping options}))
    options))

(def ^:private native-training-keys
  #{:loss :optim :device :devices :listeners :evaluators :initializer :parameter-type :executor})

(defn- native-training-config [network options]
  (when (and (contains? options :config) (some #(contains? options %) native-training-keys))
    (invalid :nn/invalid-training-options
             "A native :config cannot be combined with training configuration options"
             {:options options}))
  (when (and (contains? options :config) (not (instance? TrainingConfig (:config options))))
    (invalid :nn/invalid-training-options ":config must be a native TrainingConfig"
             {:config (:config options)}))
  (let [^TrainingConfig configuration
        (if (contains? options :config)
          (:config options)
          (training/config
           (cond-> (select-keys options native-training-keys)
             (contains? options :loss) (assoc :loss (resolve-loss (:loss options)))
             (not (contains? options :devices)) (assoc :device (:device network)))))
        devices (vec (.getDevices configuration))]
    (when-not (some #(= (:device network) %) devices)
      (invalid :nn/device-mismatch
               "Training devices must include the network device"
               {:model-device (:device network) :training-devices devices}))
    (when (and (contains? options :device)
               (not= (:device network) (tensor/device (:device options) (:engine network))))
      (invalid :nn/device-mismatch "Training :device must match the network device"
               {:model-device (:device network) :device (:device options)}))
    configuration))

(defn- trainer-options [context source options]
  (let [native? (instance? Dataset source)
        allowed (into native-training-keys
                      (cond-> #{:config :epochs :validation :seed :early-stopping :observe :metrics :stop? :forward}
                        (not native?) (into #{:x :y :sequence :standardize :batch :shuffle?})))
        options (merge {:epochs 10 :seed (:seed (session/defaults context)) :metrics []}
                       (when-not native? {:batch 32 :shuffle? true})
                       (validate-options options allowed :trainer))]
    (when-not (and (positive-int? (:epochs options)) (integer? (:seed options))
                   (or native? (and (positive-int? (:batch options)) (boolean? (:shuffle? options))))
                   (or (not native?) (nil? (:validation options)) (instance? Dataset (:validation options))))
      (invalid :nn/invalid-training-options "Invalid epochs, seed, sampling, or validation source"
               {:options options}))
    (doseq [extension [:observe :stop? :forward]
            :when (contains? options extension)]
      (when-not (fn? (get options extension))
        (invalid :nn/invalid-training-options "Training extension must be a function"
                 {:key extension :value (get options extension)})))
    (examples/metric-options (:metrics options))
    (cond-> options
      (contains? options :early-stopping)
      (update :early-stopping early-stopping-options))))

(defrecord TrainingHandle [context network network-name ^Trainer native-trainer dataset validation
                           options specification ^NDManager snapshot-manager snapshots]
  java.lang.AutoCloseable
  (close [_]
    (try
      (when (.isOpen (.getManager native-trainer)) (.close native-trainer))
      (finally
        (when (.isOpen snapshot-manager) (.close snapshot-manager))))))

(defmethod session/describe-value :trainer [_ handle]
  (let [options (:options handle)]
    {:network (:network-name handle)
     :epochs (:epochs options)
     :batch (:batch options)
     :validation? (some? (:validation options))
     :early-stopping (:early-stopping options)}))

(extend-type TrainingHandle
  protocols/Datafiable
  (datafy [handle] (session/describe-value :trainer handle)))

(defmethod print-method TrainingHandle [handle ^Writer writer]
  (.write writer "#garun/trainer ")
  (print-method (session/describe-value :trainer handle) writer))

(defn trainer
  "Create a session-owned training handle. Release it with session/release! or
  leave it to session close; do not use with-open on this registered resource.
  Table sources become ArrayDatasets on the DJL Trainer manager. Native Dataset
  sources and arrays remain borrowed. :config accepts native TrainingConfig.
  :x/:y/:sequence default to stored inputs; training always recomputes requested
  standardization from this training source, using one DuckDB aggregate query.

  Pass {} as the initial epoch! state. Epoch state is immutable and returned;
  best tensors live device-natively on this handle, not in Clojure accumulators.
  :metrics accepts metric keywords or functions (named :metric-0, :metric-1, ...).
  :forward receives [native-trainer inputs training?]; :loss functions receive
  [labels predictions] NDLists and must return a scalar NDArray."
  ^TrainingHandle [context network source options]
  (let [network-name (if (keyword? network) network
                         (some-> ^Resource (session/resource context network :network) .-name))]
    (session/exclusive
     context network :network
     (fn [network]
       (let [options (trainer-options context source options)
             native? (instance? Dataset source)
             ^TrainingConfig configuration (native-training-config network options)
             loss (get options :loss :mse)
             loss-tag (if (vector? loss) (first loss) loss)
             specification (when-not native?
                             (examples/training-inputs context network source options loss))
             examples (when-not native? (examples/table-examples context source specification true))
             validation-examples (when (and (not native?) (:validation options))
                                   (examples/table-examples context (:validation options) specification true))
             _ (when examples (examples/label-layout! network examples loss-tag))
             _ (when validation-examples (examples/label-layout! network validation-examples loss-tag))
             ^Model model (:model network)]
         (.setRandomSeed (.getEngine (.getNDManager model)) (int (:seed options)))
         (let [^Trainer native-trainer (.newTrainer model configuration)
               ^NDManager snapshot-manager (.newSubManager (.getNDManager model))]
           (try
             (.initialize native-trainer (into-array Shape (map tensor/shape (:input-shapes network))))
             (let [dataset (if native? source
                               (examples/array-dataset (.getManager native-trainer) examples specification
                                                       options (:shuffle? options) loss-tag))
                   validation (if native? (:validation options)
                                  (when validation-examples
                                    (examples/array-dataset (.getManager native-trainer) validation-examples
                                                            specification options false loss-tag)))
                   snapshots (mapv (fn [^Parameter parameter]
                                     (let [^NDArray snapshot (.duplicate (.getArray parameter))]
                                       (.setRequiresGradient snapshot false)
                                       (.attach snapshot snapshot-manager)
                                       snapshot))
                                   (.values (.getParameters (.getBlock model))))
                   handle (->TrainingHandle context network network-name native-trainer dataset validation
                                            options specification snapshot-manager snapshots)]
               (.prepare ^Dataset dataset)
               (when validation (.prepare ^Dataset validation))
               (when specification (examples/store-inputs! model specification))
               (session/register! context handle {:kind :trainer}))
             (catch Throwable error
               (try (.close native-trainer) (finally (.close snapshot-manager)))
               (throw error)))))))))

(defn- forward [handle ^NDList inputs ^NDList labels training?]
  (let [^Trainer native-trainer (:native-trainer handle)
        result (if-let [custom (:forward (:options handle))]
                 (custom native-trainer inputs training?)
                 (if training? (.forward native-trainer inputs labels) (.evaluate native-trainer inputs)))]
    (when-not (instance? NDList result)
      (invalid :nn/invalid-forward "Forward functions must return an NDList" {:result result}))
    result))

(defn- batch-result! [handle ^Batch batch training? collector collect-metrics?]
  (let [^Trainer native-trainer (:native-trainer handle)
        ^Loss loss (.getLoss native-trainer)
        result (reduce
                (fn [result ^Batch split]
                  (let [labels (.getLabels split)
                        ^NDList predictions (forward handle (.getData split) labels training?)
                        ^NDArray value (.evaluate loss labels predictions)
                        _ (when-not (= 1 (.size value))
                            (invalid :nn/invalid-loss "Loss must return one scalar" {:shape (.getShape value)}))
                        scalar (double (aget ^"[Ljava.lang.Number;" (.toArray value) 0))
                        size (.getSize split)
                        device (.getDevice (.head (.getData split)))]
                    (when training? (.backward ^GradientCollector collector value))
                    (cond-> (-> result
                                (update :loss-sum + (* size scalar))
                                (update :count + size)
                                (assoc-in [:labels-by-device device] labels)
                                (assoc-in [:predictions-by-device device] predictions))
                      collect-metrics?
                      (-> (update :predictions into (examples/output-rows (.singletonOrThrow predictions) :raw))
                          (update :truth into (examples/label-rows labels))))))
                {:loss-sum 0.0 :count 0 :labels-by-device {} :predictions-by-device {}
                 :predictions [] :truth []}
                (.split batch (.getDevices native-trainer) false))
        batch-data (TrainingListener$BatchData. batch (:labels-by-device result) (:predictions-by-device result))]
    (.notifyListeners native-trainer
                      (reify Consumer
                        (accept [_ listener]
                          (if training?
                            (.onTrainingBatch ^TrainingListener listener native-trainer batch-data)
                            (.onValidationBatch ^TrainingListener listener native-trainer batch-data)))))
    result))

(defn- run-dataset! [handle ^Dataset dataset training? collect-metrics?]
  (let [^Trainer native-trainer (:native-trainer handle)
        batches (.iterateDataset native-trainer dataset)
        result (reduce
                (fn [result ^Batch batch]
                  (with-open [batch batch]
                    (let [batch-result (if training?
                                         (with-open [collector (.newGradientCollector native-trainer)]
                                           (batch-result! handle batch true collector collect-metrics?))
                                         (batch-result! handle batch false nil collect-metrics?))]
                      ;; Gradients are applied only after the collector closes.
                      (when training? (.step native-trainer))
                      (-> result
                          (update :loss-sum + (:loss-sum batch-result))
                          (update :count + (:count batch-result))
                          (update :predictions into (:predictions batch-result))
                          (update :truth into (:truth batch-result))))))
                {:loss-sum 0.0 :count 0 :predictions [] :truth []}
                ;; Reducing Iterable directly avoids iterator-seq's chunked
                ;; lookahead exhausting DJL's auto-closing data iterator early.
                batches)]
    (when (zero? (:count result))
      (invalid :nn/empty-data "Dataset must yield at least one example" {}))
    (assoc result :loss (/ (:loss-sum result) (:count result)))))

(defn- validation-result! [handle]
  (let [requested (:metrics (:options handle))
        result (run-dataset! handle (or (:validation handle) (:dataset handle)) false (boolean (seq requested)))
        statistics (get-in handle [:specification :statistics :y])]
    (cond-> {:loss (:loss result)}
      (seq requested) (assoc :metrics
                             (examples/metric-results requested
                                                      (examples/inverse-targets (:predictions result) statistics)
                                                      (examples/inverse-targets (:truth result) statistics))))))

(defn validate
  "Return {:loss number :metrics {...}} for the handle's validation dataset.
  Without :validation, evaluate its training dataset without gradients instead.
  Keyword regression metrics and custom functions receive original target units."
  [handle]
  (session/exclusive (:context handle) (:network handle) :network
                     (fn [_]
                       (session/exclusive (:context handle) handle :trainer
                                          (fn [handle] (validation-result! handle))))))

(defn- copy-parameters! [handle restore?]
  (let [^Model model (:model (:network handle))
        parameters (.values (.getParameters (.getBlock model)))]
    (doseq [[^Parameter parameter ^NDArray snapshot] (map vector parameters (:snapshots handle))]
      (let [^NDArray array (.getArray parameter)
            ^NDArray destination (if restore? array snapshot)
            ^NDArray source (if restore? snapshot array)
            index (.addAllDim (NDIndex.))]
        ;; copyTo's default implementation copies through host memory; slice
        ;; assignment and PyTorch duplicate are device-native. Scalar views
        ;; provide a real slice, since PyTorch rejects an empty index list.
        (if (.isScalar destination)
          (.set (.reshape destination (long-array [1])) index (.reshape source (long-array [1])))
          (.set destination index source))))))

(defn restore-best!
  "Restore the latest best epoch's device-native parameters in place; return state.
  State is the most recently returned epoch! state for this handle. Optimizer
  moments are not rewound; restoring weights is normally the final training step."
  [handle state]
  (session/exclusive
   (:context handle) (:network handle) :network
   (fn [_]
     (session/exclusive (:context handle) handle :trainer
                        (fn [handle]
                          (when (:best state) (copy-parameters! handle true))
                          state)))))

(defn epoch!
  "Run one epoch and return immutable state with :epoch/:history/:best.
  Initial state is {}. History entries include loss, best status, elapsed time,
  and optional validation :metrics. :observe receives the corresponding :epoch
  event after each epoch; exceptions propagate. This function does not stop or
  restore automatically, so user reductions can supply their own schedule."
  [handle state]
  (session/exclusive
   (:context handle) (:network handle) :network
   (fn [_]
     (session/exclusive
      (:context handle) handle :trainer
      (fn [handle]
        (when-not (map? state)
          (invalid :nn/invalid-state "Epoch state must be a map" {:state state}))
        (let [start (System/nanoTime)
              ^Trainer native-trainer (:native-trainer handle)
              training (run-dataset! handle (:dataset handle) true false)
              validation (when (or (:validation handle) (seq (:metrics (:options handle))))
                           (validation-result! handle))
              monitor (if (:validation handle) :validation-loss :training-loss)
              monitored-loss (if (:validation handle) (:loss validation) (:loss training))
              best? (or (nil? (:best state))
                        (< monitored-loss (- (get-in state [:best :loss])
                                             (get-in handle [:options :early-stopping :min-delta] 0.0))))
              epoch (inc (get state :epoch 0))
              entry (cond-> {:epoch epoch :training-loss (:loss training)
                             :validation-loss (when (:validation handle) (:loss validation))
                             :best? best? :elapsed-milliseconds (/ (- (System/nanoTime) start) 1.0e6)}
                      (:metrics validation) (assoc :metrics (:metrics validation)))
              next-state (cond-> (-> state
                                     (assoc :epoch epoch :stopped :completed)
                                     (update :history (fnil conj []) entry)
                                     (assoc :epochs-without-improvement
                                            (if best? 0 (inc (get state :epochs-without-improvement 0)))))
                           best? (assoc :best {:epoch epoch :loss monitored-loss :monitor monitor}))]
          (when best? (copy-parameters! handle false))
          (.notifyListeners native-trainer
                            (reify Consumer
                              (accept [_ listener] (.onEpoch ^TrainingListener listener native-trainer))))
          (when-let [observe (:observe (:options handle))]
            (observe (assoc entry :event :epoch :network (:network-name handle))))
          next-state))))))

(defn train!
  "The default reduction over trainer/epoch!, for tables and native Datasets.
  Return {:history [...] :best {:epoch :loss :monitor} :stopped ...}.
  :early-stopping monitors validation loss when available, else training loss.
  :stop? receives epoch state and adds a custom stop rule. Either stop returns
  :early-stopping. :observe receives epoch events then one :finished event.
  :restore-best? restores parameters before :finished; observer errors abort."
  [context network source options]
  (session/exclusive
   context network :network
   (fn [_]
     (let [handle (trainer context network source options)
           options (:options handle)]
       (try
         (let [state (reduce (fn [state _]
                               (let [state (epoch! handle state)
                                     patience (get-in options [:early-stopping :patience])]
                                 (if (or (and patience (>= (:epochs-without-improvement state) patience))
                                         (when-let [stop? (:stop? options)] (stop? state)))
                                   (reduced (assoc state :stopped :early-stopping))
                                   state)))
                             {} (range (:epochs options)))
               _ (when (get-in options [:early-stopping :restore-best?]) (restore-best! handle state))
               result (select-keys state [:history :best :stopped])]
           (when-let [observe (:observe options)]
             (observe (assoc (dissoc result :history) :event :finished)))
           result)
         (finally (session/release! context handle)))))))

(defn- predict-native [network inputs options default-manager]
  (let [^NDManager manager (:manager (validate-options options #{:manager} :predict-native)
                                     default-manager)
        ^Model model (:model network)]
    (when-not (instance? NDManager manager)
      (invalid :nn/invalid-manager "Native prediction requires an NDManager"
               {:manager manager}))
    (with-open [^NDManager input-manager (.newSubManager (.getNDManager model))]
      (let [input (tensor/ndlist input-manager inputs)
            translator (reify Translator
                         (getBatchifier [_] nil)
                         (processInput [_ _ value] value)
                         (processOutput [_ _ output]
                           ;; Copy before DJL closes its per-prediction context.
                           (tensor/ndlist manager output)))]
        (with-open [^Predictor predictor (.newPredictor model translator ^Device (:device network))]
          (.predict predictor input))))))

(defn- table-output [^Context context network examples specification output]
  (if (zero? (:rows examples))
    []
    (let [^Model model (:model network)]
      (with-open [^NDManager manager (.newSubManager (.getNDManager model))]
        (let [features (examples/standardized-array manager (:features examples)
                                                    (get-in specification [:statistics :x]) false)]
          (with-open [^NDList result (predict-native network features {:manager manager} (.manager context))]
            (when-not (= 1 (.size result))
              (invalid :nn/non-tabular-output "Table predictions require a single output tensor"
                       {:outputs (.size result)}))
            (let [rows (examples/output-rows (.singletonOrThrow result) output)]
              (if (= output :raw)
                (examples/inverse-targets rows (get-in specification [:statistics :y]))
                rows))))))))

(defn- predict-table [context network source options]
  (let [{:keys [output] :or {output :raw}}
        (validate-options options #{:x :sequence :standardize :output} :predict)
        specification (examples/inference-inputs network options)]
    (when-not (#{:raw :class :probability} output)
      (invalid :nn/invalid-output "Unknown prediction output" {:output output}))
    (table-output context network (examples/table-examples context source specification false)
                  specification output)))

(defn predict
  "Predict with a Network or registered keyword name through context's shared lock.
  Explicit :x or a table source with stored inputs selects table inference.
  :x/:sequence default to training's stored specification; features are scaled
  with train-only statistics and raw outputs are restored to original units.
  :class applies argmax/zero threshold; :probability applies softmax/sigmoid.
  Sequence tables yield one prediction per full window, in partition/order order.

  Otherwise inputs are NDList, tensors or tensor descriptors. Return independent
  tensors in an NDList owned by :manager, defaulting to context's manager. Inputs
  are copied, never closed, detached or stolen. Results survive Network release,
  but not closure of their owning manager."
  [^Context context network input options]
  (session/shared context network :network
                  (fn [network]
                    (when-not (map? options)
                      (invalid :nn/invalid-options "Options must be a map"
                               {:context :predict :options options}))
                    (if (or (contains? options :x)
                            (and (examples/stored-inputs (:model network))
                                 (or (table/table? input) (keyword? input)
                                     (and (map? input) (not (contains? input :data)))
                                     (and (vector? input) (or (keyword? (first input))
                                                              (map? (first input)))))))
                      (predict-table context network input options)
                      (predict-native network input options (.manager context))))))

(defn- evaluate-native! [network ^Dataset source requested loss-specification]
  (let [^Model model (:model network)
        configuration (training/config {:loss (resolve-loss loss-specification)
                                        :device (:device network) :listeners []})]
    (with-open [^Trainer native-trainer (.newTrainer model ^TrainingConfig configuration)]
      (.initialize native-trainer (into-array Shape (map tensor/shape (:input-shapes network))))
      (.prepare source)
      (let [handle {:native-trainer native-trainer :options {:metrics requested}}
            result (run-dataset! handle source false (boolean (seq requested)))]
        (merge {:loss (:loss result)}
               (examples/metric-results requested (:predictions result) (:truth result)))))))

(defn- evaluate-table [^Context context network source specification requested loss-specification]
  (let [prepared (examples/table-examples context source specification true)
        loss-tag (if (vector? loss-specification) (first loss-specification) loss-specification)
        ^Loss loss (resolve-loss loss-specification)
        ^Model model (:model network)]
    (examples/label-layout! network prepared loss-tag)
    (with-open [^NDManager manager (.newSubManager (.getNDManager model))]
      (let [features (examples/standardized-array manager (:features prepared)
                                                  (get-in specification [:statistics :x]) false)
            labels (examples/standardized-array manager (:labels prepared)
                                                (get-in specification [:statistics :y]) (= loss-tag :ce))]
        (with-open [^NDList result (predict-native network features {:manager manager} (.manager context))]
          (let [^NDArray value (.evaluate loss
                                          (NDList. ^"[Lai.djl.ndarray.NDArray;"
                                           (into-array NDArray [labels])) result)
                predictions (examples/inverse-targets (examples/output-rows (.singletonOrThrow result) :raw)
                                                      (get-in specification [:statistics :y]))]
            (merge {:loss (double (aget ^"[Ljava.lang.Number;" (.toArray value) 0))}
                   (examples/metric-results requested predictions (examples/target-rows prepared)))))))))

(defn evaluate
  "Evaluate aligned table examples/windows or a native Dataset.
  :x/:y/:sequence default to the stored input specification. :metrics is a vector
  of :mse/:mae/:r2/:accuracy/:precision/:recall/:f1/:log-loss or custom functions.
  Return {:loss number, metric-key number, ...}; loss retains DJL's training-scale
  semantics, while regression metrics/predictions use original target units.
  :loss overrides the saved loss (required to reproduce a custom/native loss)."
  [^Context context network source options]
  (let [native? (instance? Dataset source)]
    ((if native? session/exclusive session/shared)
     context network :network
     (fn [network]
       (validate-options options
                         (if native? #{:metrics :loss}
                             #{:x :y :sequence :standardize :metrics :loss}) :evaluate)
       (let [requested (examples/metric-options (get options :metrics []))
             loss-specification (get options :loss
                                     (get (examples/stored-inputs (:model network)) :loss :mse))]
         (if native?
           (evaluate-native! network source requested loss-specification)
           (evaluate-table context network source (examples/inference-inputs network options)
                           requested loss-specification)))))))

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
  "Run a borrowed native Translator through context's shared network lock.
  Close the internal Predictor; direct NDArray/NDList outputs are independent
  copies owned by context's manager. They survive Network release, not context
  closure. Other outputs follow the translator's resource contract; nested
  tensors and input ownership remain the translator's responsibility."
  [^Context context network translator input]
  (session/shared
   context network :network
   (fn [network]
     (when-not (instance? Translator translator)
       (invalid :nn/invalid-translator "predict-with requires a native Translator"
                {:translator translator}))
     (let [^Translator native translator
           wrapper (reify Translator
                     (getBatchifier [_] (.getBatchifier native))
                     (getExpansions [_] (.getExpansions native))
                     (prepare [_ context] (.prepare native context))
                     (processInput [_ context value] (.processInput native context value))
                     (batchProcessInput [_ context values] (.batchProcessInput native context values))
                     (processOutput [_ context output]
                       (independent-output! (.processOutput native context output)
                                            (.getPredictorManager ^TranslatorContext context)))
                     (batchProcessOutput [_ context output]
                       (mapv #(independent-output! % (.getPredictorManager ^TranslatorContext context))
                             (.batchProcessOutput native context output))))
           ^Model model (:model network)]
       (with-open [^Predictor predictor (.newPredictor model wrapper ^Device (:device network))]
         (let [result (.predict predictor input)]
           (when (or (instance? NDArray result) (instance? NDList result))
             (.attach ^NDResource result (.manager context)))
           result))))))

(defn save!
  "Save a borrowed Network or keyword name under context's shared lock, returning nil.
  Plain Hiccup is persisted as spec.edn alongside DJL parameters. Native/custom
  blocks require {:spec reconstruction-spec}; non-EDN reconstruction requires
  :spec again on load. Missing reconstruction fails before writing any files.
  Paths are trusted caller input; filesystem/DJL failures propagate."
  ([context network directory] (save! context network directory {}))
  ([context network directory options]
   (session/shared
    context network :network
    (fn [network]
      (validate-options options #{:spec} :save)
      (let [spec (if (contains? options :spec) (:spec options) (:spec network))
            serializable? (blocks/serializable-spec? spec)
            _ (when (and (not serializable?) (not (contains? options :spec)))
                (invalid :nn/nonserializable-spec
                         "Native/custom blocks require an explicit reconstruction :spec when saving"
                         {}))
            _ (when (nil? spec)
                (invalid :nn/nonserializable-spec "Reconstruction :spec cannot be nil" {}))
            ^Device device (:device network)
            ^DataType data-type (:dtype network)
            metadata (cond-> {:input-shapes (:input-shapes network)
                              :dtype (.name data-type) :engine (:engine network)
                              :device (if (.isGpu device) [:gpu (.getDeviceId device)] :cpu)}
                       serializable? (assoc :spec spec)
                       (not serializable?) (assoc :requires-spec? true))
            file (io/file directory)
            ^Model model (:model network)]
        (.mkdirs file)
        (.save model (.toPath file) "garun")
        (spit (io/file file "spec.edn") (pr-str metadata))
        nil)))))

(defn- cast-parameters! [^Model model ^DataType data-type]
  ;; Block.cast is unimplemented in DJL's base blocks. Reinitialize each loaded
  ;; parameter from its converted array using the public Initializer contract.
  (doseq [^Parameter parameter (.values (.getParameters (.getBlock model)))
          :let [^NDArray array (.getArray parameter)]
          :when (not= data-type (.getDataType array))]
    (let [converted (.toType array data-type true)
          initializer (.getInitializer parameter)]
      (try
        (.setInitializer parameter
                         (reify Initializer (initialize [_ _ _ _] converted)))
        (.close parameter)
        (.initialize parameter (.getNDManager model) data-type)
        (finally (.setInitializer parameter initializer)))))
  (.setDataType model data-type))

(defn load
  "Restore and register a session-owned Network through context.
  :spec overrides reconstruction (required by custom saved networks).
  :device/:dtype/:engine override saved metadata; :device :cpu restores a GPU
  save on CPU. Saved dtype is preserved unless overridden. Optional :name and
  :replace? control session registration. Failed loads close the new Model."
  ([context directory] (load context directory {}))
  ([^Context context directory options]
   (.call context
          (fn []
            (validate-options options #{:spec :device :dtype :engine :name :replace?} :load)
            (let [file (io/file directory)
                  metadata (validate-options
                            (edn/read-string (slurp (io/file file "spec.edn")))
                            #{:spec :input-shapes :dtype :engine :device :requires-spec?} :saved-spec)
                  spec (if (contains? options :spec) (:spec options) (:spec metadata))
                  _ (when (nil? spec)
                      (invalid :nn/missing-reconstruction-spec
                               "Saved custom network requires :spec on load" {}))
                  build-options (merge (select-keys metadata [:input-shapes :engine :device])
                                       {:dtype (DataType/valueOf ^String (:dtype metadata))}
                                       (select-keys options [:dtype :engine :device]))
                  network (initialize-network! spec build-options (session/defaults context))
                  ^Model model (:model network)]
              (try
                (.load model (.toPath file) "garun")
                (when (contains? options :dtype)
                  (cast-parameters! model (:dtype network)))
                (session/register! context network
                                   (merge {:kind :network} (select-keys options [:name :replace?])))
                (catch Throwable error
                  (.close model)
                  (throw error))))))))
