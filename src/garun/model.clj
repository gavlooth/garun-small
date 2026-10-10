(ns garun.model
  "DJL model-zoo loading and typed translator inference.
  load! registers an owned ZooModel in the context's session; session close or
  session/release! closes it. Each prediction uses and closes its own Predictor.
  Direct NDArray/NDList results belong to the context's manager and remain valid
  until that manager closes. Other result types retain their translator-defined
  ownership; custom objects containing tensors manage them in processOutput.
  Registered models use session shared locks; unregistered native models remain
  borrowed and require caller synchronization."
  (:require [clojure.java.io :as io]
            [garun.context :as context]
            [garun.session :as session]
            [garun.tensor :as tensor])
  (:import [ai.djl Application Model]
           [ai.djl.inference Predictor]
           [ai.djl.ndarray NDArray NDList NDManager NDResource]
           [ai.djl.nn Block]
           [ai.djl.repository.zoo Criteria Criteria$Builder ModelZoo ZooModel]
           [ai.djl.translate Translator TranslatorContext TranslatorFactory]
           [ai.djl.util Progress]
           [garun.context Context]
           [java.nio.file Path]
           [java.util List Map]))

(set! *warn-on-reflection* true)

(def ^:private criteria-options
  #{:input-class :output-class :translator :translator-factory :block
    :model-path :model-url :model-zoo :filters :arguments :options
    :group-id :artifact-id :application :name :engine :device :progress})

(defn- invalid [error message data]
  (throw (ex-info message (assoc data :error error))))

(defn- criteria-map [options]
  (when-not (map? options)
    (invalid :model/invalid-options "Criteria options must be a map or native Criteria"
             {:options options}))
  (when-let [unknown (seq (remove criteria-options (keys options)))]
    (invalid :model/unknown-option "Unknown model criteria option" {:keys (vec unknown)}))
  (doseq [[key expected] [[:input-class Class] [:output-class Class]
                          [:translator Translator] [:translator-factory TranslatorFactory]
                          [:block Block] [:model-zoo ModelZoo] [:application Application]
                          [:progress Progress]]]
    (when (and (or (contains? options key) (#{:input-class :output-class} key))
               (not (instance? expected (get options key))))
      (invalid :model/invalid-option "Model criteria option has the wrong type"
               {:key key :expected expected :value (get options key)})))
  (doseq [key [:model-url :group-id :artifact-id :name :engine]]
    (when (and (contains? options key)
               (not (and (string? (get options key)) (seq (get options key)))))
      (invalid :model/invalid-option "Model criteria option requires a nonempty string"
               {:key key :value (get options key)})))
  (doseq [keys [[:translator :translator-factory]
                [:model-path :model-url :model-zoo]]]
    (when (< 1 (count (filter #(contains? options %) keys)))
      (invalid :model/conflicting-options "Conflicting model criteria options"
               {:keys keys})))
  options)

(defn- named-map [value string-values? key]
  (when-not (and (map? value)
                 (every? #(or (string? %) (keyword? %)) (keys value))
                 (or (not string-values?) (every? string? (vals value))))
    (invalid :model/invalid-option
             "Criteria maps require string/keyword keys; filters/options require string values"
             {:key key :value value}))
  (let [entries (mapv (fn [[key value]]
                        [(if (keyword? key) (name key) key) value]) value)
        result (into {} entries)]
    (when-not (= (count entries) (count result))
      (invalid :model/invalid-option "Criteria map keys collide after keyword normalization"
               {:key key :value value}))
    result))

(defn criteria
  "Build real DJL Criteria, or return native Criteria unchanged.
  Map options require :input-class/:output-class (Java Classes). Optional native
  objects: :translator, :translator-factory, :block, :model-zoo, :application,
  :progress. :translator and :translator-factory are mutually exclusive.
  :model-path (Path/File/string), :model-url (string), and :model-zoo are mutually
  exclusive. :name is the model parameter prefix; :group-id/:artifact-id select
  zoo artifacts. :engine is a DJL engine name; :device uses garun.tensor/device.
  :filters/:options have string values; :arguments accepts arbitrary values.
  Their keys are strings or keywords (normalized with name). Unknown options
  fail loudly. Native Criteria is the escape for all detailed DJL settings."
  ^Criteria [options]
  (if (instance? Criteria options)
    options
    (let [options (criteria-map options)
          ^Criteria$Builder builder (.setTypes (Criteria/builder)
                                               ^Class (:input-class options)
                                               ^Class (:output-class options))]
      (when-let [translator (:translator options)]
        (.optTranslator builder ^Translator translator))
      (when-let [factory (:translator-factory options)]
        (.optTranslatorFactory builder ^TranslatorFactory factory))
      (when-let [block (:block options)]
        (.optBlock builder ^Block block))
      (when (contains? options :model-path)
        (let [path (:model-path options)]
          (when-not (or (instance? Path path) (instance? java.io.File path) (string? path))
            (invalid :model/invalid-option "Model path requires a Path, File, or string"
                     {:key :model-path :value path}))
          (.optModelPath builder (if (instance? Path path) path (.toPath (io/file path))))))
      (when-let [url (:model-url options)]
        (.optModelUrls builder ^String url))
      (when-let [zoo (:model-zoo options)]
        (.optModelZoo builder ^ModelZoo zoo))
      (when (contains? options :filters)
        (.optFilters builder ^Map (named-map (:filters options) true :filters)))
      (when (contains? options :arguments)
        (.optArguments builder ^Map (named-map (:arguments options) false :arguments)))
      (when (contains? options :options)
        (.optOptions builder ^Map (named-map (:options options) true :options)))
      (when-let [group (:group-id options)]
        (.optGroupId builder ^String group))
      (when-let [artifact (:artifact-id options)]
        (.optArtifactId builder ^String artifact))
      (when-let [application (:application options)]
        (.optApplication builder ^Application application))
      (when-let [model-name (:name options)]
        (.optModelName builder ^String model-name))
      (when-let [engine (:engine options)]
        (.optEngine builder ^String engine))
      (when (contains? options :device)
        (.optDevice builder (tensor/device (:device options) (:engine options))))
      (when-let [progress (:progress options)]
        (.optProgress builder ^Progress progress))
      (.build builder))))

(defn- registration-options [options]
  (when-not (map? options)
    (invalid :model/invalid-options "Model registration options must be a map"
             {:options options}))
  (when-let [unknown (seq (remove #{:name :replace?} (keys options)))]
    (invalid :model/unknown-option "Unknown model registration option"
             {:keys (vec unknown)}))
  (when (and (contains? options :name) (not (keyword? (:name options))))
    (invalid :model/invalid-option "Model registration name must be a keyword"
             {:key :name :value (:name options)}))
  (when (and (contains? options :replace?) (not (boolean? (:replace? options))))
    (invalid :model/invalid-option "Model replacement option must be boolean"
             {:key :replace? :value (:replace? options)}))
  options)

(defn load!
  "Load a ZooModel owned by the context's session through Criteria.loadModel.
  Criteria maps without :device use the session default device; native Criteria
  is unchanged. Optional registration options are :name (keyword) and :replace?
  (boolean, default false), separate from the criteria's parameter-prefix :name.
  Session close or session/release! closes the model. Local and remote/zoo loading
  failures propagate unchanged; no replacement model or fallback is fabricated."
  (^ZooModel [context options-or-criteria]
   (load! context options-or-criteria {}))
  (^ZooModel [^Context context options-or-criteria options]
   (context/call context
                 (fn []
                   (let [options (registration-options options)
                         selection (if (and (map? options-or-criteria)
                                            (not (contains? options-or-criteria :device)))
                                     (assoc options-or-criteria :device
                                            (:device (session/defaults context)))
                                     options-or-criteria)
                         loaded (.loadModel (criteria selection))]
                     (try
                       (session/register! context loaded
                                          (assoc options :kind :model :owned? true))
                       (catch Throwable failure
                         (try
                           (.close loaded)
                           (catch Throwable close-failure
                             (.addSuppressed failure close-failure)))
                         (throw failure))))))))

(defn- owned-output [^TranslatorContext context output]
  ;; Stage independent tensors in the predictor manager until inference succeeds.
  ;; This preserves borrowed inputs and lets failed predictions clean up copies.
  (cond
    (instance? NDList output)
    (NDList. ^java.util.Collection
     (mapv (fn [^NDArray array]
             (doto (.duplicate array)
               (.attach (.getPredictorManager context))))
           output))

    (instance? NDArray output)
    (doto (.duplicate ^NDArray output)
      (.attach (.getPredictorManager context)))

    :else output))

(defn- attach-output! [^NDManager manager output]
  (when (or (instance? NDList output) (instance? NDArray output))
    (.attach ^NDResource output manager))
  output)

(defn- owning-translator [^Translator translator]
  (when-not (instance? Translator translator)
    (invalid :model/invalid-translator "Prediction requires a native DJL Translator"
             {:translator translator}))
  (reify Translator
    (prepare [_ context] (.prepare translator context))
    (getBatchifier [_] (.getBatchifier translator))
    (getExpansions [_] (.getExpansions translator))
    (processInput [_ context input] (.processInput translator context input))
    (batchProcessInput [_ context inputs] (.batchProcessInput translator context inputs))
    (processOutput [_ context output]
      (owned-output context (.processOutput translator context output)))
    (batchProcessOutput [_ context output]
      (mapv #(owned-output context %) (.batchProcessOutput translator context output)))))

(defn- predict-output! [^Context context ^Model model translator input]
  (with-open [^Predictor predictor (.newPredictor model (owning-translator translator))]
    (attach-output! (context/manager context) (.predict predictor input))))

(defn predict
  "Predict one typed input under the context's session and model shared locks.
  Model may be a registered keyword name or an object. Three arguments use a
  ZooModel's packaged Translator; four accept any native Model and an explicit
  Translator: (predict context model translator input). Unregistered models are
  borrowed and caller-synchronized. Each Predictor closes before returning.
  Direct NDArray/NDList outputs belong to the context's manager; other typed
  outputs retain translator-defined ownership and are not coerced into tables."
  ([context model input]
   (session/shared context model :model
                   (fn [^ZooModel value]
                     (predict-output! context value (.getTranslator value) input))))
  ([context model translator input]
   (session/shared context model :model
                   (fn [value]
                     (predict-output! context value translator input)))))

(defn- batch-output! [^Context context ^Model model translator inputs]
  (with-open [^Predictor predictor (.newPredictor model (owning-translator translator))]
    (mapv #(attach-output! (context/manager context) %)
          (.batchPredict predictor ^List (vec inputs)))))

(defn batch-predict
  "Predict typed inputs under the context's session and model shared locks.
  Returns a vector with the same model and result ownership as predict. Model
  may be a registered keyword name or an object. Three arguments use a ZooModel's
  packaged Translator; four accept any native Model and an explicit Translator.
  A Translator without a batchifier processes each item individually, as DJL
  specifies. Direct NDArray/NDList results belong to the context's manager."
  ([context model inputs]
   (session/shared context model :model
                   (fn [^ZooModel value]
                     (batch-output! context value (.getTranslator value) inputs))))
  ([context model translator inputs]
   (session/shared context model :model
                   (fn [value]
                     (batch-output! context value translator inputs)))))

(defmethod session/describe-value :model [_ ^Model model]
  {:model-name (.getName model)
   :class (.getName (class model))
   :model-path (str (.getModelPath model))
   :engine (.getEngineName (.getEngine (.getNDManager model)))})
