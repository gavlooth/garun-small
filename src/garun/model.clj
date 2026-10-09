(ns garun.model
  "DJL model-zoo loading and typed translator inference.
  load! returns a caller-owned ZooModel; close it with with-open. Each prediction
  closes its own Predictor, not the model. NDArray/NDList results are independent,
  detached copies: callers must close them (or attach them to their own manager).
  Other result types retain their translator-defined ownership. Custom result
  objects containing tensors must manage those tensors in processOutput."
  (:require [clojure.java.io :as io]
            [garun.tensor :as tensor])
  (:import [ai.djl Application Model]
           [ai.djl.inference Predictor]
           [ai.djl.ndarray NDArray NDList]
           [ai.djl.nn Block]
           [ai.djl.repository.zoo Criteria Criteria$Builder ModelZoo ZooModel]
           [ai.djl.translate Translator TranslatorContext TranslatorFactory]
           [ai.djl.util Progress]
           [java.nio.file Path]
           [java.util List Map]))

(set! *warn-on-reflection* true)

(def ^:private criteria-options
  #{:input-class :output-class :translator :translator-factory :block
    :model-path :model-url :model-zoo :filters :arguments :options
    :group-id :artifact-id :application :name :engine :device :progress})

(defn- invalid [error message data]
  (throw (ex-info message (assoc data :error error))))

(defn- criteria-map [opts]
  (when-not (map? opts)
    (invalid :model/invalid-options "Criteria options must be a map or native Criteria"
             {:options opts}))
  (when-let [unknown (seq (remove criteria-options (keys opts)))]
    (invalid :model/unknown-option "Unknown model criteria option" {:keys (vec unknown)}))
  (doseq [[key expected] [[:input-class Class] [:output-class Class]
                          [:translator Translator] [:translator-factory TranslatorFactory]
                          [:block Block] [:model-zoo ModelZoo] [:application Application]
                          [:progress Progress]]]
    (when (and (or (contains? opts key) (#{:input-class :output-class} key))
               (not (instance? expected (get opts key))))
      (invalid :model/invalid-option "Model criteria option has the wrong type"
               {:key key :expected expected :value (get opts key)})))
  (doseq [key [:model-url :group-id :artifact-id :name :engine]]
    (when (and (contains? opts key)
               (not (and (string? (get opts key)) (seq (get opts key)))))
      (invalid :model/invalid-option "Model criteria option requires a nonempty string"
               {:key key :value (get opts key)})))
  (doseq [keys [[:translator :translator-factory]
                [:model-path :model-url :model-zoo]]]
    (when (< 1 (count (filter #(contains? opts %) keys)))
      (invalid :model/conflicting-options "Conflicting model criteria options"
               {:keys keys})))
  opts)

(defn- named-map [value string-values? key]
  (when-not (and (map? value)
                 (every? #(or (string? %) (keyword? %)) (keys value))
                 (or (not string-values?) (every? string? (vals value))))
    (invalid :model/invalid-option
             "Criteria maps require string/keyword keys; filters/options require string values"
             {:key key :value value}))
  (let [entries (mapv (fn [[k v]] [(if (keyword? k) (name k) k) v]) value)
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
  ^Criteria [opts]
  (if (instance? Criteria opts)
    opts
    (let [opts (criteria-map opts)
          ^Criteria$Builder builder (.setTypes (Criteria/builder)
                                               ^Class (:input-class opts)
                                               ^Class (:output-class opts))]
      (when-let [translator (:translator opts)]
        (.optTranslator builder ^Translator translator))
      (when-let [factory (:translator-factory opts)]
        (.optTranslatorFactory builder ^TranslatorFactory factory))
      (when-let [block (:block opts)]
        (.optBlock builder ^Block block))
      (when (contains? opts :model-path)
        (let [path (:model-path opts)]
          (when-not (or (instance? Path path) (instance? java.io.File path) (string? path))
            (invalid :model/invalid-option "Model path requires a Path, File, or string"
                     {:key :model-path :value path}))
          (.optModelPath builder (if (instance? Path path) path (.toPath (io/file path))))))
      (when-let [url (:model-url opts)]
        (.optModelUrls builder ^String url))
      (when-let [zoo (:model-zoo opts)]
        (.optModelZoo builder ^ModelZoo zoo))
      (when (contains? opts :filters)
        (.optFilters builder ^Map (named-map (:filters opts) true :filters)))
      (when (contains? opts :arguments)
        (.optArguments builder ^Map (named-map (:arguments opts) false :arguments)))
      (when (contains? opts :options)
        (.optOptions builder ^Map (named-map (:options opts) true :options)))
      (when-let [group (:group-id opts)]
        (.optGroupId builder ^String group))
      (when-let [artifact (:artifact-id opts)]
        (.optArtifactId builder ^String artifact))
      (when-let [application (:application opts)]
        (.optApplication builder ^Application application))
      (when-let [model-name (:name opts)]
        (.optModelName builder ^String model-name))
      (when-let [engine (:engine opts)]
        (.optEngine builder ^String engine))
      (when (contains? opts :device)
        (.optDevice builder (tensor/device (:device opts) (:engine opts))))
      (when-let [progress (:progress opts)]
        (.optProgress builder ^Progress progress))
      (.build builder))))

(defn load!
  "Load a caller-owned AutoCloseable ZooModel through actual Criteria.loadModel.
  Supports local parameter files and DJL remote/zoo locations. Loading failures
  propagate unchanged; no replacement model or download fallback is fabricated."
  ^ZooModel [opts-or-criteria]
  (.loadModel (criteria opts-or-criteria)))

(defn- owned-output [^TranslatorContext context output]
  ;; Predictor closes its per-call context before predict returns. Move copies
  ;; to its longer-lived manager first; failed predictions then close them too.
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

(defn- detach-output! [output]
  (when (or (instance? NDList output) (instance? NDArray output))
    (.detach ^ai.djl.ndarray.NDResource output))
  output)

(defn- owning-translator [^Translator translator]
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

(defn predict
  "Predict one typed input, closing the Predictor but preserving the caller's model.
  Two arguments use the ZooModel's packaged Translator. Three arguments accept
  any DJL Model and an explicit native Translator: (predict model translator input).
  Returned NDArray/NDList copies must be closed by the caller; ordinary typed
  outputs are returned unchanged and are not coerced into tabular values."
  ([^ZooModel model input]
   (predict model (.getTranslator model) input))
  ([^Model model translator input]
   (when-not (instance? Translator translator)
     (invalid :model/invalid-translator "Prediction requires a native DJL Translator"
              {:translator translator}))
   (with-open [^Predictor predictor (.newPredictor model (owning-translator translator))]
     (detach-output! (.predict predictor input)))))

(defn batch-predict
  "Predict a sequence of typed inputs using DJL's Translator batch semantics.
  Returns a vector. Model and result ownership match predict. A Translator with
  no batchifier processes each item individually, as specified by DJL."
  ([^ZooModel model inputs]
   (batch-predict model (.getTranslator model) inputs))
  ([^Model model translator inputs]
   (when-not (instance? Translator translator)
     (invalid :model/invalid-translator "Prediction requires a native DJL Translator"
              {:translator translator}))
   (with-open [^Predictor predictor (.newPredictor model (owning-translator translator))]
     (mapv detach-output! (.batchPredict predictor ^List (vec inputs))))))
