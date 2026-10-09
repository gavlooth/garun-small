(ns garun.model-test
  (:require [clojure.datafy :as datafy]
            [clojure.edn :as edn]
            [clojure.test :refer [deftest is]]
            [garun.model :as model]
            [garun.session :as session]
            [garun.tensor :as tensor])
  (:import [ai.djl Application Device Model]
           [ai.djl.ndarray NDArray NDList NDManager]
           [ai.djl.ndarray.types DataType Shape]
           [ai.djl.nn Block Blocks Parameter$Type]
           [ai.djl.nn.core Linear]
           [ai.djl.repository.zoo Criteria]
           [ai.djl.training.initializer ConstantInitializer]
           [ai.djl.translate Batchifier DefaultTranslatorFactory NoopTranslator
            TranslateException Translator TranslatorContext]
           [garun.session Session]
           [java.nio.file Files Path]
           [java.nio.file.attribute FileAttribute]))

(set! *warn-on-reflection* true)

(defn- linear-block []
  (-> (Linear/builder) (.setUnits (long 1)) (.optBias false) .build))

(defn- with-local-model! [operation]
  (let [^Path path (Files/createTempDirectory "garun-model-test-" (make-array FileAttribute 0))
        directory (.toFile path)]
    (try
      (with-open [source (Model/newInstance "fixture" (Device/cpu) "PyTorch")]
        (let [^Block block (linear-block)]
          (.setBlock source block)
          (.setInitializer block (ConstantInitializer. (float 2)) Parameter$Type/WEIGHT)
          (.initialize block (.getNDManager source) DataType/FLOAT32
                       (into-array Shape [(Shape. (long-array [1 2 2]))]))
          (.save source path "fixture")))
      (operation path)
      (finally
        (doseq [^java.io.File file (reverse (file-seq directory))]
          (.delete file))))))

(defn- local-options [path]
  {:input-class NDList :output-class NDList
   :model-path path :name "fixture" :engine "PyTorch" :device :cpu
   :translator (NoopTranslator.) :block (linear-block)})

(defn- error-key [operation]
  (try
    (operation)
    nil
    (catch clojure.lang.ExceptionInfo error
      (:error (ex-data error)))))

(defn- text-translator []
  (reify Translator
    (getBatchifier [_] Batchifier/STACK)
    (processInput [_ context input]
      (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray
                                                        [(tensor/array (.getNDManager ^TranslatorContext context)
                                                                       (edn/read-string input) {:dtype :float32})])))
    (processOutput [_ context output]
      (let [^NDArray array (.singletonOrThrow ^NDList output)]
        {:values (vec (.toFloatArray array))
         :shape (vec (.getShape (.getShape array)))
         :context-manager (.getNDManager ^TranslatorContext context)
         :predictor-manager (.getPredictorManager ^TranslatorContext context)}))))

(defn- assert-closed-prediction-managers [result]
  (is (false? (.isOpen ^NDManager (:context-manager result))))
  (is (false? (.isOpen ^NDManager (:predictor-manager result)))))

(deftest local-criteria-loads-real-parameters-and-preserves-session-ownership
  (with-local-model!
    (fn [path]
      (let [{:keys [loaded output manager]}
            (with-open [^Session current-session (session/open {:device :cpu :engine "PyTorch"})]
              (let [loaded (model/load! current-session (dissoc (local-options path) :device)
                                        {:name :fixture})
                    input (tensor/ndlist current-session
                                         [{:data [[[1 2] [3 4]]] :dtype :float32}])
                    ^NDList output (model/predict current-session :fixture input)
                    ^NDList again (model/predict current-session loaded input)
                    description (session/describe current-session)
                    resource (first (:resources description))]
                (is (instance? java.lang.AutoCloseable loaded))
                (is (.isOpen (.getNDManager loaded)))
                (is (= (:device (session/defaults current-session))
                       (.getDevice (.getNDManager loaded))))
                (is (= [6.0 14.0] (vec (.toFloatArray (.singletonOrThrow again)))))
                (is (= [6.0 14.0] (vec (.toFloatArray (.singletonOrThrow output)))))
                (is (= [1 2 1] (vec (.getShape (.getShape (.singletonOrThrow output))))))
                (is (= [1.0 2.0 3.0 4.0]
                       (vec (.toFloatArray (.singletonOrThrow input)))))
                (is (not (identical? (.singletonOrThrow input) (.singletonOrThrow output))))
                (is (identical? (.manager current-session) (.getManager output)))
                (is (false? (.isReleased (.singletonOrThrow output))))
                (is (= :fixture (:name resource)))
                (is (= :model (:kind resource)))
                (is (true? (:owned? resource)))
                (is (false? (:released? resource)))
                (is (= (.getName loaded) (:model-name resource)))
                (is (= (.getName (class loaded)) (:class resource)))
                (is (= (str (.getModelPath loaded)) (:model-path resource)))
                (is (= "PyTorch" (:engine resource)))
                (is (= description (datafy/datafy current-session)))
                {:loaded loaded :output (.singletonOrThrow output)
                 :manager (.manager current-session)}))]
        (is (false? (.isOpen (.getNDManager ^Model loaded))))
        (is (false? (.isOpen ^NDManager manager)))
        (is (true? (.isReleased ^NDArray output)))))))

(deftest native-criteria-and-translator-factory-load-offline
  (with-local-model!
    (fn [^Path path]
      (let [factory (DefaultTranslatorFactory.)
            translator (text-translator)
            native (-> (Criteria/builder)
                       (.setTypes String clojure.lang.IPersistentMap)
                       (.optModelPath path)
                       (.optModelName "fixture")
                       (.optEngine "PyTorch")
                       (.optDevice (Device/cpu))
                       (.optBlock (linear-block))
                       (.optTranslator translator)
                       .build)]
        (is (identical? native (model/criteria native)))
        (.registerTranslator factory String clojure.lang.IPersistentMap translator)
        (doseq [selection [native
                           {:input-class String :output-class clojure.lang.IPersistentMap
                            :model-path (.toFile path) :name "fixture" :engine "PyTorch"
                            :device :cpu :block (linear-block) :translator-factory factory}]]
          (with-open [^Session current-session (session/open)]
            (let [loaded (model/load! current-session selection)
                  result (model/predict current-session loaded "[[[1 2] [3 4]]]")
                  batch (model/batch-predict current-session loaded
                                             ["[[[1 2] [3 4]]]" "[[[5 6] [7 8]]]"])]
              (is (= [6.0 14.0] (:values result)))
              (is (= [1 2 1] (:shape result)))
              (is (= [[6.0 14.0] [22.0 30.0]] (mapv :values batch)))
              (assert-closed-prediction-managers result)
              (doseq [prediction batch]
                (assert-closed-prediction-managers prediction))
              (is (.isOpen (.getNDManager loaded))))))))))

(deftest direct-tensor-results-and-batches-survive-predictor-close
  (with-local-model!
    (fn [path]
      (with-open [^Session current-session (session/open)]
        (let [loaded (model/load! current-session (local-options path) {:name :fixture})
              input (tensor/ndlist current-session
                                   [{:data [[[1 2] [3 4]]] :dtype :float32}])
              translator (reify Translator
                           (getBatchifier [_] nil)
                           (processInput [_ _ input] input)
                           (processOutput [_ _ output] (.singletonOrThrow ^NDList output)))
              ^NDArray output (model/predict current-session :fixture translator input)
              batch (model/batch-predict current-session :fixture [input input])
              array-batch (model/batch-predict current-session :fixture translator [input input])]
          (is (identical? loaded (session/value current-session :fixture :model)))
          (is (= [6.0 14.0] (vec (.toFloatArray output))))
          (is (identical? (.manager current-session) (.getManager output)))
          (is (= [[6.0 14.0] [6.0 14.0]]
                 (mapv (fn [^NDList output]
                         (is (identical? (.manager current-session) (.getManager output)))
                         (vec (.toFloatArray (.singletonOrThrow output)))) batch)))
          (is (= [[6.0 14.0] [6.0 14.0]]
                 (mapv (fn [^NDArray output]
                         (is (identical? (.manager current-session) (.getManager output)))
                         (vec (.toFloatArray output))) array-batch)))
          (is (= [1.0 2.0 3.0 4.0]
                 (vec (.toFloatArray (.singletonOrThrow input))))))))))

(deftest translator-errors-close-predictors-without-closing-models
  (with-local-model!
    (fn [path]
      (with-open [^Session current-session (session/open)]
        (let [loaded (model/load! current-session (local-options path))
              input (tensor/ndlist current-session
                                   [{:data [[[1 2] [3 4]]] :dtype :float32}])
              translator (reify Translator
                           (getBatchifier [_] nil)
                           (processInput [_ _ input] input)
                           (processOutput [_ context _]
                             (throw (ex-info "Translator rejected output"
                                             {:context-manager (.getNDManager ^TranslatorContext context)
                                              :predictor-manager (.getPredictorManager ^TranslatorContext context)}))))
              failure (try
                        (model/predict current-session loaded translator input)
                        nil
                        (catch TranslateException error error))]
          (is (instance? TranslateException failure))
          (is (= "Translator rejected output" (.getMessage (.getCause ^TranslateException failure))))
          (assert-closed-prediction-managers (ex-data (.getCause ^TranslateException failure)))
          (is (.isOpen (.getNDManager loaded)))
          (let [^NDList result (model/predict current-session loaded input)]
            (is (= [6.0 14.0] (vec (.toFloatArray (.singletonOrThrow result)))))))))))

(deftest local-loading-failures-are-not-replaced-with-models
  (with-local-model!
    (fn [^Path path]
      (with-open [^Session current-session (session/open)]
        (is (thrown? Exception
                     (model/load! current-session
                                  (assoc (local-options (.resolve path "missing"))
                                         :name "missing"))))))))

(deftest criteria-options-map-to-native-settings-and-reject-silent-knobs
  (let [options {:input-class NDList :output-class NDList :translator (NoopTranslator.)
                 :application Application/UNDEFINED :engine "PyTorch" :device :cpu
                 :name "weights" :group-id "example" :artifact-id "artifact"
                 :filters {:variant "tiny"} :arguments {:threshold 0.5}
                 :options {:epoch "3"}}
        criteria (model/criteria options)]
    (is (= {"variant" "tiny"} (.getFilters criteria)))
    (is (= {"threshold" 0.5} (.getArguments criteria)))
    (is (= {"epoch" "3"} (.getOptions criteria)))
    (is (= "weights" (.getModelName criteria)))
    (is (= (Device/cpu) (.getDevice criteria)))
    (is (= :model/unknown-option (error-key #(model/criteria (assoc options :typo true)))))
    (is (= :model/invalid-option (error-key #(model/criteria (dissoc options :input-class)))))
    (is (= :model/invalid-option (error-key #(model/criteria (assoc options :filters {:variant 1})))))
    (is (= :model/invalid-option
           (error-key #(model/criteria (assoc options :arguments {:x 1 "x" 2})))))
    (is (= :model/conflicting-options
           (error-key #(model/criteria (assoc options :translator-factory (DefaultTranslatorFactory.))))))
    (is (= :model/conflicting-options
           (error-key #(model/criteria (assoc options :model-path "local" :model-url "https://example.invalid/model")))))))

(deftest noop-identity-output-does-not-steal-input-ownership
  (with-open [^Session current-session (session/open)
              native (Model/newInstance "identity" (Device/cpu) "PyTorch")
              manager (.newSubManager (.manager current-session))
              input (tensor/ndlist manager [{:data [[1 2]] :dtype :float32}])]
    (let [^Block block (Blocks/identityBlock)]
      (.setBlock native block)
      (.initialize block (.getNDManager native) DataType/FLOAT32
                   (into-array Shape [(Shape. (long-array [1 2]))])))
    (with-open [^NDList result (model/predict current-session native (NoopTranslator.) input)]
      (is (= [1.0 2.0] (vec (.toFloatArray (.singletonOrThrow result))))))
    (is (= [1.0 2.0] (vec (.toFloatArray (.singletonOrThrow input)))))))

(deftest named-model-replacement-and-release-preserve-returned-tensors
  (with-local-model!
    (fn [path]
      (with-open [^Session current-session (session/open)]
        (let [options (local-options path)
              original (model/load! current-session options {:name :fixture})
              input (tensor/ndlist current-session
                                   [{:data [[[1 2] [3 4]]] :dtype :float32}])
              ^NDList output (model/predict current-session :fixture input)]
          (is (= :session/duplicate-name
                 (error-key #(model/load! current-session options {:name :fixture}))))
          (is (.isOpen (.getNDManager original)))
          (is (= 1 (count (:resources (session/describe current-session)))))
          (let [replacement (model/load! current-session options
                                         {:name :fixture :replace? true})
                ^NDList result (model/predict current-session :fixture input)]
            (is (false? (.isOpen (.getNDManager original))))
            (is (.isOpen (.getNDManager replacement)))
            (is (= [6.0 14.0] (vec (.toFloatArray (.singletonOrThrow output)))))
            (is (= [6.0 14.0] (vec (.toFloatArray (.singletonOrThrow result)))))
            (is (= 1 (count (:resources (session/describe current-session)))))
            (session/release! current-session :fixture)
            (is (false? (.isOpen (.getNDManager replacement))))
            (is (= [6.0 14.0] (vec (.toFloatArray (.singletonOrThrow result)))))
            (is (empty? (:resources (session/describe current-session))))
            (is (= :session/unknown-name
                   (error-key #(model/predict current-session :fixture input))))))))))

(deftest model-registration-options-reject-silent-knobs
  (with-open [^Session current-session (session/open)]
    (doseq [[options expected] [[nil :model/invalid-options]
                                [{:typo true} :model/unknown-option]
                                [{:name "fixture"} :model/invalid-option]
                                [{:name nil} :model/invalid-option]
                                [{:replace? :yes} :model/invalid-option]]]
      (is (= expected (error-key #(model/load! current-session {} options)))))
    (is (empty? (:resources (session/describe current-session))))))
