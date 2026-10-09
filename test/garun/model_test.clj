(ns garun.model-test
  (:require [clojure.edn :as edn]
            [clojure.test :refer [deftest is]]
            [garun.model :as model]
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
           [java.nio.file Files Path]
           [java.nio.file.attribute FileAttribute]))

(set! *warn-on-reflection* true)

(defn- linear-block []
  (-> (Linear/builder) (.setUnits (long 1)) (.optBias false) .build))

(defn- with-local-model! [operation]
  (let [^Path path (Files/createTempDirectory "garun-model-test-" (make-array FileAttribute 0))
        dir (.toFile path)]
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
        (doseq [^java.io.File file (reverse (file-seq dir))]
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

(deftest local-criteria-loads-real-parameters-and-preserves-ownership
  (with-local-model!
    (fn [path]
      (with-open [manager (tensor/manager {:device :cpu :engine "PyTorch"})
                  input (tensor/ndlist manager [{:data [[[1 2] [3 4]]] :dtype :float32}])]
        (let [^NDList output
              (with-open [loaded (model/load! (local-options path))]
                (is (instance? java.lang.AutoCloseable loaded))
                (let [prediction (model/predict loaded input)]
                  (is (.isOpen (.getNDManager loaded)))
                  (with-open [^NDList again (model/predict loaded input)]
                    (is (= [6.0 14.0] (vec (.toFloatArray (.singletonOrThrow again))))))
                  prediction))]
          (with-open [result output]
            (is (= [6.0 14.0] (vec (.toFloatArray (.singletonOrThrow result)))))
            (is (= [1 2 1] (vec (.getShape (.getShape (.singletonOrThrow result))))))
            (is (= [1.0 2.0 3.0 4.0]
                   (vec (.toFloatArray (.singletonOrThrow input)))))
            (is (not (identical? (.singletonOrThrow input) (.singletonOrThrow result))))))))))

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
          (with-open [loaded (model/load! selection)]
            (let [result (model/predict loaded "[[[1 2] [3 4]]]")
                  batch (model/batch-predict loaded
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
      (with-open [loaded (model/load! (local-options path))
                  manager (tensor/manager {:device :cpu :engine "PyTorch"})
                  input (tensor/ndlist manager [{:data [[[1 2] [3 4]]] :dtype :float32}])]
        (let [translator (reify Translator
                           (getBatchifier [_] nil)
                           (processInput [_ _ input] input)
                           (processOutput [_ _ output] (.singletonOrThrow ^NDList output)))
              batch (model/batch-predict loaded [input input])]
          (try
            (with-open [^NDArray output (model/predict loaded translator input)]
              (is (= [6.0 14.0] (vec (.toFloatArray output)))))
            (is (= [[6.0 14.0] [6.0 14.0]]
                   (mapv (fn [^NDList output]
                           (vec (.toFloatArray (.singletonOrThrow output)))) batch)))
            (is (= [1.0 2.0 3.0 4.0]
                   (vec (.toFloatArray (.singletonOrThrow input)))))
            (finally
              (doseq [^NDList output batch]
                (.close output)))))))))

(deftest translator-errors-close-predictors-without-closing-models
  (with-local-model!
    (fn [path]
      (with-open [loaded (model/load! (local-options path))
                  manager (tensor/manager {:device :cpu :engine "PyTorch"})
                  input (tensor/ndlist manager [{:data [[[1 2] [3 4]]] :dtype :float32}])]
        (let [translator (reify Translator
                           (getBatchifier [_] nil)
                           (processInput [_ _ input] input)
                           (processOutput [_ context _]
                             (throw (ex-info "Translator rejected output"
                                             {:context-manager (.getNDManager ^TranslatorContext context)
                                              :predictor-manager (.getPredictorManager ^TranslatorContext context)}))))
              failure (try
                        (model/predict loaded translator input)
                        nil
                        (catch TranslateException error error))]
          (is (instance? TranslateException failure))
          (is (= "Translator rejected output" (.getMessage (.getCause ^TranslateException failure))))
          (assert-closed-prediction-managers (ex-data (.getCause ^TranslateException failure)))
          (is (.isOpen (.getNDManager loaded)))
          (with-open [^NDList result (model/predict loaded input)]
            (is (= [6.0 14.0] (vec (.toFloatArray (.singletonOrThrow result)))))))))))

(deftest local-loading-failures-are-not-replaced-with-models
  (with-local-model!
    (fn [^Path path]
      (is (thrown? Exception
                   (model/load! (assoc (local-options (.resolve path "missing"))
                                       :name "missing")))))))

(deftest criteria-options-map-to-native-settings-and-reject-silent-knobs
  (let [opts {:input-class NDList :output-class NDList :translator (NoopTranslator.)
              :application Application/UNDEFINED :engine "PyTorch" :device :cpu
              :name "weights" :group-id "example" :artifact-id "artifact"
              :filters {:variant "tiny"} :arguments {:threshold 0.5}
              :options {:epoch "3"}}
        criteria (model/criteria opts)]
    (is (= {"variant" "tiny"} (.getFilters criteria)))
    (is (= {"threshold" 0.5} (.getArguments criteria)))
    (is (= {"epoch" "3"} (.getOptions criteria)))
    (is (= "weights" (.getModelName criteria)))
    (is (= (Device/cpu) (.getDevice criteria)))
    (is (= :model/unknown-option (error-key #(model/criteria (assoc opts :typo true)))))
    (is (= :model/invalid-option (error-key #(model/criteria (dissoc opts :input-class)))))
    (is (= :model/invalid-option (error-key #(model/criteria (assoc opts :filters {:variant 1})))))
    (is (= :model/invalid-option
           (error-key #(model/criteria (assoc opts :arguments {:x 1 "x" 2})))))
    (is (= :model/conflicting-options
           (error-key #(model/criteria (assoc opts :translator-factory (DefaultTranslatorFactory.))))))
    (is (= :model/conflicting-options
           (error-key #(model/criteria (assoc opts :model-path "local" :model-url "https://example.invalid/model")))))))

(deftest noop-identity-output-does-not-steal-input-ownership
  (with-open [native (Model/newInstance "identity" (Device/cpu) "PyTorch")
              manager (tensor/manager {:device :cpu :engine "PyTorch"})
              input (tensor/ndlist manager [{:data [[1 2]] :dtype :float32}])]
    (let [^Block block (Blocks/identityBlock)]
      (.setBlock native block)
      (.initialize block (.getNDManager native) DataType/FLOAT32
                   (into-array Shape [(Shape. (long-array [1 2]))])))
    (with-open [^NDList result (model/predict native (NoopTranslator.) input)]
      (is (= [1.0 2.0] (vec (.toFloatArray (.singletonOrThrow result))))))
    (is (= [1.0 2.0] (vec (.toFloatArray (.singletonOrThrow input)))))))
