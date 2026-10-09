(ns garun.nn-test
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.test :refer [deftest is testing]]
            [garun.blocks :as blocks]
            [garun.nn :as nn]
            [garun.table :as table]
            [garun.tensor :as tensor]
            [garun.training :as training])
  (:import [ai.djl Device Model]
           [ai.djl.nn Block]
           [ai.djl.ndarray NDArray NDList NDManager]
           [ai.djl.ndarray.types DataType Shape]
           [ai.djl.training DefaultTrainingConfig TrainingResult]
           [ai.djl.training.dataset ArrayDataset$Builder]
           [ai.djl.translate NoopTranslator Translator TranslatorContext]
           [garun.nn Network]
           [java.nio.file Files]
           [java.nio.file.attribute FileAttribute]))

(set! *warn-on-reflection* true)

(defn- squared-error [prediction actual]
  (let [difference (- prediction actual)]
    (* difference difference)))

(defn- mse [predictions truth]
  (/ (reduce + (map squared-error predictions truth)) (count truth)))

(defn- accuracy [predictions truth]
  (/ (count (filter true? (map = predictions truth))) (double (count truth))))

(defn- error-key [operation]
  (try
    (operation)
    nil
    (catch clojure.lang.ExceptionInfo error
      (:error (ex-data error)))))

(defn- regression-table []
  (let [x (mapv #(/ % 20.0) (range -20 21))]
    (table/from {:x x :y (mapv #(+ 1 (* 2 %)) x)})))

(deftest linear-regression-and-validation
  (with-open [db (table/open)
              net (nn/build [:linear 1] {:input-shape [1] :seed 11})]
    (let [source (regression-table)
          before (nn/predict net db source {:x [:x]})
          {:keys [history]} (nn/train! net db source
                                       {:x [:x] :y :y :epochs 180 :batch 16
                                        :optim [:adam {:lr 0.05}] :validation source})
          predictions (nn/predict net db source {:x [:x]})
          truth (mapv :y (table/rows db source))]
      (is (= 41 (count before)))
      (is (= 180 (count history)))
      (is (< (mse predictions truth) 0.05))
      (is (< (:loss (last history)) (:loss (first history))))
      (is (< (:validation-loss (last history)) 0.05)))))

(deftest binary-classification-uses-logits-and-sigmoid-probabilities
  (let [x (vec (concat (range -20 -1) (range 2 21)))
        truth (mapv #(if (pos? %) 1 0) x)
        source (table/from {:x (mapv #(/ % 10.0) x) :label truth})]
    (with-open [db (table/open)
                net (nn/build [:linear 1] {:input-shape [1] :seed 13})]
      (nn/train! net db source
                 {:x [:x] :y :label :loss :bce :epochs 90 :batch 16 :optim [:adam {:lr 0.08}]})
      (is (>= (accuracy (nn/predict net db source {:x [:x] :output :class}) truth) 0.9))
      (is (every? #(<= 0 % 1) (nn/predict net db source {:x [:x] :output :probability}))))))

(deftest three-class-classification-uses-sparse-labels-and-softmax
  (let [centres [[-2.0 -2.0] [2.0 -2.0] [0.0 2.0]]
        points (vec (for [[label [a b]] (map-indexed vector centres)
                          offset (range -6 6)]
                      {:a (+ a (* 0.03 offset)) :b (+ b (* 0.02 offset)) :label label}))
        source (table/from {:a (mapv :a points) :b (mapv :b points) :label (mapv :label points)})
        truth (mapv :label points)]
    (with-open [db (table/open)
                net (nn/build [:linear 3] {:input-shape [2] :seed 17})]
      (nn/train! net db source
                 {:x [:a :b] :y :label :loss :ce :epochs 70 :batch 12 :optim [:adam {:lr 0.05}]})
      (is (>= (accuracy (nn/predict net db source {:x [:a :b] :output :class}) truth) 0.9))
      (is (every? #(= 3 (count %)) (nn/predict net db source {:x [:a :b]})))
      (is (every? #(< (Math/abs (- 1.0 (reduce + %))) 1.0e-12)
                  (nn/predict net db source {:x [:a :b] :output :probability}))))))

(def ^:private composite-spec
  [:sequential
   [:skip {:combine :add} [:linear 2] [:tanh]]
   [:parallel {:combine :add} [:linear 2] [:identity]]
   [:skip {:combine :concat} [:linear 2]]
   [:parallel {:combine :concat} [:linear 2] [:linear 1]]
   [:linear 1]])

(defn- network-output-shape [net input-shape]
  (let [^Model model (:model net)
        ^Block block (.getBlock model)
        shapes (.getOutputShapes block (into-array Shape [(Shape. (long-array input-shape))]))
        ^Shape output-shape (first shapes)]
    (vec (.getShape output-shape))))

(defn- saved-network-predictions! [net db source features]
  (let [dir (.toFile (Files/createTempDirectory "garun-nn-test-" (make-array FileAttribute 0)))]
    (try
      (nn/save! net dir)
      (with-open [^Network restored (nn/load dir)]
        {:spec (:spec restored)
         :predictions (nn/predict restored db source {:x features})})
      (finally
        (doseq [^java.io.File file (reverse (file-seq dir))]
          (.delete file))))))

(deftest fnblock-gradients-shapes-and-child-serialization
  (with-open [db (table/open)
              net (nn/build composite-spec {:input-shape [2] :seed 19})]
    (let [x (mapv #(/ % 10.0) (range -10 11))
          source (table/from {:a x :b (mapv #(* 0.5 %) x) :y (mapv #(+ 1 (* 2 %)) x)})
          {:keys [history]} (nn/train! net db source {:x [:a :b] :y :y :epochs 80 :batch 21 :optim [:adam {:lr 0.025}]})
          predictions (nn/predict net db source {:x [:a :b]})]
      (is (= [7 1] (network-output-shape net [7 2])))
      (is (= 21 (count predictions)))
      (is (every? double? predictions))
      (is (< (:loss (last history)) (:loss (first history))))
      (is (= {:spec composite-spec :predictions predictions}
             (saved-network-predictions! net db source [:a :b]))))))

(deftest direct-combination-output-shapes
  (with-open [db (table/open)]
    (doseq [[spec width] [[[:parallel {:combine :add} [:linear 2] [:linear 2]] 2]
                          [[:parallel {:combine :concat} [:linear 2] [:linear 3]] 5]
                          [[:skip {:combine :add} [:linear 2]] 2]
                          [[:skip {:combine :concat} [:linear 3]] 5]]]
      (with-open [net (nn/build spec {:input-shape [2] :seed 4})]
        (is (= [width width]
               (mapv count (nn/predict net db (table/from {:a [1 2] :b [3 4]}) {:x [:a :b]}))))))))

(deftest normalization-dropout-flatten-and-activations
  (with-open [db (table/open)
              net (nn/build [:sequential [:linear 4] [:batchnorm] [:layernorm]
                             [:dropout {:p 0.2}] [:flatten] [:activation :relu]
                             [:tanh] [:sigmoid] [:gelu] [:identity] [:linear 1]]
                            {:input-shape [1] :seed 21})]
    (let [source (regression-table)
          {:keys [history]} (nn/train! net db source {:x [:x] :y [:y] :epochs 3 :batch 16})]
      (is (= 3 (count history)))
      (is (every? #(Double/isFinite (double (:loss %))) history))
      (is (= 41 (count (nn/predict net db source {:x [:x]})))))))

(deftest optimizer-and-mae-options
  (with-open [db (table/open)]
    (doseq [optim [[:sgd {:lr 0.05 :momentum 0.5}] [:adamw {:lr 0.05 :weight-decay 0.01}]]]
      (with-open [net (nn/build [:linear {:units 1 :bias? false}] {:input-shape [1]})]
        (let [history (:history (nn/train! net db (regression-table)
                                           {:x [:x] :y :y :loss :mae :epochs 3 :batch 16 :shuffle? false :optim optim}))]
          (is (= 3 (count history)))
          (is (every? #(Double/isFinite (double (:loss %))) history)))))))

(deftest multi-output-label-matrices
  (with-open [db (table/open)
              net (nn/build [:linear 2] {:input-shape [1] :seed 23})]
    (let [x (mapv #(/ % 10.0) (range -10 11))
          ya (mapv #(+ 1 (* 2 %)) x)
          yb (mapv #(- (* 3 %) 1) x)
          source (table/from {:x x :ya ya :yb yb})
          _ (nn/train! net db source
                       {:x [:x] :y [:ya :yb] :epochs 140 :batch 21 :optim [:adam {:lr 0.08}]})
          predictions (nn/predict net db source {:x [:x]})]
      (is (< (mse (mapv first predictions) ya) 0.05))
      (is (< (mse (mapv second predictions) yb) 0.05)))))

(defn- train-seeded-network! [db source]
  (with-open [net (nn/build [:sequential [:linear 3] [:tanh] [:dropout {:p 0.2}] [:linear 1]]
                            {:input-shape [1] :seed 29})]
    (nn/train! net db source {:x [:x] :y :y :epochs 5 :batch 13 :seed 31})))

(deftest seeds-reproduce-initialization-shuffling-and-dropout
  (with-open [db (table/open)]
    (let [source (regression-table)]
      (is (= (train-seeded-network! db source) (train-seeded-network! db source))))))

(deftest scalar-predictions-use-zero-threshold-and-stable-sigmoid
  (let [source (table/from {:x [-1000.0 -1.0 0.0 1.0 1000.0]})
        exponential (Math/exp -1.0)]
    (with-open [db (table/open)
                net (nn/build [:identity] {:input-shape [1]})]
      (is (= [-1000.0 -1.0 0.0 1.0 1000.0]
             (nn/predict net db source {:x [:x]})))
      (is (= [0 0 0 1 1] (nn/predict net db source {:x [:x] :output :class})))
      (is (= [0.0 (/ exponential (+ 1.0 exponential)) 0.5
              (/ 1.0 (+ 1.0 exponential)) 1.0]
             (nn/predict net db source {:x [:x] :output :probability}))))))

(deftest vector-predictions-use-first-argmax-and-stable-softmax
  (let [source (table/from {:a [0.0 10.0 -1000.0 1000.0]
                            :b [0.0 10.0 1000.0 -1000.0]})]
    (with-open [db (table/open)
                net (nn/build [:identity] {:input-shape [2]})]
      (is (= [[0.0 0.0] [10.0 10.0] [-1000.0 1000.0] [1000.0 -1000.0]]
             (nn/predict net db source {:x [:a :b]})))
      (is (= [0 0 1 0] (nn/predict net db source {:x [:a :b] :output :class})))
      (is (= [[0.5 0.5] [0.5 0.5] [0.0 1.0] [1.0 0.0]]
             (nn/predict net db source {:x [:a :b] :output :probability}))))))

(deftest block-argument-validation-is-consistent
  (testing "parameterized blocks report missing or surplus arguments"
    (doseq [[spec error] [[[:activation] :nn/invalid-spec]
                          [[:activation :relu :tanh] :nn/invalid-spec]
                          [[:linear 1 2] :nn/invalid-linear]
                          [[:dropout {} {}] :nn/invalid-dropout]
                          [[:parallel {:combine :add}] :nn/invalid-combination]
                          [[:skip {:combine :add}] :nn/invalid-combination]
                          [[:skip {:combine :multiply} [:identity]] :nn/invalid-combination]]]
      (is (= error (error-key #(nn/build spec {:input-shape [1]}))) (pr-str spec)))))

(deftest invalid-dsl-and-options
  (doseq [[spec error] [[[:unknown] :nn/unknown-tag]
                        [[:linear {:units 2 :typo true}] :nn/unknown-option]
                        [[:linear 0] :nn/invalid-linear]
                        [[:linear 1.5] :nn/invalid-linear]
                        [[:linear {:units 1 :bias? :yes}] :nn/invalid-linear]
                        [[:dropout {:p 1}] :nn/invalid-dropout]
                        [[:parallel {:combine :multiply} [:linear 1]] :nn/invalid-combination]
                        [[:parallel {:combine :add :oops 1} [:linear 1]] :nn/unknown-option]
                        [[:activation :unknown] :nn/invalid-activation]]]
    (is (= error (error-key #(nn/build spec {:input-shape [1]})))))
  (is (= :nn/unknown-option (error-key #(nn/build [:linear 1] {:input-shape [1] :oops 1}))))
  (is (= :nn/invalid-input-shape (error-key #(nn/build [:linear 1] {:input-shape [0]}))))
  (with-open [db (table/open)
              net (nn/build [:linear 1] {:input-shape [1]})]
    (let [source (regression-table)]
      (is (= :nn/unknown-option (error-key #(nn/train! net db source {:x [:x] :y :y :oops true}))))
      (is (= :nn/unknown-option (error-key #(nn/predict net db source {:x [:x] :oops true}))))
      (is (= :nn/invalid-output (error-key #(nn/predict net db source {:x [:x] :output :unknown}))))
      (is (= :nn/unknown-option (error-key #(nn/train! net db source {:x [:x] :y :y :optim [:adam {:lr 0.01 :oops 1}]}))))
      (is (= :nn/invalid-features (error-key #(nn/predict net db source {:x [:x :y]})))))))

(deftest native-dataset-training-improves-real-predictions
  (with-open [^NDManager manager (tensor/manager {})
              net (nn/build [:linear 1] {:input-shapes [[1 1]] :seed 41})]
    (let [x (mapv #(/ % 20.0) (range -20 21))
          truth (mapv #(+ 1 (* 2 %)) x)
          ^NDArray features (tensor/array manager x {:shape [41 1] :dtype :float32})
          ^NDArray labels (tensor/array manager truth {:shape [41 1] :dtype :float32})
          ^ArrayDataset$Builder builder (doto (ArrayDataset$Builder.)
                                          (.setData (into-array NDArray [features]))
                                          (.optLabels (into-array NDArray [labels]))
                                          (.setSampling (int 16) false))
          dataset (.build builder)
          before (with-open [^NDList output (nn/predict net features {:manager manager})]
                   (vec (.toFloatArray (.singletonOrThrow output))))
          ^TrainingResult result (nn/train! net dataset
                                            {:epochs 100 :validation dataset
                                             :config (training/config
                                                      {:device :cpu :optim [:adam {:lr 0.05}]})})
          after (with-open [^NDList output (nn/predict net features {:manager manager})]
                  (vec (.toFloatArray (.singletonOrThrow output))))]
      (is (< (mse after truth) (* 0.05 (mse before truth))))
      (is (< (mse after truth) 0.05))
      (is (= 100 (.getEpoch result)))
      (is (some? (.getTrainLoss result)))
      (is (< (.getValidateLoss result) 0.05))
      (is (= (mapv float x) (vec (.toFloatArray features))))
      (is (= 41 (.size labels)))
      ;; Reusing the same caller dataset exercises its lifetime after Trainer.close.
      (is (= :nn/device-mismatch
             (error-key #(nn/train! net dataset
                                    {:config (-> (DefaultTrainingConfig. (training/loss :mse))
                                                 (.optDevices (into-array Device
                                                                          [(Device/gpu Integer/MAX_VALUE)])))}))))
      (is (= :nn/invalid-training-options
             (error-key #(nn/train! net dataset {:config :invalid}))))
      (is (= :nn/invalid-manager (error-key #(nn/predict net features {}))))
      (is (= :nn/unknown-option
             (error-key #(nn/train! net dataset {:epochs 1 :batch 16}))))
      (is (instance? TrainingResult (nn/train! net dataset {:epochs 1}))))))

(deftest multidimensional-native-predictions-outlive-network-and-preserve-inputs
  (with-open [^NDManager manager (tensor/manager {})]
    (let [^NDArray input (tensor/array manager (range 12)
                                       {:shape [2 2 3] :dtype :float64})
          ^NDList output (with-open [net (nn/build [:identity]
                                                   {:input-shape [2 3] :dtype :float64})]
                           (is (= [[1 2 3]] (:input-shapes net)))
                           (nn/predict net (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray [input]))
                                       {:manager manager}))]
      (with-open [output output]
        (let [^NDArray result (.singletonOrThrow output)]
          (is (= [2 2 3] (vec (.getShape (.getShape result)))))
          (is (= DataType/FLOAT64 (.getDataType result)))
          (is (= manager (.getManager result)))
          (.addi result (Double/valueOf (double 100)))
          (is (= (mapv double (range 12)) (vec (.toDoubleArray input))))
          (is (= (mapv double (range 100 112)) (vec (.toDoubleArray result))))))
      (is (= (mapv double (range 12)) (vec (.toDoubleArray input)))))))

(deftest multiple-native-inputs-and-outputs
  (let [spec [:lambda (fn [^NDList inputs]
                        (let [^NDArray left (.get inputs (int 0))
                              ^NDArray right (.get inputs (int 1))]
                          (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray [(.add left right) (.sub left right)]))))]]
    (with-open [^NDManager manager (tensor/manager {})
                net (nn/build spec {:input-shapes [[1 2] [1 2]]})]
      (let [^NDArray left (tensor/array manager [[1 2] [3 4]] {:dtype :float32})
            ^NDArray right (tensor/array manager [[10 20] [30 40]] {:dtype :float32})]
        (with-open [^NDList output (nn/predict net [left right] {:manager manager})]
          (is (= 2 (.size output)))
          (is (= [11.0 22.0 33.0 44.0] (vec (.toFloatArray ^NDArray (.get output (int 0))))))
          (is (= [-9.0 -18.0 -27.0 -36.0] (vec (.toFloatArray ^NDArray (.get output (int 1)))))))
        (is (= [1.0 2.0 3.0 4.0] (vec (.toFloatArray left))))
        (is (= [10.0 20.0 30.0 40.0] (vec (.toFloatArray right))))))))

(deftest translator-pipelines-and-native-tensor-output-lifetimes
  (let [translator (reify Translator
                     (getBatchifier [_] nil)
                     (processInput [_ context input]
                       (tensor/ndlist (.getNDManager ^TranslatorContext context)
                                      {:data input :shape [1 2] :dtype :float32}))
                     (processOutput [_ _ output]
                       (reduce + (.toFloatArray (.singletonOrThrow ^NDList output)))))]
    (with-open [net (nn/build [:identity] {:input-shape [2]})]
      (is (= 7.0 (nn/predict-with net translator [3 4])))))
  (with-open [^NDManager manager (tensor/manager {})]
    (let [^NDArray input (tensor/array manager [[3 4]] {:dtype :float32})
          ^NDList result (with-open [net (nn/build [:identity] {:input-shape [2]})]
                           (nn/predict-with net (NoopTranslator. nil)
                                            (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray [input]))))]
      (with-open [result result]
        (is (= [3.0 4.0] (vec (.toFloatArray (.singletonOrThrow result))))))
      (is (= [3.0 4.0] (vec (.toFloatArray input)))))))

(defn- with-temporary-directory! [operation]
  (let [dir (.toFile (Files/createTempDirectory "garun-native-test-"
                                                (make-array FileAttribute 0)))]
    (try
      (operation dir)
      (finally
        (doseq [^java.io.File file (reverse (file-seq dir))]
          (.delete file))))))

(deftest custom-reconstruction-is-explicit-and-round-trips-parameters
  (let [spec [:factory #(blocks/compile-block [:linear 1])]]
    (with-temporary-directory!
      (fn [dir]
        (with-open [db (table/open)
                    ^NDManager manager (tensor/manager {})
                    net (nn/build spec {:input-shape [1]})]
          (is (= :nn/nonserializable-spec (error-key #(nn/save! net dir))))
          (is (empty? (.listFiles ^java.io.File dir)))
          (nn/train! net db (regression-table)
                     {:x [:x] :y :y :epochs 100 :optim [:adam {:lr 0.05}]})
          (let [input {:data [[0.0] [1.0]] :dtype :float32}
                before (with-open [^NDList output (nn/predict net input {:manager manager})]
                         (vec (.toFloatArray (.singletonOrThrow output))))]
            (nn/save! net dir {:spec spec})
            (is (= :nn/missing-reconstruction-spec (error-key #(nn/load dir))))
            (with-open [^Network restored (nn/load dir {:spec spec :device :cpu})
                        ^NDList output (nn/predict restored input {:manager manager})]
              (is (= before (vec (.toFloatArray (.singletonOrThrow output))))))
            (nn/save! net dir {:spec [:linear 1]})
            (with-open [^Network restored (nn/load dir)
                        ^NDList output (nn/predict restored input {:manager manager})]
              (is (= [:linear 1] (:spec restored)))
              (is (= before (vec (.toFloatArray (.singletonOrThrow output))))))))))))

(deftest dtype-device-validation-and-saved-overrides
  (is (= :tensor/invalid-dtype
         (error-key #(nn/build [:identity] {:input-shape [1] :dtype :decimal128}))))
  (is (= :tensor/unsupported-dtype
         (error-key #(nn/build [:linear 1] {:input-shape [1] :dtype :bfloat16}))))
  (is (= :tensor/unavailable-device
         (error-key #(nn/build [:identity] {:input-shape [1] :device [:gpu Integer/MAX_VALUE]}))))
  (is (= :nn/invalid-input-shape
         (error-key #(nn/build [:identity] {:input-shape [1] :input-shapes [[1 1]]}))))
  (with-open [db (table/open)
              net (nn/build [:identity] {:input-shape [2 3]})]
    (is (= :nn/non-tabular-input
           (error-key #(nn/predict net db (regression-table) {:x [:x]}))))
    (is (= :nn/non-tabular-input
           (error-key #(nn/train! net db (regression-table) {:x [:x] :y :y})))))
  (with-open [db (table/open)
              net (nn/build [:identity] {:input-shape [1] :dtype :float64})]
    (is (= :nn/non-tabular-dtype
           (error-key #(nn/predict net db (regression-table) {:x [:x]})))))
  (with-temporary-directory!
    (fn [dir]
      (with-open [net (nn/build [:linear 1] {:input-shape [1] :dtype :float64
                                             :engine "PyTorch"})]
        (nn/save! net dir))
      ;; Simulate a portable saved GPU preference without requiring CUDA hardware.
      (let [file (io/file dir "spec.edn")
            metadata (edn/read-string (slurp file))]
        (spit file (pr-str (assoc metadata :device [:gpu Integer/MAX_VALUE]))))
      (is (= :tensor/unavailable-device (error-key #(nn/load dir))))
      (with-open [^NDManager manager (tensor/manager {})
                  ^Network restored (nn/load dir {:device (Device/cpu)})
                  ^NDList output (nn/predict restored {:data [[1.0]] :dtype :float64}
                                             {:manager manager})]
        (is (= DataType/FLOAT64 (:dtype restored)))
        (is (= (Device/cpu) (.getDevice (.getNDManager ^Model (:model restored)))))
        (is (= DataType/FLOAT64 (.getDataType (.singletonOrThrow output)))))
      (with-open [^NDManager manager (tensor/manager {})
                  ^Network restored (nn/load dir {:device :cpu :dtype :float32})
                  ^NDList output (nn/predict restored {:data [[1.0]] :dtype :float32}
                                             {:manager manager})]
        (is (= DataType/FLOAT32 (.getDataType (.singletonOrThrow output))))))))

(deftest native-integer-identity-preserves-exact-values
  (with-open [^NDManager manager (tensor/manager {})
              net (nn/build [:identity] {:input-shapes [(Shape. (long-array [1 2]))]
                                         :dtype DataType/INT64})]
    (let [values [9007199254740993 9007199254740995]
          ^NDArray input (tensor/array manager values {:shape [1 2] :dtype :int64})]
      (with-open [^NDList output (nn/predict net input {:manager manager})]
        (is (= DataType/INT64 (.getDataType (.singletonOrThrow output))))
        (is (= values (vec (.toLongArray (.singletonOrThrow output))))))
      (is (= values (vec (.toLongArray input)))))))

(deftest native-scalar-and-empty-tensor-shapes
  (with-open [manager (tensor/manager {:device :cpu})
              scalar (nn/build [:identity] {:input-shapes [[]]})
              empty (nn/build [:identity] {:input-shapes [[0 2]]})]
    (with-open [^NDList output (nn/predict scalar {:data 7.5 :dtype :float32} {:manager manager})]
      (is (= [] (vec (.getShape (.getShape (.singletonOrThrow output))))))
      (is (= [7.5] (vec (.toFloatArray (.singletonOrThrow output))))))
    (with-open [^NDList output (nn/predict empty {:data [] :shape [0 2] :dtype :float32} {:manager manager})]
      (is (= [0 2] (vec (.getShape (.getShape (.singletonOrThrow output))))))
      (is (= [] (vec (.toFloatArray (.singletonOrThrow output))))))))
