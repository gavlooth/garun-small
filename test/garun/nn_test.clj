(ns garun.nn-test
  (:require [clojure.datafy :as datafy]
            [garun.context :as context]
            [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as string]
            [clojure.test :refer [deftest is testing]]
            [garun.blocks :as blocks]
            [garun.nn :as nn]
            [garun.ndarray :as nd]
            [garun.nn.examples :as examples]
            [garun.session :as session]
            [garun.table :as table]
            [garun.tensor :as tensor]
            [garun.training :as training])
  (:import [ai.djl Device Model]
           [ai.djl.nn Block Parameter]
           [ai.djl.ndarray NDArray NDList NDManager]
           [ai.djl.ndarray.types DataType Shape]
           [ai.djl.training DefaultTrainingConfig Trainer]
           [ai.djl.training.dataset ArrayDataset$Builder Batch]
           [ai.djl.training.listener TrainingListenerAdapter]
           [ai.djl.translate NoopTranslator Translator TranslatorContext]
           [garun.session Session]
           [java.nio.file Files]
           [java.nio.file.attribute FileAttribute]
           [java.util.concurrent CompletableFuture ConcurrentLinkedQueue CountDownLatch TimeUnit TimeoutException]
           [java.util.function Supplier]))

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
  (with-open [current-session (session/open)]
    (let [network (nn/build current-session [:linear 1] {:input-shape [1] :seed 11})
          source (regression-table)
          before (nn/predict current-session network source {:x [:x]})
          {:keys [history]} (nn/train! current-session network source
                                       {:x [:x] :y :y :epochs 180 :batch 16
                                        :optim [:adam {:lr 0.05}] :validation source})
          predictions (nn/predict current-session network source {:x [:x]})
          truth (mapv :y (table/rows current-session source))]
      (is (= 41 (count before)))
      (is (= 180 (count history)))
      (is (< (mse predictions truth) 0.05))
      (is (< (:training-loss (last history)) (:training-loss (first history))))
      (is (< (:validation-loss (last history)) 0.05)))))

(deftest binary-classification-uses-logits-and-sigmoid-probabilities
  (let [values (vec (concat (range -20 -1) (range 2 21)))
        truth (mapv #(if (pos? %) 1 0) values)
        source (table/from {:x (mapv #(/ % 10.0) values) :label truth})]
    (with-open [current-session (session/open)]
      (let [network (nn/build current-session [:linear 1] {:input-shape [1] :seed 13})]
        (nn/train! current-session network source
                   {:x [:x] :y :label :loss :bce :epochs 90 :batch 16 :optim [:adam {:lr 0.08}]})
        (is (>= (accuracy (nn/predict current-session network source {:x [:x] :output :class}) truth) 0.9))
        (is (every? #(<= 0 % 1) (nn/predict current-session network source {:x [:x] :output :probability})))))))

(deftest three-class-classification-uses-sparse-labels-and-softmax
  (let [centres [[-2.0 -2.0] [2.0 -2.0] [0.0 2.0]]
        points (vec (for [[label [horizontal vertical]] (map-indexed vector centres)
                          offset (range -6 6)]
                      {:a (+ horizontal (* 0.03 offset)) :b (+ vertical (* 0.02 offset)) :label label}))
        source (table/from {:a (mapv :a points) :b (mapv :b points) :label (mapv :label points)})
        truth (mapv :label points)]
    (with-open [current-session (session/open)]
      (let [network (nn/build current-session [:linear 3] {:input-shape [2] :seed 17})]
        (nn/train! current-session network source
                   {:x [:a :b] :y :label :loss :ce :epochs 70 :batch 12 :optim [:adam {:lr 0.05}]})
        (is (>= (accuracy (nn/predict current-session network source {:x [:a :b] :output :class}) truth) 0.9))
        (is (every? #(= 3 (count %)) (nn/predict current-session network source {:x [:a :b]})))
        (let [evaluation (nn/evaluate current-session network source
                                      {:metrics [:accuracy :precision :recall :f1 :log-loss]})]
          (is (every? #(Double/isFinite (double %)) (vals evaluation)))
          (is (>= (:accuracy evaluation) 0.9)))
        (is (every? #(< (Math/abs (- 1.0 (reduce + %))) 1.0e-12)
                    (nn/predict current-session network source {:x [:a :b] :output :probability})))))))

(def ^:private composite-spec
  [:sequential
   [:skip {:combine :add} [:linear 2] [:tanh]]
   [:parallel {:combine :add} [:linear 2] [:identity]]
   [:skip {:combine :concat} [:linear 2]]
   [:parallel {:combine :concat} [:linear 2] [:linear 1]]
   [:linear 1]])

(defn- network-output-shape [network input-shape]
  (let [^Model model (:model network)
        ^Block block (.getBlock model)
        shapes (.getOutputShapes block (into-array Shape [(Shape. (long-array input-shape))]))
        ^Shape output-shape (first shapes)]
    (vec (.getShape output-shape))))

(defn- saved-network-predictions! [context network source features]
  (let [directory (.toFile (Files/createTempDirectory "garun-nn-test-" (make-array FileAttribute 0)))]
    (try
      (nn/save! context network directory)
      (let [restored (nn/load context directory)]
        {:spec (:spec restored)
         :predictions (nn/predict context restored source {:x features})})
      (finally
        (doseq [^java.io.File file (reverse (file-seq directory))]
          (.delete file))))))

(deftest fnblock-gradients-shapes-and-child-serialization
  (with-open [current-session (session/open)]
    (let [network (nn/build current-session composite-spec {:input-shape [2] :seed 19})
          values (mapv #(/ % 10.0) (range -10 11))
          source (table/from {:a values :b (mapv #(* 0.5 %) values) :y (mapv #(+ 1 (* 2 %)) values)})
          {:keys [history]} (nn/train! current-session network source
                                       {:x [:a :b] :y :y :epochs 80 :batch 21 :optim [:adam {:lr 0.025}]})
          predictions (nn/predict current-session network source {:x [:a :b]})]
      (is (= [7 1] (network-output-shape network [7 2])))
      (is (= 21 (count predictions)))
      (is (every? double? predictions))
      (is (< (:training-loss (last history)) (:training-loss (first history))))
      (is (= {:spec composite-spec :predictions predictions}
             (saved-network-predictions! current-session network source [:a :b]))))))

(deftest bidirectional-recurrent-networks-save-and-load-identical-predictions
  (let [directory (.toFile (Files/createTempDirectory "garun-bidirectional-"
                                                      (make-array FileAttribute 0)))]
    (try
      (with-open [current-session (session/open {:seed 43})]
        (doseq [tag [:rnn :gru :lstm]]
          (let [spec [:sequential
                      [tag {:state-size 2 :layers 2 :bidirectional? true :batch-first? true}]
                      [:index ":, -1"] [:linear 1]]
                network (nn/build current-session spec {:input-shape [4 3]})
                inputs (nd/arange current-session [2 4 3] {:dtype :float32})
                before (nn/predict current-session network inputs {})
                _ (nn/save! current-session network directory)
                restored (nn/load current-session directory)
                after (nn/predict current-session restored inputs {})]
            (is (= [[2 1]] (mapv nd/shape before) (mapv nd/shape after)) (name tag))
            (is (= (mapv nd/to-vector before) (mapv nd/to-vector after)) (name tag))
            (is (= spec (:spec restored))))))
      (finally
        (doseq [^java.io.File file (reverse (file-seq directory))]
          (.delete file))))))

(deftest direct-combination-output-shapes
  (with-open [current-session (session/open)]
    (doseq [[spec width] [[[:parallel {:combine :add} [:linear 2] [:linear 2]] 2]
                          [[:parallel {:combine :concat} [:linear 2] [:linear 3]] 5]
                          [[:skip {:combine :add} [:linear 2]] 2]
                          [[:skip {:combine :concat} [:linear 3]] 5]]]
      (let [network (nn/build current-session spec {:input-shape [2] :seed 4})]
        (is (= [width width]
               (mapv count (nn/predict current-session network
                                       (table/from {:a [1 2] :b [3 4]}) {:x [:a :b]}))))))))

(deftest normalization-dropout-flatten-and-activations
  (with-open [current-session (session/open)]
    (let [network (nn/build current-session
                            [:sequential [:linear 4] [:batchnorm] [:layernorm]
                             [:dropout {:p 0.2}] [:flatten] [:activation :relu]
                             [:tanh] [:sigmoid] [:gelu] [:identity] [:linear 1]]
                            {:input-shape [1] :seed 21})
          source (regression-table)
          {:keys [history]} (nn/train! current-session network source
                                       {:x [:x] :y [:y] :epochs 3 :batch 16})]
      (is (= 3 (count history)))
      (is (every? #(Double/isFinite (double (:training-loss %))) history))
      (is (= 41 (count (nn/predict current-session network source {:x [:x]})))))))

(deftest optimizer-and-mae-options
  (with-open [current-session (session/open)]
    (doseq [optimizer [[:sgd {:lr 0.05 :momentum 0.5}] [:adamw {:lr 0.05 :weight-decay 0.01}]]]
      (let [network (nn/build current-session [:linear {:units 1 :bias? false}] {:input-shape [1]})
            history (:history (nn/train! current-session network (regression-table)
                                         {:x [:x] :y :y :loss :mae :epochs 3 :batch 16
                                          :shuffle? false :optim optimizer}))]
        (is (= 3 (count history)))
        (is (every? #(Double/isFinite (double (:training-loss %))) history))))))

(deftest multi-output-label-matrices
  (with-open [current-session (session/open)]
    (let [network (nn/build current-session [:linear 2] {:input-shape [1] :seed 23})
          values (mapv #(/ % 10.0) (range -10 11))
          first-labels (mapv #(+ 1 (* 2 %)) values)
          second-labels (mapv #(- (* 3 %) 1) values)
          source (table/from {:x values :ya first-labels :yb second-labels})
          _ (nn/train! current-session network source
                       {:x [:x] :y [:ya :yb] :epochs 140 :batch 21 :optim [:adam {:lr 0.08}]})
          predictions (nn/predict current-session network source {:x [:x]})]
      (is (< (mse (mapv first predictions) first-labels) 0.05))
      (is (< (mse (mapv second predictions) second-labels) 0.05)))))

(defn- train-seeded-network! [context source]
  (let [network (nn/build context [:sequential [:linear 3] [:tanh] [:dropout {:p 0.2}] [:linear 1]]
                          {:input-shape [1] :seed 29})]
    (nn/train! context network source {:x [:x] :y :y :epochs 5 :batch 13 :seed 31})))

(deftest seeds-reproduce-initialization-shuffling-and-dropout
  (with-open [current-session (session/open)]
    (let [source (regression-table)]
      (is (= (update (train-seeded-network! current-session source) :history
                     #(mapv (fn [entry] (dissoc entry :elapsed-milliseconds)) %))
             (update (train-seeded-network! current-session source) :history
                     #(mapv (fn [entry] (dissoc entry :elapsed-milliseconds)) %)))))))

(deftest scalar-predictions-use-zero-threshold-and-stable-sigmoid
  (let [source (table/from {:x [-1000.0 -1.0 0.0 1.0 1000.0]})
        exponential (Math/exp -1.0)]
    (with-open [current-session (session/open)]
      (let [network (nn/build current-session [:identity] {:input-shape [1]})]
        (is (= [-1000.0 -1.0 0.0 1.0 1000.0]
               (nn/predict current-session network source {:x [:x]})))
        (is (= [0 0 0 1 1] (nn/predict current-session network source {:x [:x] :output :class})))
        (is (= [0.0 (/ exponential (+ 1.0 exponential)) 0.5
                (/ 1.0 (+ 1.0 exponential)) 1.0]
               (nn/predict current-session network source {:x [:x] :output :probability})))))))

(deftest vector-predictions-use-first-argmax-and-stable-softmax
  (let [source (table/from {:a [0.0 10.0 -1000.0 1000.0]
                            :b [0.0 10.0 1000.0 -1000.0]})]
    (with-open [current-session (session/open)]
      (let [network (nn/build current-session [:identity] {:input-shape [2]})]
        (is (= [[0.0 0.0] [10.0 10.0] [-1000.0 1000.0] [1000.0 -1000.0]]
               (nn/predict current-session network source {:x [:a :b]})))
        (is (= [0 0 1 0] (nn/predict current-session network source {:x [:a :b] :output :class})))
        (is (= [[0.5 0.5] [0.5 0.5] [0.0 1.0] [1.0 0.0]]
               (nn/predict current-session network source {:x [:a :b] :output :probability})))))))

(deftest block-argument-validation-is-consistent
  (with-open [current-session (session/open)]
    (testing "parameterized blocks report missing or surplus arguments"
      (doseq [[spec error] [[[:activation] :nn/invalid-spec]
                            [[:activation :relu :tanh] :nn/invalid-spec]
                            [[:linear 1 2] :nn/invalid-linear]
                            [[:dropout {} {}] :nn/invalid-dropout]
                            [[:parallel {:combine :add}] :nn/invalid-combination]
                            [[:skip {:combine :add}] :nn/invalid-combination]
                            [[:skip {:combine :multiply} [:identity]] :nn/invalid-combination]]]
        (is (= error (error-key #(nn/build current-session spec {:input-shape [1]})))
            (pr-str spec))))))

(deftest invalid-dsl-and-options
  (with-open [current-session (session/open)]
    (doseq [[spec error] [[[:unknown] :nn/unknown-tag]
                          [[:linear {:units 2 :typo true}] :nn/unknown-option]
                          [[:linear 0] :nn/invalid-linear]
                          [[:linear 1.5] :nn/invalid-linear]
                          [[:linear {:units 1 :bias? :yes}] :nn/invalid-linear]
                          [[:dropout {:p 1}] :nn/invalid-dropout]
                          [[:parallel {:combine :multiply} [:linear 1]] :nn/invalid-combination]
                          [[:parallel {:combine :add :oops 1} [:linear 1]] :nn/unknown-option]
                          [[:activation :unknown] :nn/invalid-activation]]]
      (is (= error (error-key #(nn/build current-session spec {:input-shape [1]})))))
    (is (= :nn/unknown-option
           (error-key #(nn/build current-session [:linear 1] {:input-shape [1] :oops 1}))))
    (is (= :nn/invalid-input-shape
           (error-key #(nn/build current-session [:linear 1] {:input-shape [0]}))))
    (let [network (nn/build current-session [:linear 1] {:input-shape [1]})
          source (regression-table)]
      (is (= :nn/unknown-option
             (error-key #(nn/train! current-session network source {:x [:x] :y :y :oops true}))))
      (is (= :nn/unknown-option
             (error-key #(nn/predict current-session network source {:x [:x] :oops true}))))
      (is (= :nn/invalid-output
             (error-key #(nn/predict current-session network source {:x [:x] :output :unknown}))))
      (is (= :nn/unknown-option
             (error-key #(nn/train! current-session network source
                                    {:x [:x] :y :y :optim [:adam {:lr 0.01 :oops 1}]}))))
      (is (= :nn/invalid-features
             (error-key #(nn/predict current-session network source {:x [:x :y]})))))))

(deftest native-dataset-training-improves-real-predictions
  (with-open [^Session current-session (session/open)]
    (let [network (nn/build current-session [:linear 1] {:input-shapes [[1 1]] :seed 41})
          values (mapv #(/ % 20.0) (range -20 21))
          truth (mapv #(+ 1 (* 2 %)) values)
          ^NDArray features (tensor/array current-session values {:shape [41 1] :dtype :float32})
          ^NDArray labels (tensor/array current-session truth {:shape [41 1] :dtype :float32})
          ^ArrayDataset$Builder builder (doto (ArrayDataset$Builder.)
                                          (.setData (into-array NDArray [features]))
                                          (.optLabels (into-array NDArray [labels]))
                                          (.setSampling (int 16) false))
          dataset (.build builder)
          before (with-open [^NDList output (nn/predict current-session network features {})]
                   (is (= (context/manager current-session) (.getManager (.singletonOrThrow output))))
                   (vec (.toFloatArray (.singletonOrThrow output))))
          result (nn/train! current-session network dataset
                            {:epochs 100 :validation dataset
                             :config (training/config {:device :cpu :optim [:adam {:lr 0.05}]})})
          after (with-open [^NDList output (nn/predict current-session network features {})]
                  (vec (.toFloatArray (.singletonOrThrow output))))]
      (is (< (mse after truth) (* 0.05 (mse before truth))))
      (is (< (mse after truth) 0.05))
      (is (= 100 (count (:history result))))
      (is (= #{:history :best :stopped} (set (keys result))))
      (is (= :completed (:stopped result)))
      (is (Double/isFinite (double (:training-loss (last (:history result))))))
      (is (< (:validation-loss (last (:history result))) 0.05))
      (is (Double/isFinite (double (:mae (nn/evaluate current-session network dataset {:metrics [:mae]})))))
      (is (= (mapv float values) (vec (.toFloatArray features))))
      (is (= 41 (.size labels)))
      ;; Reusing the same caller dataset exercises its lifetime after Trainer.close.
      (is (= :nn/device-mismatch
             (error-key #(nn/train! current-session network dataset
                                    {:config (-> (DefaultTrainingConfig. (training/loss :mse))
                                                 (.optDevices (into-array Device
                                                                          [(Device/gpu Integer/MAX_VALUE)])))}))))
      (is (= :nn/invalid-training-options
             (error-key #(nn/train! current-session network dataset {:config :invalid}))))
      (is (= :nn/invalid-manager
             (error-key #(nn/predict current-session network features {:manager :invalid}))))
      (is (= :nn/unknown-option
             (error-key #(nn/train! current-session network dataset {:epochs 1 :batch 16}))))
      (is (= :completed (:stopped (nn/train! current-session network dataset {:epochs 1}))))
      (let [handle (nn/trainer current-session network dataset {:epochs 4})]
        (try
          (is (= {:network nil :epochs 4 :batch nil :validation? false :early-stopping nil}
                 (datafy/datafy handle)))
          (finally (session/release! current-session handle))))
      (is (= [:network] (mapv :kind (:resources (session/describe current-session))))))))

(deftest multidimensional-native-predictions-outlive-network-and-preserve-inputs
  (with-open [^Session owner (session/open)]
    (let [^NDManager manager (context/manager owner)
          ^NDArray input (tensor/array owner (range 12) {:shape [2 2 3] :dtype :float64})
          ^NDList output
          (with-open [current-session (session/open)]
            (let [network (nn/build current-session [:identity]
                                    {:input-shape [2 3] :dtype :float64})]
              (is (= [[1 2 3]] (:input-shapes network)))
              (nn/predict current-session network
                          (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray [input]))
                          {:manager manager})))]
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
  (let [spec [:lambda (fn [inputs]
                        (let [[array other] inputs]
                          (nd/ndlist [(nd/add array other) (nd/sub array other)])))]]
    (with-open [current-session (session/open)]
      (let [network (nn/build current-session spec {:input-shapes [[1 2] [1 2]]})
            ^NDArray left (tensor/array current-session [[1 2] [3 4]] {:dtype :float32})
            ^NDArray right (tensor/array current-session [[10 20] [30 40]] {:dtype :float32})]
        (with-open [^NDList output (nn/predict current-session network [left right] {})]
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
    (with-open [current-session (session/open)]
      (let [network (nn/build current-session [:identity] {:input-shape [2]})]
        (is (= 7.0 (nn/predict-with current-session network translator [3 4]))))))
  (with-open [^Session owner (session/open)]
    (let [^NDArray input (tensor/array owner [[3 4]] {:dtype :float32})
          {:keys [manager outputs]}
          (with-open [^Session current-session (session/open)]
            (let [network (nn/build current-session [:identity] {:input-shape [2]})
                  inputs (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray [input]))
                  array-translator (reify Translator
                                     (getBatchifier [_] nil)
                                     (processInput [_ _ value] value)
                                     (processOutput [_ _ output] (.singletonOrThrow ^NDList output)))
                  ^NDList list-output (nn/predict-with current-session network (NoopTranslator. nil) inputs)
                  ^NDArray array-output (nn/predict-with current-session network array-translator inputs)
                  ^NDArray list-array (.singletonOrThrow list-output)
                  outputs [list-array array-output]]
              (is (every? #(identical? (context/manager current-session) (.getManager ^NDArray %)) outputs))
              (session/release! current-session network)
              (is (= [[3.0 4.0] [3.0 4.0]]
                     (mapv #(vec (.toFloatArray ^NDArray %)) outputs)))
              (.addi array-output (Float/valueOf (float 10)))
              (is (= [3.0 4.0] (vec (.toFloatArray input))))
              (is (= [3.0 4.0] (vec (.toFloatArray list-array))))
              {:manager (context/manager current-session) :outputs outputs}))]
      (is (false? (.isOpen ^NDManager manager)))
      (is (every? #(.isReleased ^NDArray %) outputs))
      (is (= [3.0 4.0] (vec (.toFloatArray input)))))))

(defn- with-temporary-directory! [operation]
  (let [directory (.toFile (Files/createTempDirectory "garun-native-test-"
                                                      (make-array FileAttribute 0)))]
    (try
      (operation directory)
      (finally
        (doseq [^java.io.File file (reverse (file-seq directory))]
          (.delete file))))))

(deftest custom-reconstruction-is-explicit-and-round-trips-parameters
  (let [spec [:factory #(blocks/compile-block [:linear 1])]]
    (with-temporary-directory!
      (fn [directory]
        (with-open [current-session (session/open)]
          (let [network (nn/build current-session spec {:input-shape [1]})]
            (is (= :nn/nonserializable-spec
                   (error-key #(nn/save! current-session network directory))))
            (is (empty? (.listFiles ^java.io.File directory)))
            (nn/train! current-session network (regression-table)
                       {:x [:x] :y :y :epochs 100 :optim [:adam {:lr 0.05}]})
            (let [input {:data [[0.0] [1.0]] :dtype :float32}
                  before (with-open [^NDList output (nn/predict current-session network input {})]
                           (vec (.toFloatArray (.singletonOrThrow output))))]
              (nn/save! current-session network directory {:spec spec})
              (is (= :nn/missing-reconstruction-spec
                     (error-key #(nn/load current-session directory))))
              (let [restored (nn/load current-session directory {:spec spec :device :cpu})]
                (with-open [^NDList output (nn/predict current-session restored input {})]
                  (is (= before (vec (.toFloatArray (.singletonOrThrow output)))))))
              (nn/save! current-session network directory {:spec [:linear 1]})
              (let [restored (nn/load current-session directory)]
                (with-open [^NDList output (nn/predict current-session restored input {})]
                  (is (= [:linear 1] (:spec restored)))
                  (is (= before (vec (.toFloatArray (.singletonOrThrow output))))))))))))))

(deftest dtype-device-validation-and-saved-overrides
  (with-open [current-session (session/open)]
    (is (= :tensor/invalid-dtype
           (error-key #(nn/build current-session [:identity] {:input-shape [1] :dtype :decimal128}))))
    (is (= :tensor/unsupported-dtype
           (error-key #(nn/build current-session [:linear 1] {:input-shape [1] :dtype :bfloat16}))))
    (is (= :tensor/unavailable-device
           (error-key #(nn/build current-session [:identity]
                                 {:input-shape [1] :device [:gpu Integer/MAX_VALUE]}))))
    (is (= :nn/invalid-input-shape
           (error-key #(nn/build current-session [:identity] {:input-shape [1] :input-shapes [[1 1]]}))))
    (let [network (nn/build current-session [:identity] {:input-shape [2 3]})]
      (is (= :nn/non-tabular-input
             (error-key #(nn/predict current-session network (regression-table) {:x [:x]}))))
      (is (= :nn/non-tabular-input
             (error-key #(nn/train! current-session network (regression-table) {:x [:x] :y :y})))))
    (let [network (nn/build current-session [:identity] {:input-shape [1] :dtype :float64})]
      (is (= :nn/non-tabular-dtype
             (error-key #(nn/predict current-session network (regression-table) {:x [:x]}))))))
  (with-temporary-directory!
    (fn [directory]
      (with-open [current-session (session/open)]
        (let [network (nn/build current-session [:linear 1]
                                {:input-shape [1] :dtype :float64 :engine "PyTorch"})]
          (nn/save! current-session network directory)))
      ;; Simulate a portable saved GPU preference without requiring CUDA hardware.
      (let [file (io/file directory "spec.edn")
            metadata (edn/read-string (slurp file))]
        (spit file (pr-str (assoc metadata :device [:gpu Integer/MAX_VALUE]))))
      (with-open [current-session (session/open)]
        (is (= :tensor/unavailable-device (error-key #(nn/load current-session directory))))
        (let [restored (nn/load current-session directory {:device (Device/cpu)})]
          (with-open [^NDList output
                      (nn/predict current-session restored {:data [[1.0]] :dtype :float64} {})]
            (is (= DataType/FLOAT64 (:dtype restored)))
            (is (= (Device/cpu) (.getDevice (.getNDManager ^Model (:model restored)))))
            (is (= DataType/FLOAT64 (.getDataType (.singletonOrThrow output))))))
        (let [restored (nn/load current-session directory {:device :cpu :dtype :float32})]
          (with-open [^NDList output
                      (nn/predict current-session restored {:data [[1.0]] :dtype :float32} {})]
            (is (= DataType/FLOAT32 (.getDataType (.singletonOrThrow output))))))))))

(deftest native-integer-identity-preserves-exact-values
  (with-open [current-session (session/open)]
    (let [network (nn/build current-session [:identity]
                            {:input-shapes [(Shape. (long-array [1 2]))] :dtype DataType/INT64})
          values [9007199254740993 9007199254740995]
          ^NDArray input (tensor/array current-session values {:shape [1 2] :dtype :int64})]
      (with-open [^NDList output (nn/predict current-session network input {})]
        (is (= DataType/INT64 (.getDataType (.singletonOrThrow output))))
        (is (= values (vec (.toLongArray (.singletonOrThrow output))))))
      (is (= values (vec (.toLongArray input)))))))

(deftest native-scalar-and-empty-tensor-shapes
  (with-open [current-session (session/open {:device :cpu})]
    (let [scalar (nn/build current-session [:identity] {:input-shapes [[]]})
          empty-network (nn/build current-session [:identity] {:input-shapes [[0 2]]})]
      (with-open [^NDList output (nn/predict current-session scalar {:data 7.5 :dtype :float32} {})]
        (is (= [] (vec (.getShape (.getShape (.singletonOrThrow output))))))
        (is (= [7.5] (vec (.toFloatArray (.singletonOrThrow output))))))
      (with-open [^NDList output
                  (nn/predict current-session empty-network {:data [] :shape [0 2] :dtype :float32} {})]
        (is (= [0 2] (vec (.getShape (.getShape (.singletonOrThrow output))))))
        (is (= [] (vec (.toFloatArray (.singletonOrThrow output)))))))))

(deftest named-networks-train-and-predict-through-the-session
  (with-open [current-session (session/open)]
    (let [network (nn/build current-session [:linear 1]
                            {:input-shape [1] :name :regression :seed 11})
          source (regression-table)
          before (nn/predict current-session :regression source {:x [:x]})
          {:keys [history]} (nn/train! current-session :regression source
                                       {:x [:x] :y :y :epochs 5 :batch 16})
          predictions (nn/predict current-session :regression source {:x [:x]})]
      (is (= 5 (count history)))
      (is (not= before predictions))
      (is (= predictions (nn/predict current-session network source {:x [:x]}))))))

(deftest network-inspection-is-readable-plain-data
  (with-open [current-session (session/open)]
    (let [network (nn/build current-session [:linear 2] {:input-shape [3]})
          description (datafy/datafy network)
          printed (with-out-str (prn network))]
      (is (string/starts-with? printed "#garun/network "))
      (is (string/includes? printed ":spec [:linear 2]"))
      (is (string/includes? printed ":dtype :float32"))
      (is (= {:spec [:linear 2] :input-shapes [[1 3]] :dtype :float32
              :device :cpu :engine "PyTorch" :parameters 8}
             description)))))

(deftest closing-a-session-closes-its-owned-network-manager
  (let [^NDManager model-manager
        (with-open [current-session (session/open)]
          (let [network (nn/build current-session [:linear 1] {:input-shape [1]})
                ^NDManager manager (.getNDManager ^Model (:model network))]
            (is (.isOpen manager))
            manager))]
    (is (not (.isOpen model-manager)))))

(deftest workers-share-networks-and-materialized-table-predictions
  (with-open [current-session (session/open)]
    (let [network (nn/build current-session [:linear 1] {:input-shape [1] :name :regression})
          source (table/materialize! current-session :training-data (regression-table))
          predictions (nn/predict current-session network source {:x [:x]})]
      (with-open [worker (session/worker current-session)]
        (is (= predictions (nn/predict worker :regression :training-data {:x [:x]})))))))

(deftest loading-registers-a-network-under-its-name
  (with-temporary-directory!
    (fn [directory]
      (with-open [current-session (session/open)]
        (let [network (nn/build current-session [:linear 1] {:input-shape [1]})
              source (regression-table)
              predictions (nn/predict current-session network source {:x [:x]})]
          (nn/save! current-session network directory)
          (let [restored (nn/load current-session directory {:name :restored})]
            (is (= [:linear 1] (:spec restored)))
            (is (= predictions (nn/predict current-session :restored source {:x [:x]})))
            (is (= predictions (nn/predict current-session restored source {:x [:x]})))))))))

(deftest build-inherits-session-defaults-and-registration-errors-are-loud
  (with-open [current-session (session/open {:dtype :float64 :seed 37})]
    (let [network (nn/build current-session [:linear 1] {:input-shape [1] :name :regression})
          ^NDManager original-manager (.getNDManager ^Model (:model network))]
      (is (= DataType/FLOAT64 (:dtype network)))
      (is (= :session/duplicate-name
             (error-key #(nn/build current-session [:linear 1]
                                   {:input-shape [1] :name :regression}))))
      (is (.isOpen original-manager))
      (nn/build current-session [:identity]
                {:input-shape [1] :name :regression :replace? true})
      (is (not (.isOpen original-manager)))
      (is (= :session/unknown-name
             (error-key #(nn/predict current-session :missing [[1.0]] {})))))))

(defn- sequence-table [shift]
  (table/from
   (vec (reverse
         (for [[partition row-count] [["battery-c" 2] ["battery-a" 6] ["battery-b" 5]]
               cycle (range row-count)
               :let [value (+ shift cycle)]]
           {:battery partition :cycle cycle :x value :y (+ 100.0 (* 20.0 value))})))))

(deftest sequence-windows-use-exact-window-end-flags-and-never-cross-string-partitions
  (with-open [current-session (session/open)]
    (let [partitions [["battery-a" 6 11] ["battery-b" 2 22] ["battery-c" 5 33]
                      ["battery-d" 1 44] ["battery-e" 3 55]]
          source (table/from
                  (vec (reverse
                        (for [[battery rows feature] partitions
                              cycle (range rows)]
                          {:battery battery :cycle cycle :x feature :y (+ (* feature 100) cycle)}))))
          specification {:x [[:x]] :y :y :sequence {:length 3 :partition-by :battery :order-by :cycle}}
          queries (ConcurrentLinkedQueue.)
          extract-matrix table/matrix
          extracted (with-redefs [table/matrix (fn [context source columns]
                                                 (.add queries (first (table/->sql context source)))
                                                 (extract-matrix context source columns))]
                      (examples/table-examples current-session source specification true))
          expected-features (mapcat (fn [[_ rows feature]] (repeat (max 0 (- rows 2)) (float feature)))
                                    partitions)
          expected-labels (mapcat (fn [[_ rows feature]]
                                    (map #(float (+ (* feature 100) %)) (range 2 rows)))
                                  partitions)
          windows (partition 3 (:data (first (:features extracted))))]
      (is (= 8 (reduce + (map (fn [[_ rows _]] (max 0 (- rows 2))) partitions))
             (:rows extracted) (count windows)))
      (is (= [8 3 1] (get-in extracted [:features 0 :shape])))
      (is (every? #(apply = %) windows))
      (is (= (vec expected-features) (mapv first windows)))
      (is (= (vec expected-labels) (vec (get-in extracted [:labels :data]))))
      (is (= 1 (.size queries)))
      (is (string/includes? (.peek queries) "ROW_NUMBER()"))
      (is (not (string/includes? (.peek queries) "DENSE_RANK"))))))

(def ^:private sequence-options
  {:x [:x] :y :y :sequence {:length 3 :partition-by :battery :order-by :cycle}
   :standardize [:x :y] :epochs 140 :batch 7 :shuffle? false
   :optim [:adam {:lr 0.08}]})

(defn- network-inputs [network]
  (edn/read-string (.getProperty ^Model (:model network) "garun.inputs")))

(deftest sequence-regression-observation-evaluation-and-stored-input-roundtrip
  (with-temporary-directory!
    (fn [directory]
      (with-open [current-session (session/open {:seed 2026})]
        (let [source (sequence-table 0.0)
              validation (sequence-table 0.2)
              network (nn/build current-session [:sequential [:index ":, -1"] [:linear 1]]
                                {:input-shape [3 1] :name :sequence})
              events (ConcurrentLinkedQueue.)
              result (nn/train! current-session :sequence source
                                (assoc sequence-options :validation validation
                                       :metrics [:mae] :observe #(.add events %)))
              history (:history result)
              predictions (nn/predict current-session :sequence validation {})
              evaluation (nn/evaluate current-session :sequence validation {:metrics [:mae :mse :r2]})
              observed (vec events)
              expected-windows (reduce + (map #(max 0 (inc (- % 3))) [6 5 2]))]
          (is (= 7 expected-windows (count predictions)))
          (is (< (:validation-loss (last history)) (:validation-loss (first history))))
          (is (Double/isFinite (double (:mae evaluation))))
          (is (< (:mae evaluation) 5.0))
          (is (every? #(> % 90.0) predictions))
          (is (= (mapv #(assoc % :event :epoch :network :sequence) history) (pop observed)))
          (is (= {:event :finished :stopped (:stopped result) :best (:best result)} (peek observed)))
          (is (every? #(Double/isFinite (double (get-in % [:metrics :mae]))) history))
          (is (= [:network] (mapv :kind (:resources (session/describe current-session)))))
          (is (nil? (.getProperty ^Model (:model network) "garun.training")))
          (nn/save! current-session :sequence directory)
          (let [loaded (nn/load current-session directory)]
            (is (= (network-inputs network) (network-inputs loaded)))
            (is (nil? (.getProperty ^Model (:model loaded) "garun.training")))
            (is (= predictions (nn/predict current-session loaded validation {})))))))))

(deftest early-stopping-restores-the-best-device-parameters
  (with-open [current-session (session/open)]
    (let [source (regression-table)
          network (nn/build current-session [:linear 1] {:input-shape [1] :seed 101})
          ^Model model (:model network)
          snapshots (mapv (fn [^Parameter parameter] (.duplicate (.getArray parameter)))
                          (.values (.getParameters (.getBlock model))))
          events (ConcurrentLinkedQueue.)
          result (nn/train! current-session network source
                            {:x [:x] :y :y :validation source :epochs 20 :batch 41
                             :shuffle? false :optim [:sgd {:lr [:fixed {:value 0.0}]}]
                             :early-stopping {:patience 1 :restore-best? true}
                             :observe (fn [event]
                                        (.add events event)
                                        ;; Perturb after the plateau is observed. Restore
                                        ;; must undo it, not merely retain final weights.
                                        (when (and (= :epoch (:event event)) (= 2 (:epoch event)))
                                          (doseq [^Parameter parameter
                                                  (.values (.getParameters (.getBlock model)))]
                                            (.addi (.getArray parameter) (Float/valueOf (float 10))))))})
          evaluation (nn/evaluate current-session network source {})]
      (try
        (is (= :early-stopping (:stopped result)))
        (is (= 2 (count (:history result))))
        (is (= {:epoch 1 :loss (:validation-loss (first (:history result))) :monitor :validation-loss}
               (:best result)))
        (is (< (Math/abs (double (- (:loss evaluation) (get-in result [:best :loss])))) 1.0e-6))
        (is (every? true? (map (fn [^Parameter parameter ^NDArray snapshot]
                                 (.allClose (.getArray parameter) snapshot 1.0e-7 1.0e-7 false))
                               (.values (.getParameters (.getBlock model))) snapshots)))
        (is (= [:epoch :epoch :finished] (mapv :event events)))
        (finally (doseq [^NDArray snapshot snapshots] (.close snapshot)))))))

(deftest standardization-uses-training-aggregates-and-original-target-units
  (with-open [current-session (session/open)]
    (let [network (nn/build current-session [:linear 1] {:input-shape [1]})
          ^Model model (:model network)
          ^Block block (.getBlock model)
          ^Parameter weight (.get (.getDirectParameters block) "weight")
          ^Parameter bias (.get (.getDirectParameters block) "bias")
          source (table/from {:x [0.0 1.0 2.0 3.0] :y [50.0 70.0 90.0 110.0]})
          validation (table/from {:x [100.0 101.0 102.0 103.0] :y [2050.0 2070.0 2090.0 2110.0]})]
      (.set (.getArray weight) (float-array [0.5]))
      (.set (.getArray bias) (float-array [0.0]))
      (nn/train! current-session network source
                 {:x [:x] :y :y :standardize [:x :y] :validation validation
                  :epochs 1 :batch 4 :optim [:sgd {:lr [:fixed {:value 0.0}]}]})
      (let [statistics (:statistics (network-inputs network))
            predictions (nn/predict current-session network source {})]
        (is (= 1.5 (get-in statistics [:x 0 0 :mean])))
        (is (= 80.0 (get-in statistics [:y 0 :mean])))
        (is (< (Math/abs (- (Math/sqrt 1.25)
                            (get-in statistics [:x 0 0 :standard-deviation]))) 1.0e-12))
        (is (< (mse predictions [65.0 75.0 85.0 95.0]) 1.0e-8))
        (is (= [0.0 0.5 1.0 1.5]
               (nn/predict current-session network source {:standardize []})))
        (is (= 4 (count (nn/predict current-session network validation {}))))))))

(deftest lower-training-layer-supports-user-reductions-and-custom-metrics
  (with-open [current-session (session/open)]
    (let [source (regression-table)
          network (nn/build current-session [:linear 1] {:input-shape [1] :name :custom})
          constant-metric (fn [_predictions _truth] 42.0)
          closed-trainers (ConcurrentLinkedQueue.)
          listener (proxy [TrainingListenerAdapter] []
                     (onTrainingEnd [native-trainer] (.add closed-trainers native-trainer)))
          handle (nn/trainer current-session :custom source
                             {:x [:x] :y :y :validation source :epochs 50 :batch 41
                              :metrics [constant-metric] :listeners [listener]
                              :optim [:sgd {:lr [:fixed {:value 0.0}]}]})
          description {:network :custom :epochs 50 :batch 41 :validation? true :early-stopping nil}]
      (try
        (is (identical? network (:network handle)))
        (is (= description (datafy/datafy handle)))
        (let [state (reduce (fn [state _]
                              (let [state (nn/epoch! handle state)]
                                (if (= 3 (:epoch state)) (reduced state) state)))
                            {} (range 50))
              validation (nn/validate handle)]
          (is (= 3 (:epoch state) (count (:history state))))
          (is (= [42.0 42.0 42.0] (mapv #(get-in % [:metrics :metric-0]) (:history state))))
          (is (= 42.0 (get-in validation [:metrics :metric-0])))
          (is (= state (nn/restore-best! handle state)))
          (is (= description (datafy/datafy handle)))
          (is (= handle (session/value current-session handle :trainer))))
        (finally (session/release! current-session handle)))
      (is (nil? (session/resource current-session handle :trainer)))
      (is (false? (.isOpen (.getManager ^Trainer (:native-trainer handle)))))
      (is (false? (.isOpen ^NDManager (:snapshot-manager handle))))
      ;; Release already closed the handle; repeated close must remain harmless.
      (.close ^java.lang.AutoCloseable handle)
      (.close ^java.lang.AutoCloseable handle)
      (is (= [(:native-trainer handle)] (vec closed-trainers)))
      (is (= description (datafy/datafy handle))))))

(deftest training-operations-synchronize-only-on-the-network
  (with-open [current-session (session/open)
              worker (session/worker current-session)]
    (let [network (nn/build current-session [:linear 1] {:input-shape [1] :name :lock-order})
          handle (nn/trainer worker :lock-order (regression-table)
                             {:x [:x] :y :y :epochs 1 :batch 41})
          exclusive session/exclusive]
      (try
        ;; A held network blocks every native training operation. The trainer
        ;; remains lockable, and no operation attempts its resource lock.
        (doseq [operation [nn/validate #(nn/epoch! % {}) #(nn/restore-best! % {})
                           #(nn/train! (:context %) (:network %) (regression-table)
                                       {:x [:x] :y :y :epochs 1 :batch 41})]]
          (let [network-attempted (CountDownLatch. (int 1))
                trainer-attempted (CountDownLatch. (int 1))]
            (with-redefs [session/exclusive
                          (fn [context value kind callback]
                            (when (and (= :network kind) (identical? network value))
                              (.countDown network-attempted))
                            (when (and (= :trainer kind) (identical? handle value))
                              (.countDown trainer-attempted))
                            (exclusive context value kind callback))]
              (let [[^CompletableFuture result ^CompletableFuture trainer-result]
                    (exclusive
                     current-session network :network
                     (fn [_]
                       (let [result (CompletableFuture/supplyAsync
                                     (reify Supplier
                                       (get [_] (operation handle))))]
                         (is (.await network-attempted (long 10) TimeUnit/SECONDS))
                         (is (thrown? TimeoutException (.get result (long 100) TimeUnit/MILLISECONDS)))
                         (is (= 1 (.getCount trainer-attempted)))
                         (let [trainer-result (CompletableFuture/supplyAsync
                                               (reify Supplier
                                                 (get [_]
                                                   (exclusive current-session handle :trainer
                                                              (constantly :available)))))]
                           (is (= :available (.get trainer-result (long 1) TimeUnit/SECONDS)))
                           [result trainer-result]))))]
                (is (map? (.get result (long 10) TimeUnit/SECONDS)))
                (.get trainer-result (long 10) TimeUnit/SECONDS)
                (is (= 1 (.getCount trainer-attempted)))))))
        (finally (session/release! current-session handle))))))

(deftest trainer-release-waits-for-network-users
  (with-open [current-session (session/open)]
    (let [network (nn/build current-session [:linear 1] {:input-shape [1]})
          handle (nn/trainer current-session network (regression-table)
                             {:x [:x] :y :y :epochs 1 :batch 41})
          release (session/exclusive
                   current-session network :network
                   (fn [_]
                     ;; While a network user (as epoch! does) holds the lock, a
                     ;; release from another thread must wait instead of closing.
                     (let [release (CompletableFuture/runAsync
                                    (reify Runnable
                                      (run [_] (session/release! current-session handle))))]
                       (is (thrown? TimeoutException (.get release (long 200) TimeUnit/MILLISECONDS)))
                       (is (map? (nn/epoch! handle {})))
                       release)))]
      (.get ^CompletableFuture release (long 10) TimeUnit/SECONDS)
      (is (nil? (session/resource current-session handle :trainer))))))

(deftest custom-stopping-functional-loss-and-forward-share-the-loop
  (with-open [current-session (session/open)]
    (let [source (regression-table)
          network (nn/build current-session [:linear 1] {:input-shape [1]})
          functional-loss (fn [labels predictions]
                            (nd/mean (nd/square (nd/sub (nd/singleton predictions)
                                                        (nd/singleton labels)))))
          result (nn/train! current-session network source
                            {:x [:x] :y :y :epochs 30 :batch 41 :validation source
                             :loss functional-loss :optim [:adam {:lr 0.02}]
                             :forward (fn [^Trainer native-trainer inputs training?]
                                        (if training? (.forward native-trainer ^NDList inputs)
                                            (.evaluate native-trainer ^NDList inputs)))
                             :stop? #(>= (:epoch %) 3)})]
      (is (= :early-stopping (:stopped result)))
      (is (= 3 (count (:history result))))
      (is (every? #(Double/isFinite (double (:training-loss %))) (:history result))))))

(deftest observers-abort-and-extension-options-fail-loudly
  (with-open [current-session (session/open)]
    (let [source (regression-table)
          network (nn/build current-session [:linear 1] {:input-shape [1]})
          options {:x [:x] :y :y :epochs 2 :batch 41}]
      (doseq [failed-event [:epoch :finished]]
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"observer-failure"
                              (nn/train! current-session network source
                                         (assoc options :observe
                                                (fn [event]
                                                  (when (= failed-event (:event event))
                                                    (throw (ex-info "observer-failure" {:sentinel true}))))))))
        (is (= [:network] (mapv :kind (:resources (session/describe current-session))))))
      (doseq [extension [:observe :forward :stop?]]
        (is (= :nn/invalid-training-options
               (error-key #(nn/trainer current-session network source (assoc options extension :invalid))))))
      (is (= :nn/invalid-metrics
             (error-key #(nn/train! current-session network source (assoc options :metrics [:unknown])))))
      (is (= :nn/missing-statistics
             (error-key #(nn/predict current-session network source {:standardize [:x]}))))
      (is (= :nn/unknown-option
             (error-key #(nn/train! current-session network source
                                    (assoc options :sequence {:length 3 :partition-by :x
                                                              :order-by :x :typo true})))))
      (is (= :nn/unknown-option
             (error-key #(nn/train! current-session network source
                                    (assoc options :early-stopping {:patience 1 :typo true}))))))))

(deftest battery-gru-workflow-is-observable-and-restores-best
  (with-open [current-session (session/open {:seed 2026})]
    (let [network (nn/build current-session
                            [:sequential [:gru {:state-size 4 :batch-first? true}]
                             [:index ":, -1"] [:linear 1]]
                            {:input-shape [3 1] :name :battery-gru})
          source (sequence-table 0.0)
          validation (sequence-table 0.2)
          events (ConcurrentLinkedQueue.)
          result (nn/train! current-session :battery-gru source
                            (assoc sequence-options :epochs 30 :optim [:adam {:lr 0.03}]
                                   :validation validation
                                   :early-stopping {:patience 5 :restore-best? true}
                                   :observe #(.add events %)))
          history (:history result)
          observed (vec events)
          evaluation (nn/evaluate current-session :battery-gru validation {:metrics [:mae]})]
      (is (< (:validation-loss (last history)) (:validation-loss (first history))))
      (is (= 7 (count (nn/predict current-session network validation {}))))
      (is (Double/isFinite (double (:mae evaluation))))
      (is (< (Math/abs (double (- (:loss evaluation) (get-in result [:best :loss])))) 1.0e-6))
      (is (= (mapv #(assoc % :event :epoch :network :battery-gru) history) (pop observed)))
      (is (= {:event :finished :stopped (:stopped result) :best (:best result)} (peek observed)))
      (is (= [:network] (mapv :kind (:resources (session/describe current-session))))))))

(def ^:private two-input-regression-spec
  [:sequential
   [:lambda (fn [inputs] (nd/ndlist [(nd/concat (vec inputs) 1)]))]
   [:linear 1]])

(deftest grouped-table-regression-trains-and-persists-input-order
  (with-temporary-directory!
    (fn [directory]
      (with-open [current-session (session/open {:seed 47})]
        (let [values (mapv #(/ % 20.0) (range -20 21))
              other (mapv #(- (/ (mod % 7) 3.0) 1.0) (range 41))
              third (mapv #(- (/ (mod % 5) 2.0) 1.0) (range 41))
              truth (mapv (fn [a b c] (+ 1.0 (* 2.0 a) (* -0.5 b) (* 3.0 c)))
                          values other third)
              source (table/from {:a values :b other :c third :y truth})
              groups [[:a :b] [:c]]
              network (nn/build current-session two-input-regression-spec
                                {:input-shapes [[1 2] [1 1]]})
              before (nn/predict current-session network source {:x groups})
              result (nn/train! current-session network source
                                {:x groups :y :y :standardize [:x :y] :validation source
                                 :epochs 100 :batch 41 :shuffle? false :optim [:adam {:lr 0.08}]})
              history (:history result)
              predictions (nn/predict current-session network source {})
              evaluation (nn/evaluate current-session network source {:metrics [:mae]})]
          (is (= 41 (count before) (count predictions)))
          (is (< (:training-loss (last history)) (:training-loss (first history))))
          (is (< (:validation-loss (last history)) (:validation-loss (first history))))
          (is (< (mse predictions truth) 0.05))
          (is (Double/isFinite (double (:mae evaluation))))
          (is (< (:mae evaluation) 0.15))
          (is (= groups (:x (network-inputs network))))
          (is (= predictions (nn/predict current-session network source {:x groups})))
          (nn/save! current-session network directory {:spec two-input-regression-spec})
          (let [restored (nn/load current-session directory {:spec two-input-regression-spec})]
            (is (= [[1 2] [1 1]] (:input-shapes restored)))
            (is (= (network-inputs network) (network-inputs restored)))
            (is (= predictions (nn/predict current-session restored source {})))
            (is (= evaluation (nn/evaluate current-session restored source {:metrics [:mae]})))))))))

(deftest grouped-random-order-table-extraction-keeps-data-and-labels-aligned
  (with-open [current-session (session/open)]
    (let [source (table/from
                  [:duckdb {:query (str "SELECT i AS a, 10 * i AS b, -3 * i AS c, 7 * i + 2 AS y "
                                        "FROM range(64) t(i) ORDER BY random()")}])
          network (nn/build current-session two-input-regression-spec
                            {:input-shapes [[1 2] [1 1]]})
          queries (ConcurrentLinkedQueue.)
          extract-matrix table/matrix
          handle (with-redefs [table/matrix (fn [context source columns]
                                              (.add queries columns)
                                              (extract-matrix context source columns))]
                   (nn/trainer current-session network source
                               {:x [[:b :a] [:c]] :y :y :batch 64 :shuffle? false}))
          ^Trainer native-trainer (:native-trainer handle)]
      (try
        (is (= [[:b :a :c :y]] (vec queries)))
        (let [rows (reduce (fn [rows ^Batch batch]
                             (with-open [batch batch]
                               (let [[left right] (.getData batch)
                                     labels (.singletonOrThrow (.getLabels batch))]
                                 (is (= [[64 2] [64 1]] (mapv nd/shape [left right])))
                                 (into rows
                                       (map (fn [[b a] c y] [a b c y])
                                            (partition 2 (nd/to-vector left))
                                            (nd/to-vector right) (nd/to-vector labels))))))
                           [] (.iterateDataset native-trainer (:dataset handle)))]
          (is (= 64 (count rows)))
          (is (= (mapv double (range 64)) (vec (sort (map first rows)))))
          (is (every? (fn [[a b c y]] (and (= (* 10.0 a) b)
                                           (= (* -3.0 a) c)
                                           (= (+ (* 7.0 a) 2.0) y))) rows)))
        (finally (session/release! current-session handle))))))

(deftest grouped-standardization-shares-per-column-training-statistics
  (with-open [current-session (session/open)]
    (let [network (nn/build current-session two-input-regression-spec
                            {:input-shapes [[1 2] [1 1]]})
          source (table/from {:a [0 1 2 3] :b [10 20 30 40] :y [50 70 90 110]})
          specification (examples/training-inputs current-session network source
                                                  {:x [[:a :b] [:a]] :y :y :standardize [:x :y]} :mse)
          statistics (:statistics specification)
          prepared (examples/table-examples current-session source specification true)
          [left right] (examples/feature-arrays (context/manager current-session) prepared specification)
          normalized (mapv #(/ (- % 1.5) (Math/sqrt 1.25)) (range 4))]
      (is (= [[1.5 25.0] [1.5]] (mapv #(mapv :mean %) (:x statistics))))
      (is (= (get-in statistics [:x 0 0]) (get-in statistics [:x 1 0])))
      (is (= 80.0 (get-in statistics [:y 0 :mean])))
      (is (= [[4 2] [4 1]] (mapv nd/shape [left right])))
      (is (< (mse (nd/to-vector right) normalized) 1.0e-12))
      (is (< (mse (nd/to-vector left) (vec (mapcat #(repeat 2 %) normalized))) 1.0e-12)))))

(deftest grouped-sequence-windows-share-layout-and-window-end-labels
  (with-open [current-session (session/open)]
    (let [spec [:sequential
                [:lambda (fn [inputs] (nd/ndlist [(nd/concat (vec inputs) 2)]))]
                [:index ":, -1"] [:linear 1]]
          network (nn/build current-session spec {:input-shapes [[1 3 2] [1 3 1]]})
          source (table/query (sequence-table 0.0) [:derive {:extra [:* :x 10]}])
          options {:x [[:x :cycle] [:extra]] :y :y
                   :sequence {:length 3 :partition-by :battery :order-by :cycle}}
          specification (examples/training-inputs current-session network source options :mse)
          prepared (examples/table-examples current-session source specification true)
          arrays (examples/feature-arrays (context/manager current-session) prepared specification)
          windows (vec (for [row-count [6 5] end (range 2 row-count)]
                         (range (- end 2) (inc end))))
          expected (mapv double (mapcat identity windows))
          history (:history (nn/train! current-session network source
                                       (assoc options :epochs 3 :batch 4 :shuffle? false)))]
      (is (= 7 (:rows prepared)))
      (is (= [[7 3 2] [7 3 1]] (mapv :shape (:features prepared)) (mapv nd/shape arrays)))
      (is (= (vec (mapcat #(repeat 2 %) expected)) (nd/to-vector (first arrays))))
      (is (= (mapv #(* 10.0 %) expected) (nd/to-vector (second arrays))))
      (is (= (mapv #(+ 100.0 (* 20.0 (last %))) windows)
             (mapv double (get-in prepared [:labels :data]))))
      (is (every? #(Double/isFinite (double (:training-loss %))) history))
      (is (= 7 (count (nn/predict current-session network source {}))))
      (is (Double/isFinite (double (:mae (nn/evaluate current-session network source {:metrics [:mae]}))))))))

(deftest grouped-feature-specifications-fail-loudly
  (with-open [current-session (session/open)]
    (let [network (nn/build current-session two-input-regression-spec
                            {:input-shapes [[1 2] [1 1]]})
          source (table/from {:a [0 1] :b [2 3] :c [4 5] :y [6 7]})
          invalid-features [nil :a [] [[:a :b] []] [[:a "b"] [:c]] [[:a :b] :c]
                            [[:a] [:c]] [[:a :b]] [[:a :b] [:c] [:a]] [:a :b]
                            {:left [:a :b] :right [:c]}]]
      (doseq [columns invalid-features]
        (testing (pr-str columns)
          (is (= :nn/invalid-features
                 (error-key #(nn/predict current-session network source {:x columns}))))
          (is (= :nn/invalid-features
                 (error-key #(nn/evaluate current-session network source {:x columns :y :y}))))
          (is (= :nn/invalid-features
                 (error-key #(nn/trainer current-session network source {:x columns :y :y})))))))))

(deftest named-graph-feature-groups-resolve-and-persist-in-network-input-order
  (with-temporary-directory!
    (fn [directory]
      (with-open [current-session (session/open)]
        (let [spec [:graph {:inputs [:left :right] :outputs [:output]}
                    [:joined [:left :right] [:concat {:axis 1}]]
                    [:output [:joined] [:linear 1]]]
              network (nn/build current-session spec {:input-shapes [[1 2] [1 1]]})
              source (table/from {:a [0 1 2 3] :b [10 20 30 40] :c [4 3 2 1] :y [5 6 7 8]})
              named-groups (array-map :right [:c] :left [:a :b])
              groups [[:a :b] [:c]]
              result (nn/train! current-session network source
                                {:x named-groups :y :y :standardize [:x]
                                 :epochs 3 :batch 4 :validation source})
              predictions (nn/predict current-session network source {})]
          (is (= 3 (count (:history result))))
          (is (= groups (:x (network-inputs network))))
          (is (= predictions
                 (nn/predict current-session network source {:x groups})
                 (nn/predict current-session network source {:x named-groups})))
          (is (Double/isFinite (double (:mae (nn/evaluate current-session network source
                                                          {:x named-groups :metrics [:mae]})))))
          (doseq [columns [{:left [:a :b]} {:left [:a :b] :right [:c] :unknown [:a]}
                           {:left [:a :b] :right []} {:left [:a "b"] :right [:c]}]]
            (is (= :nn/invalid-features
                   (error-key #(nn/predict current-session network source {:x columns})))))
          (nn/save! current-session network directory)
          (let [restored (nn/load current-session directory)]
            (is (= spec (:spec restored)))
            (is (= (network-inputs network) (network-inputs restored)))
            (is (= predictions (nn/predict current-session restored source {})))
            (is (= predictions (nn/predict current-session restored source {:x named-groups})))))))))
