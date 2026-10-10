(ns garun.examples.networks
  "Sixteen copyable network specs, built and predicted natively in one CPU session.
  Shapes use N=batch, C=channels, T=time and H=hidden features. Convolutions use
  [N,C,...]; recurrent and attention inputs use [N,T,H]. Token IDs are int64,
  but embedding weights remain float32. Trusted DuckDB range() SQL generates the
  synthetic tensor values. Function-valued specs are runnable data, not serializable
  EDN: save them only with an explicit reconstruction spec."
  (:refer-clojure :exclude [run!])
  (:require [garun.ndarray :as nd]
            [garun.nn :as nn]
            [garun.session :as session]
            [garun.table :as table]
            [garun.tensor :as tensor]))

(set! *warn-on-reflection* true)

;; Tabular [N,4] -> two logits. Dropout is disabled during prediction.
(def mlp
  [:sequential
   [:linear 8] [:batchnorm] [:relu] [:dropout {:p 0.2}]
   [:linear 2]])

;; A residual branch must preserve the input width for :add.
(def residual
  [:skip {:combine :add}
   [:linear 4] [:relu] [:linear 4]])

;; The branches see the same image; concatenate channels, not the batch axis.
(def inception
  [:sequential
   [:parallel {:combine :concat :axis 1}
    [:sequential [:conv2d {:filters 2 :kernel [1 1]}] [:relu]]
    [:sequential [:conv2d {:filters 3 :kernel [3 3] :padding [1 1]}] [:relu]]]
   [:global-avg-pool2d] [:linear 3]])

;; A conventional small image classifier: [N,1,8,8] -> [N,3].
(def cnn
  [:sequential
   [:conv2d {:filters 4 :kernel [3 3] :padding [1 1]}] [:relu]
   [:max-pool2d {:kernel [2 2] :stride [2 2]}]
   [:flatten] [:linear 3]])

;; Separate image and measurement tensors join after image feature extraction.
(def image-measurements-classifier
  [:graph {:inputs [:image :measurements] :outputs [:logits]}
   [:features [:image]
    [:sequential [:conv2d {:filters 8 :kernel [3 3]}] [:relu] [:global-avg-pool2d]]]
   [:joined [:features :measurements] [:concat {:axis 1}]]
   [:hidden [:joined] [:sequential [:linear 32] [:relu]]]
   [:logits [:hidden] [:linear 3]]])

;; The first hidden tensor skips two later nodes before the elementwise merge.
(def long-skip
  [:graph {:inputs [:input] :outputs [:logits]}
   [:first [:input] [:sequential [:linear 4] [:relu]]]
   [:second [:first] [:sequential [:linear 4] [:relu]]]
   [:third [:second] [:linear 4]]
   [:sum [:first :third] [:add]]
   [:logits [:sum] [:linear 2]]])

;; 1-D convolution expects channels before time: [N,2,8] -> [N,1].
(def sequence-convolution
  [:sequential
   [:conv1d {:filters 4 :kernel [3] :padding [1]}] [:relu]
   [:global-avg-pool1d] [:linear 1]])

;; Stride 2 encodes 8x8 to 4x4; output padding restores exactly 8x8.
(def autoencoder
  [:sequential
   [:conv2d {:filters 2 :kernel [3 3] :stride [2 2] :padding [1 1]}]
   [:relu]
   [:conv2d-transpose {:filters 1 :kernel [3 3] :stride [2 2]
                       :padding [1 1] :output-padding [1 1]}]
   [:sigmoid]])

;; Native indexing takes the last timestep: [N,5,4] -> [N,4] -> [N,1].
(def gru-regressor
  [:sequential
   [:gru {:state-size 4 :batch-first? true}]
   [:index ":, -1"] [:linear 1]])

(def lstm-regressor
  [:sequential
   [:lstm {:state-size 4 :batch-first? true}]
   [:index ":, -1"] [:linear 1]])

;; Bidirectionality doubles the recurrent output width to 8.
(def bidirectional-lstm
  [:sequential
   [:lstm {:state-size 4 :batch-first? true :bidirectional? true}]
   [:index ":, -1"] [:linear 2]])

;; Token IDs [N,4] -> embeddings [N,4,6] -> three class logits.
(def text-classifier
  [:sequential
   [:embedding {:dictionary-size 16 :embedding-size 6}]
   [:lstm {:state-size 4 :batch-first? true}]
   [:index ":, -1"] [:linear 3]])

;; Classify already-embedded tokens. Position zero plays the summary-token role.
(def transformer-classifier
  [:sequential
   [:transformer-encoder {:embedding-size 6 :heads 2 :hidden-size 12
                          :dropout 0.0 :activation :gelu}]
   [:index ":, 0"] [:linear 3]])

;; Standalone self-attention preserves [N,4,6]. No mask means all tokens attend.
(def attention
  [:attention {:embedding-size 6 :heads 2 :dropout 0.0}])

;; A differentiable Clojure lambda using reflection-free NDArray operations.
(def centered-square
  [:lambda {:singleton? true}
   (fn [array]
     (nd/square (nd/sub array (nd/mean array [1] true))))])

;; A custom :parallel combine receives child NDLists and the original NDList.
;; The optional shape function returns a vector of output dimension vectors.
(def averaged-branches
  [:parallel
   {:combine (fn [children _inputs]
               (let [[left right] (mapv nd/singleton children)]
                 (nd/ndlist [(nd/div (nd/add left right) 2)])))
    :shape-fn (fn [children _input-shapes] (first children))}
   [:linear 4] [:linear 4]])

(def input-data
  [:duckdb {:query "SELECT i % 16 AS value FROM range(128) AS samples(i)"}])

(def gallery
  [{:name :mlp :spec mlp :input-shape [4]}
   {:name :residual :spec residual :input-shape [4]}
   {:name :inception :spec inception :input-shape [1 8 8]}
   {:name :cnn :spec cnn :input-shape [1 8 8]}
   {:name :image-measurements-classifier :spec image-measurements-classifier
    :input-shapes [[1 1 8 8] [1 4]]}
   {:name :long-skip :spec long-skip :input-shape [4]}
   {:name :sequence-convolution :spec sequence-convolution :input-shape [2 8]}
   {:name :autoencoder :spec autoencoder :input-shape [1 8 8]}
   {:name :gru-regressor :spec gru-regressor :input-shape [5 2]}
   {:name :lstm-regressor :spec lstm-regressor :input-shape [5 2]}
   {:name :bidirectional-lstm :spec bidirectional-lstm :input-shape [5 2]}
   {:name :text-classifier :spec text-classifier :input-shape [4] :input-dtype :int64}
   {:name :transformer-classifier :spec transformer-classifier :input-shape [4 6]}
   {:name :attention :spec attention :input-shape [4 6]}
   {:name :centered-square :spec centered-square :input-shape [4]}
   {:name :averaged-branches :spec averaged-branches :input-shape [4]}])

(defn run!
  "Build every spec, predict a batch of two samples per input, and report shapes."
  []
  (with-open [current-session (session/open {:seed 7 :device :cpu})]
    (into (sorted-map)
          (map (fn [{:keys [name spec input-dtype] :as entry}]
                 (let [network (nn/build current-session spec
                                         (assoc (select-keys entry [:input-shape :input-shapes])
                                                :name name))
                       ;; These arrays and predictions are owned by the session manager.
                       inputs (mapv (fn [input-shape]
                                      (let [dimensions (assoc input-shape 0 2)
                                            matrix (table/matrix
                                                    current-session
                                                    (table/query input-data
                                                                 [:limit (reduce * dimensions)])
                                                    [:value])]
                                        (tensor/array current-session (:data matrix)
                                                      {:shape dimensions
                                                       :dtype (or input-dtype :float32)})))
                                    (:input-shapes network))
                       outputs (nn/predict current-session network inputs {})]
                   [name {:input-shapes (mapv nd/shape inputs)
                          :output-shapes (mapv nd/shape outputs)
                          :finite? (every? #(Double/isFinite (double %))
                                           (mapcat nd/to-vector outputs))}])))
          gallery)))

(defn -main
  "Print and return all native prediction shapes."
  [& _]
  (let [summary (run!)]
    (prn summary)
    summary))
