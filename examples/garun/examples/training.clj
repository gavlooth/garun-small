(ns garun.examples.training
  "Seven small CPU training stories: binary and multiclass logits, standardized
  regression, a multi-input graph, partitioned battery sequences, a hybrid battery
  graph and a custom epoch loop. A model save/load round trip preserves regression
  predictions. DuckDB range() generates all data without downloads.
  Trusted SQL only generates rows; Garun Hiccup selects and splits them.
  Tabular examples use seeded, sample-keyed held-out validation; the battery
  examples hold out whole batteries."
  (:refer-clojure :exclude [run!])
  (:require [clojure.java.io :as io]
            [garun.nn :as nn]
            [garun.session :as session]
            [garun.table :as table])
  (:import [java.nio.file Files]
           [java.nio.file.attribute FileAttribute]))

(set! *warn-on-reflection* true)

(def binary-data
  [:duckdb {:query (str "SELECT i AS sample_id, (i - 15.5) / 8.0 AS x, "
                        "CASE WHEN i >= 16 THEN 1 ELSE 0 END AS label "
                        "FROM range(32) AS samples(i)")}])

(def multiclass-data
  [:duckdb {:query (str "SELECT i AS sample_id, CASE i % 3 WHEN 0 THEN -2.0 "
                        "WHEN 1 THEN 2.0 ELSE 0.0 END AS horizontal, "
                        "CASE i % 3 WHEN 2 THEN 2.0 ELSE -2.0 END AS vertical, "
                        "i % 3 AS label FROM range(30) AS samples(i)")}])

(def regression-data
  [:duckdb {:query (str "SELECT i AS sample_id, (i - 15.5) / 16.0 AS x, "
                        "100.0 + 20.0 * (i - 15.5) / 16.0 AS target "
                        "FROM range(32) AS samples(i)")}])

(def multi-input-data
  [:duckdb {:query (str "SELECT i AS sample_id, (i - 15.5) / 16.0 AS horizontal, "
                        "(i % 4 - 1.5) / 2.0 AS vertical, "
                        "20.0 + (i % 8) / 4.0 AS temperature, "
                        "15.0 + 2.0 * (i - 15.5) / 16.0 - (i % 4 - 1.5) / 2.0 "
                        "+ (i % 8) / 8.0 AS target FROM range(32) AS samples(i)")}])

(def multi-input-network
  [:graph {:inputs [:position :measurements] :outputs [:prediction]}
   [:joined [:position :measurements] [:concat {:axis 1}]]
   [:prediction [:joined] [:linear 1]]])

(def binary-partitions
  (table/query binary-data [:split {:fractions [0.8 0.2] :seed 7 :key :sample_id}]))

(def multiclass-partitions
  (table/query multiclass-data [:split {:fractions [0.8 0.2] :seed 7 :key :sample_id}]))

(def regression-partitions
  (table/query regression-data [:split {:fractions [0.8 0.2] :seed 7 :key :sample_id}]))

(def multi-input-partitions
  (table/query multi-input-data [:split {:fractions [0.8 0.2] :seed 7 :key :sample_id}]))

(def battery-data
  [:duckdb {:query (str "SELECT i // 8 AS battery_id, i % 8 AS cycle, "
                        "1.0 - (i % 8) / 16.0 AS voltage, "
                        "20.0 + (i % 8) / 10.0 AS temperature, "
                        "8.0 - (i % 8) AS remaining_life "
                        "FROM range(64) AS readings(i)")}])

(def battery-network
  [:sequential
   [:gru {:state-size 4 :batch-first? true}]
   [:index ":, -1"]
   [:linear 1]])

(def binary-options
  {:x [:x] :y :label :loss :bce
   :optim [:adam {:lr 0.15}] :epochs 8 :batch 32 :shuffle? false
   :metrics [:accuracy :f1] :observe tap>})

(def battery-options
  {:x [:voltage :temperature] :y :remaining_life
   :sequence {:length 3 :partition-by :battery_id :order-by :cycle}
   :standardize [:x :y] :loss :mse
   :optim [:adam {:lr 0.05}] :epochs 8 :batch 48 :shuffle? false
   :early-stopping {:patience 2 :min-delta 0.0001 :restore-best? true}
   :metrics [:mae]})

(def battery-hybrid-data
  [:duckdb {:query (str "SELECT i // 12 AS battery_id, i % 12 AS cycle, "
                        "4.2 - (i % 12) * 0.03 - (i // 12 % 4) * 0.01 AS voltage, "
                        "20.0 + (i % 12) * 0.4 + (i // 12 % 5) * 0.3 AS temperature, "
                        "2.0 + (i // 12 % 6) * 0.05 - (i % 12) * 0.02 AS capacity, "
                        "2.0 + (i // 12 % 6) * 0.05 AS nominal_capacity, "
                        "0.05 + (i // 12 % 4) * 0.005 AS resistance, "
                        "24.0 + (i // 12 % 6) * 0.8 "
                        "- (i % 12) * (1.0 + (i // 12 % 4) * 0.05) AS remaining_life "
                        "FROM range(384) AS readings(i)")}])

(def battery-hybrid-partitions
  (table/query battery-hybrid-data
               [:split {:fractions [0.7 0.15 0.15] :seed 7 :key :battery_id}]))

(def battery-hybrid-network
  [:graph {:inputs [:history :static] :outputs [:remaining]}
   [:encoded [:history]
    [:sequential [:gru {:state-size 16 :batch-first? true}] [:index ":, -1"]]]
   [:context [:static]
    [:sequential [:index ":, -1"] [:linear 8] [:relu]]]
   [:joined [:encoded :context] [:concat {:axis 1}]]
   [:hidden [:joined] [:sequential [:linear 24] [:relu]]]
   [:residual [:hidden :joined] [:add]]
   [:remaining [:residual] [:linear 1]]])

(def battery-hybrid-options
  {:x {:history [:voltage :temperature :capacity]
       :static [:nominal_capacity :resistance]}
   :y :remaining_life
   :sequence {:length 4 :partition-by :battery_id :order-by :cycle}
   :standardize [:x :y] :loss :mse
   :optim [:adam {:lr 0.03}] :epochs 8 :batch 256 :shuffle? false
   :early-stopping {:patience 3 :min-delta 0.0001 :restore-best? true}
   :metrics [:mae]})

(defn- custom-training!
  "Drive immutable epoch state ourselves; epoch! does not apply :stop? for us."
  [current-session]
  (nn/build current-session [:linear 1] {:input-shape [1] :name :custom})
  (let [options {:x [:x] :y :target :standardize [:x :y]
                 :optim [:adam {:lr 0.1}] :batch 32 :shuffle? false :epochs 8
                 :stop? (fn [state] (>= (:epoch state) 4))}
        handle (nn/trainer current-session :custom regression-data options)]
    (try
      (let [state (reduce (fn [state _]
                            (let [next-state (nn/epoch! handle state)]
                              (if ((:stop? options) next-state)
                                (reduced (assoc next-state :stopped :custom-stop))
                                next-state)))
                          {} (range (:epochs options)))]
        (nn/restore-best! handle state)
        (select-keys state [:epoch :history :best :stopped]))
      (finally
        (session/release! current-session handle)))))

(defn run!
  "Train all seven examples and return losses, predictions, metrics and persistence evidence."
  []
  (let [[binary-training binary-validation] binary-partitions
        [multiclass-training multiclass-validation] multiclass-partitions
        [regression-training regression-validation] regression-partitions
        [multi-input-training multi-input-validation] multi-input-partitions
        [battery-hybrid-training battery-hybrid-validation battery-hybrid-test]
        battery-hybrid-partitions
        directory (str (Files/createTempDirectory "garun-gallery-model-"
                                                  (make-array FileAttribute 0)))]
    (try
      (with-open [current-session (session/open {:seed 7 :device :cpu})]
        ;; BCE takes logits: no sigmoid in the network. Metrics convert logits for us.
        (nn/build current-session [:linear 1] {:input-shape [1] :name :binary})
        (let [binary-result (nn/train! current-session :binary binary-training
                                       (assoc binary-options :validation binary-validation))
              binary-classes (nn/predict current-session :binary binary-validation {:output :class})
              binary-probabilities (nn/predict current-session :binary binary-validation
                                               {:output :probability})

              ;; CE uses sparse integer labels and three logits per sample.
              multiclass-network (nn/build current-session [:linear 3]
                                           {:input-shape [2] :name :multiclass})
              multiclass-result (nn/train! current-session multiclass-network multiclass-training
                                           {:x [:horizontal :vertical] :y :label :loss :ce
                                            :optim [:adam {:lr 0.1}] :epochs 8 :batch 30
                                            :shuffle? false :validation multiclass-validation
                                            :metrics [:accuracy]})
              multiclass-probabilities (nn/predict current-session :multiclass multiclass-validation
                                                   {:output :probability})

              ;; Both feature and target scaling are learned from training rows only.
              regression-network (nn/build current-session [:linear 1]
                                           {:input-shape [1] :name :regression})
              regression-result (nn/train! current-session regression-network regression-training
                                           {:x [:x] :y :target :standardize [:x :y]
                                            :loss :mse :optim [:adam {:lr 0.1}]
                                            :epochs 8 :batch 32 :shuffle? false
                                            :validation regression-validation :metrics [:mae]})
              regression-predictions (nn/predict current-session :regression regression-validation {})

              ;; Named column groups resolve in the graph's declared input order.
              graph-network (nn/build current-session multi-input-network
                                      {:input-shapes [[1 2] [1 1]] :name :multi-input})
              multi-input-result (nn/train! current-session graph-network multi-input-training
                                            {:x {:measurements [:temperature]
                                                 :position [:horizontal :vertical]}
                                             :y :target :standardize [:x :y] :loss :mse
                                             :optim [:adam {:lr 0.1}] :epochs 8 :batch 32
                                             :shuffle? false :validation multi-input-validation
                                             :metrics [:mae :mse]})
              multi-input-predictions (nn/predict current-session :multi-input
                                                  multi-input-validation {})

              ;; Keep whole batteries out of training; no window crosses a partition.
              training-batteries (table/query battery-data [:where [:< :battery_id 6]])
              validation-batteries (table/query battery-data [:where [:>= :battery_id 6]])
              sequence-network (nn/build current-session battery-network
                                         {:input-shape [3 2] :name :battery})
              sequence-result (nn/train! current-session sequence-network training-batteries
                                         (assoc battery-options :validation validation-batteries))
              sequence-predictions (nn/predict current-session :battery validation-batteries {})

              ;; Both inputs share cycle windows; the static branch selects repeated attributes.
              hybrid-network (nn/build current-session battery-hybrid-network
                                       {:input-shapes [[1 4 3] [1 4 2]] :name :battery-hybrid})
              hybrid-result (nn/train! current-session hybrid-network battery-hybrid-training
                                       (assoc battery-hybrid-options
                                              :validation battery-hybrid-validation))
              hybrid-evaluation (nn/evaluate current-session :battery-hybrid battery-hybrid-test
                                             {:metrics [:mae]})
              hybrid-predictions (nn/predict current-session :battery-hybrid battery-hybrid-test {})

              custom-result (custom-training! current-session)]
          ;; Save both weights and standardization; loading needs no feature overrides.
          (nn/save! current-session :regression directory)
          (nn/load current-session directory {:name :restored})
          {:binary (assoc binary-result :classes binary-classes
                          :probabilities binary-probabilities)
           :multiclass (assoc multiclass-result :probabilities multiclass-probabilities)
           :regression (assoc regression-result :predictions regression-predictions
                              :evaluation (nn/evaluate current-session :regression regression-validation
                                                       {:metrics [:mae :mse]}))
           :multi-input (assoc multi-input-result :predictions multi-input-predictions
                               :evaluation (nn/evaluate current-session :multi-input
                                                        multi-input-validation {:metrics [:mae :mse]}))
           :sequence (assoc sequence-result :predictions sequence-predictions
                            :evaluation (nn/evaluate current-session :battery validation-batteries
                                                     {:metrics [:mae]}))
           :battery-hybrid (assoc hybrid-result :predictions hybrid-predictions
                                  :evaluation hybrid-evaluation)
           :custom custom-result
           :persistence {:before regression-predictions
                         :after (nn/predict current-session :restored regression-validation {})}}))
      (finally
        (doseq [file (reverse (file-seq (io/file directory)))]
          (io/delete-file file))))))

(defn -main
  "Print and return the complete training gallery summary."
  [& _]
  (let [summary (run!)]
    (prn summary)
    summary))
