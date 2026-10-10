(ns garun.examples-test
  "Run the README's gallery against real DuckDB and native CPU predictions."
  (:require [clojure.set :as set]
            [clojure.test :refer [deftest is testing]]
            [garun.examples.networks :as networks]
            [garun.examples.tables :as tables]
            [garun.examples.training :as training]
            [garun.session :as session]
            [garun.table :as table]))

(set! *warn-on-reflection* true)

(defn- finite? [value]
  (and (number? value) (Double/isFinite (double value))))

(def expected-output-shapes
  {:mlp [[2 2]]
   :residual [[2 4]]
   :inception [[2 3]]
   :cnn [[2 3]]
   :sequence-convolution [[2 1]]
   :autoencoder [[2 1 8 8]]
   :gru-regressor [[2 1]]
   :lstm-regressor [[2 1]]
   :bidirectional-lstm [[2 2]]
   :text-classifier [[2 3]]
   :transformer-classifier [[2 3]]
   :attention [[2 4 6]]
   :centered-square [[2 4]]
   :averaged-branches [[2 4]]})

(deftest runnable-gallery
  (let [started (System/nanoTime)
        table-summary (tables/run!)
        network-summary (networks/run!)
        training-summary (training/run!)
        training-partitions
        (with-open [current-session (session/open)]
          (into {}
                (map (fn [[example source partitions]]
                       (let [source-rows (table/rows current-session source)
                             reordered-partitions
                             (table/query (vec (reverse source-rows))
                                          [:split {:fractions [0.8 0.2] :seed 7 :key :sample_id}])]
                         [example {:source source-rows
                                   :rows (mapv #(table/rows current-session %) partitions)
                                   :reordered-rows (mapv #(table/rows current-session %)
                                                         reordered-partitions)}])))
                [[:binary training/binary-data training/binary-partitions]
                 [:multiclass training/multiclass-data training/multiclass-partitions]
                 [:regression training/regression-data training/regression-partitions]]))
        elapsed-milliseconds (/ (- (System/nanoTime) started) 1.0e6)]
    (println (format "Gallery examples runtime: %.1f ms" elapsed-milliseconds))

    (testing "Table pipelines preserve data, partition keys and persisted values"
      (let [{:keys [rows totals left semi-count anti-count split-rows sample sample-again
                    large-orders parquet-rows materialized-count]} table-summary
            split-keys (mapv #(set (map :customer_id %)) split-rows)]
        (is (= 24 materialized-count (count rows) (count left)))
        (is (= rows parquet-rows))
        (is (= (set (range 24)) (set (map :order_id (apply concat split-rows)))))
        (is (= 24 (count (apply concat split-rows))))
        (is (empty? (apply set/intersection split-keys)))
        (is (= (set (range 6)) (apply set/union split-keys)))
        (is (= 16 semi-count))
        (is (= 8 anti-count))
        (is (= 8 (count (filter #(= "unregistered" (:tier %)) left))))
        (is (= 5 (count sample)))
        (is (= sample sample-again))
        (is (= 3 (count large-orders)))
        (is (every? #(>= (:gross %) 60) large-orders))
        (is (= 1232 (reduce + (map :revenue totals))))
        (is (= [360 288 224 176 104 80] (mapv :revenue totals)))
        (is (every? #(= 4 (:orders %)) totals))
        (is (= (mapv #(inc (quot % 6)) (range 24)) (mapv :customer_order rows)))
        (is (= [80 176 288 104 224 360] (mapv :running_total (take-last 6 rows))))
        (is (every? #(= (if (>= (:gross %) 60) "large" "small") (:size %)) rows))))

    (testing "Every network performs an actual native prediction at its declared shape"
      (is (= expected-output-shapes
             (into {} (map (fn [[name summary]] [name (:output-shapes summary)]))
                   network-summary)))
      (is (every? :finite? (vals network-summary)))
      (is (every? #(= 2 (first (:input-shape %))) (vals network-summary))))

    (testing "Tabular training and validation are disjoint, complete and keyed deterministically"
      (doseq [[example {:keys [source rows reordered-rows]}] training-partitions]
        (let [partition-keys (mapv #(set (map :sample_id %)) rows)
              reordered-keys (mapv #(set (map :sample_id %)) reordered-rows)]
          (is (every? seq rows) (name example))
          (is (= (count source) (reduce + (map count rows))) (name example))
          (is (= (set (map :sample_id source)) (apply set/union partition-keys))
              (name example))
          (is (empty? (apply set/intersection partition-keys)) (name example))
          (is (= partition-keys reordered-keys) (name example)))))

    (testing "Short training runs produce finite, decreasing losses"
      (doseq [example [:binary :multiclass :regression :sequence :custom]]
        (let [history (get-in training-summary [example :history])]
          (is (<= 2 (count history) 8) (name example))
          (is (every? finite? (map :training-loss history)) (name example))
          (is (every? finite? (keep :validation-loss history)) (name example))
          (is (every? finite? (mapcat #(vals (:metrics %)) history)) (name example))
          (is (< (:training-loss (last history)) (:training-loss (first history)))
              (name example))))
      (is (= #{:accuracy :f1}
             (set (keys (get-in training-summary [:binary :history 0 :metrics])))))
      (is (= (count (second (get-in training-partitions [:binary :rows])))
             (count (get-in training-summary [:binary :classes]))))
      (is (every? #{0 1} (get-in training-summary [:binary :classes])))
      (is (every? #(<= 0.0 % 1.0) (get-in training-summary [:binary :probabilities])))
      (is (= (count (second (get-in training-partitions [:multiclass :rows])))
             (count (get-in training-summary [:multiclass :probabilities]))))
      (is (every? #(and (= 3 (count %))
                        (every? finite? %)
                        (< (abs (- 1.0 (reduce + %))) 1.0e-6))
                  (get-in training-summary [:multiclass :probabilities])))
      (is (= (count (second (get-in training-partitions [:regression :rows])))
             (count (get-in training-summary [:regression :predictions]))))
      (is (every? finite? (get-in training-summary [:regression :predictions])))
      (is (every? finite? (vals (get-in training-summary [:regression :evaluation]))))
      (is (= 12 (count (get-in training-summary [:sequence :predictions]))))
      (is (every? finite? (get-in training-summary [:sequence :predictions])))
      (is (every? finite? (vals (get-in training-summary [:sequence :evaluation]))))
      (is (= :validation-loss (get-in training-summary [:sequence :best :monitor])))
      (is (= 4 (get-in training-summary [:custom :epoch])))
      (is (= :custom-stop (get-in training-summary [:custom :stopped])))
      (is (= (get-in training-summary [:persistence :before])
             (get-in training-summary [:persistence :after]))))))
