(ns garun.examples.tables
  "A tiny sales story: expressions, CASE, aggregates, windows, three joins,
  keyed splits, sampling, named materialization and a temporary parquet round trip.
  DuckDB's range() generates the synthetic input; the trusted source SQL is the
  data generator, while all transformations use Garun or native HoneySQL Hiccup."
  (:refer-clojure :exclude [run!])
  (:require [garun.session :as session]
            [garun.table :as table])
  (:import [java.nio.file Files Path]
           [java.nio.file.attribute FileAttribute]))

(set! *warn-on-reflection* true)

(def orders
  [:duckdb {:query (str "SELECT i AS order_id, i % 6 AS customer_id, "
                        "(i + 1) * 2 AS amount, i % 3 + 1 AS quantity "
                        "FROM range(24) AS orders(i)")}])

(def customers
  {:customer_id [0 1 2 3]
   :customer ["Ada" "Bo" "Cy" "Dee"]
   :tier ["gold" "silver" "bronze" "gold"]})

(def enriched-clauses
  [[:derive {:gross [:* :amount :quantity]
             :discounted [:* :amount 0.9]}]
   ;; CASE belongs to HoneySQL, not Garun's smaller expression vocabulary.
   [:honeysql {:select [:* [[:case [:>= :gross 60] "large"
                             :else "small"] :size]]}]
   [:derive {:customer_order [:over [:row-number]
                              {:partition-by [:customer_id]
                               :order-by [:order_id]}]
             ;; SUM(BIGINT) is HUGEINT in DuckDB; parquet has no 128-bit integer.
             :running_total [:cast
                             [:over [:sum :gross]
                              {:partition-by [:customer_id]
                               :order-by [:order_id]
                               :rows [:unbounded :current]}]
                             :int64]}]
   [:order-by :order_id]])

(def totals-clauses
  [[:group-by [:customer_id] {:orders [:count]
                              :revenue [:sum :gross]
                              :average [:avg :gross]}]
   [:rename {:average :average_order}]
   [:order-by [:revenue :desc]]])

(def split-clause
  [:split {:fractions [0.75 0.25] :seed 7 :key :customer_id}])

(defn run!
  "Run the complete table story and return only portable Clojure data."
  []
  (let [directory (str (Files/createTempDirectory "garun-gallery-tables-"
                                                  (make-array FileAttribute 0)))
        parquet-path (str directory "/orders.parquet")]
    (try
      (with-open [current-session (session/open {:seed 7 :device :cpu})]
        ;; Freeze the SQL source once. Subsequent plans use the registered keyword.
        (table/materialize! current-session :orders orders)
        (let [enriched (table/query :orders enriched-clauses)
              totals (table/query enriched totals-clauses)
              left (table/query enriched
                                [[:join customers {:on [:customer_id] :how :left}]
                                 [:derive {:tier [:coalesce :tier "unregistered"]}]
                                 [:order-by :order_id]])
              semi (table/query enriched [:join customers {:on [:customer_id] :how :semi}])
              anti (table/query enriched [:join customers {:on [:customer_id] :how :anti}])
              [training validation] (table/query enriched split-clause)
              sample (table/query enriched
                                  [[:sample {:n 5 :seed 7 :key :order_id}]
                                   [:select :order_id :gross]
                                   [:order-by :order_id]])
              large-orders (table/query enriched
                                        [[:where [:= :size "large"]]
                                         [:select :order_id :gross]
                                         [:limit 3]])
              rows (table/rows current-session enriched)]
          ;; Parquet is an ordinary source on read; the temp directory is cleaned below.
          (table/write-parquet! current-session enriched parquet-path)
          {:rows rows
           :totals (table/rows current-session totals)
           :left (table/rows current-session left)
           :semi-count (table/row-count current-session semi)
           :anti-count (table/row-count current-session anti)
           :split-rows (mapv #(table/rows current-session %) [training validation])
           :sample (table/rows current-session sample)
           :sample-again (table/rows current-session sample)
           :large-orders (table/rows current-session large-orders)
           :parquet-rows (table/rows current-session
                                     (table/query [:parquet {:path parquet-path}]
                                                  [:order-by :order_id]))
           :materialized-count (table/row-count current-session :orders)}))
      (finally
        (Files/deleteIfExists (Path/of parquet-path (make-array String 0)))
        (Files/deleteIfExists (Path/of directory (make-array String 0)))))))

(defn -main
  "Print and return the table gallery summary."
  [& _]
  (let [summary (run!)]
    (prn summary)
    summary))
