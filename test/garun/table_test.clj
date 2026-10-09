(ns garun.table-test
  (:require [clojure.test :refer [deftest is testing]]
            [garun.table :as table])
  (:import [java.nio.file Files Path]
           [java.nio.file.attribute FileAttribute]))

(set! *warn-on-reflection* true)

(defn- error-code [f]
  (try
    (f)
    nil
    (catch clojure.lang.ExceptionInfo error
      (:error (ex-data error)))))

(defn- temp-path! [suffix]
  (str (Files/createTempFile "garun-table-" suffix (make-array FileAttribute 0))))

(defn- delete-file! [path]
  (Files/deleteIfExists (Path/of path (make-array String 0))))

(defn- round-trip! [db source format writer]
  (let [path (temp-path! (str "'?." (name format)))]
    (try
      (writer db source path)
      (let [restored (table/from [format {:path path}])]
        {:rows (table/rows db restored) :schema (table/schema db restored)})
      (finally
        (delete-file! path)))))

(deftest nested-bound-values-survive-execution-and-copy
  (with-open [db (table/open)]
    (let [source (table/from {:x [1 2 3] :s ["a" "b" "b"]})
          filtered (table/query source [[:where [:> :x 1]]
                                        [:derive {:z [:+ :x 10]}]
                                        [:where [:= :s "b"]]])
          expected-rows [{:s "b" :x 2 :z 12} {:s "b" :x 3 :z 13}]]
      (testing "Derived values and filters bind correctly across nested clauses"
        (is (= expected-rows (table/rows db filtered))))
      (testing "COPY materializes the same bound values"
        (doseq [[format writer] [[:csv table/write-csv!] [:parquet table/write-parquet!]]]
          (is (= expected-rows (:rows (round-trip! db filtered format writer))) (name format)))))))

(deftest copy-preserves-scalar-types-and-escapes-text
  (with-open [db (table/open)]
    (let [source (table/from {:id [1 2]
                              :large [9007199254740993 Long/MAX_VALUE]
                              :number [3.125 -4.5]
                              :flag [true false]
                              :nullable [nil 7]
                              :text ["O'Brien? -- not SQL" "a \"quoted\" value"]
                              :a?b ["first?" "second?"]})
          expected-rows [{:id 1 :large 9007199254740993 :number 3.125 :flag true
                          :nullable nil :text "O'Brien? -- not SQL" :a?b "first?"}
                         {:id 2 :large Long/MAX_VALUE :number -4.5 :flag false
                          :nullable 7 :text "a \"quoted\" value" :a?b "second?"}]]
      (doseq [[format writer] [[:csv table/write-csv!] [:parquet table/write-parquet!]]]
        (testing (str (name format) " preserves booleans, NULL, BIGINT, DOUBLE and escaped strings")
          (let [restored (round-trip! db source format writer)]
            (is (= expected-rows (:rows restored)))
            (when (= :parquet format)
              (is (= (table/schema db source) (:schema restored)))))))
      (testing "FLOAT stays FLOAT in parquet"
        (let [floats (table/from {:x [(float 1.25) (float -2.5)]})
              restored (round-trip! db floats :parquet table/write-parquet!)]
          (is (= (table/rows db floats) (:rows restored)))
          (is (= [{:name "x" :type :float32}] (:schema restored)))))
      (testing "Trusted SQL question marks are not generated parameter placeholders"
        (let [trusted (table/query
                       (table/from [:duckdb {:query "SELECT 'what?''s' AS text, 2 AS x, 4 AS \"a?b\" -- ?\n"}])
                       [:derive {:z [:+ :x 10]}])]
          (is (= [{:text "what?'s" :x 2 :a?b 4 :z 12}]
                 (:rows (round-trip! db trusted :parquet table/write-parquet!)))))))))

(deftest copy-drops-temporary-tables-after-success-and-failure
  (with-open [db (table/open)]
    (let [source (table/from {:x [1]})
          copy-tables (table/from
                       [:duckdb {:query (str "SELECT table_name FROM duckdb_tables() "
                                             "WHERE temporary AND starts_with(table_name, '__garun_copy_')")}])
          path (temp-path! ".parquet")]
      (try
        (doseq [[format writer] [[:csv table/write-csv!] [:parquet table/write-parquet!]]]
          (testing (str (name format) " drops its temporary table after success")
            (is (= [{:x 1}] (:rows (round-trip! db source format writer))))
            (is (zero? (table/row-count db copy-tables))))
          (testing (str (name format) " drops its temporary table when COPY fails")
            ;; A regular file cannot be a parent directory for the output file.
            (is (thrown? java.sql.SQLException (writer db source (str path "/output"))))
            (is (zero? (table/row-count db copy-tables)))))
        (finally
          (delete-file! path))))))

(deftest sources-and-execution
  (with-open [db (table/open)]
    (let [source (table/from {:x [1 2] :y [3.0 4.0]})]
      (is (table/table? source))
      (is (identical? source (table/from source)))
      (is (= [{:x 1 :y 3.0} {:x 2 :y 4.0}] (table/rows db source)))
      (is (= {:x [1 2] :y [3.0 4.0]} (table/columns db source)))
      (is (instance? clojure.lang.PersistentArrayMap (table/columns db source)))
      (is (= [{:name "x" :type :int64} {:name "y" :type :float64}] (table/schema db source)))
      (is (= 2 (table/row-count db source)))
      (is (= [{:x 1 :y 3.0}] (table/rows db source {:max-rows 1})))
      (is (= [] (table/rows db source {:max-rows 0})))
      (is (= [{:x 1 :y nil} {:x 2 :y "a"}]
             (table/rows db (table/from [{:x 1} {:x 2 :y "a"}]))))
      (is (= [] (table/rows db (table/from {:x []}))))
      (is (= {:x []} (table/columns db (table/from {:x []}))))
      (is (= [{:n 3}] (table/rows db (table/from [:duckdb {:query "SELECT 3 AS n"}])))))))

(deftest file-sources-and-round-trips
  (let [csv (temp-path! ".csv") json (temp-path! ".json") parquet (temp-path! ".parquet")
        source (table/from {:x [1 2] :y [3.0 4.0]})]
    (try
      (spit csv "x;y\n1;3.0\n2;4.0\n")
      (spit json "[{\"x\":1,\"y\":3.0},{\"x\":2,\"y\":4.0}]")
      (with-open [db (table/open)]
        (is (= (table/rows db source)
               (table/rows db (table/from [:csv {:path csv :header? true :delimiter ";"}]))))
        (is (= (table/rows db source) (table/rows db (table/from [:json {:path json}]))))
        (table/write-parquet! db source parquet)
        (is (= (table/rows db source) (table/rows db (table/from [:parquet {:path parquet}]))))
        (table/write-csv! db source csv)
        (is (= (table/rows db source) (table/rows db (table/from [:csv {:path csv}]))))
        (testing "Derived values add or replace columns even when file names are unknown"
          (is (= [{:x 8 :y 3.0 :z 1} {:x 8 :y 4.0 :z 1}]
                 (table/rows db (table/query (table/from [:parquet {:path parquet}])
                                             [:derive {:x 8 :z 1}])))))
        (testing "Every column of an unknown SQL schema can be replaced"
          (is (= [{:x 8}]
                 (table/rows db (table/query (table/from [:duckdb {:query "SELECT 1 AS x"}])
                                             [:derive {:x 8}]))))))
      (finally (doseq [p [csv json parquet]] (delete-file! p))))))

(deftest clauses
  (with-open [db (table/open)]
    (let [source (table/from [{:k "a" :x 1 :y 4} {:k "a" :x 2 :y 5} {:k "b" :x 3 :y 6}])]
      (is (= [{:x 1 :y 4} {:x 2 :y 5} {:x 3 :y 6}]
             (table/rows db (table/query source [:select :x :y]))))
      (is (= [{:y 4} {:y 5} {:y 6}] (table/rows db (table/query source [:select [:y]]))))
      (is (= [{:k "a" :a 1 :y 4} {:k "a" :a 2 :y 5} {:k "b" :a 3 :y 6}]
             (table/rows db (table/query source [:rename {:x :a}]))))
      (is (= [{:k "a" :x 1 :y 4 :z 10} {:k "a" :x 2 :y 5 :z 10} {:k "b" :x 3 :y 6 :z 10}]
             (table/rows db (table/query source [:derive {:z 10}]))))
      (is (= [{:k "a" :x 5 :y 4} {:k "a" :x 7 :y 5} {:k "b" :x 9 :y 6}]
             (table/rows db (table/query source [:derive {:x [:+ :x :y]}]))))
      (is (= [{:k "a" :m 1.5 :n 2} {:k "b" :m 3.0 :n 1}]
             (table/rows db (table/query source [[:group-by [:k] {:n [:count] :m [:avg :x]}]
                                                 [:order-by :k]]))))
      (is (= [{:x 3} {:x 2}]
             (table/rows db (table/query source [[:where [:> :x 1]] [:order-by [:x :desc :first]]
                                                 [:limit 2] [:select :x]]))))
      (is (= 2 (table/row-count db (table/query source [:sample {:n 2 :seed 11}]))))
      (is (= (table/rows db (table/query source [:sample {:n 2 :seed 11 :key :x}]))
             (table/rows db (table/query source [:sample {:n 2 :seed 11 :key :x}]))))
      (is (= [1 2 1]
             (mapv :rank (table/rows db (table/query source
                                                     [[:derive {:rank [:over [:row-number]
                                                                       {:partition-by [:k] :order-by [[:x :asc]]}]}]
                                                      [:order-by :x]])))))
      (is (= [1 3 3]
             (mapv :running (table/rows db (table/query source
                                                        [[:derive {:running [:over [:sum :x]
                                                                             {:partition-by [:k] :order-by [:x]
                                                                              :rows [:unbounded :current]}]}]
                                                         [:order-by :x]]))))))))

(deftest expressions-and-null-semantics
  (with-open [db (table/open)]
    (let [source (table/from {:x [1 2 nil] :text ["Alpha" "beta" nil]})
          xs (fn [expr] (mapv :x (table/rows db (table/query source [:where expr]))))]
      (is (= [1] (xs [:and [:= [:+ :x [:* 2 3]] 7] [:not [:is-null :x]]])))
      (is (= [1 nil] (xs [:or [:= :x 1] [:is-null :x]])))
      (is (= [] (xs [:= :x nil])))
      (is (= [1 2] (xs [:not-null :x])))
      (is (= [1] (xs [:in :x [1 5]])))
      (is (= [] (xs [:in :x []])))
      (is (= [1 2] (xs [:between :x 1 2])))
      (is (= [1] (xs [:ilike :text "a%"])))
      (is (= [2] (xs [:like :text "b%"])))
      (is (= ["1" "2" nil] (mapv :s (table/rows db (table/query source [:derive {:s [:cast :x :string]}])))))
      (is (= ["alpha" "beta" "none"]
             (mapv :s (table/rows db (table/query source [:derive {:s [:coalesce [:lower :text] "none"]}])))))
      (is (= [{:distinct 2 :n 2 :total 3}]
             (table/rows db (table/query source [:group-by [] {:n [:count :x] :total [:sum :x]
                                                               :distinct [:count-distinct :x]}]))))
      (is (= [true true true]
             (mapv :b (table/rows db (table/query source [:derive {:b [:lit true]}])))))
      (is (= [nil 2 1] (mapv :x (table/rows db (table/query source [:order-by [[:x :desc :first]]])))))
      (doseq [[expr expected] [[[:!= :x 1] [2]] [[:not= :x 1] [2]] [[:>= :x 2] [2]]
                               [[:< :x 2] [1]] [[:<= :x 1] [1]] [[:= [:- :x 1] 0] [1]]
                               [[:= [:/ :x 2] 1] [2]] [[:= [:% :x 2] 1] [1]]
                               [[:= [:col "x"] 1] [1]]]]
        (is (= expected (xs expr)))))))

(deftest expression-scalars-preserve-types-and-null-comparisons
  (with-open [db (table/open)]
    (let [source (table/from {:x [1 2 nil]})
          result (table/query source
                              [[:derive {:large [:+ :x 9007199254740993]
                                         :single [:* [:cast :x :float32] (float 1.25)]
                                         :flag [:and true [:> :x 1]]
                                         :equal [:= :x nil]
                                         :equal-left [:= nil :x]
                                         :different [:!= :x nil]
                                         :missing [:is-null :x]}]
                               [:select :large :single :flag :equal :equal-left :different :missing]])]
      (is (= [{:large 9007199254740994 :single (float 1.25) :flag false
               :equal nil :equal-left nil :different nil :missing false}
              {:large 9007199254740995 :single (float 2.5) :flag true
               :equal nil :equal-left nil :different nil :missing false}
              {:large nil :single nil :flag nil
               :equal nil :equal-left nil :different nil :missing true}]
             (table/rows db result)))
      (is (= [{:name "large" :type :int64}
              {:name "single" :type :float32}
              {:name "flag" :type :bool}
              {:name "equal" :type :bool}
              {:name "equal-left" :type :bool}
              {:name "different" :type :bool}
              {:name "missing" :type :bool}]
             (table/schema db result))))))

(deftest literal-identifier-names-survive-clauses-and-joins
  (with-open [db (table/open)]
    (doseq [column ["part.value" "dash-name" "quote\"name" "question?mark"
                    "literal/slash" "select" "*" "x); DROP TABLE victims; --"]]
      (let [renamed (str "renamed." column)
            aggregate (str column "?total")
            source (table/from {column [2 1 2] "discard" [8 9 10]})]
        (testing (str "Literal name through select, replacement, rename, group and order: " column)
          (is (= [{(keyword renamed) 12 (keyword aggregate) 24}
                  {(keyword renamed) 11 (keyword aggregate) 11}]
                 (table/rows db
                             (table/query source
                                          [[:select column]
                                           [:derive {column [:+ [:col column] 10]}]
                                           [:rename {column renamed}]
                                           [:group-by [renamed] {aggregate [:sum [:col renamed]]}]
                                           [:order-by [renamed :desc]]])))))
        (testing (str "Join keys and collision aliases remain literal names: " column)
          (let [right (table/from {column [2 3] "payload" [20 30]})
                joined (table/query source
                                    [[:join right {:on [column]}]
                                     [:select column (str column "_right") "payload"]
                                     [:group-by [column (str column "_right")]
                                      {"matched" [:sum :payload]}]])]
            (is (= [{(keyword column) 2 (keyword (str column "_right")) 2 :matched 40}]
                   (table/rows db joined)))))))))

(deftest chained-projections-bind-both-join-inputs
  (with-open [db (table/open)]
    (let [left (table/query
                (table/from {:id [1 2 3] :text ["left's ?" "keep?" "also keep"] :x [2 4 8]})
                [[:where [:> :id 1]]
                 [:derive {:shift [:+ :x 10]}]
                 [:select :id :shift :text]
                 [:derive {:result [:* :shift 3]}]
                 [:select :id :result :text]
                 [:where [:< :result 60]]])
          right (table/query
                 (table/from {:rid [2 3 4] :value [5 6 7]})
                 [[:derive {:adjusted [:+ :value 100]}]
                  [:where [:< :rid 4]]
                  [:select :rid :adjusted]
                  [:derive {:score [:- :adjusted 7]}]
                  [:select :rid :score]])
          joined (table/query left
                              [[:join right {:on {:id :rid}}]
                               [:derive {:combined [:+ [:+ :result :score] 1000]}]
                               [:where [:> :combined 1100]]
                               [:select :id :text :combined]
                               [:order-by :id]])]
      (is (= [{:id 2 :text "keep?" :combined 1140}
              {:id 3 :text "also keep" :combined 1153}]
             (table/rows db joined))))))

(deftest native-honeysql-transformations-compose-with-garun
  (with-open [db (table/open)]
    (let [source (table/from {:category ["a" "a" "b"]
                              :amount [1 2 4]
                              :nullable [nil "x" nil]})]
      (testing "Native CASE expressions bind text and NULL equality uses HoneySQL semantics"
        (let [result (table/query source
                                  [[:where [:> :amount 1]]
                                   [:honeysql {:select [:amount [[:case [:>= :amount 4] "O'Brien?"
                                                                  :else "small"] :band]]
                                               :where [:= :nullable nil]}]
                                   [:derive {:doubled [:* :amount 2]}]
                                   [:select :amount :band :doubled]])]
          (is (= [{:amount 4 :band "O'Brien?" :doubled 8}] (table/rows db result)))
          (is (= [{:amount 4 :band "O'Brien?" :doubled 8}]
                 (:rows (round-trip! db result :parquet table/write-parquet!))))))
      (testing "WHERE-only transformations retain the input projection"
        (is (= [{:category "a" :amount 2 :nullable "x"}]
               (table/rows db (table/query source [:honeysql {:where [:= :amount 2]}])))))
      (testing "Native DISTINCT replaces the default SELECT rather than duplicating it"
        (is (= [{:category "a"} {:category "b"}]
               (table/rows db
                           (table/query source [:honeysql {:select-distinct [:category]
                                                           :order-by [[:category :asc]]}])))))
      (testing "Native HAVING and aggregates remain composable with Garun clauses"
        (is (= [{:category "b" :total 4}]
               (table/rows db
                           (table/query source
                                        [[:honeysql {:select [:category [[:sum :amount] :total]]
                                                     :group-by [:category]
                                                     :having [:> [:sum :amount] 3]}]
                                         [:where [:> :total 3]]]))))))))

(deftest native-honeysql-transformations-reject-source-replacement-and-statements
  (let [source (table/from {:amount [1 2]})]
    (doseq [clause [[:honeysql] [:honeysql nil] [:honeysql {} {}]]]
      (is (= :table/invalid (error-code #(table/query source clause)))))
    (doseq [transformation [{:from [:other]}
                            {:delete-from :__input__}
                            {:create-table-as :other :select [:*]}
                            {:update :__input__ :set {:amount 0}}]]
      (is (= :table/unknown-option
             (error-code #(table/query source [:honeysql transformation])))))))

(deftest finite-window-frames-respect-bounds-and-partitions
  (with-open [db (table/open)]
    (let [source (table/from {:partition ["a" "b" "a" "b" "a" "a"]
                              :position [1 1 2 2 3 4]
                              :x [1 10 2 20 4 8]})]
      (doseq [[frame expected] [[[-1 1] [3 7 14 12 30 30]]
                                [[-2 :current] [1 3 7 14 10 30]]
                                [[:current 2] [7 14 12 8 30 20]]
                                [[0 0] [1 2 4 8 10 20]]]]
        (testing (str "ROWS frame " frame)
          (is (= expected
                 (mapv :total
                       (table/rows db
                                   (table/query source
                                                [[:derive {:total [:over [:sum :x]
                                                                   {:partition-by [:partition]
                                                                    :order-by [:position]
                                                                    :rows frame}]}]
                                                 [:order-by :partition :position]]))))))))))

(deftest join-types-and-collisions
  (with-open [db (table/open)]
    (let [left (table/from {:id [1 2] :v ["a" "b"]})
          right (table/from {:id [2 3] :v ["B" "C"] :z [20 30]})
          joined (fn [how] (table/query left [:join right {:on [:id] :how how}]))]
      (is (= [{:id 1 :v "a" :id_right nil :v_right nil :z nil}
              {:id 2 :v "b" :id_right 2 :v_right "B" :z 20}]
             (table/rows db (table/query (joined :left) [:order-by :id]))))
      (is (= 1 (table/row-count db (joined :inner))))
      (is (= 2 (table/row-count db (joined :right))))
      (is (= 3 (table/row-count db (joined :full))))
      (is (= [{:id 2 :v "b"}] (table/rows db (joined :semi))))
      (is (= [{:id 1 :v "a"}] (table/rows db (joined :anti))))
      (is (= 4 (table/row-count db (table/query left [:join right {:how :cross}]))))
      (is (= [{:id 2 :v "b" :id_other 2 :v_other "B" :z 20}]
             (table/rows db (table/query left [:join right {:on {:id :id} :suffix "_other"}]))))
      (is (= [{:id 2 :v "b" :k 2 :z 20}]
             (table/rows db (table/query left [:join (table/from {:k [2] :z [20]}) {:on {:id :k}}]))))
      (is (= [{:id 2 :v "b" :id_right 2}]
             (table/rows db (table/query left
                                         [:join (table/from [:duckdb {:query "SELECT 2 AS id"}])
                                          {:on [:id]}]))))
      (is (= [{:id 2 :v "b" :id_right 2 :z 20}]
             (table/rows db (table/query left
                                         [:join (table/from [:duckdb {:query "SELECT 2 AS id, 20 AS z"}])
                                          {:on [:id]}])))))))

(deftest deterministic-covering-splits
  (with-open [db (table/open)]
    (let [source (table/from {:id (vec (range 200)) :x (vec (range 200))})
          opts {:fractions [0.6 0.3 0.1] :seed 12 :key :id}
          parts (mapv #(mapv :id (table/rows db %)) (table/query source [:split opts]))
          again (mapv #(mapv :id (table/rows db %)) (table/query source [:split opts]))
          no-key (mapv #(mapv :id (table/rows db %))
                       (table/query source [:split {:fractions [0.5 0.5] :seed 3}]))]
      (is (= parts again))
      (is (every? seq parts))
      (is (= (vec (range 200)) (vec (sort (apply concat parts)))))
      (is (= 200 (count (distinct (apply concat parts)))))
      (is (= 200 (count (apply concat no-key))))
      (is (= no-key (mapv #(mapv :id (table/rows db %))
                          (table/query source [:split {:fractions [0.5 0.5] :seed 3}])))))))

(deftest split-helpers-do-not-shadow-user-columns
  (with-open [db (table/open)]
    (let [source (table/from [:duckdb {:query "SELECT i AS id, i AS __garun_row__, i AS __garun_bucket__, i AS __garun_split_0 FROM range(20) t(i)"}])
          parts (table/query source [:split {:fractions [0.5 0.5] :seed 9}])
          combined (mapcat #(table/rows db %) parts)]
      (is (= (vec (range 20)) (vec (sort (map :id combined)))))
      (is (every? #(= (:id %) (:__garun_row__ %) (:__garun_bucket__ %) (:__garun_split_0 %)) combined)))))

(deftest matrix-order-and-errors
  (with-open [db (table/open)]
    (let [source (table/from {:a [1 2] :b [3.5 4.5]})
          matrix (table/matrix db (table/query source [:order-by [:a :desc]]) [:b :a])]
      (is (= 2 (:rows matrix)))
      (is (= 2 (:cols matrix)))
      (is (= [4.5 2.0 3.5 1.0] (vec (:data matrix))))
      (is (= Float/TYPE (.getComponentType (class (:data matrix)))))
      (is (= :table/null-value (error-code #(table/matrix db (table/from {:x [nil 1]}) [:x]))))
      (is (= :table/non-numeric (error-code #(table/matrix db (table/from {:x ["a"]}) [:x])))))))

(deftest invalid-dsl-is-rejected
  (let [source (table/from {:x [1]})]
    (is (= :table/unknown-clause (error-code #(table/query source [:wat 1]))))
    (is (= :table/unknown-operator (error-code #(table/query source [:where [:wat :x]]))))
    (doseq [f [#(table/from [:csv {:path "x" :wat true}])
               #(table/query source [:join source {:on [:x] :wat true}])
               #(table/query source [:sample {:n 1 :wat true}])
               #(table/query source [:split {:fractions [1] :wat true}])
               #(table/query source [:derive {:x [:over [:count] {:wat true}]}])]]
      (is (= :table/unknown-option (error-code f))))
    (doseq [clauses [[[:split {:fractions [1]}] [:limit 1]]
                     [:split {:fractions [0.4 0.4]}] [:split {:fractions [1] :seed 1.0}]
                     [:sample {:n 0}] [:limit -1] [:select] [:rename []] [:derive []]
                     [:group-by :x {}] [:order-by [[:x :up]]] [:join source {:how :inner}]
                     [:join source {:on [:x] :how :cross}] [:join source {:on [:x] :how :wat}]
                     [:where [:+ :x]] [:where [:count :x :x]] [:where [:in :x 1]]
                     [:where [:cast :x :wat]] [:derive {:x [:avg :x :x]}]
                     [:derive {:x [:quantile :x 2]}] [:derive {:x [:lag]}]
                     [:derive {:x [:over [:count] {:rows [:unbounded :current]}]}]]]
      (is (= :table/invalid (error-code #(table/query source clauses))) (str clauses)))
    (is (= :table/invalid (error-code #(table/from {:x [1] :y [1 2]}))))
    (is (= :table/invalid (error-code #(table/from {:x [1] :X [2]}))))
    (with-open [db (table/open)]
      (is (= :table/unknown-option (error-code #(table/rows db source {:wat 1})))))))
