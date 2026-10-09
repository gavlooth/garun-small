(ns garun.table-test
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.string :as string]
            [garun.session :as session]
            [garun.table :as table])
  (:import [java.nio.file Files Path]
           [java.nio.file.attribute FileAttribute]
           [java.sql ResultSet]
           [java.util.concurrent CompletableFuture CountDownLatch TimeUnit TimeoutException]
           [java.util.function Supplier]))

(set! *warn-on-reflection* true)

(defn- error-code [operation]
  (try
    (operation)
    nil
    (catch clojure.lang.ExceptionInfo error
      (:error (ex-data error)))))

(defn- temp-path! [suffix]
  (str (Files/createTempFile "garun-table-" suffix (make-array FileAttribute 0))))

(defn- delete-file! [path]
  (Files/deleteIfExists (Path/of path (make-array String 0))))

(defn- round-trip! [current-session source format writer]
  (let [path (temp-path! (str "'?." (name format)))]
    (try
      (writer current-session source path)
      (let [restored (table/from [format {:path path}])]
        {:rows (table/rows current-session restored) :schema (table/schema current-session restored)})
      (finally
        (delete-file! path)))))

(deftest nested-bound-values-survive-execution-and-copy
  (with-open [current-session (session/open)]
    (let [source (table/from {:x [1 2 3] :s ["a" "b" "b"]})
          filtered (table/query source [[:where [:> :x 1]]
                                        [:derive {:z [:+ :x 10]}]
                                        [:where [:= :s "b"]]])
          expected-rows [{:s "b" :x 2 :z 12} {:s "b" :x 3 :z 13}]]
      (testing "Derived values and filters bind correctly across nested clauses"
        (is (= expected-rows (table/rows current-session filtered))))
      (testing "COPY materializes the same bound values"
        (doseq [[format writer] [[:csv table/write-csv!] [:parquet table/write-parquet!]]]
          (is (= expected-rows (:rows (round-trip! current-session filtered format writer))) (name format)))))))

(deftest copy-preserves-scalar-types-and-escapes-text
  (with-open [current-session (session/open)]
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
          (let [restored (round-trip! current-session source format writer)]
            (is (= expected-rows (:rows restored)))
            (when (= :parquet format)
              (is (= (table/schema current-session source) (:schema restored)))))))
      (testing "FLOAT stays FLOAT in parquet"
        (let [floats (table/from {:x [(float 1.25) (float -2.5)]})
              restored (round-trip! current-session floats :parquet table/write-parquet!)]
          (is (= (table/rows current-session floats) (:rows restored)))
          (is (= [{:name "x" :type :float32}] (:schema restored)))))
      (testing "Trusted SQL question marks are not generated parameter placeholders"
        (let [trusted (table/query
                       (table/from [:duckdb {:query "SELECT 'what?''s' AS text, 2 AS x, 4 AS \"a?b\" -- ?\n"}])
                       [:derive {:z [:+ :x 10]}])]
          (is (= [{:text "what?'s" :x 2 :a?b 4 :z 12}]
                 (:rows (round-trip! current-session trusted :parquet table/write-parquet!)))))))))

(deftest copy-drops-temporary-tables-after-success-and-failure
  (with-open [current-session (session/open)]
    (let [source (table/from {:x [1]})
          copy-tables (table/from
                       [:duckdb {:query (str "SELECT table_name FROM duckdb_tables() "
                                             "WHERE temporary AND starts_with(table_name, '__garun_copy_')")}])
          path (temp-path! ".parquet")]
      (try
        (doseq [[format writer] [[:csv table/write-csv!] [:parquet table/write-parquet!]]]
          (testing (str (name format) " drops its temporary table after success")
            (is (= [{:x 1}] (:rows (round-trip! current-session source format writer))))
            (is (zero? (table/row-count current-session copy-tables))))
          (testing (str (name format) " drops its temporary table when COPY fails")
            ;; A regular file cannot be a parent directory for the output file.
            (is (thrown? java.sql.SQLException (writer current-session source (str path "/output"))))
            (is (zero? (table/row-count current-session copy-tables)))))
        (finally
          (delete-file! path))))))

(deftest sources-and-execution
  (with-open [current-session (session/open)]
    (let [source (table/from {:x [1 2] :y [3.0 4.0]})]
      (is (table/table? source))
      (is (identical? source (table/from source)))
      (is (= [{:x 1 :y 3.0} {:x 2 :y 4.0}] (table/rows current-session source)))
      (is (= {:x [1 2] :y [3.0 4.0]} (table/columns current-session source)))
      (is (instance? clojure.lang.PersistentArrayMap (table/columns current-session source)))
      (is (= [{:name "x" :type :int64} {:name "y" :type :float64}] (table/schema current-session source)))
      (is (= 2 (table/row-count current-session source)))
      (is (= [{:x 1 :y 3.0}] (table/rows current-session source {:max-rows 1})))
      (is (= [] (table/rows current-session source {:max-rows 0})))
      (is (= [{:x 1 :y nil} {:x 2 :y "a"}]
             (table/rows current-session (table/from [{:x 1} {:x 2 :y "a"}]))))
      (is (= [] (table/rows current-session (table/from {:x []}))))
      (is (= {:x []} (table/columns current-session (table/from {:x []}))))
      (is (= [{:n 3}] (table/rows current-session (table/from [:duckdb {:query "SELECT 3 AS n"}])))))))

(deftest file-sources-and-round-trips
  (let [csv (temp-path! ".csv") json (temp-path! ".json") parquet (temp-path! ".parquet")
        source (table/from {:x [1 2] :y [3.0 4.0]})]
    (try
      (spit csv "x;y\n1;3.0\n2;4.0\n")
      (spit json "[{\"x\":1,\"y\":3.0},{\"x\":2,\"y\":4.0}]")
      (with-open [current-session (session/open)]
        (is (= (table/rows current-session source)
               (table/rows current-session (table/from [:csv {:path csv :header? true :delimiter ";"}]))))
        (is (= (table/rows current-session source) (table/rows current-session (table/from [:json {:path json}]))))
        (table/write-parquet! current-session source parquet)
        (is (= (table/rows current-session source) (table/rows current-session (table/from [:parquet {:path parquet}]))))
        (table/write-csv! current-session source csv)
        (is (= (table/rows current-session source) (table/rows current-session (table/from [:csv {:path csv}]))))
        (testing "Derived values add or replace columns even when file names are unknown"
          (is (= [{:x 8 :y 3.0 :z 1} {:x 8 :y 4.0 :z 1}]
                 (table/rows current-session (table/query (table/from [:parquet {:path parquet}])
                                                          [:derive {:x 8 :z 1}])))))
        (testing "Every column of an unknown SQL schema can be replaced"
          (is (= [{:x 8}]
                 (table/rows current-session (table/query (table/from [:duckdb {:query "SELECT 1 AS x"}])
                                                          [:derive {:x 8}]))))))
      (finally (doseq [path [csv json parquet]] (delete-file! path))))))

(deftest clauses
  (with-open [current-session (session/open)]
    (let [source (table/from [{:k "a" :x 1 :y 4} {:k "a" :x 2 :y 5} {:k "b" :x 3 :y 6}])]
      (is (= [{:x 1 :y 4} {:x 2 :y 5} {:x 3 :y 6}]
             (table/rows current-session (table/query source [:select :x :y]))))
      (is (= [{:y 4} {:y 5} {:y 6}] (table/rows current-session (table/query source [:select [:y]]))))
      (is (= [{:k "a" :a 1 :y 4} {:k "a" :a 2 :y 5} {:k "b" :a 3 :y 6}]
             (table/rows current-session (table/query source [:rename {:x :a}]))))
      (is (= [{:k "a" :x 1 :y 4 :z 10} {:k "a" :x 2 :y 5 :z 10} {:k "b" :x 3 :y 6 :z 10}]
             (table/rows current-session (table/query source [:derive {:z 10}]))))
      (is (= [{:k "a" :x 5 :y 4} {:k "a" :x 7 :y 5} {:k "b" :x 9 :y 6}]
             (table/rows current-session (table/query source [:derive {:x [:+ :x :y]}]))))
      (is (= [{:k "a" :m 1.5 :n 2} {:k "b" :m 3.0 :n 1}]
             (table/rows current-session (table/query source [[:group-by [:k] {:n [:count] :m [:avg :x]}]
                                                              [:order-by :k]]))))
      (is (= [{:x 3} {:x 2}]
             (table/rows current-session (table/query source [[:where [:> :x 1]] [:order-by [:x :desc :first]]
                                                              [:limit 2] [:select :x]]))))
      (is (= 2 (table/row-count current-session (table/query source [:sample {:n 2 :seed 11}]))))
      (is (= (table/rows current-session (table/query source [:sample {:n 2 :seed 11 :key :x}]))
             (table/rows current-session (table/query source [:sample {:n 2 :seed 11 :key :x}]))))
      (is (= [1 2 1]
             (mapv :rank (table/rows current-session (table/query source
                                                                  [[:derive {:rank [:over [:row-number]
                                                                                    {:partition-by [:k] :order-by [[:x :asc]]}]}]
                                                                   [:order-by :x]])))))
      (is (= [1 3 3]
             (mapv :running (table/rows current-session (table/query source
                                                                     [[:derive {:running [:over [:sum :x]
                                                                                          {:partition-by [:k] :order-by [:x]
                                                                                           :rows [:unbounded :current]}]}]
                                                                      [:order-by :x]]))))))))

(deftest expressions-and-null-semantics
  (with-open [current-session (session/open)]
    (let [source (table/from {:x [1 2 nil] :text ["Alpha" "beta" nil]})
          matching-values (fn [input-expression] (mapv :x (table/rows current-session (table/query source [:where input-expression]))))]
      (is (= [1] (matching-values [:and [:= [:+ :x [:* 2 3]] 7] [:not [:is-null :x]]])))
      (is (= [1 nil] (matching-values [:or [:= :x 1] [:is-null :x]])))
      (is (= [] (matching-values [:= :x nil])))
      (is (= [1 2] (matching-values [:not-null :x])))
      (is (= [1] (matching-values [:in :x [1 5]])))
      (is (= [] (matching-values [:in :x []])))
      (is (= [1 2] (matching-values [:between :x 1 2])))
      (is (= [1] (matching-values [:ilike :text "a%"])))
      (is (= [2] (matching-values [:like :text "b%"])))
      (is (= ["1" "2" nil] (mapv :s (table/rows current-session (table/query source [:derive {:s [:cast :x :string]}])))))
      (is (= ["alpha" "beta" "none"]
             (mapv :s (table/rows current-session (table/query source [:derive {:s [:coalesce [:lower :text] "none"]}])))))
      (is (= [{:distinct 2 :n 2 :total 3}]
             (table/rows current-session (table/query source [:group-by [] {:n [:count :x] :total [:sum :x]
                                                                            :distinct [:count-distinct :x]}]))))
      (is (= [true true true]
             (mapv :b (table/rows current-session (table/query source [:derive {:b [:lit true]}])))))
      (is (= [nil 2 1] (mapv :x (table/rows current-session (table/query source [:order-by [[:x :desc :first]]])))))
      (doseq [[input-expression expected] [[[:!= :x 1] [2]] [[:not= :x 1] [2]] [[:>= :x 2] [2]]
                                           [[:< :x 2] [1]] [[:<= :x 1] [1]] [[:= [:- :x 1] 0] [1]]
                                           [[:= [:/ :x 2] 1] [2]] [[:= [:% :x 2] 1] [1]]
                                           [[:= [:col "x"] 1] [1]]]]
        (is (= expected (matching-values input-expression)))))))

(deftest expression-scalars-preserve-types-and-null-comparisons
  (with-open [current-session (session/open)]
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
             (table/rows current-session result)))
      (is (= [{:name "large" :type :int64}
              {:name "single" :type :float32}
              {:name "flag" :type :bool}
              {:name "equal" :type :bool}
              {:name "equal-left" :type :bool}
              {:name "different" :type :bool}
              {:name "missing" :type :bool}]
             (table/schema current-session result))))))

(deftest literal-identifier-names-survive-clauses-and-joins
  (with-open [current-session (session/open)]
    (doseq [column ["part.value" "dash-name" "quote\"name" "question?mark"
                    "literal/slash" "select" "*" "x); DROP TABLE victims; --"]]
      (let [renamed (str "renamed." column)
            aggregate (str column "?total")
            source (table/from {column [2 1 2] "discard" [8 9 10]})]
        (testing (str "Literal name through select, replacement, rename, group and order: " column)
          (is (= [{(keyword renamed) 12 (keyword aggregate) 24}
                  {(keyword renamed) 11 (keyword aggregate) 11}]
                 (table/rows current-session
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
                   (table/rows current-session joined)))))))))

(deftest chained-projections-bind-both-join-inputs
  (with-open [current-session (session/open)]
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
             (table/rows current-session joined))))))

(deftest native-honeysql-transformations-compose-with-garun
  (with-open [current-session (session/open)]
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
          (is (= [{:amount 4 :band "O'Brien?" :doubled 8}] (table/rows current-session result)))
          (is (= [{:amount 4 :band "O'Brien?" :doubled 8}]
                 (:rows (round-trip! current-session result :parquet table/write-parquet!))))))
      (testing "WHERE-only transformations retain the input projection"
        (is (= [{:category "a" :amount 2 :nullable "x"}]
               (table/rows current-session (table/query source [:honeysql {:where [:= :amount 2]}])))))
      (testing "Native DISTINCT replaces the default SELECT rather than duplicating it"
        (is (= [{:category "a"} {:category "b"}]
               (table/rows current-session
                           (table/query source [:honeysql {:select-distinct [:category]
                                                           :order-by [[:category :asc]]}])))))
      (testing "Native HAVING and aggregates remain composable with Garun clauses"
        (is (= [{:category "b" :total 4}]
               (table/rows current-session
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
  (with-open [current-session (session/open)]
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
                       (table/rows current-session
                                   (table/query source
                                                [[:derive {:total [:over [:sum :x]
                                                                   {:partition-by [:partition]
                                                                    :order-by [:position]
                                                                    :rows frame}]}]
                                                 [:order-by :partition :position]]))))))))))

(deftest join-types-and-collisions
  (with-open [current-session (session/open)]
    (let [left (table/from {:id [1 2] :v ["a" "b"]})
          right (table/from {:id [2 3] :v ["B" "C"] :z [20 30]})
          joined (fn [how] (table/query left [:join right {:on [:id] :how how}]))]
      (is (= [{:id 1 :v "a" :id_right nil :v_right nil :z nil}
              {:id 2 :v "b" :id_right 2 :v_right "B" :z 20}]
             (table/rows current-session (table/query (joined :left) [:order-by :id]))))
      (is (= 1 (table/row-count current-session (joined :inner))))
      (is (= 2 (table/row-count current-session (joined :right))))
      (is (= 3 (table/row-count current-session (joined :full))))
      (is (= [{:id 2 :v "b"}] (table/rows current-session (joined :semi))))
      (is (= [{:id 1 :v "a"}] (table/rows current-session (joined :anti))))
      (is (= 4 (table/row-count current-session (table/query left [:join right {:how :cross}]))))
      (is (= [{:id 2 :v "b" :id_other 2 :v_other "B" :z 20}]
             (table/rows current-session (table/query left [:join right {:on {:id :id} :suffix "_other"}]))))
      (is (= [{:id 2 :v "b" :k 2 :z 20}]
             (table/rows current-session (table/query left [:join (table/from {:k [2] :z [20]}) {:on {:id :k}}]))))
      (is (= [{:id 2 :v "b" :id_right 2}]
             (table/rows current-session (table/query left
                                                      [:join (table/from [:duckdb {:query "SELECT 2 AS id"}])
                                                       {:on [:id]}]))))
      (is (= [{:id 2 :v "b" :id_right 2 :z 20}]
             (table/rows current-session (table/query left
                                                      [:join (table/from [:duckdb {:query "SELECT 2 AS id, 20 AS z"}])
                                                       {:on [:id]}])))))))

(deftest deterministic-covering-splits
  (with-open [current-session (session/open)]
    (let [source (table/from {:id (vec (range 200)) :x (vec (range 200))})
          options {:fractions [0.6 0.3 0.1] :seed 12 :key :id}
          parts (mapv #(mapv :id (table/rows current-session %)) (table/query source [:split options]))
          again (mapv #(mapv :id (table/rows current-session %)) (table/query source [:split options]))
          no-key (mapv #(mapv :id (table/rows current-session %))
                       (table/query source [:split {:fractions [0.5 0.5] :seed 3}]))]
      (is (= parts again))
      (is (every? seq parts))
      (is (= (vec (range 200)) (vec (sort (apply concat parts)))))
      (is (= 200 (count (distinct (apply concat parts)))))
      (is (= 200 (count (apply concat no-key))))
      (is (= no-key (mapv #(mapv :id (table/rows current-session %))
                          (table/query source [:split {:fractions [0.5 0.5] :seed 3}])))))))

(deftest split-helpers-do-not-shadow-user-columns
  (with-open [current-session (session/open)]
    (let [source (table/from [:duckdb {:query "SELECT i AS id, i AS __garun_row__, i AS __garun_bucket__, i AS __garun_split_0 FROM range(20) t(i)"}])
          parts (table/query source [:split {:fractions [0.5 0.5] :seed 9}])
          combined (mapcat #(table/rows current-session %) parts)]
      (is (= (vec (range 20)) (vec (sort (map :id combined)))))
      (is (every? #(= (:id %) (:__garun_row__ %) (:__garun_bucket__ %) (:__garun_split_0 %)) combined)))))

(deftest matrix-order-and-errors
  (with-open [current-session (session/open)]
    (let [source (table/from {:a [1 2] :b [3.5 4.5]})
          matrix (table/matrix current-session (table/query source [:order-by [:a :desc]]) [:b :a])]
      (is (= 2 (:rows matrix)))
      (is (= 2 (:cols matrix)))
      (is (= [4.5 2.0 3.5 1.0] (vec (:data matrix))))
      (is (= Float/TYPE (.getComponentType (class (:data matrix)))))
      (is (= :table/null-value (error-code #(table/matrix current-session (table/from {:x [nil 1]}) [:x]))))
      (is (= :table/non-numeric (error-code #(table/matrix current-session (table/from {:x ["a"]}) [:x])))))))

(deftest invalid-dsl-is-rejected
  (let [source (table/from {:x [1]})]
    (is (= :table/unknown-clause (error-code #(table/query source [:wat 1]))))
    (is (= :table/unknown-operator (error-code #(table/query source [:where [:wat :x]]))))
    (doseq [operation [#(table/from [:csv {:path "x" :wat true}])
                       #(table/query source [:join source {:on [:x] :wat true}])
                       #(table/query source [:sample {:n 1 :wat true}])
                       #(table/query source [:split {:fractions [1] :wat true}])
                       #(table/query source [:derive {:x [:over [:count] {:wat true}]}])]]
      (is (= :table/unknown-option (error-code operation))))
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
    (with-open [current-session (session/open)]
      (is (= :table/unknown-option (error-code #(table/rows current-session source {:wat 1})))))))

(deftest named-materialized-tables-compose-and-describe
  (with-open [current-session (session/open)]
    (let [materialized (table/materialize! current-session :observations
                                           {:id [1 2]
                                            :amount [2 4]
                                            :text ["O'Brien? -- not SQL" "other"]})
          description (session/describe current-session)
          resource (first (:resources description))
          qualified-name (str "\"" (string/replace (:sql-name resource) "." "\".\"") "\"")
          filtered (table/query materialized
                                [[:where [:= :id 2]]
                                 [:derive {:total [:+ :amount 10]}]])]
      (is (table/table? materialized))
      (is (= (table/from :observations) materialized))
      (is (= [{:amount 4 :id 2 :text "other" :total 14}]
             (table/rows current-session filtered)))
      (is (= :observations (:name resource)))
      (is (= :table (:kind resource)))
      (is (true? (:owned? resource)))
      (is (false? (:released? resource)))
      (is (= [{:name "amount" :type :int64}
              {:name "id" :type :int64}
              {:name "text" :type :string}]
             (:columns resource)))
      (is (= 2 (:rows resource)))
      (is (= (:columns resource) (table/schema current-session :observations)))
      (is (= 2 (table/row-count current-session :observations)))
      (is (= {:amount [2 4] :id [1 2] :text ["O'Brien? -- not SQL" "other"]}
             (table/columns current-session :observations)))
      (is (= [2.0 1.0 4.0 2.0]
             (vec (:data (table/matrix current-session :observations [:amount :id])))))

      (testing "Named sources remain pure until a context resolves them"
        (doseq [source [:observations materialized filtered
                        (table/query {:id [2]} [:join :observations {:on [:id]}])]]
          (is (= :table/unresolved-session-table (error-code #(table/->sql source)))))
        (is (string/includes? (first (table/->sql current-session filtered)) qualified-name))
        (is (= [2] (rest (table/->sql current-session
                                      (table/query :observations [:where [:= :id 2]]))))))

      (testing "Either side of a join can reference a named session table"
        (is (= [{:amount 4 :id 2 :text "other" :id_right 2 :label "matched"}]
               (table/rows current-session
                           (table/query materialized
                                        [:join {:id [2 3] :label ["matched" "unmatched"]}
                                         {:on [:id]}]))))
        (is (= [{:id 2 :label "matched" :amount 4 :id_right 2 :text "other"}]
               (table/rows current-session
                           (table/query {:id [2 3] :label ["matched" "unmatched"]}
                                        [:join :observations {:on [:id]}])))))

      (testing "COPY accepts named plans while keeping temporary tables context-local"
        (doseq [[format writer] [[:csv table/write-csv!] [:parquet table/write-parquet!]]]
          (is (= (table/rows current-session filtered)
                 (:rows (round-trip! current-session filtered format writer)))))))))

(deftest materialized-results-are-stable-and-visible-to-workers
  (with-open [current-session (session/open)]
    (let [materialized (table/materialize!
                        current-session :random-values
                        [:duckdb {:query "SELECT i AS id, random() AS value FROM range(8) t(i)"}])
          ordered (table/query materialized [:order-by :id])
          first-results (table/rows current-session ordered)]
      (is (= 8 (count first-results)))
      (is (= first-results (table/rows current-session ordered)))
      (is (= 8 (:rows (first (:resources (session/describe current-session))))))
      (with-open [worker (session/worker current-session)]
        (is (= first-results (table/rows worker ordered)))
        (is (= (filterv #(> (:id %) 3) first-results)
               (table/rows worker (table/query :random-values
                                               [[:where [:> :id 3]] [:order-by :id]]))))
        (let [worker-table (table/materialize! worker :worker-values {:x [7 8]})]
          (is (= [{:x 7} {:x 8}] (table/rows current-session worker-table))))))))

(deftest materialization-replacement-and-release-drop-physical-tables
  (with-open [current-session (session/open)]
    (let [schema-name (:schema (session/describe current-session))
          inventory (table/query
                     [:duckdb {:query "SELECT schema_name, table_name FROM duckdb_tables() WHERE NOT temporary"}]
                     [:where [:= :schema_name schema-name]])
          materialized (table/materialize! current-session :values {:x [1 2]})
          first-sql-name (:sql-name (first (:resources (session/describe current-session))))]
      (is (= 1 (table/row-count current-session inventory)))
      (is (= :session/duplicate-name
             (error-code #(table/materialize! current-session :values {:x [9]}))))
      (is (= [{:x 1} {:x 2}] (table/rows current-session materialized)))
      (is (= 1 (table/row-count current-session inventory)))

      (is (= (table/from :values)
             (table/materialize! current-session :values {:x [9]} {:replace? true})))
      (is (= [{:x 9}] (table/rows current-session materialized)))
      (is (= 1 (table/row-count current-session inventory)))
      (let [description (session/describe current-session)
            resource (first (:resources description))
            resolved-sql (first (table/->sql current-session materialized))]
        (is (= 1 (count (:resources description))))
        (is (not= first-sql-name (:sql-name resource)))
        (is (= 1 (:rows resource)))
        (is (nil? (session/release! current-session :values)))
        (is (= [] (:resources (session/describe current-session))))
        (is (zero? (table/row-count current-session inventory)))
        (is (= :session/unknown-name (error-code #(table/rows current-session materialized))))
        (is (thrown? java.sql.SQLException
                     (table/rows current-session [:duckdb {:query resolved-sql}])))))))

(deftest worker-table-cleanup-and-child-workers-do-not-interfere-with-root-queries
  (with-open [current-session (session/open)
              worker (session/worker current-session)]
    (table/materialize! current-session :replace-values {:x [1]})
    ;; Keep the root JDBC cursor open while another context runs DDL and duplicates
    ;; its connection. Latches test the native boundary without relying on query speed.
    (let [root-query-entered (CountDownLatch. (int 1))
          worker-finished (CountDownLatch. (int 1))
          read-rows! @#'table/result-rows!
          source [:duckdb {:query "SELECT i AS session_number FROM range(10000) t(i)"}]
          expected (mapv #(hash-map :session_number %) (range 10000))]
      (with-redefs-fn
        {#'table/result-rows!
         (fn [^ResultSet result max-rows]
           (when (= "session_number" (.getColumnLabel (.getMetaData result) (int 1)))
             (.countDown root-query-entered)
             (when-not (.await worker-finished (long 10) TimeUnit/SECONDS)
               (throw (ex-info "Worker cleanup timed out" {:error :test/timeout}))))
           (read-rows! result max-rows))}
        (fn []
          (let [root-result (CompletableFuture/supplyAsync
                             (reify Supplier
                               (get [_]
                                 (mapv (fn [_] (table/rows current-session source)) (range 2)))))
                worker-result (CompletableFuture/supplyAsync
                               (reify Supplier
                                 (get [_]
                                   (try
                                     (when-not (.await root-query-entered (long 10) TimeUnit/SECONDS)
                                       (throw (ex-info "Root query admission timed out" {:error :test/timeout})))
                                     (table/materialize! worker :worker-values {:x [7 8]})
                                     (session/release! worker :worker-values)
                                     (table/materialize! worker :replace-values {:x [9]} {:replace? true})
                                     (let [replacement (table/rows worker :replace-values)
                                           child-rows (with-open [child (session/worker worker)]
                                                        (table/rows child {:nested_number [17]}))]
                                       (session/release! worker :replace-values)
                                       {:replacement replacement :child child-rows})
                                     (finally (.countDown worker-finished))))))]
            (try
              (is (= {:replacement [{:x 9}] :child [{:nested_number 17}]}
                     (.get worker-result (long 15) TimeUnit/SECONDS)))
              (is (= [expected expected] (.get root-result (long 15) TimeUnit/SECONDS)))
              (finally
                (.countDown worker-finished)
                (.get worker-result (long 15) TimeUnit/SECONDS)
                (.get root-result (long 15) TimeUnit/SECONDS))))))
      (is (zero? (table/row-count
                  current-session
                  (table/query
                   [:duckdb {:query "SELECT schema_name FROM duckdb_tables() WHERE NOT temporary"}]
                   [:where [:= :schema_name (:schema (session/describe current-session))]]))))
      (is (= :session/unknown-name (error-code #(table/rows current-session :worker-values))))
      (is (= :session/unknown-name (error-code #(table/rows current-session :replace-values)))))))

(deftest named-table-query-holds-shared-lock-until-its-result-is-consumed
  (with-open [current-session (session/open)
              worker (session/worker current-session)]
    (table/materialize! current-session :locked-values {:x [3 4]})
    ;; Pause after JDBC execution but before consuming the cursor: release must wait
    ;; for the whole result lifetime, not just name resolution or statement creation.
    (let [query-entered (CountDownLatch. (int 1))
          finish-query (CountDownLatch. (int 1))
          release-attempted (CountDownLatch. (int 1))
          read-rows! @#'table/result-rows!
          physical-query (first (table/->sql current-session :locked-values))]
      (with-redefs-fn
        {#'table/result-rows!
         (fn [result max-rows]
           (.countDown query-entered)
           (when-not (.await finish-query (long 10) TimeUnit/SECONDS)
             (throw (ex-info "Query completion timed out" {:error :test/timeout})))
           (read-rows! result max-rows))}
        (fn []
          (let [query-result (CompletableFuture/supplyAsync
                              (reify Supplier
                                (get [_] (table/rows current-session :locked-values))))
                release-result (CompletableFuture/supplyAsync
                                (reify Supplier
                                  (get [_]
                                    (when-not (.await query-entered (long 10) TimeUnit/SECONDS)
                                      (throw (ex-info "Query admission timed out" {:error :test/timeout})))
                                    (.countDown release-attempted)
                                    (session/release! worker :locked-values))))]
            (try
              (is (.await query-entered (long 10) TimeUnit/SECONDS))
              (is (.await release-attempted (long 10) TimeUnit/SECONDS))
              (is (thrown? TimeoutException (.get release-result (long 100) TimeUnit/MILLISECONDS)))
              (.countDown finish-query)
              (is (= [{:x 3} {:x 4}] (.get query-result (long 10) TimeUnit/SECONDS)))
              (is (nil? (.get release-result (long 10) TimeUnit/SECONDS)))
              (finally
                (.countDown finish-query)
                (.get query-result (long 10) TimeUnit/SECONDS)
                (.get release-result (long 10) TimeUnit/SECONDS))))))
      (is (= :session/unknown-name (error-code #(table/rows current-session :locked-values))))
      (is (thrown? java.sql.SQLException
                   (table/rows current-session [:duckdb {:query physical-query}]))))))

(deftest materialization-validates-names-and-closed-options
  (with-open [current-session (session/open)]
    (doseq [resource-name [nil "values" 7]]
      (is (= :table/invalid
             (error-code #(table/materialize! current-session resource-name {:x [1]})))))
    (doseq [options [nil [] {:replace? nil} {:replace? "true"}]]
      (is (= :table/invalid
             (error-code #(table/materialize! current-session :values {:x [1]} options)))))
    (is (= :table/unknown-option
           (error-code #(table/materialize! current-session :values {:x [1]} {:replace true}))))
    (is (= :session/unknown-name
           (error-code #(table/materialize! current-session :values :missing))))
    (is (= :session/unknown-name (error-code #(table/rows current-session :missing))))
    (is (= [] (:resources (session/describe current-session))))))

(deftest table-execution-rejects-a-closed-session
  (with-open [current-session (session/open)]
    (let [source (table/from {:x [1]})]
      (.close ^java.lang.AutoCloseable current-session)
      (doseq [operation [#(table/rows current-session source)
                         #(table/columns current-session source)
                         #(table/schema current-session source)
                         #(table/row-count current-session source)
                         #(table/matrix current-session source [:x])
                         #(table/write-csv! current-session source "closed.csv")
                         #(table/write-parquet! current-session source "closed.parquet")
                         #(table/materialize! current-session :values source)
                         #(table/->sql current-session :values)]]
        (is (= :session/closed (error-code operation)))))))
