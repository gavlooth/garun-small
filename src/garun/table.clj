(ns garun.table
  "Immutable Garun table plans executed through DuckDB JDBC.

  A plan holds SQL text and its ordered parameters; each clause selects from the
  previous plan as __input__. Nothing runs until a connection is supplied. This
  small renderer owns DuckDB-specific syntax, quotes identifiers, and binds DSL
  values. Only [:duckdb {:query ...}] accepts trusted SQL text. COPY cannot bind
  parameters, so exports materialize the bound query in a temporary table and
  escape only the output path; the temporary table is always dropped."
  (:require [clojure.string :as str])
  (:import [java.sql Connection DriverManager PreparedStatement ResultSet ResultSetMetaData]
           [java.util UUID]))

(set! *warn-on-reflection* true)

(defn- invalid! [path message value]
  (throw (ex-info message {:error :table/invalid :path path :value value})))

(defn- options! [op opts allowed]
  (when-not (map? opts)
    (invalid! [op] "Options must be a map" opts))
  (when-let [unknown (seq (remove (set allowed) (keys opts)))]
    (throw (ex-info "Unknown table option"
                    {:error :table/unknown-option :op op :keys (vec unknown)}))))

(defn- column-name [column]
  (let [text (cond (keyword? column) (name column)
                   (string? column) column
                   :else nil)]
    (when (or (nil? text) (empty? text) (str/includes? text "\u0000"))
      (invalid! [:column] "Column names must be nonempty keywords or strings without NUL" column))
    text))

(defn- quote-name [column]
  (str "\"" (str/replace (column-name column) "\"" "\"\"") "\""))

(defn- distinct-names! [columns]
  (when-not (= (count columns) (count (distinct (map str/lower-case columns))))
    (invalid! [:columns] "Column names collide after case normalization" columns))
  columns)

(defn- arity! [op args expected]
  (when-not (= (count args) expected)
    (invalid! [op] "Wrong expression or clause arity" args)))

(defn- combine-sql [prefix separator suffix pieces]
  {:sql (str prefix (str/join separator (map :sql pieces)) suffix)
   :params (vec (mapcat :params pieces))})

(defn- fragment [& parts]
  (combine-sql "" "" ""
               (map #(if (string? %) {:sql % :params []} %) parts)))

(defn- literal [value]
  (when-not (or (nil? value) (number? value) (string? value) (boolean? value))
    (invalid! [:literal] "Expected a scalar literal" value))
  {:sql "?" :params [value]})

(def ^:private sql-types
  {:bool "BOOLEAN" :boolean "BOOLEAN" :int8 "TINYINT" :int16 "SMALLINT" :int32 "INTEGER"
   :int64 "BIGINT" :i64 "BIGINT" :uint8 "UTINYINT" :uint16 "USMALLINT" :uint32 "UINTEGER"
   :uint64 "UBIGINT" :float32 "FLOAT" :f32 "FLOAT" :float64 "DOUBLE" :f64 "DOUBLE"
   :string "VARCHAR" :binary "BLOB" :date "DATE" :timestamp "TIMESTAMP"})

(def ^:private binary-ops
  {:= "=" :not= "<>" :!= "<>" :> ">" :>= ">=" :< "<" :<= "<="
   :+ "+" :- "-" :* "*" :/ "/" :% "%" :like "LIKE" :ilike "ILIKE"})

(def ^:private unary-ops
  {:not ["(NOT " ")"] :is-null ["(" " IS NULL)"]
   :not-null ["(" " IS NOT NULL)"] :count-distinct ["count(DISTINCT " ")"]})

(def ^:private function-ops
  {:sum ["sum" 1] :avg ["avg" 1] :min ["min" 1] :max ["max" 1]
   :stddev ["stddev_samp" 1] :stddev-samp ["stddev_samp" 1]
   :variance ["var_samp" 1] :var-samp ["var_samp" 1]
   :stddev-pop ["stddev_pop" 1] :variance-pop ["var_pop" 1] :var-pop ["var_pop" 1]
   :median ["median" 1] :mad ["mad" 1] :mode ["mode" 1] :sem ["sem" 1]
   :entropy ["entropy" 1] :skewness ["skewness" 1] :kurtosis ["kurtosis" 1]
   :covar-pop ["covar_pop" 2] :covar-samp ["covar_samp" 2] :corr ["corr" 2]
   :quantile-cont ["quantile_cont" 2] :quantile-disc ["quantile_disc" 2]
   :quantile ["quantile" 2] :approx-quantile ["approx_quantile" 2]
   :quantile-reservoir ["reservoir_quantile" [2 3]]
   :regr-slope ["regr_slope" 2] :regr-intercept ["regr_intercept" 2]
   :regr-r2 ["regr_r2" 2] :regr-count ["regr_count" 2] :regr-avgx ["regr_avgx" 2]
   :regr-avgy ["regr_avgy" 2] :regr-sxx ["regr_sxx" 2] :regr-sxy ["regr_sxy" 2] :regr-syy ["regr_syy" 2]
   :abs ["abs" 1] :sqrt ["sqrt" 1] :lower ["lower" 1] :upper ["upper" 1] :length ["length" 1]
   :coalesce ["coalesce" :many] :round ["round" [1 2]] :lag ["lag" [1 3]] :lead ["lead" [1 3]]
   :row-number ["row_number" 0] :rank ["rank" 0] :dense-rank ["dense_rank" 0]})

(defn- order-item [item]
  (when (and (vector? item) (not (<= 1 (count item) 3)))
    (invalid! [:order-by] "Order item is [column direction nulls]" item))
  (let [[column direction nulls] (if (vector? item)
                                  [(first item) (get item 1 :asc) (get item 2 :last)]
                                  [item :asc :last])]
    (when-not (#{:asc :desc} direction)
      (invalid! [:order-by] "Invalid direction" direction))
    (when-not (#{:first :last} nulls)
      (invalid! [:order-by] "Invalid null ordering" nulls))
    (fragment (quote-name column) " " (str/upper-case (name direction))
              " NULLS " (str/upper-case (name nulls)))))

(defn- frame-bound [bound start?]
  (cond
    (= :unbounded bound) (if start? "UNBOUNDED PRECEDING" "UNBOUNDED FOLLOWING")
    (= :current bound) "CURRENT ROW"
    (integer? bound) (str (abs bound) (if (neg? bound) " PRECEDING" " FOLLOWING"))
    :else (invalid! [:window :rows] "Invalid window frame bound" bound)))

(defn- window-spec [opts]
  (options! :window opts [:partition-by :order-by :rows])
  (let [partition (get opts :partition-by [])
        order (get opts :order-by [])
        frame (:rows opts)
        frame? (contains? opts :rows)]
    (when-not (and (vector? partition) (vector? order))
      (invalid! [:window] "Partition and order must be vectors" opts))
    (when (and frame? (or (empty? order) (not (and (vector? frame) (= 2 (count frame))))))
      (invalid! [:window :rows] "Ordered frame must be [start end]" frame))
    (fragment
     (if (seq partition)
       (combine-sql "PARTITION BY " ", " " " (map #(fragment (quote-name %)) partition)) "")
     (if (seq order) (combine-sql "ORDER BY " ", " " " (map order-item order)) "")
     (if frame?
       (str "ROWS BETWEEN " (frame-bound (first frame) true) " AND " (frame-bound (second frame) false))
       ""))))

(defn- expression-kind [expr]
  (let [op (when (vector? expr) (first expr))]
    (cond
      (keyword? expr) ::column
      (not (vector? expr)) ::literal
      (contains? binary-ops op) ::binary
      (contains? unary-ops op) ::unary
      (contains? function-ops op) ::function
      :else op)))

(defmulti ^:private expression expression-kind)

(defmethod expression ::column [column]
  (fragment (quote-name column)))

(defmethod expression ::literal [value]
  (literal value))

(defmethod expression :col [[op & args]]
  (arity! op args 1)
  (fragment (quote-name (first args))))

(defmethod expression :lit [[op & args]]
  (arity! op args 1)
  (literal (first args)))

(defmethod expression ::binary [[op & args]]
  (arity! op args 2)
  (combine-sql "(" (str " " (get binary-ops op) " ") ")" (map expression args)))

(defmethod expression ::unary [[op & args]]
  (arity! op args 1)
  (let [[prefix suffix] (get unary-ops op)]
    (combine-sql prefix "" suffix [(expression (first args))])))

(defn- boolean-expression [op args]
  (when (empty? args)
    (invalid! [op] "Boolean operator needs operands" (into [op] args)))
  (combine-sql "(" (if (= :and op) " AND " " OR ") ")" (map expression args)))

(defmethod expression :and [[op & args]]
  (boolean-expression op args))

(defmethod expression :or [[op & args]]
  (boolean-expression op args))

(defmethod expression :in [[op & args :as expr]]
  (arity! op args 2)
  (when-not (vector? (second args))
    (invalid! [op] "IN requires a literal vector" expr))
  (if (empty? (second args))
    (fragment "FALSE")
    (fragment "(" (expression (first args)) " IN "
              (combine-sql "(" ", " ")" (map literal (second args))) ")")))

(defmethod expression :between [[op & args]]
  (arity! op args 3)
  (fragment "(" (expression (first args)) " BETWEEN " (expression (second args))
            " AND " (expression (nth args 2)) ")"))

(defmethod expression :cast [[op & args]]
  (arity! op args 2)
  (when-not (contains? sql-types (second args))
    (invalid! [op] "Unsupported cast type" (second args)))
  (fragment "CAST(" (expression (first args)) " AS " (get sql-types (second args)) ")"))

(defmethod expression :count [[op & args]]
  (when (> (count args) 1)
    (invalid! [op] "COUNT takes zero or one argument" args))
  (fragment "count(" (if (seq args) (expression (first args)) "*") ")"))

(defmethod expression :over [[op & args]]
  (arity! op args 2)
  (fragment (expression (first args)) " OVER (" (window-spec (second args)) ")"))

(defn- literal-value [value]
  (if (and (vector? value) (= :lit (first value)) (= 2 (count value)))
    (second value)
    value))

(defn- function-arguments! [op args arity]
  (let [n (count args)
        valid? (cond (vector? arity) (<= (first arity) n (second arity))
                     (= :many arity) (pos? n)
                     :else (= arity n))
        quantile? (#{:quantile-cont :quantile-disc :quantile :approx-quantile :quantile-reservoir} op)]
    (when-not valid?
      (invalid! [op] "Wrong function arity" args))
    (when quantile?
      (let [quantile (literal-value (second args))]
        (when-not (and (number? quantile) (Double/isFinite (double quantile)) (<= 0 quantile 1))
          (invalid! [op] "Quantile must be a finite literal in [0,1]" quantile))))
    (when (and (= :quantile-reservoir op) (= 3 n))
      (let [size (literal-value (nth args 2))]
        (when-not (and (integer? size) (pos? size))
          (invalid! [op] "Reservoir size must be positive" size))))))

(defmethod expression ::function [[op & args]]
  (let [[function arity] (get function-ops op)]
    (function-arguments! op args arity)
    (combine-sql (str function "(") ", " ")" (map expression args))))

(defmethod expression :default [[op]]
  (throw (ex-info "Unknown expression operator" {:error :table/unknown-operator :operator op})))

(defn table?
  "Return whether x is a Garun table plan with SQL and an ordered parameter vector."
  [x]
  (and (map? x) (= :garun/table (:type x)) (string? (:sql x)) (vector? (:params x))))

(defn- plan [piece columns]
  (with-meta (assoc piece :type :garun/table)
    (assoc (meta piece) ::columns columns)))

(defn- names [table]
  (::columns (meta table)))

(defn- values-source [source]
  (let [column-input? (map? source)]
    (when column-input?
      (when-not (every? vector? (vals source))
        (invalid! [:source] "Column values must be vectors" source))
      (when (> (count (distinct (map count (vals source)))) 1)
        (invalid! [:source] "Column lengths must match" source)))
    (when (and (not column-input?) (not (every? map? source)))
      (invalid! [:source] "Rows must be maps" source))
    (let [ordered (sort-by column-name (if column-input? (keys source) (distinct (mapcat keys source))))
          columns (distinct-names! (mapv column-name ordered))
          rows (if column-input?
                 (map (fn [index] (map #(nth (get source %) index) ordered))
                      (range (if (seq source) (count (first (vals source))) 0)))
                 (map (fn [row] (map #(get row %) ordered)) source))]
      (when (empty? columns)
        (invalid! [:source] "Source requires at least one named column" source))
      (if (seq rows)
        (plan (fragment "SELECT * FROM (VALUES "
                        (combine-sql "" ", " "" (map #(combine-sql "(" ", " ")" (map literal %)) rows))
                        ") AS __input__(" (str/join ", " (map quote-name columns)) ")") columns)
        (plan (fragment (combine-sql "SELECT " ", " ""
                                    (map #(fragment (literal nil) " AS " (quote-name %)) columns))
                        " WHERE FALSE") columns)))))

(def ^:private source-options
  {:csv [:path :header? :delimiter] :parquet [:path] :json [:path] :duckdb [:query]})

(defn- source-options! [tag opts]
  (let [allowed (get source-options tag)
        text (get opts (if (= :duckdb tag) :query :path))]
    (when-not allowed
      (invalid! [:source] "Unknown source tag" tag))
    (options! tag opts allowed)
    (when (or (not (string? text)) (empty? text) (str/includes? text "\u0000"))
      (invalid! [:source] "Source needs nonempty text without NUL" text))
    (when (and (contains? opts :header?) (not (boolean? (:header? opts))))
      (invalid! [:source :header?] "Header must be boolean" (:header? opts)))
    (when (and (contains? opts :delimiter)
               (or (not (string? (:delimiter opts))) (empty? (:delimiter opts))
                   (str/includes? (:delimiter opts) "\u0000")))
      (invalid! [:source :delimiter] "Delimiter must be nonempty text without NUL" (:delimiter opts)))))

(defn- descriptor-source [[tag opts :as source]]
  (arity! :source source 2)
  (source-options! tag opts)
  (if (= :duckdb tag)
    (plan (fragment "SELECT * FROM (" (:query opts) ") AS __input__") nil)
    (plan (fragment "SELECT * FROM "
                    (get {:csv "read_csv_auto(" :parquet "read_parquet(" :json "read_json_auto("} tag)
                    (literal (:path opts))
                    (if (contains? opts :header?) (fragment ", header=" (literal (:header? opts))) "")
                    (if (contains? opts :delimiter) (fragment ", delim=" (literal (:delimiter opts))) "")
                    ")") nil)))

(defn from
  "Build a pure table plan from columns, rows, a file descriptor or trusted SQL.
  File descriptors are [:csv {:path p :header? b :delimiter s}], [:parquet
  {:path p}] and [:json {:path p}]. [:duckdb {:query sql}] is trusted SQL.
  Invalid source descriptions throw an ExceptionInfo with :error :table/invalid."
  [source]
  (cond
    (table? source) source
    (map? source) (values-source source)
    (and (vector? source) (or (empty? source) (map? (first source)))) (values-source source)
    (vector? source) (descriptor-source source)
    :else (invalid! [:source] "Unsupported table source" source)))

(defn- input-sql [table]
  (fragment "(" table ") AS __input__"))

(defn- select-sql [table projection]
  (fragment "SELECT " projection " FROM " (input-sql table)))

(defn- aliased-expression [[column expr]]
  (fragment (expression expr) " AS " (quote-name column)))

(defn- join-keys [on]
  (let [pairs (cond (map? on) (sort-by #(column-name (first %)) on)
                    (vector? on) (mapv #(if (vector? %) % [% %]) on)
                    (some? on) [[on on]]
                    :else [])]
    (doseq [pair pairs]
      (arity! :join-key pair 2))
    pairs))

(def ^:private join-types
  {:inner "INNER JOIN" :left "LEFT JOIN" :right "RIGHT JOIN" :full "FULL JOIN" :cross "CROSS JOIN"})

(defn- join-options! [opts how suffix pairs]
  (when-not (or (#{:semi :anti} how) (contains? join-types how))
    (invalid! [:join :how] "Unknown join type" how))
  (when-not (and (string? suffix) (not (str/includes? suffix "\u0000")))
    (invalid! [:join :suffix] "Suffix must be text without NUL" suffix))
  (when (and (not= :cross how) (empty? pairs))
    (invalid! [:join :on] "Join requires explicit keys" (:on opts)))
  (when (and (= :cross how) (seq pairs))
    (invalid! [:join :on] "Cross join cannot have keys" (:on opts))))

(defn- join-condition [[left right]]
  (fragment "__input__." (quote-name left) " = __right__." (quote-name right)))

(defn- right-column [[column alias]]
  (fragment "__right__." (quote-name column) " AS " (quote-name alias)))

(defn- join-projection [left right aliases pairs suffix left-only?]
  (let [right-keys (distinct (map (comp column-name second) pairs))]
    (cond
      left-only? (fragment "__input__.*")
      (and left right)
      (combine-sql "__input__.*, " ", " "" (map right-column (map vector right aliases)))
      (seq right-keys)
      (combine-sql "__input__.*, __right__.* RENAME (" ", " ")"
                   (map #(fragment (quote-name %) " AS " (quote-name (str % suffix))) right-keys))
      :else (fragment "__input__.*, __right__.*"))))

(defn- joined-sql [table other how projection condition]
  (let [left (select-sql table projection)
        right (fragment "(" other ") AS __right__")]
    (cond
      (#{:semi :anti} how)
      (fragment left (if (= :semi how) " WHERE EXISTS (" " WHERE NOT EXISTS (")
                "SELECT 1 FROM " right " WHERE " condition ")")
      (= :cross how) (fragment left " CROSS JOIN " right)
      :else (fragment left " " (get join-types how) " " right " ON " condition))))

(defn- join-plan [table source opts]
  (options! :join opts [:on :how :suffix])
  (let [other (from source)
        how (get opts :how :inner)
        suffix (get opts :suffix "_right")
        pairs (join-keys (:on opts))
        left (names table)
        right (names other)
        normalized-left (set (map str/lower-case left))
        left-only? (#{:semi :anti} how)]
    (join-options! opts how suffix pairs)
    (let [aliases (when right
                    (mapv #(if (contains? normalized-left (str/lower-case %)) (str % suffix) %) right))
          output (cond left-only? left
                       (and left right) (distinct-names! (into left aliases))
                       :else nil)
          projection (join-projection left right aliases pairs suffix left-only?)
          condition (combine-sql "(" " AND " ")" (map join-condition pairs))]
      (plan (joined-sql table other how projection condition) output))))

(defn- seeded-key [opts]
  (if (contains? opts :key)
    (expression [:col (:key opts)])
    (expression [:over [:row-number] {}])))

(defn- split-options! [opts]
  (options! :split opts [:fractions :seed :key])
  (let [fractions (:fractions opts)
        seed (get opts :seed 0)]
    (when (or (not (vector? fractions)) (empty? fractions)
              (some #(or (not (number? %)) (not (Double/isFinite (double %))) (<= % 0)) fractions)
              (> (abs (- 1.0 (double (reduce + fractions)))) 1.0e-9))
      (invalid! [:split :fractions] "Fractions must be positive and sum to one" fractions))
    (when-not (integer? seed)
      (invalid! [:split :seed] "Seed must be an integer" seed))))

(defn- bucketed-input [table opts]
  (let [bucket (fragment "(hash(" (seeded-key opts) ", " (literal (get opts :seed 0)) ") % 1000000000)")]
    ;; Nest the original row in a struct: helper columns then cannot shadow user
    ;; columns, including columns of file/SQL sources whose names are unknown.
    (select-sql table (fragment "__input__ AS __garun_row__, " bucket " AS __garun_bucket__"))))

(defn- split-range [bucketed columns [lower upper]]
  (plan (fragment (select-sql bucketed (fragment "__garun_row__.*"))
                  " WHERE __garun_bucket__ >= " (literal (long (* 1000000000 lower)))
                  " AND __garun_bucket__ < " (literal (long (* 1000000000 upper)))) columns))

(defn- split-plan [table opts]
  (split-options! opts)
  (let [bucketed (bucketed-input table opts)
        boundaries (vec (reductions + 0.0 (:fractions opts)))
        ;; Pin the final boundary to one despite floating-point addition.
        boundaries (assoc boundaries (dec (count boundaries)) 1.0)]
    (mapv #(split-range bucketed (names table) %) (partition 2 1 boundaries))))

(defn- clause-items [op args]
  (cond
    (and (= :order-by op) (= 1 (count args)) (vector? (first args))
         (#{:asc :desc} (second (first args)))) [(first args)]
    (and (= 1 (count args)) (vector? (first args))) (first args)
    :else args))

(defmulti ^:private clause (fn [_ [op]] op))

(defmethod clause :select [table [op & args]]
  (let [items (clause-items op args)
        output (distinct-names! (mapv column-name items))]
    (when (empty? items)
      (invalid! [op] "Select requires columns" items))
    (plan (select-sql table (combine-sql "" ", " "" (map #(expression [:col %]) items))) output)))

(defmethod clause :rename [table [op & args]]
  (arity! op args 1)
  (when-not (map? (first args))
    (invalid! [op] "Rename requires a map" args))
  (let [entries (sort-by #(column-name (first %)) (first args))
        renames (into {} (map (fn [[old new]] [(str/lower-case (column-name old)) (column-name new)]) entries))
        output (when (names table)
                 (distinct-names! (mapv #(get renames (str/lower-case %) %) (names table))))
        projection (if (seq entries)
                     (combine-sql "* RENAME (" ", " ")"
                                  (map (fn [[old new]] (fragment (quote-name old) " AS " (quote-name new))) entries))
                     (fragment "*"))]
    (plan (select-sql table projection) output)))

(defmethod clause :where [table [op & args]]
  (arity! op args 1)
  (plan (fragment (select-sql table (fragment "*")) " WHERE " (expression (first args))) (names table)))

(defn- struct-field [[column expr]]
  (fragment (quote-name column) " := " (expression expr)))

(defn- derived-plan [table op args]
  (arity! op args 1)
  (when-not (map? (first args))
    (invalid! [op] "Derived columns require a map" args))
  (let [entries (sort-by #(column-name (first %)) (first args))
        aliases (distinct-names! (mapv #(column-name (first %)) entries))
        replaced (set (map str/lower-case aliases))
        input-names (names table)
        retained (when input-names (filterv #(not (contains? replaced (str/lower-case %))) input-names))
        projection (cond
                     (empty? entries) (fragment "*")
                     input-names
                     (combine-sql "" ", " ""
                                  (concat (map #(fragment (quote-name %)) retained)
                                          (map aliased-expression entries)))
                     ;; SELECT *, value AS x duplicates existing x; * REPLACE
                     ;; cannot add a new x. Updating the row struct handles both
                     ;; cases without inspecting an unknown file/SQL schema.
                     :else
                     (fragment "unnest(struct_update(__input__, "
                               (combine-sql "" ", " "" (map struct-field entries)) "))"))]
    (plan (select-sql table projection) (when input-names (into retained aliases)))))

(defmethod clause :derive [table [op & args]]
  (derived-plan table op args))

(defmethod clause :window [table [op & args]]
  (derived-plan table op args))

(defmethod clause :group-by [table [op & args]]
  (arity! op args 2)
  (let [[keys aggregates] args]
    (when-not (and (vector? keys) (map? aggregates))
      (invalid! [op] "Group-by requires key vector and aggregate map" args))
    (let [entries (sort-by #(column-name (first %)) aggregates)
          output (distinct-names! (into (mapv column-name keys) (map #(column-name (first %)) entries)))
          projection (concat (map #(expression [:col %]) keys) (map aliased-expression entries))]
      (when (empty? projection)
        (invalid! [op] "Group-by requires keys or aggregates" args))
      (plan (fragment (select-sql table (combine-sql "" ", " "" projection))
                      (if (seq keys)
                        (combine-sql " GROUP BY " ", " "" (map #(expression [:col %]) keys)) "")) output))))

(defmethod clause :order-by [table [op & args]]
  (let [items (clause-items op args)]
    (when (empty? items)
      (invalid! [op] "Order-by requires columns" items))
    (plan (fragment (select-sql table (fragment "*"))
                    (combine-sql " ORDER BY " ", " "" (map order-item items))) (names table))))

(defmethod clause :limit [table [op & args]]
  (arity! op args 1)
  (when-not (and (integer? (first args)) (not (neg? (first args))))
    (invalid! [op] "Limit must be nonnegative" args))
  (plan (fragment (select-sql table (fragment "*")) " LIMIT " (literal (first args))) (names table)))

(defmethod clause :sample [table [op & args]]
  (arity! op args 1)
  (let [opts (first args)]
    (options! op opts [:n :seed :key])
    (let [n (:n opts)
          seed (get opts :seed 0)]
      (when-not (and (integer? n) (pos? n))
        (invalid! [op :n] "Sample size must be positive" n))
      (when-not (integer? seed)
        (invalid! [op :seed] "Seed must be integer" seed))
      (plan (fragment (select-sql table (fragment "*"))
                      " ORDER BY hash(" (seeded-key opts) ", " (literal seed) ") LIMIT " (literal n))
            (names table)))))

(defmethod clause :join [table [op & args]]
  (arity! op args 2)
  (join-plan table (first args) (second args)))

(defmethod clause :split [table [op & args]]
  (arity! op args 1)
  (split-plan table (first args)))

(defmethod clause :default [_ [op]]
  (throw (ex-info "Unknown table clause" {:error :table/unknown-clause :clause op})))

(defn query
  "Apply clause vectors to a table without I/O; a final :split returns tables.
  Invalid clauses throw ExceptionInfo before execution.

  Joins suffix every colliding right column when both plans have known names
  (in-memory sources, select/group-by and their descendants). For unknown file
  or SQL schemas, explicit right :on keys are suffixed unconditionally; other
  right columns must have names disjoint from the left columns. Cross joins of
  unknown schemas likewise require disjoint names. DuckDB binds column names."
  [table clauses]
  (let [table (from table)
        clauses (if (and (vector? clauses) (keyword? (first clauses))) [clauses] clauses)]
    (when-not (vector? clauses)
      (invalid! [:query] "Query requires a vector of clauses" clauses))
    (when-not (every? #(and (vector? %) (seq %)) clauses)
      (invalid! [:query] "Clauses must be nonempty vectors" clauses))
    (when (some #(= :split (first %)) (butlast clauses))
      (invalid! [:query :split] "Split must be the final clause" clauses))
    (reduce clause table clauses)))

(defn ->sql
  "Return a table as [SQL & bound-parameters], in SQL placeholder order.
  Source descriptions are accepted too; invalid descriptions throw ExceptionInfo."
  [table]
  (let [{:keys [sql params]} (from table)]
    (into [sql] params)))

(defn open
  "Open a DuckDB JDBC connection, in memory or at path. Close with with-open.
  Invalid paths throw ExceptionInfo; JDBC connection failures propagate."
  (^Connection [] (DriverManager/getConnection "jdbc:duckdb:"))
  (^Connection [path]
   (when (or (not (string? path)) (empty? path) (str/includes? path "\u0000"))
     (invalid! [:open] "Database path must be nonempty text without NUL" path))
   (DriverManager/getConnection (str "jdbc:duckdb:" path))))

(defn- prepare! ^PreparedStatement [^Connection db [sql & params]]
  (let [statement (.prepareStatement db ^String sql)]
    (try
      (doseq [[index value] (map-indexed vector params)]
        (.setObject statement (int (inc index)) value))
      statement
      (catch Throwable error
        (.close statement)
        (throw error)))))

(defn- result-labels [^ResultSetMetaData metadata]
  (mapv #(keyword (.getColumnLabel metadata (int %))) (range 1 (inc (.getColumnCount metadata)))))

(defn- read-row! [^ResultSet result labels]
  (into {} (map-indexed (fn [index label] [label (.getObject result (int (inc index)))]) labels)))

(defn- result-rows! [^ResultSet result max-rows]
  (let [labels (result-labels (.getMetaData result))]
    ;; Eagerly read each row before advancing the JDBC cursor again.
    (into [] (comp (take-while (fn [_] (.next result)))
                   (map (fn [_] (read-row! result labels))))
          (range max-rows))))

(defn rows
  "Execute a table and return a vector of keyword-keyed row maps.
  Optional :max-rows caps returned rows (zero returns no rows). Invalid options
  throw ExceptionInfo; JDBC execution failures propagate."
  ([db table] (rows db table {}))
  ([db table opts]
   (options! :rows opts [:max-rows])
   (let [max-rows (get opts :max-rows Long/MAX_VALUE)]
     (when-not (and (integer? max-rows) (<= 0 max-rows Long/MAX_VALUE))
       (invalid! [:rows :max-rows] "Expected a nonnegative row cap" max-rows))
     (with-open [statement (prepare! db (->sql table))
                 result (.executeQuery statement)]
       (result-rows! result max-rows)))))

(defn columns
  "Execute a table and return an ordered array-map of keyword to column vector.
  JDBC execution failures propagate."
  [db table]
  (with-open [statement (prepare! db (->sql table))
              result (.executeQuery statement)]
    (let [labels (result-labels (.getMetaData result))
          values (result-rows! result Long/MAX_VALUE)]
      (apply array-map (mapcat (fn [label] [label (mapv label values)]) labels)))))

(def ^:private column-types
  {"BIGINT" :int64 "INTEGER" :int32 "DOUBLE" :float64 "FLOAT" :float32
   "VARCHAR" :string "BOOLEAN" :bool "DATE" :date "TIMESTAMP" :timestamp})

(defn- column-descriptor [^ResultSetMetaData metadata index]
  (let [type-name (.getColumnTypeName metadata (int index))]
    {:name (.getColumnLabel metadata (int index))
     :type (get column-types type-name (keyword (str/lower-case type-name)))}))

(defn schema
  "Return ordered {:name string :type keyword} DuckDB column descriptors.
  JDBC execution failures propagate."
  [db table]
  (with-open [statement (prepare! db (->sql (query table [:limit 0])))
              result (.executeQuery statement)]
    (let [metadata (.getMetaData result)]
      (mapv #(column-descriptor metadata %) (range 1 (inc (.getColumnCount metadata)))))))

(defn row-count
  "Execute COUNT(*) on a table and return a long. JDBC failures propagate."
  [db table]
  (long (:n (first (rows db (query table [:group-by [] {:n [:count]}]))))))

(defn- copy! [^Connection db table path format]
  (when (or (not (string? path)) (empty? path) (str/includes? path "\u0000"))
    (invalid! [:copy :path] "Output path must be nonempty text without NUL" path))
  (let [{:keys [sql params]} (from table)
        table-name (str "__garun_copy_" (str/replace (str (UUID/randomUUID)) "-" ""))
        create-sql (str "CREATE TEMP TABLE " table-name " AS " sql)
        copy-sql (str "COPY " table-name " TO '" (str/replace path "'" "''")
                      "' (FORMAT " format (when (= "CSV" format) ", HEADER TRUE") ")")]
    (with-open [statement (prepare! db (into [create-sql] params))]
      (.execute statement))
    (try
      (with-open [statement (prepare! db [copy-sql])]
        (.execute statement))
      nil
      (finally
        (with-open [statement (prepare! db [(str "DROP TABLE " table-name)])]
          (.execute statement))))))

(defn write-csv!
  "Write a table to a CSV file with a header via DuckDB COPY.
  Invalid output paths throw ExceptionInfo; JDBC failures propagate."
  [db table path]
  (copy! db table path "CSV"))

(defn write-parquet!
  "Write a table to a parquet file via DuckDB COPY.
  Invalid output paths throw ExceptionInfo; JDBC failures propagate."
  [db table path]
  (copy! db table path "PARQUET"))

(defn- matrix-cell! [^ResultSet result [index column]]
  (let [value (.getObject result (int index))]
    (cond
      (nil? value) (throw (ex-info "NULL matrix value" {:error :table/null-value :column column}))
      (not (number? value)) (throw (ex-info "Nonnumeric matrix value"
                                          {:error :table/non-numeric :column column :value value}))
      :else (float value))))

(defn- matrix-row! [result indexed-columns]
  (mapv #(matrix-cell! result %) indexed-columns))

(defn matrix
  "Return numeric columns as {:rows n :cols k :data float-array}, row-major.
  Column order follows cols and row order follows the table. NULL and nonnumeric
  cells throw :table/null-value and :table/non-numeric respectively. Invalid
  column selections throw ExceptionInfo; JDBC failures propagate."
  [db table cols]
  (when-not (and (vector? cols) (seq cols) (every? keyword? cols))
    (invalid! [:matrix :cols] "Matrix requires a nonempty keyword column vector" cols))
  (with-open [statement (prepare! db (->sql (query table [:select cols])))
              result (.executeQuery statement)]
    (let [indexed-columns (mapv vector (range 1 (inc (count cols))) cols)
          ;; Realize each row while its cursor is current; no row maps or Java
          ;; helper/build step are needed for this functional JDBC traversal.
          values (sequence (comp (take-while (fn [_] (.next result)))
                                 (mapcat (fn [_] (matrix-row! result indexed-columns))))
                           (range))
          data (float-array values)]
      {:rows (quot (alength data) (count cols)) :cols (count cols) :data data})))
