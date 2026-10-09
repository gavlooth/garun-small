(ns garun.table
  "Immutable Garun table plans executed through session-owned DuckDB contexts.

  A plan holds a HoneySQL :query map; each clause selects from the previous
  plan as __input__. Nothing runs until a context is supplied. Each context
  owns its connection; use one Worker per thread and close the owning Session
  to release its tables, managers and connections. Keyword sources reference
  named session tables. Execution holds their shared resource locks in sorted
  name order until JDBC results are consumed, excluding release and replacement.

  Scalar DSL values are lifted parameters, including NULL, numbers and
  booleans. Only [:duckdb {:query ...}] accepts trusted SQL. Local HoneySQL
  extensions handle literal identifiers (including '*'), DuckDB named
  arguments and COPY paths. COPY uses a temporary table on the same context
  connection; materialize! creates immutable tables in the session schema."
  (:require [clojure.string :as str]
            [clojure.walk :as walk]
            [honey.sql :as sql])
  (:import [garun.session Context Kind MaterializedTable Resource Session]
           [java.sql Connection PreparedStatement ResultSet ResultSetMetaData]
           [java.util UUID]))

(set! *warn-on-reflection* true)

(defn- invalid! [path message value]
  (throw (ex-info message {:error :table/invalid :path path :value value})))

(defn- options! [operation options allowed]
  (when-not (map? options)
    (invalid! [operation] "Options must be a map" options))
  (when-let [unknown (seq (remove (set allowed) (keys options)))]
    (throw (ex-info "Unknown table option"
                    {:error :table/unknown-option :op operation :keys (vec unknown)}))))

(defn- column-name [column]
  (let [text (cond (keyword? column) (name column)
                   (string? column) column
                   :else nil)]
    (when (or (nil? text) (empty? text) (str/includes? text "\u0000"))
      (invalid! [:column] "Column names must be nonempty keywords or strings without NUL" column))
    text))

(defn- identifier [column]
  [:garun/identifier (column-name column)])

;; HoneySQL's entity formatter interprets '*' as a wildcard, even in aliases.
;; Garun names are always literal; qualification and wildcards are separate ASTs.
(sql/register-fn! :garun/identifier
                  (fn [_ [text]]
                    [(str "\"" (str/replace text "\"" "\"\"") "\"")]))

(defn- named-argument [operation [column value]]
  (let [[value-sql & parameters] (sql/format-expr value)
        [column-sql] (sql/format-expr column)]
    (into [(str column-sql (if (= :garun/struct-field operation) " := " " = ") value-sql)]
          parameters)))

(sql/register-fn! :garun/file-option named-argument)

(sql/register-fn! :garun/struct-field named-argument)

;; DuckDB COPY does not accept bound paths. Only this extension escapes a
;; validated output path; the query itself is bound through CREATE TABLE AS.
(sql/register-clause!
 :garun/copy
 (fn [_ [table-name path format]]
   [(str "COPY " (sql/format-entity table-name) " TO '"
         (str/replace path "'" "''") "' (FORMAT " format
         (when (= "CSV" format) ", HEADER TRUE") ")")])
 nil)

(defn- distinct-names! [columns]
  (when-not (= (count columns) (count (distinct (map str/lower-case columns))))
    (invalid! [:columns] "Column names collide after case normalization" columns))
  columns)

(defn- arity! [operation arguments expected]
  (when-not (= (count arguments) expected)
    (invalid! [operation] "Wrong expression or clause arity" arguments)))

(defn- literal [value]
  (when-not (or (nil? value) (number? value) (string? value) (boolean? value))
    (invalid! [:literal] "Expected a scalar literal" value))
  [:lift value])

(def ^:private sql-types
  {:bool :boolean :boolean :boolean :int8 :tinyint :int16 :smallint :int32 :integer
   :int64 :bigint :i64 :bigint :uint8 :utinyint :uint16 :usmallint :uint32 :uinteger
   :uint64 :ubigint :float32 :float :f32 :float :float64 :double :f64 :double
   :string :varchar :binary :blob :date :date :timestamp :timestamp})

(def ^:private binary-operators
  #{:= :not= :!= :> :>= :< :<= :+ :- :* :/ :% :like :ilike})

(def ^:private unary-operators
  #{:not :is-null :not-null :count-distinct})

(def ^:private function-operators
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
    [(identifier column) (keyword (str (name direction) "-nulls-" (name nulls)))]))

(defn- frame-bound [bound start?]
  (cond
    (= :unbounded bound) (if start? :unbounded-preceding :unbounded-following)
    (= :current bound) :current-row
    (integer? bound) [(literal (abs bound)) (if (neg? bound) :preceding :following)]
    :else (invalid! [:window :rows] "Invalid window frame bound" bound)))

(defn- window-spec [options]
  (options! :window options [:partition-by :order-by :rows])
  (let [partition (get options :partition-by [])
        order (get options :order-by [])
        frame (:rows options)
        frame? (contains? options :rows)]
    (when-not (and (vector? partition) (vector? order))
      (invalid! [:window] "Partition and order must be vectors" options))
    (when (and frame? (or (empty? order) (not (and (vector? frame) (= 2 (count frame))))))
      (invalid! [:window :rows] "Ordered frame must be [start end]" frame))
    (cond-> {}
      (seq partition) (assoc :partition-by (mapv #(vector (identifier %)) partition))
      (seq order) (assoc :order-by (mapv order-item order))
      frame? (assoc :frame [:rows :between (frame-bound (first frame) true)
                            (frame-bound (second frame) false)]))))

(defn- literal-value [value]
  (if (and (vector? value) (= :lit (first value)) (= 2 (count value)))
    (second value)
    value))

(defn- function-arguments! [operation arguments arity]
  (let [argument-count (count arguments)
        valid? (cond (vector? arity) (<= (first arity) argument-count (second arity))
                     (= :many arity) (pos? argument-count)
                     :else (= arity argument-count))
        quantile? (#{:quantile-cont :quantile-disc :quantile :approx-quantile :quantile-reservoir} operation)]
    (when-not valid?
      (invalid! [operation] "Wrong function arity" arguments))
    (when quantile?
      (let [quantile (literal-value (second arguments))]
        (when-not (and (number? quantile) (Double/isFinite (double quantile)) (<= 0 quantile 1))
          (invalid! [operation] "Quantile must be a finite literal in [0,1]" quantile))))
    (when (and (= :quantile-reservoir operation) (= 3 argument-count))
      (let [size (literal-value (nth arguments 2))]
        (when-not (and (integer? size) (pos? size))
          (invalid! [operation] "Reservoir size must be positive" size))))))

(defn- expression [input-expression]
  (cond
    (keyword? input-expression) (identifier input-expression)
    (not (vector? input-expression)) (literal input-expression)
    :else
    (let [[operation & arguments] input-expression
          arity (cond
                  (or (contains? binary-operators operation) (#{:in :cast :over} operation)) 2
                  (or (contains? unary-operators operation) (#{:col :lit} operation)) 1
                  (= :between operation) 3)]
      (when arity (arity! operation arguments arity))
      (cond
        (contains? binary-operators operation) (into [operation] (map expression arguments))
        (contains? function-operators operation)
        (let [[function arity] (get function-operators operation)]
          (function-arguments! operation arguments arity)
          (into [(keyword function)] (map expression arguments)))
        :else
        (case operation
          :col (identifier (first arguments))
          :lit (literal (first arguments))
          :not [:not (expression (first arguments))]
          :is-null [:is (expression (first arguments)) nil]
          :not-null [:is-not (expression (first arguments)) nil]
          :count-distinct [:count [:distinct (expression (first arguments))]]
          (:and :or)
          (do
            (when (empty? arguments)
              (invalid! [operation] "Boolean operator needs operands" input-expression))
            (into [operation] (map expression arguments)))
          :in
          (do
            (when-not (vector? (second arguments))
              (invalid! [operation] "IN requires a literal vector" input-expression))
            (if (empty? (second arguments))
              (literal false)
              [:in (expression (first arguments)) (mapv literal (second arguments))]))
          :between (into [:between] (map expression arguments))
          :cast
          (do
            (when-not (contains? sql-types (second arguments))
              (invalid! [operation] "Unsupported cast type" (second arguments)))
            [:cast (expression (first arguments)) (get sql-types (second arguments))])
          :count
          (do
            (when (> (count arguments) 1)
              (invalid! [operation] "COUNT takes zero or one argument" arguments))
            [:count (if (seq arguments) (expression (first arguments)) :*)])
          :over [:over [(expression (first arguments)) (window-spec (second arguments))]]
          (throw (ex-info "Unknown expression operator"
                          {:error :table/unknown-operator :operator operation})))))))

(defn table?
  "Return whether value is a Garun table plan with a HoneySQL query map."
  [value]
  (and (map? value) (= :garun/table (:type value)) (map? (:query value))))

(defn- plan [query columns]
  (with-meta {:type :garun/table :query query}
    {::columns columns}))

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
        (plan {:select [:*]
               :from [[{:values (mapv #(mapv literal %) rows)}
                       [:__input__ (mapv identifier columns)]]]} columns)
        (plan {:select (mapv #(vector (literal nil) [(identifier %)]) columns)
               :where (literal false)} columns)))))

(def ^:private source-options
  {:csv [:path :header? :delimiter] :parquet [:path] :json [:path] :duckdb [:query]})

(defn- source-options! [tag options]
  (let [allowed (get source-options tag)
        text (get options (if (= :duckdb tag) :query :path))]
    (when-not allowed
      (invalid! [:source] "Unknown source tag" tag))
    (options! tag options allowed)
    (when (or (not (string? text)) (empty? text) (str/includes? text "\u0000"))
      (invalid! [:source] "Source needs nonempty text without NUL" text))
    (when (and (contains? options :header?) (not (boolean? (:header? options))))
      (invalid! [:source :header?] "Header must be boolean" (:header? options)))
    (when (and (contains? options :delimiter)
               (or (not (string? (:delimiter options))) (empty? (:delimiter options))
                   (str/includes? (:delimiter options) "\u0000")))
      (invalid! [:source :delimiter] "Delimiter must be nonempty text without NUL" (:delimiter options)))))

(defn- descriptor-source [[tag options :as source]]
  (arity! :source source 2)
  (source-options! tag options)
  (if (= :duckdb tag)
    (plan {:select [:*] :from [[[:nest [:raw (:query options)]] :__input__]]} nil)
    (let [function (get {:csv :read_csv_auto :parquet :read_parquet :json :read_json_auto} tag)
          arguments (cond-> [(literal (:path options))]
                      (contains? options :header?)
                      (conj [:garun/file-option (identifier :header) (literal (:header? options))])
                      (contains? options :delimiter)
                      (conj [:garun/file-option (identifier :delim) (literal (:delimiter options))]))]
      (plan {:select [:*] :from [[(into [function] arguments)]]} nil))))

(defn from
  "Build a pure table plan from columns, rows, a descriptor or a session name.
  Keyword names resolve to owned materialized tables only during execution.
  File descriptors are [:csv {:path path :header? boolean :delimiter string}],
  [:parquet {:path path}] and [:json {:path path}]. [:duckdb {:query sql}] is
  trusted SQL. Invalid sources throw ExceptionInfo with :error :table/invalid."
  [source]
  (cond
    (table? source) source
    (keyword? source) (plan {:select [:*] :from [[[:garun/session-table source]]]} nil)
    (map? source) (values-source source)
    (and (vector? source) (or (empty? source) (map? (first source)))) (values-source source)
    (vector? source) (descriptor-source source)
    :else (invalid! [:source] "Unsupported table source" source)))

(defn- select-query [table projection]
  {:select projection :from [[(:query table) :__input__]]})

(defn- aliased-expression [[column input-expression]]
  [(expression input-expression) [(identifier column)]])

(defn- join-keys [on]
  (let [pairs (cond (map? on) (sort-by #(column-name (first %)) on)
                    (vector? on) (mapv #(if (vector? %) % [% %]) on)
                    (some? on) [[on on]]
                    :else [])]
    (doseq [pair pairs]
      (arity! :join-key pair 2))
    pairs))

(def ^:private join-types
  {:inner :join :left :left-join :right :right-join :full :full-join :cross :cross-join})

(defn- join-options! [options how suffix pairs]
  (when-not (or (#{:semi :anti} how) (contains? join-types how))
    (invalid! [:join :how] "Unknown join type" how))
  (when-not (and (string? suffix) (not (str/includes? suffix "\u0000")))
    (invalid! [:join :suffix] "Suffix must be text without NUL" suffix))
  (when (and (not= :cross how) (empty? pairs))
    (invalid! [:join :on] "Join requires explicit keys" (:on options)))
  (when (and (= :cross how) (seq pairs))
    (invalid! [:join :on] "Cross join cannot have keys" (:on options))))

(defn- join-condition [[left right]]
  [:= [:. :__input__ (identifier left)] [:. :__right__ (identifier right)]])

(defn- join-projection [left right aliases pairs suffix left-only?]
  (let [right-keys (distinct (map (comp column-name second) pairs))]
    (cond
      left-only? [:__input__.*]
      (and left right)
      (into [:__input__.*]
            (map (fn [column alias] [[:. :__right__ (identifier column)] [(identifier alias)]])
                 right aliases))
      (seq right-keys)
      [:__input__.*
       [:__right__.* {:rename (mapv #(vector (identifier %) [(identifier (str % suffix))]) right-keys)}]]
      :else [:__input__.* :__right__.*])))

(defn- joined-query [table other how projection condition]
  (let [left (select-query table projection)
        right [(:query other) :__right__]]
    (cond
      (#{:semi :anti} how)
      (assoc left :where [(if (= :semi how) :exists :not-exists)
                          {:select [[(literal 1)]] :from [right] :where condition}])
      (= :cross how) (assoc left :cross-join [right])
      :else (assoc left (get join-types how) [right condition]))))

(defn- join-plan [table source options]
  (options! :join options [:on :how :suffix])
  (let [other (from source)
        how (get options :how :inner)
        suffix (get options :suffix "_right")
        pairs (join-keys (:on options))
        left (names table)
        right (names other)
        normalized-left (set (map str/lower-case left))
        left-only? (#{:semi :anti} how)]
    (join-options! options how suffix pairs)
    (let [aliases (when right
                    (mapv #(if (contains? normalized-left (str/lower-case %)) (str % suffix) %) right))
          output (cond left-only? left
                       (and left right) (distinct-names! (into left aliases))
                       :else nil)
          projection (join-projection left right aliases pairs suffix left-only?)
          condition (into [:and] (map join-condition pairs))]
      (plan (joined-query table other how projection condition) output))))

(defn- seeded-key [options]
  (if (contains? options :key)
    (expression [:col (:key options)])
    (expression [:over [:row-number] {}])))

(defn- split-options! [options]
  (options! :split options [:fractions :seed :key])
  (let [fractions (:fractions options)
        seed (get options :seed 0)]
    (when (or (not (vector? fractions)) (empty? fractions)
              (some #(or (not (number? %)) (not (Double/isFinite (double %))) (<= % 0)) fractions)
              (> (abs (- 1.0 (double (reduce + fractions)))) 1.0e-9))
      (invalid! [:split :fractions] "Fractions must be positive and sum to one" fractions))
    (when-not (integer? seed)
      (invalid! [:split :seed] "Seed must be an integer" seed))))

(defn- bucketed-input [table options]
  ;; Nest the original row in a struct so helpers cannot shadow user columns,
  ;; including columns of file/SQL sources whose names are unknown.
  (select-query table
                [[:__input__ :__garun_row__]
                 [[:% [:hash (seeded-key options) (literal (get options :seed 0))]
                   (literal 1000000000)] :__garun_bucket__]]))

(defn- split-range [bucketed columns [lower upper]]
  (plan {:select [:__garun_row__.*]
         :from [[bucketed :__input__]]
         :where [:and [:>= :__garun_bucket__ (literal (long (* 1000000000 lower)))]
                 [:< :__garun_bucket__ (literal (long (* 1000000000 upper)))]]} columns))

(defn- split-plan [table options]
  (split-options! options)
  (let [bucketed (bucketed-input table options)
        boundaries (vec (reductions + 0.0 (:fractions options)))
        ;; Pin the final boundary to one despite floating-point addition.
        boundaries (assoc boundaries (dec (count boundaries)) 1.0)]
    (mapv #(split-range bucketed (names table) %) (partition 2 1 boundaries))))

(defn- clause-items [operation arguments]
  (cond
    (and (= :order-by operation) (= 1 (count arguments)) (vector? (first arguments))
         (#{:asc :desc} (second (first arguments)))) [(first arguments)]
    (and (= 1 (count arguments)) (vector? (first arguments))) (first arguments)
    :else arguments))

(defmulti ^:private clause (fn [_ [operation]] operation))

(defmethod clause :select [table [operation & arguments]]
  (let [items (clause-items operation arguments)
        output (distinct-names! (mapv column-name items))]
    (when (empty? items)
      (invalid! [operation] "Select requires columns" items))
    (plan (select-query table (mapv #(vector (identifier %)) items)) output)))

(defmethod clause :rename [table [operation & arguments]]
  (arity! operation arguments 1)
  (when-not (map? (first arguments))
    (invalid! [operation] "Rename requires a map" arguments))
  (let [entries (sort-by #(column-name (first %)) (first arguments))
        renames (into {} (map (fn [[old new]] [(str/lower-case (column-name old)) (column-name new)]) entries))
        output (when (names table)
                 (distinct-names! (mapv #(get renames (str/lower-case %) %) (names table))))
        projection (if (seq entries)
                     [[:* {:rename (mapv (fn [[old new]] [(identifier old) [(identifier new)]]) entries)}]]
                     [:*])]
    (plan (select-query table projection) output)))

(defmethod clause :where [table [operation & arguments]]
  (arity! operation arguments 1)
  (plan (assoc (select-query table [:*]) :where (expression (first arguments))) (names table)))

(defn- struct-field [[column input-expression]]
  [:garun/struct-field (identifier column) (expression input-expression)])

(defn- derived-plan [table operation arguments]
  (arity! operation arguments 1)
  (when-not (map? (first arguments))
    (invalid! [operation] "Derived columns require a map" arguments))
  (let [entries (sort-by #(column-name (first %)) (first arguments))
        aliases (distinct-names! (mapv #(column-name (first %)) entries))
        replaced (set (map str/lower-case aliases))
        input-names (names table)
        retained (when input-names (filterv #(not (contains? replaced (str/lower-case %))) input-names))
        projection (cond
                     (empty? entries) [:*]
                     input-names
                     (into (mapv #(vector (identifier %)) retained)
                           (map aliased-expression entries))
                     ;; Updating the row struct replaces and adds fields without
                     ;; inspecting an unknown file/SQL schema.
                     :else
                     [[[:unnest (into [:struct_update :__input__] (map struct-field entries))]]])]
    (plan (select-query table projection) (when input-names (into retained aliases)))))

(defmethod clause :derive [table [operation & arguments]]
  (derived-plan table operation arguments))

(defmethod clause :window [table [operation & arguments]]
  (derived-plan table operation arguments))

(defmethod clause :group-by [table [operation & arguments]]
  (arity! operation arguments 2)
  (let [[keys aggregates] arguments]
    (when-not (and (vector? keys) (map? aggregates))
      (invalid! [operation] "Group-by requires key vector and aggregate map" arguments))
    (let [entries (sort-by #(column-name (first %)) aggregates)
          output (distinct-names! (into (mapv column-name keys) (map #(column-name (first %)) entries)))
          projection (into (mapv #(vector (identifier %)) keys) (map aliased-expression entries))]
      (when (empty? projection)
        (invalid! [operation] "Group-by requires keys or aggregates" arguments))
      (plan (cond-> (select-query table projection)
              (seq keys) (assoc :group-by (mapv identifier keys))) output))))

(defmethod clause :order-by [table [operation & arguments]]
  (let [items (clause-items operation arguments)]
    (when (empty? items)
      (invalid! [operation] "Order-by requires columns" items))
    (plan (assoc (select-query table [:*]) :order-by (mapv order-item items)) (names table))))

(defmethod clause :limit [table [operation & arguments]]
  (arity! operation arguments 1)
  (when-not (and (integer? (first arguments)) (not (neg? (first arguments))))
    (invalid! [operation] "Limit must be nonnegative" arguments))
  (plan (assoc (select-query table [:*]) :limit (literal (first arguments))) (names table)))

(defmethod clause :sample [table [operation & arguments]]
  (arity! operation arguments 1)
  (let [options (first arguments)]
    (options! operation options [:n :seed :key])
    (let [sample-size (:n options)
          seed (get options :seed 0)]
      (when-not (and (integer? sample-size) (pos? sample-size))
        (invalid! [operation :n] "Sample size must be positive" sample-size))
      (when-not (integer? seed)
        (invalid! [operation :seed] "Seed must be integer" seed))
      (plan (assoc (select-query table [:*])
                   :order-by [[[:hash (seeded-key options) (literal seed)]]]
                   :limit (literal sample-size)) (names table)))))

(defmethod clause :join [table [operation & arguments]]
  (arity! operation arguments 2)
  (join-plan table (first arguments) (second arguments)))

(defmethod clause :split [table [operation & arguments]]
  (arity! operation arguments 1)
  (split-plan table (first arguments)))

(defmethod clause :honeysql [table [operation & arguments]]
  (arity! operation arguments 1)
  (let [transformation (first arguments)]
    (options! operation transformation
              [:select :select-distinct :select-distinct-on :select-top :select-distinct-top
               :where :group-by :having :window :qualify :order-by :limit :offset :fetch
               :join :left-join :right-join :full-join :cross-join :join-by :with :with-recursive])
    (let [projection? (some #(contains? transformation %)
                            [:select :select-distinct :select-distinct-on :select-top :select-distinct-top])
          base (cond-> (select-query table [:*]) projection? (dissoc :select))]
      (plan (merge base transformation) nil))))

(defmethod clause :default [_ [operation]]
  (throw (ex-info "Unknown table clause" {:error :table/unknown-clause :clause operation})))

(defn query
  "Apply clause vectors to a table without I/O; a final :split returns tables.
  Invalid clauses throw ExceptionInfo before execution.

  [:honeysql clause-map] applies native HoneySQL clauses to the previous table.
  FROM is supplied by Garun; SELECT defaults to *. HoneySQL expressions retain
  their own identifier/NULL conventions and :raw forms are trusted SQL.

  Joins suffix every colliding right column when both plans have known names
  (in-memory sources, select/group-by and their descendants). For unknown file,
  SQL or named session schemas, explicit right :on keys are suffixed
  unconditionally; other right columns must have names disjoint from the left.
  Cross joins of unknown schemas likewise require disjoint names. DuckDB binds
  column names."
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

(defn- resolved-query [query tables]
  (walk/postwalk
   (fn [form]
     (if (and (vector? form) (= :garun/session-table (first form)))
       (or (get tables (second form))
           (throw (ex-info "Session table references require a context"
                           {:error :table/unresolved-session-table :name (second form)})))
       form))
   query))

(defn- with-resolved-query [^Context context table operation]
  (let [query (:query (from table))
        names (->> (tree-seq coll? seq query)
                   (filter #(and (vector? %) (= :garun/session-table (first %))))
                   (map second)
                   distinct
                   sort)
        ^Session owner (.session context)]
    ;; Hold all source resources through execution; deterministic ordering prevents
    ;; opposite join orders from acquiring multiple table locks in opposite orders.
    (letfn [(resolve-tables [names tables]
              (if-let [resource-name (first names)]
                (let [^Resource resource (.resolve owner resource-name Kind/TABLE)]
                  (.shared resource
                           (fn [^MaterializedTable materialized]
                             (resolve-tables
                              (next names)
                              (assoc tables resource-name
                                     (keyword (str (.-schema materialized) "." (.-table materialized))))))))
                (operation (resolved-query query tables))))]
      (resolve-tables names {}))))

(defn ->sql
  "Return [SQL & bound-parameters], in SQL placeholder order.
  The pure arity rejects named session sources with
  :table/unresolved-session-table; the context arity resolves their qualified
  SQL names. Both accept plans and source descriptions."
  ([table]
   (sql/format (resolved-query (:query (from table)) {}) {:quoted true}))
  ([^Context context table]
   (.call context
          (fn []
            (with-resolved-query context table #(sql/format % {:quoted true}))))))

(defn- prepare! ^PreparedStatement [^Connection connection [statement-sql & parameters]]
  (let [statement (.prepareStatement connection ^String statement-sql)]
    (try
      (doseq [[index value] (map-indexed vector parameters)]
        (.setObject statement (int (inc index)) value))
      statement
      (catch Throwable error
        (.close statement)
        (throw error)))))

(defn- with-result! [^Context context table operation]
  (with-resolved-query
    context table
    (fn [query]
      (with-open [statement (prepare! (.connection context) (sql/format query {:quoted true}))
                  result (.executeQuery statement)]
        (operation result)))))

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
  "Execute a table through its context and return keyword-keyed row maps.
  Optional :max-rows caps returned rows (zero returns no rows). The context
  retains its connection; JDBC resources close before returning. Invalid
  options throw ExceptionInfo; JDBC execution failures propagate."
  ([context table] (rows context table {}))
  ([^Context context table options]
   (.call context
          (fn []
            (options! :rows options [:max-rows])
            (let [max-rows (get options :max-rows Long/MAX_VALUE)]
              (when-not (and (integer? max-rows) (<= 0 max-rows Long/MAX_VALUE))
                (invalid! [:rows :max-rows] "Expected a nonnegative row cap" max-rows))
              (with-result! context table #(result-rows! % max-rows)))))))

(defn columns
  "Execute through the context and return an ordered keyword-to-vector map.
  The context retains its connection; JDBC execution failures propagate."
  [^Context context table]
  (.call context
         (fn []
           (with-result!
             context table
             (fn [^ResultSet result]
               (let [labels (result-labels (.getMetaData result))
                     values (result-rows! result Long/MAX_VALUE)]
                 (apply array-map (mapcat (fn [label] [label (mapv label values)]) labels))))))))

(def ^:private column-types
  {"BIGINT" :int64 "INTEGER" :int32 "DOUBLE" :float64 "FLOAT" :float32
   "VARCHAR" :string "BOOLEAN" :bool "DATE" :date "TIMESTAMP" :timestamp})

(defn- column-descriptor [^ResultSetMetaData metadata index]
  (let [type-name (.getColumnTypeName metadata (int index))]
    {:name (.getColumnLabel metadata (int index))
     :type (get column-types type-name (keyword (str/lower-case type-name)))}))

(defn schema
  "Return ordered {:name string :type keyword} DuckDB column descriptors.
  The context retains its connection; JDBC execution failures propagate."
  [^Context context table]
  (.call context
         (fn []
           (with-result!
             context (query table [:limit 0])
             (fn [^ResultSet result]
               (let [metadata (.getMetaData result)]
                 (mapv #(column-descriptor metadata %)
                       (range 1 (inc (.getColumnCount metadata))))))))))

(defn row-count
  "Execute COUNT(*) through the context and return a long.
  The context retains its connection; JDBC execution failures propagate."
  [^Context context table]
  (.call context
         (fn []
           (long (:n (first (rows context (query table [:group-by [] {:n [:count]}]))))))))

(defn- drop-table! [^Connection connection table-name]
  (with-open [statement (prepare! connection
                                  (sql/format {:drop-table [table-name]} {:quoted true}))]
    (.execute statement)))

(defn materialize!
  "Execute source once and register an immutable named table owned by the session.
  Return a pure plan referencing resource-name for further query composition.
  Options are {:replace? boolean}; duplicate names otherwise fail. CTAS holds
  shared locks on named source tables. Release/replacement drops tables on the
  calling context's connection; session close drops them under its write lock."
  ([context resource-name source]
   (materialize! context resource-name source {}))
  ([^Context context resource-name source options]
   (.call context
          (fn []
            (options! :materialize! options [:replace?])
            (when-not (keyword? resource-name)
              (invalid! [:materialize! :name] "Materialized tables require a keyword name" resource-name))
            (let [replace? (get options :replace? false)
                  ^Session owner (.session context)]
              (when-not (boolean? replace?)
                (invalid! [:materialize! :replace?] "Replace must be boolean" replace?))
              (let [table-name (.nextTableName owner)
                    schema-name (.schema owner)
                    qualified-name (keyword (str schema-name "." table-name))
                    connection (.connection context)
                    created-plan (plan {:select [:*] :from [[qualified-name]]} nil)]
                (with-resolved-query
                  context source
                  (fn [query]
                    (with-open [statement (prepare! connection
                                                    (sql/format (assoc query :create-table-as [qualified-name])
                                                                {:quoted true}))]
                      (.execute statement))))
                (try
                  (let [materialized (MaterializedTable. owner schema-name table-name
                                                         (schema context created-plan)
                                                         (long (row-count context created-plan)))]
                    (.register owner resource-name Kind/TABLE materialized true (boolean replace?) context)
                    (from resource-name))
                  (catch Throwable failure
                    (try
                      (drop-table! connection qualified-name)
                      (catch Throwable cleanup-failure
                        (.addSuppressed failure cleanup-failure)))
                    (throw failure)))))))))

(defn- copy! [^Context context table path format]
  (when (or (not (string? path)) (empty? path) (str/includes? path "\u0000"))
    (invalid! [:copy :path] "Output path must be nonempty text without NUL" path))
  (let [connection (.connection context)
        table (from table)
        table-name (keyword (str "__garun_copy_" (str/replace (str (UUID/randomUUID)) "-" "")))
        materialized (assoc-in table [:query :create-table-as] [:temp table-name])]
    (with-resolved-query
      context materialized
      (fn [query]
        (with-open [statement (prepare! connection (sql/format query {:quoted true}))]
          (.execute statement))))
    (try
      (with-open [statement (prepare! connection (sql/format {:garun/copy [table-name path format]}
                                                             {:quoted true}))]
        (.execute statement))
      nil
      (finally
        (drop-table! connection table-name)))))

(defn write-csv!
  "Write a table to CSV with a header through the context's DuckDB connection.
  COPY's temporary table is dropped before returning; the context retains
  its connection. Invalid paths throw ExceptionInfo; JDBC failures propagate."
  [^Context context table path]
  (.call context (fn [] (copy! context table path "CSV"))))

(defn write-parquet!
  "Write a table to parquet through the context's DuckDB connection.
  COPY's temporary table is dropped before returning; the context retains
  its connection. Invalid paths throw ExceptionInfo; JDBC failures propagate."
  [^Context context table path]
  (.call context (fn [] (copy! context table path "PARQUET"))))

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
  Column order follows column-selection and row order follows the table.
  The context retains its connection. NULL and nonnumeric cells throw
  :table/null-value and :table/non-numeric respectively. Invalid selections
  throw ExceptionInfo; JDBC failures propagate."
  [^Context context table column-selection]
  (.call context
         (fn []
           (when-not (and (vector? column-selection) (seq column-selection)
                          (every? keyword? column-selection))
             (invalid! [:matrix :cols] "Matrix requires a nonempty keyword column vector"
                       column-selection))
           (with-result!
             context (query table [:select column-selection])
             (fn [^ResultSet result]
               (let [indexed-columns (mapv vector (range 1 (inc (count column-selection)))
                                           column-selection)
                    ;; Realize each row while its cursor is current; no row maps or Java
                    ;; helper/build step are needed for this functional JDBC traversal.
                     values (sequence (comp (take-while (fn [_] (.next result)))
                                            (mapcat (fn [_] (matrix-row! result indexed-columns))))
                                      (range))
                     data (float-array values)]
                 {:rows (quot (alength data) (count column-selection))
                  :cols (count column-selection) :data data}))))))
