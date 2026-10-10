(ns garun.nn.examples
  "Prepare table examples and align their labels, predictions, and metrics.

  DuckDB computes training-only standardization statistics and orders sequence
  rows. All feature groups, labels and window flags share one extraction. This
  namespace owns the persisted grouped input specification, window layouts,
  and manager-owned arrays/ArrayDatasets. It accepts Model/network data without
  depending on garun.nn; network lifecycles and training remain in that namespace."
  (:require [clojure.edn :as edn]
            [garun.metrics :as metrics]
            [garun.table :as table]
            [garun.tensor :as tensor])
  (:import [ai.djl Model]
           [ai.djl.ndarray NDArray NDList NDManager]
           [ai.djl.ndarray.types DataType Shape]
           [ai.djl.training.dataset ArrayDataset$Builder BatchSampler
            RandomAccessDataset Sampler$SubSampler]
           [java.util List Random UUID]))

(set! *warn-on-reflection* true)

(defn- invalid [error message data]
  (throw (ex-info message (assoc data :error error))))

(def ^:private inputs-property "garun.inputs")

(defn stored-inputs
  "Read the Model's persisted table input specification, or return nil."
  [^Model model]
  (when-let [specification (.getProperty model inputs-property)]
    (edn/read-string specification)))

(defn store-inputs!
  "Persist a prepared table specification alongside the Model's parameters."
  [^Model model specification]
  (.setProperty model inputs-property (pr-str specification)))

(defn- column-vector [columns error]
  (let [columns (if (keyword? columns) [columns] columns)]
    (when-not (and (vector? columns) (seq columns) (every? keyword? columns))
      (invalid error "Columns must be a keyword or nonempty vector of keywords" {:columns columns}))
    columns))

(defn- input-groups [network columns]
  (let [spec (:spec network)
        graph-inputs (when (and (vector? spec) (= :graph (first spec)))
                       (:inputs (second spec)))
        groups (cond
                 (map? columns)
                 (do
                   (when-not graph-inputs
                     (invalid :nn/invalid-features "Named feature groups require a graph network"
                              {:x columns}))
                   (when-not (= (set graph-inputs) (set (keys columns)))
                     (invalid :nn/invalid-features "Named feature groups must match the graph inputs"
                              {:x columns :inputs graph-inputs}))
                   (mapv columns graph-inputs))

                 (and (vector? columns) (every? keyword? columns)) [columns]
                 (vector? columns) columns
                 :else nil)]
    (when-not (and (seq groups)
                   (every? #(and (vector? %) (seq %) (every? keyword? %)) groups)
                   (= (count groups) (count (:input-shapes network))))
      (invalid :nn/invalid-features "Features require one nonempty keyword vector per network input"
               {:x columns :input-shapes (:input-shapes network)}))
    groups))

(defn- input-specification [network options]
  (let [{:keys [sequence standardize] :as specification}
        (merge (stored-inputs (:model network)) (select-keys options [:x :y :sequence :standardize]))
        groups (input-groups network (:x specification))
        _ (when (some? sequence)
            (when-not (map? sequence)
              (invalid :nn/invalid-options "Options must be a map" {:context :sequence :options sequence}))
            (when-let [unknown (seq (remove #{:length :partition-by :order-by} (keys sequence)))]
              (invalid :nn/unknown-option "Unknown neural network option"
                       {:context :sequence :keys (vec unknown)}))
            (when-not (and (integer? (:length sequence))
                           (pos? (:length sequence))
                           (<= (:length sequence) Integer/MAX_VALUE)
                           (keyword? (:partition-by sequence)))
              (invalid :nn/invalid-sequence "Sequence requires positive :length and a partition column"
                       {:sequence sequence}))
            (column-vector (:order-by sequence) :nn/invalid-sequence))
        expected-shapes (mapv #(if sequence [(:length sequence) (count %)] [(count %)]) groups)
        input-shapes (mapv #(vec (rest %)) (:input-shapes network))]
    (when-not (= (mapv count expected-shapes) (mapv count input-shapes))
      (invalid :nn/non-tabular-input "Table input ranks must match the input specification"
               {:input-shapes (:input-shapes network) :sequence sequence}))
    (when-not (= DataType/FLOAT32 (:dtype network))
      (invalid :nn/non-tabular-dtype "Table operations require a float32 network" {:dtype (:dtype network)}))
    (when-not (= expected-shapes input-shapes)
      (invalid :nn/invalid-features "Features/sequence must match the network input shapes"
               {:x groups :sequence sequence :input-shapes (:input-shapes network)}))
    (when-not (or (nil? standardize)
                  (and (vector? standardize) (every? #{:x :y} standardize)
                       (= (count standardize) (count (distinct standardize)))))
      (invalid :nn/invalid-standardization ":standardize must be a subset of [:x :y]"
               {:standardize standardize}))
    (assoc specification :x groups)))

(defn inference-inputs
  "Resolve table inputs against network shape/dtype and stored training statistics."
  [network options]
  (let [specification (input-specification network options)
        requested (:standardize specification)
        statistics (select-keys (:statistics specification) requested)]
    (when-not (every? #(seq (get statistics %)) requested)
      (invalid :nn/missing-statistics "Inference standardization requires stored training statistics"
               {:standardize requested}))
    (assoc specification :statistics statistics)))

(defn- training-statistics [context source specification]
  (let [groups (into {}
                     (map (fn [group]
                            [group (if (= group :x)
                                     (vec (mapcat identity (:x specification)))
                                     (column-vector (:y specification) :nn/invalid-labels))]))
                     (:standardize specification))
        columns (vec (distinct (mapcat identity (vals groups))))
        aliases (mapv (fn [index column]
                        {:column column :mean (keyword (str "mean_" index))
                         :standard-deviation (keyword (str "standard_deviation_" index))})
                      (range) columns)]
    (when (seq columns)
      (let [aggregates (into {} (mapcat (fn [{:keys [column mean standard-deviation]}]
                                          [[mean [:avg column]] [standard-deviation [:stddev-pop column]]]))
                             aliases)
            statistics (first (table/rows context (table/query source [:group-by [] aggregates])))
            per-column (into {} (map (fn [{:keys [column mean standard-deviation]}]
                                       (let [average (get statistics mean)
                                             deviation (get statistics standard-deviation)]
                                         (when-not (and (number? average) (number? deviation)
                                                        (Double/isFinite (double average))
                                                        (Double/isFinite (double deviation)))
                                           (invalid :nn/invalid-statistics
                                                    "Standardization requires nonempty finite numeric training columns"
                                                    {:column column :statistics statistics}))
                                         [column {:mean (double average)
                                                  :standard-deviation (max 1.0e-9 (double deviation))}])))
                             aliases)]
        (into {}
              (map (fn [[group group-columns]]
                     [group (if (= group :x)
                              (mapv #(mapv per-column %) (:x specification))
                              (mapv per-column group-columns))]))
              groups)))))

(defn training-inputs
  "Resolve table training inputs and compute fresh statistics from training alone.
  Classification targets cannot be standardized; serializable loss specs persist."
  [context network source options loss]
  (let [specification (input-specification network options)
        loss-tag (if (vector? loss) (first loss) loss)]
    (column-vector (:y specification) :nn/invalid-labels)
    (when (and (#{:ce :bce} loss-tag) (some #{:y} (:standardize specification)))
      (invalid :nn/invalid-standardization "Classification targets cannot be standardized" {}))
    (cond-> (assoc specification :statistics (training-statistics context source specification))
      (or (keyword? loss) (vector? loss)) (assoc :loss loss))))

(defn- matrix-row [matrix row]
  (let [^floats values (:data matrix)
        width (:cols matrix)
        offset (* row width)]
    (map #(aget values (int (+ offset %))) (range width))))

(defn table-examples
  "Extract features and optional labels, one example per row or full sequence window.
  SQL marks each full window's end with an exact float32 0/1 flag; windows stay
  within partitions and labels come from the window's last row. One query extracts
  features, labels and flags together without coercing partition identities."
  [context source specification labels?]
  (let [input-groups (:x specification)
        label-columns (when labels? (column-vector (:y specification) :nn/invalid-labels))
        {:keys [length partition-by order-by] :as sequence} (:sequence specification)
        ;; Only an exact 0/1 window-end flag enters float32 storage, never a partition ID.
        window-end-column (when sequence (keyword (str "__garun_window_end_" (UUID/randomUUID))))
        source (if sequence
                 (table/query source
                              [[:derive {window-end-column
                                         [:cast
                                          [:>= [:over [:row-number]
                                                {:partition-by [partition-by]
                                                 :order-by (column-vector order-by :nn/invalid-sequence)}]
                                           length]
                                          :int32]}]
                               [:order-by (into [partition-by]
                                                (column-vector order-by :nn/invalid-sequence))]])
                 source)
        groups (cond-> input-groups labels? (conj label-columns) sequence (conj [window-end-column]))
        matrices (table/matrices context source groups)
        features (subvec matrices 0 (count input-groups))
        labels (when labels? (nth matrices (count input-groups)))
        window-ends (when sequence (last matrices))
        windows (when sequence
                  (into []
                        (comp
                         (filter #(== 1.0 (aget ^floats (:data window-ends) (int %))))
                         (map #(range (inc (- % length)) (inc %))))
                        (range (:rows (first features)))))
        row-count (if sequence (count windows) (:rows (first features)))]
    {:rows row-count
     :features (mapv (fn [matrix]
                       (if sequence
                         {:rows row-count :cols (:cols matrix)
                          :shape [row-count length (:cols matrix)]
                          :data (float-array
                                 (mapcat (fn [window] (mapcat #(matrix-row matrix %) window)) windows))}
                         (assoc matrix :shape [row-count (:cols matrix)])))
                     features)
     :labels (when labels?
               (if sequence
                 {:rows row-count :cols (count label-columns)
                  :data (float-array (mapcat #(matrix-row labels (last %)) windows))}
                 labels))}))

(defn label-layout!
  "Check nonempty table examples against the network output shape and loss labels."
  [network examples loss]
  (let [^Model model (:model network)
        outputs (.getOutputShapes (.getBlock model)
                                  (into-array Shape (map tensor/shape (:input-shapes network))))
        labels (:labels examples)
        ^floats label-data (:data labels)
        _ (when-not (and (= 1 (count outputs)) (= 2 (.dimension ^Shape (first outputs))))
            (invalid :nn/non-tabular-output "Table training requires one rank-2 output"
                     {:output-shapes (mapv #(vec (.getShape ^Shape %)) outputs)}))
        width (.getLastDimension ^Shape (first outputs))]
    (when-not (pos? (:rows examples))
      (invalid :nn/empty-data "Training/evaluation requires at least one full example" {}))
    (when-not (if (= loss :ce) (= 1 (:cols labels)) (= width (:cols labels)))
      (invalid :nn/invalid-labels "Label width does not match output/loss"
               {:label-width (:cols labels) :output-width width :loss loss}))
    (when (and (= loss :ce)
               (not-every? #(and (<= 0 %) (< % width) (== (double %) (Math/floor (double %)))) label-data))
      (invalid :nn/invalid-labels "CE labels must be integral class indices within the output width"
               {:output-width width}))
    (when (and (= loss :bce) (not-every? #(<= 0 % 1) label-data))
      (invalid :nn/invalid-labels "BCE labels must be in [0, 1]" {}))))

(defn standardized-array
  "Create a manager-owned matrix/window tensor, optionally standardized or sparse."
  [^NDManager manager matrix statistics sparse?]
  (let [^NDArray array (if sparse?
                         (.create manager (long-array (map long (:data matrix)))
                                  (tensor/shape [(:rows matrix)]))
                         (.create manager ^floats (:data matrix)
                                  (tensor/shape (or (:shape matrix) [(:rows matrix) (:cols matrix)]))))]
    (if (seq statistics)
      (let [means (.create manager (float-array (map :mean statistics)))
            deviations (.create manager (float-array (map :standard-deviation statistics)))]
        (.div (.sub array means) deviations))
      array)))

(defn feature-arrays
  "Create one manager-owned input tensor per aligned feature group."
  [^NDManager manager examples specification]
  (let [statistics (get-in specification [:statistics :x])]
    (mapv (fn [index matrix]
            (standardized-array manager matrix (get statistics index) false))
          (range) (:features examples))))

(defn- shuffle-indices! [indices ^Random random]
  ;; RNG is the mutable SDK boundary; the permutation is a persistent reduction.
  (reduce (fn [order index]
            (let [shuffled-index (.nextInt random (int (inc index)))]
              (assoc order index (nth order shuffled-index) shuffled-index (nth order index))))
          indices (range (dec (count indices)) 0 -1)))

(defn array-dataset
  "Build a manager-owned ArrayDataset with reproducible sampling and scaled labels."
  [^NDManager manager examples specification options shuffle? loss]
  (let [statistics (:statistics specification)
        features (feature-arrays manager examples specification)
        labels (standardized-array manager (:labels examples) (:y statistics) (= loss :ce))
        random (Random. (long (:seed options)))
        sampler (reify Sampler$SubSampler
                  (sample [_ dataset]
                    (let [indices (vec (range (.size ^RandomAccessDataset dataset)))
                          ^List order (if shuffle? (shuffle-indices! indices random) indices)]
                      (.iterator order))))
        ^ArrayDataset$Builder builder (ArrayDataset$Builder.)]
    (.setData builder (into-array NDArray features))
    (.optLabels builder (into-array NDArray [labels]))
    (.setSampling builder (BatchSampler. sampler (int (:batch options)) false))
    (.build builder)))

(defn inverse-targets
  "Restore scalar or vector target rows from training-scale to original units."
  [rows statistics]
  (if (seq statistics)
    (mapv (fn [row]
            (let [values (if (vector? row) row [row])
                  restored (mapv (fn [value {:keys [mean standard-deviation]}]
                                   (+ mean (* standard-deviation value)))
                                 values statistics)]
              (if (= 1 (count restored)) (first restored) restored)))
          rows)
    rows))

(defn- argmax [values]
  (let [maximum (reduce max values)]
    ;; Keep the first maximum, including when two logits tie.
    (first (keep-indexed (fn [index value] (when (= value maximum) index)) values))))

(defn- sigmoid [value]
  ;; The negative branch avoids overflowing exp for large negative logits.
  (if (neg? value)
    (let [exponential (Math/exp value)]
      (/ exponential (+ 1.0 exponential)))
    (/ 1.0 (+ 1.0 (Math/exp (- value))))))

(defn- softmax [values]
  (let [maximum (reduce max values)
        exponentials (mapv #(Math/exp (- % maximum)) values)
        total (reduce + exponentials)]
    (mapv #(/ % total) exponentials)))

(defn- output-value [values output]
  (let [single? (= 1 (count values))
        value (first values)]
    (cond
      (= output :raw) (if single? value values)
      (and (= output :class) single?) (if (pos? value) 1 0)
      (= output :class) (argmax values)
      single? (sigmoid value)
      :else (softmax values))))

(defn output-rows
  "Read rank-2 predictions as scalar/vector rows, raw, classes, or probabilities."
  [^NDArray array output]
  (when-not (= 2 (.dimension (.getShape array)))
    (invalid :nn/non-tabular-output "Table predictions require a rank-2 output"
             {:shape (vec (.getShape (.getShape array)))}))
  (let [width (.getLastDimension (.getShape array))
        values (if (= DataType/FLOAT32 (.getDataType array)) (.toFloatArray array) (.toArray array))
        rows (partition (int width) values)]
    (mapv #(output-value (mapv double %) output) rows)))

(defn label-rows
  "Read a singleton label NDList as matrix rows or sparse class labels."
  [^NDList labels]
  (let [^NDArray array (.singletonOrThrow labels)
        dimensions (.dimension (.getShape array))]
    (if (= 2 dimensions)
      (output-rows array :raw)
      (mapv double (.toArray array)))))

(defn target-rows
  "Read extracted table labels in example/window order and original target units."
  [examples]
  (mapv (fn [index]
          (let [row (mapv double (matrix-row (:labels examples) index))]
            (if (= 1 (count row)) (first row) row)))
        (range (:rows examples))))

(def ^:private metric-functions
  {:mse metrics/mse :mae metrics/mae :r2 metrics/r2 :accuracy metrics/accuracy
   :precision metrics/precision :recall metrics/recall :f1 metrics/f1 :log-loss metrics/log-loss})

(defn metric-options
  "Check requested metric keywords/functions, preserving their configured order."
  [requested]
  (when-not (and (vector? requested) (every? #(or (contains? metric-functions %) (fn? %)) requested))
    (invalid :nn/invalid-metrics ":metrics must contain metric keywords or functions" {:metrics requested}))
  requested)

(defn metric-results
  "Align prediction/label rows for named metrics; custom metrics receive raw rows."
  [requested predictions truth]
  (into {}
        (map-indexed
         (fn [index metric]
           (let [classification? (#{:accuracy :precision :recall :f1 :log-loss} metric)
                 values (cond
                          (= metric :log-loss) (mapv #(output-value (if (vector? %) % [%]) :probability) predictions)
                          classification? (mapv #(output-value (if (vector? %) % [%]) :class) predictions)
                          :else (vec (mapcat #(if (vector? %) % [%]) predictions)))
                 targets (if classification? (mapv long truth)
                             (vec (mapcat #(if (vector? %) % [%]) truth)))
                 value (cond
                         (fn? metric) (metric predictions truth)
                         (and (#{:precision :recall :f1} metric)
                              (> (count (into #{} (concat values targets))) 2))
                         ((get metric-functions metric) values targets {:average :macro})
                         :else ((get metric-functions metric) values targets))]
             (when-not (number? value)
               (invalid :nn/invalid-metric "Metric functions must return a number" {:metric metric :value value}))
             [(if (keyword? metric) metric (keyword (str "metric-" index))) value])))
        requested))
