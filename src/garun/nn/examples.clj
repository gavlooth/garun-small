(ns garun.nn.examples
  "Prepare table examples and align their labels, predictions, and metrics.

  DuckDB orders sequence rows. All feature groups, labels and window flags share
  one extraction, and training-only standardization statistics are measured in
  that same read. This namespace owns the persisted grouped input specification,
  window layouts, and manager-owned arrays/ArrayDatasets. It accepts
  Model/network data without depending on garun.nn; network lifecycles and
  training remain in that namespace."
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

(defn training-inputs
  "Resolve table training inputs against the network; training-statistics adds
  the statistics once the training rows are extracted. Classification targets
  cannot be standardized; serializable loss specs persist."
  [network options loss]
  (let [specification (input-specification network options)
        loss-tag (if (vector? loss) (first loss) loss)]
    (column-vector (:y specification) :nn/invalid-labels)
    (when (and (#{:ce :bce} loss-tag) (some #{:y} (:standardize specification)))
      (invalid :nn/invalid-standardization "Classification targets cannot be standardized" {}))
    (cond-> specification
      (or (keyword? loss) (vector? loss)) (assoc :loss loss))))

(defn- usable-statistics [columns measured]
  (mapv (fn [column {:keys [mean standard-deviation] :as statistics}]
          (when-not (and (Double/isFinite (double mean)) (Double/isFinite (double standard-deviation)))
            (invalid :nn/invalid-statistics
                     "Standardization requires nonempty finite numeric training columns"
                     {:column column :statistics statistics}))
          {:mean mean :standard-deviation (max 1.0e-9 (double standard-deviation))})
        columns measured))

(defn training-statistics
  "Add to specification the mean and population standard deviation of every
  column in the groups it standardizes, as measured while examples were
  extracted from the training source: over every training row, before sequence
  windows repeat rows. A column used by several groups gets the same values."
  [specification examples]
  (let [measured (:statistics examples)]
    (assoc specification :statistics
           (when (seq (:standardize specification))
             (into {}
                   (map (fn [group]
                          [group (if (= :x group)
                                   (mapv usable-statistics (:x specification) (:x measured))
                                   (usable-statistics (column-vector (:y specification) :nn/invalid-labels)
                                                      (:y measured)))]))
                   (:standardize specification))))))

(defn- window-rows
  "Copy, for each window, the length consecutive rows that end at its end row
  into one row-major array. A window is one contiguous block of the matrix."
  ^floats [matrix ^ints end-rows ^long length]
  (let [^floats data (:data matrix)
        width (long (:cols matrix))
        span (* length width)
        windows (float-array (* (alength end-rows) span))]
    (dotimes [window (alength end-rows)]
      (System/arraycopy data (int (* (- (inc (aget end-rows window)) length) width))
                        windows (int (* window span)) (int span)))
    windows))

(defn table-examples
  "Extract features and optional labels, one example per row or full sequence window.
  SQL marks each full window's end with an exact float32 0/1 flag; windows stay
  within partitions and labels come from the window's last row. One query extracts
  features, labels and flags together without coercing partition identities.
  :statistics holds every extracted column's mean and population standard
  deviation, per feature group under :x and for the labels under :y."
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
        ^ints window-end-rows (when sequence
                                (let [^floats flags (:data (last matrices))]
                                  (int-array (filter #(== 1.0 (aget flags (int %)))
                                                     (range (alength flags))))))
        row-count (if sequence (alength window-end-rows) (:rows (first features)))]
    {:rows row-count
     :features (mapv (fn [matrix]
                       (if sequence
                         {:rows row-count :cols (:cols matrix)
                          :shape [row-count length (:cols matrix)]
                          :data (window-rows matrix window-end-rows length)}
                         (assoc matrix :shape [row-count (:cols matrix)])))
                     features)
     :labels (when labels?
               (if sequence
                 {:rows row-count :cols (count label-columns)
                  :data (window-rows labels window-end-rows 1)}
                 labels))
     :statistics {:x (mapv :statistics features) :y (:statistics labels)}}))

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

(defn- tensor-values [^NDArray array]
  (let [shape (.getShape array)
        cols (if (= 2 (.dimension shape)) (.getLastDimension shape) 1)
        ^doubles data (if (= DataType/FLOAT32 (.getDataType array))
                        (let [^floats values (.toFloatArray array)
                              widened (double-array (alength values))]
                          (dotimes [index (alength values)]
                            (aset widened index (double (aget values index))))
                          widened)
                        (double-array (map double (.toArray array))))]
    {:rows (if (zero? cols) 0 (quot (alength data) cols)) :cols cols :data data}))

(defn prediction-values
  "Copy rank-2 predictions to host as {:rows :cols :data double-array}, row-major."
  [^NDArray array]
  (when-not (= 2 (.dimension (.getShape array)))
    (invalid :nn/non-tabular-output "Table predictions require a rank-2 output"
             {:shape (vec (.getShape (.getShape array)))}))
  (tensor-values array))

(defn label-values
  "Copy a singleton label NDList to host as a value matrix; labels of any other
  rank than 2 (sparse class indices) become one column."
  [^NDList labels]
  (tensor-values (.singletonOrThrow labels)))

(defn matrix-values
  "Widen an extracted float32 matrix, such as table labels, to a value matrix."
  [{:keys [rows cols] ^floats data :data}]
  (let [widened (double-array (alength data))]
    (dotimes [index (alength data)]
      (aset widened index (double (aget data index))))
    {:rows rows :cols cols :data widened}))

(defn join-values
  "Concatenate value matrices with equal column counts, such as one per batch."
  [chunks]
  (let [rows (long (reduce + 0 (map :rows chunks)))
        cols (long (:cols (first chunks) 0))
        joined (double-array (* rows cols))]
    (reduce (fn [offset {^doubles data :data}]
              (System/arraycopy data 0 joined (int offset) (alength data))
              (+ (long offset) (alength data)))
            0 chunks)
    {:rows rows :cols cols :data joined}))

(defn restore-targets
  "Return values in original target units: each column times its training
  standard deviation plus its training mean. Unchanged without statistics."
  [values statistics]
  (if (seq statistics)
    (let [^doubles data (:data values)
          cols (long (:cols values))
          means (double-array (map :mean statistics))
          deviations (double-array (map :standard-deviation statistics))
          restored (double-array (alength data))]
      (dotimes [index (alength data)]
        (let [column (rem index cols)]
          (aset restored index (+ (aget means column) (* (aget deviations column) (aget data index))))))
      (assoc values :data restored))
    values))

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

(defn value-rows
  "Return one entry per row of a value matrix: raw values, classes or
  probabilities. A single column gives scalars, several give vectors."
  [{:keys [rows cols] ^doubles data :data} output]
  (let [cols (long cols)]
    (if (and (= 1 cols) (= :raw output))
      (vec data)
      (mapv (fn [row]
              (output-value (mapv #(aget data (+ (* (long row) cols) (long %))) (range cols)) output))
            (range rows)))))

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
  "Compute requested metrics from prediction and label value matrices in the
  same units. Regression metrics read every value; classification metrics read
  predicted classes or probabilities against integer labels; custom functions
  receive one scalar or vector per row. Each form is built once and only when
  a requested metric needs it."
  [requested predictions truth]
  (let [all-predictions (delay (vec (:data predictions)))
        all-truth (delay (vec (:data truth)))
        classes (delay (value-rows predictions :class))
        probabilities (delay (value-rows predictions :probability))
        class-labels (delay (mapv long (:data truth)))
        prediction-rows (delay (value-rows predictions :raw))
        truth-rows (delay (value-rows truth :raw))]
    (into {}
          (map-indexed
           (fn [index metric]
             (let [named (get metric-functions metric)
                   value (cond
                           (fn? metric) (metric @prediction-rows @truth-rows)
                           (= :log-loss metric) (named @probabilities @class-labels)
                           (and (#{:precision :recall :f1} metric)
                                (> (count (into #{} (concat @classes @class-labels))) 2))
                           (named @classes @class-labels {:average :macro})
                           (#{:accuracy :precision :recall :f1} metric) (named @classes @class-labels)
                           :else (named @all-predictions @all-truth))]
               (when-not (number? value)
                 (invalid :nn/invalid-metric "Metric functions must return a number" {:metric metric :value value}))
               [(if (keyword? metric) metric (keyword (str "metric-" index))) value])))
          requested)))
