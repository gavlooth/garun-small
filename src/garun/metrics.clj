(ns garun.metrics
  "Pure vector metrics with argument order (metric y-pred y-true).

  Metrics follow sklearn semantics: zero division gives 0.0 for precision,
  recall and F1; R² uses finite scores for constant targets and returns NaN for
  fewer than two observations. Probability columns use integer class labels
  starting at zero."
  (:refer-clojure :exclude [precision]))

(set! *warn-on-reflection* true)

(defn- sample-count [y-pred y-true]
  (when-not (and (vector? y-pred)
                 (vector? y-true)
                 (pos? (count y-pred))
                 (= (count y-pred) (count y-true)))
    (throw (ex-info "Metrics require equally sized, nonempty vectors"
                    {:error :metrics/shape-mismatch
                     :predictions (count y-pred)
                     :targets (count y-true)})))
  (count y-pred))

(defn- finite-number? [x]
  (and (number? x) (Double/isFinite (double x))))

(defn- numeric-sample-count [y-pred y-true]
  (let [n (sample-count y-pred y-true)]
    (when-not (and (every? finite-number? y-pred)
                   (every? finite-number? y-true))
      (throw (ex-info "Regression metrics require finite numeric values"
                      {:error :metrics/non-numeric})))
    n))

(defn- validate-options [opts allowed]
  (when-not (map? opts)
    (throw (ex-info "Metric options must be a map"
                    {:error :metrics/invalid-options :options opts})))
  (let [unknown (into #{} (remove allowed) (keys opts))]
    (when (seq unknown)
      (throw (ex-info "Unknown metric options"
                      {:error :metrics/invalid-options :unknown unknown})))))

(defn- squared-error-sum [y-pred y-true]
  (reduce + 0.0
          (map (fn [prediction truth]
                 (let [difference (- (double prediction) (double truth))]
                   (* difference difference)))
               y-pred y-true)))

(defn accuracy
  "Return correct predictions / sample count for two nonempty label vectors.
  Throw ExceptionInfo if the vectors are empty or have different lengths."
  [y-pred y-true]
  (let [sample-size (sample-count y-pred y-true)
        correct-count (reduce + 0 (map #(if (= %1 %2) 1 0) y-pred y-true))]
    (/ (double correct-count) sample-size)))

(defn mse
  "Return sum of squared errors / sample count for two numeric vectors.
  Throw ExceptionInfo for invalid shapes or nonfinite, nonnumeric values."
  [y-pred y-true]
  (let [sample-size (numeric-sample-count y-pred y-true)]
    (/ (squared-error-sum y-pred y-true) sample-size)))

(defn mae
  "Return sum of absolute errors / sample count for two numeric vectors.
  Throw ExceptionInfo for invalid shapes or nonfinite, nonnumeric values."
  [y-pred y-true]
  (let [sample-size (numeric-sample-count y-pred y-true)
        absolute-error-sum (reduce + 0.0
                                   (map #(Math/abs (- (double %1) (double %2)))
                                        y-pred y-true))]
    (/ absolute-error-sum sample-size)))

(defn r2
  "Return 1 - residual sum of squares / target sum of squares for numeric vectors.
  Constant targets score 1.0 when perfect, else 0.0; one observation scores NaN.
  Throw ExceptionInfo for invalid shapes or nonfinite, nonnumeric values."
  [y-pred y-true]
  (let [sample-size (numeric-sample-count y-pred y-true)
        target-mean (/ (reduce + 0.0 y-true) sample-size)
        residual-sum-of-squares (squared-error-sum y-pred y-true)
        target-sum-of-squares (reduce + 0.0
                                      (map (fn [truth]
                                             (let [deviation (- (double truth) target-mean)]
                                               (* deviation deviation)))
                                           y-true))]
    (cond
      (< sample-size 2) Double/NaN
      (zero? residual-sum-of-squares) 1.0
      (zero? target-sum-of-squares) 0.0
      :else (- 1.0 (/ residual-sum-of-squares target-sum-of-squares)))))

(defn- confusion-matrix-labels [classes]
  (let [labels (if (and (integer? classes) (pos? classes))
                 (vec (range classes))
                 classes)]
    (when-not (and (vector? labels) (seq labels)
                   (= (count labels) (count (distinct labels))))
      (throw (ex-info "Classes must be a positive count or nonempty distinct label vector"
                      {:error :metrics/invalid-classes :classes classes})))
    labels))

(defn- increment-confusion-count [indices matrix [prediction truth]]
  (let [row (get indices truth)
        column (get indices prediction)]
    (if (and (some? row) (some? column))
      (update-in matrix [row column] inc)
      matrix)))

(defn confusion-matrix
  "Return class counts with true-class rows and predicted-class columns.
  Classes is an ordered vector of labels or a positive count for labels 0..n-1.
  Observations with either label outside the selected classes are omitted.
  Throw ExceptionInfo for invalid shapes or classes."
  [y-pred y-true classes]
  (sample-count y-pred y-true)
  (let [labels (confusion-matrix-labels classes)
        indices (zipmap labels (range))
        class-count (count labels)
        empty-matrix (vec (repeat class-count (vec (repeat class-count 0))))]
    (reduce (partial increment-confusion-count indices)
            empty-matrix
            (map vector y-pred y-true))))

(defn- ratio-or-zero [numerator denominator]
  (if (zero? denominator) 0.0 (/ (double numerator) denominator)))

(defn- classification-labels [y-pred y-true opts]
  (validate-options opts #{:average :positive})
  (let [{:keys [average positive] :or {average :binary positive 1}} opts
        labels (into #{} (concat y-pred y-true))]
    (when-not (#{:binary :macro} average)
      (throw (ex-info "Average must be :binary or :macro"
                      {:error :metrics/invalid-options :average average})))
    (when (and (= average :binary)
               (or (> (count labels) 2)
                   (and (= 2 (count labels)) (not (contains? labels positive)))))
      (throw (ex-info "Binary averaging requires at most two labels and a valid positive class"
                      {:error :metrics/invalid-classes :positive positive :classes labels})))
    (if (= average :binary) [positive] labels)))

(defn- count-true-positive [counts [prediction truth]]
  (if (= prediction truth)
    (update counts truth (fnil inc 0))
    counts))

(defn- counts-for-class [predicted-counts actual-counts true-positive-counts label]
  (let [true-positives (get true-positive-counts label 0)]
    {:true-positives true-positives
     :false-positives (- (get predicted-counts label 0) true-positives)
     :false-negatives (- (get actual-counts label 0) true-positives)}))

(defn- classification-counts [y-pred y-true opts]
  (sample-count y-pred y-true)
  (let [labels (classification-labels y-pred y-true opts)
        predicted-counts (frequencies y-pred)
        actual-counts (frequencies y-true)
        true-positive-counts (reduce count-true-positive {} (map vector y-pred y-true))]
    (mapv (partial counts-for-class predicted-counts actual-counts true-positive-counts)
          labels)))

(defn- classification-score [score-per-class y-pred y-true opts]
  (let [counts-per-class (classification-counts y-pred y-true opts)
        score-sum (reduce + 0.0 (map score-per-class counts-per-class))]
    (/ score-sum (count counts-per-class))))

(defn precision
  "Return TP / (TP + FP) for the positive class, or its macro average.
  Options: :average (:binary by default, or :macro), :positive (default 1).
  Zero division gives 0.0. Throw ExceptionInfo for invalid shapes, options or classes."
  ([y-pred y-true] (precision y-pred y-true {}))
  ([y-pred y-true opts]
   (classification-score
    (fn [{:keys [true-positives false-positives]}]
      (ratio-or-zero true-positives (+ true-positives false-positives)))
    y-pred y-true opts)))

(defn recall
  "Return TP / (TP + FN) for the positive class, or its macro average.
  Options: :average (:binary by default, or :macro), :positive (default 1).
  Zero division gives 0.0. Throw ExceptionInfo for invalid shapes, options or classes."
  ([y-pred y-true] (recall y-pred y-true {}))
  ([y-pred y-true opts]
   (classification-score
    (fn [{:keys [true-positives false-negatives]}]
      (ratio-or-zero true-positives (+ true-positives false-negatives)))
    y-pred y-true opts)))

(defn f1
  "Return 2TP / (2TP + FP + FN) for the positive class, or its macro average.
  Options: :average (:binary by default, or :macro), :positive (default 1).
  Macro F1 averages per-class F1, not precision and recall. Zero division gives 0.0.
  Throw ExceptionInfo for invalid shapes, options or classes."
  ([y-pred y-true] (f1 y-pred y-true {}))
  ([y-pred y-true opts]
   (classification-score
    (fn [{:keys [true-positives false-positives false-negatives]}]
      (ratio-or-zero (* 2 true-positives)
                     (+ (* 2 true-positives) false-positives false-negatives)))
    y-pred y-true opts)))

(defn- probability? [x]
  (and (finite-number? x) (<= 0.0 (double x) 1.0)))

(defn- validated-log-loss-input [y-proba y-true opts]
  (let [sample-size (sample-count y-proba y-true)
        _ (validate-options opts #{:epsilon})
        epsilon (get opts :epsilon 1e-15)
        matrix? (vector? (first y-proba))
        width (if matrix? (count (first y-proba)) 2)]
    (when-not (and (finite-number? epsilon) (< 0.0 (double epsilon) 0.5))
      (throw (ex-info "Epsilon must be finite and between zero and 0.5"
                      {:error :metrics/invalid-options :epsilon epsilon})))
    (when (and matrix?
               (or (< width 2)
                   (not-every? #(and (vector? %) (= width (count %))) y-proba)))
      (throw (ex-info "Probability rows must have the same width of at least two"
                      {:error :metrics/shape-mismatch})))
    (when-not (every? probability? (if matrix? (mapcat identity y-proba) y-proba))
      (throw (ex-info "Probabilities must be finite numbers between zero and one"
                      {:error :metrics/invalid-probability})))
    (when-not (every? #(and (integer? %) (<= 0 % (dec width))) y-true)
      (throw (ex-info "True classes must index probability columns"
                      {:error :metrics/invalid-classes :classes y-true :width width})))
    {:sample-size sample-size :matrix? matrix? :epsilon (double epsilon)}))

(defn- negative-log-likelihood [matrix? lower upper probabilities truth]
  (let [true-class-probability (cond
                                 matrix? (nth probabilities truth)
                                 (= truth 1) probabilities
                                 :else (- 1.0 (double probabilities)))
        clipped-probability (max lower (min upper (double true-class-probability)))]
    (- (Math/log clipped-probability))))

(defn log-loss
  "Return sum of -log(true-class probability) / sample count.
  Supply binary positive-class probabilities or equal-width probability vectors
  with integer labels indexing columns. :epsilon (default 1e-15) clips probabilities
  to [epsilon, 1-epsilon]; it must be finite and strictly between zero and 0.5.
  Throw ExceptionInfo for invalid shapes, probabilities, classes or options."
  ([y-proba y-true] (log-loss y-proba y-true {}))
  ([y-proba y-true opts]
   (let [{:keys [sample-size matrix? epsilon]} (validated-log-loss-input y-proba y-true opts)
         negative-log-likelihood-sum (reduce + 0.0
                                             (map (partial negative-log-likelihood
                                                           matrix? epsilon (- 1.0 epsilon))
                                                  y-proba y-true))]
     (/ negative-log-likelihood-sum sample-size))))
