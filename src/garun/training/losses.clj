(ns garun.training.losses
  "Engine-portable repairs for the named training loss DSL.
  Constructors receive the parent factory's validated options and return native
  Loss instances. All math stays on the input engine and its autograd graph."
  (:import [ai.djl.ndarray NDArray NDArrays NDList NDManager]
           [ai.djl.ndarray.types DataType Shape]
           [ai.djl.training.loss Loss]))

(set! *warn-on-reflection* true)

(def ^:private ^Double epsilon (Double/valueOf 1.0e-12))

(def ^:private ^Double normalizer (Double/valueOf (double (+ 1.0 epsilon))))

(def ^:private ^Double one (Double/valueOf 1.0))

(defn binary-cross-entropy
  "Return mean binary cross entropy, scaled by weight.
  Raw logits use DJL's stable relu/softplus implementation unchanged. Probability
  inputs use -y*log((p+eps)/(1+eps))-(1-y)*log((1-p+eps)/(1+eps)), eps=1e-12.
  This normalized endpoint regularization keeps both terms nonnegative, finite,
  and differentiable at p=0 and p=1 without clipping away corrective gradients.
  Half-precision probabilities are evaluated in float32 to retain epsilon."
  ^Loss [name weight from-sigmoid?]
  (if-not from-sigmoid?
    (Loss/sigmoidBinaryCrossEntropyLoss name (float weight) false)
    (let [native-weight (Float/valueOf (float weight))]
      (proxy [Loss] [name]
        (evaluate [^NDList labels ^NDList predictions]
          (let [^NDArray input (.singletonOrThrow predictions)
                ^NDArray probability (if (= DataType/FLOAT64 (.getDataType input))
                                       input
                                       (.toType input DataType/FLOAT32 false))
                ^NDArray label (-> (.singletonOrThrow labels)
                                   (.reshape (.getShape probability))
                                   (.toType (.getDataType probability) false))
                ^NDArray positive-log (-> probability (.add epsilon) (.div normalizer) .log)
                ^NDArray negative-log (-> (NDArrays/sub one probability)
                                          (.add epsilon) (.div normalizer) .log)
                ^NDArray terms (.add (.mul positive-log label)
                                     (.mul negative-log (NDArrays/sub one label)))]
            (-> terms .neg .mean (.mul native-weight))))))))

(defn- masked-values
  "Replace padding before nonlinear math or indexing. PyTorch's where requires
  an explicitly broadcast condition; the replacement is only a scalar tensor."
  ^NDArray [^NDArray values ^NDArray mask]
  (let [^NDManager manager (.getManager values)
        ^NDArray zero (.zeros manager (Shape. (long-array []))
                              (.getDataType values) (.getDevice values))]
    (NDArrays/where (.broadcast mask (.getShape values)) values zero)))

(defn- masked-sequence-loss
  ^NDArray [^NDList labels ^NDList predictions ^Number weight axis sparse? from-logits?]
  (when-not (= 2 (.size labels))
    (throw (IllegalArgumentException. "Masked cross entropy needs labels [classes valid-lengths]")))
  (let [^NDArray input (.singletonOrThrow predictions)
        ^Shape input-shape (.getShape input)
        rank (.dimension input-shape)]
    (when (< rank 3)
      (throw (IllegalArgumentException. "Masked cross entropy needs batch, sequence, and class dimensions")))
    (let [class-axis (Math/floorMod (int axis) (int rank))
          ;; Keep batch/sequence order while moving the configurable class axis
          ;; last. Default-axis outputs retain DJL's trailing singleton class dim.
          permutation (int-array (concat (remove #{class-axis} (range rank)) [class-axis]))
          ^NDArray prediction (if (= class-axis (dec rank)) input (.transpose input permutation))
          ^Shape prediction-shape (.getShape prediction)
          ^Shape token-shape (.slice prediction-shape 0 (int (dec rank)))
          ^NDArray lengths (.get labels (int 1))
          ^NDManager manager (.getManager prediction)
          ^NDArray positions (-> (.arange manager (int 0)
                                          (Math/toIntExact (.get prediction-shape (int 1)))
                                          (int 1) DataType/INT64)
                                 (.toDevice (.getDevice prediction) false)
                                 (.reshape (long-array (concat [1 (.get prediction-shape (int 1))]
                                                               (repeat (- rank 2) 1)))))
          ^NDArray sequence-lengths (.reshape lengths
                                              (long-array (cons (.get prediction-shape (int 0))
                                                                (repeat (dec rank) 1))))
          ^NDArray mask (.lt positions sequence-lengths)
          ^NDArray safe-prediction (masked-values prediction mask)
          ^NDArray log-probability (if from-logits?
                                     (.logSoftmax safe-prediction (int -1))
                                     safe-prediction)
          ^NDArray label (.head labels)
          ^NDArray token-loss
          (if sparse?
            (let [^NDArray indices (-> label (.reshape token-shape)
                                       (.toType DataType/INT64 false) (.expandDims (int -1)))
                  ^NDArray safe-indices (masked-values indices mask)]
              (.neg (.gather log-probability safe-indices (int -1))))
            (let [^NDArray dense (.reshape label input-shape)
                  ^NDArray aligned (if (= class-axis (dec rank)) dense (.transpose dense permutation))
                  ^NDArray safe-label (masked-values aligned mask)]
              (-> log-probability (.mul safe-label) .neg (.sum (int-array [-1]) true))))]
      ;; Average over padded width, NOT valid lengths, including empty sequences.
      (-> (masked-values token-loss mask) (.mean (int-array [1])) (.mul weight)))))

(defn masked-cross-entropy
  "Return per-sequence masked cross entropy, scaled by weight.
  Labels are NDList[classes, valid-lengths], predictions NDList[input]. Sparse
  labels omit the class axis; dense labels follow the prediction layout. After
  omitting the class axis, the first axes are batch and padded sequence width.
  :from-logits? true applies logSoftmax; false consumes log probabilities.
  Padding is replaced before indexing and math, so padded sparse indices need
  not be valid classes and padded values have exactly zero gradient. Reduction
  retains DJL's singleton class dimension and divides by padded sequence width,
  not valid length. Zero valid lengths produce zero, without division by zero."
  ^Loss [name weight axis sparse? from-logits?]
  (let [native-weight (Float/valueOf (float weight))]
    (proxy [Loss] [name]
      (evaluate [^NDList labels ^NDList predictions]
        (masked-sequence-loss labels predictions native-weight axis sparse? from-logits?)))))
