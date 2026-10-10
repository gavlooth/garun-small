(ns garun.blocks
  "Readable DJL block specifications. Native Blocks and zero-argument factories
  are escape hatches for any core/custom DJL builder. Native blocks are not cloned:
  use a factory when separate networks need separate parameters and ownership.
  Lambda functions receive/return NDList, or NDArray with :singleton? true.
  Custom combinations receive child NDLists and the original NDList. Their shape
  must be value-independent; :shape-fn can describe dtype/value-sensitive outputs
  without allocating probe tensors. Caller functions must preserve input ownership.
  [:index \"DJL NDIndex string\"] borrows a singleton input and applies DJL indexing,
  including slices, steps, negative indices and ellipses, without rewriting the string."
  (:import [ai.djl.ndarray NDArray NDArrays NDList]
           [ai.djl.ndarray.types Shape SparseFormat]
           [ai.djl.nn Activation Block Blocks LambdaBlock SequentialBlock]
           [ai.djl.nn.convolutional Conv1d Conv1d$Builder Conv2d Conv2d$Builder
            Conv3d Conv3d$Builder Conv1dTranspose Conv2dTranspose
            Convolution$ConvolutionBuilder Deconvolution$DeconvolutionBuilder]
           [ai.djl.nn.core ConstantEmbedding Linear]
           [ai.djl.nn.norm BatchNorm BatchNorm$BaseBuilder Dropout GhostBatchNorm
            GhostBatchNorm$Builder LayerNorm]
           [ai.djl.nn.pooling Pool]
           [ai.djl.nn.recurrent GRU GRU$Builder LSTM LSTM$Builder RNN RNN$Builder
            RNN$Activation RecurrentBlock$BaseBuilder]
           [ai.djl.nn.transformer BertBlock BertBlock$Builder BertNextSentenceBlock
            BertPretrainingBlock IdEmbedding$Builder ScaledDotProductAttentionBlock]
           [ai.djl.modality.nlp DefaultVocabulary Vocabulary]
           [ai.djl.modality.nlp.embedding TrainableWordEmbedding TrainableWordEmbedding$Builder]
           [garun.djl FnBlock MaskedTransformerEncoder TransposeConvolution]
           [java.util Collection List]
           [java.util.function Function]))

(set! *warn-on-reflection* true)

(defn- invalid [error message data]
  (throw (ex-info message (assoc data :error error))))

(defn- options [opts allowed context]
  (when-not (map? opts)
    (invalid :nn/invalid-options "Options must be a map" {:context context :options opts}))
  (when-let [unknown (seq (remove allowed (keys opts)))]
    (invalid :nn/unknown-option "Unknown neural network option"
             {:context context :keys (vec unknown)}))
  opts)

(defn- one-options [args allowed tag]
  (when-not (= 1 (count args))
    (invalid :nn/invalid-spec "Block requires one options map" {:tag tag :args args}))
  (options (first args) allowed tag))

(defn- positive-int? [x]
  (and (integer? x) (pos? x) (<= x Integer/MAX_VALUE)))

(defn- finite-number? [x]
  (and (number? x) (Double/isFinite (double x))))

(defn- validate-options [opts predicates tag]
  (doseq [[key predicate] predicates
          :when (contains? opts key)]
    (when-not (predicate (get opts key))
      (invalid :nn/invalid-spec "Invalid block option" {:tag tag :key key :value (get opts key)})))
  opts)

(defn- probability? [x]
  (and (finite-number? x) (<= 0 x) (< x 1)))

(defn- positive-number? [x]
  (and (finite-number? x) (pos? x)))

(defn- dimensions [dims rank zero? tag]
  (when-not (and (vector? dims) (= rank (count dims))
                 (every? #(and (integer? %) (if zero? (<= 0 %) (pos? %))) dims))
    (invalid :nn/invalid-spec "Invalid spatial dimensions" {:tag tag :dimensions dims :rank rank}))
  (Shape. (long-array dims)))

(defn- sam
  "An explicit SAM bridge: a Clojure IFn is not assumed to implement Function."
  ^Function [f]
  (reify Function
    (apply [_ input] (f input))))

(defn- activation-function [tag]
  (case tag
    :relu (fn [^NDList inputs] (Activation/relu inputs))
    :relu6 (fn [^NDList inputs] (Activation/relu6 inputs))
    :tanh (fn [^NDList inputs] (Activation/tanh inputs))
    :sigmoid (fn [^NDList inputs] (Activation/sigmoid inputs))
    :gelu (fn [^NDList inputs] (Activation/gelu inputs))
    :selu (fn [^NDList inputs] (Activation/selu inputs))
    :mish (fn [^NDList inputs] (Activation/mish inputs))
    :softplus (fn [^NDList inputs] (Activation/softPlus inputs))
    :softsign (fn [^NDList inputs] (Activation/softSign inputs))
    :identity identity
    (invalid :nn/invalid-activation "Unknown activation" {:activation tag})))

(defn- activation [tag]
  (case tag
    :relu (Activation/reluBlock)
    :relu6 (Activation/relu6Block)
    :tanh (Activation/tanhBlock)
    :sigmoid (Activation/sigmoidBlock)
    :gelu (Activation/geluBlock)
    :selu (Activation/seluBlock)
    :mish (Activation/mishBlock)
    :softplus (Activation/softPlusBlock)
    :softsign (Activation/softSignBlock)
    :prelu (Activation/preluBlock)
    :identity (Blocks/identityBlock)
    (invalid :nn/invalid-activation "Unknown activation" {:activation tag})))

(declare compile-block)

(defn- sequential! [children]
  (.addAll (SequentialBlock.) ^Collection (mapv compile-block children)))

(defn- combine [mode axis]
  (fn [^List outputs ^NDList _input]
    (let [^NDList first-output (.get outputs 0)
          output-count (.size first-output)
          _ (when-not (every? (fn [^NDList output] (= output-count (.size output))) outputs)
              (invalid :nn/invalid-combination "Combination branches must have equal output counts"
                       {:output-counts (mapv (fn [^NDList output] (.size output)) outputs)}))
          arrays (mapv (fn [index]
                         (case mode
                           :concat (NDArrays/concat
                                    (NDList. ^Collection
                                     (mapv (fn [^NDList output] (.get output (int index))) outputs))
                                    (int axis))
                           :add (reduce (fn [^NDArray sum ^NDList output]
                                          (.add sum ^NDArray (.get output (int index))))
                                        (.get first-output (int index)) (rest outputs))))
                       (range output-count))]
      (NDList. ^Collection arrays))))

(defn- combination! [tag args spec]
  (let [opts (options (first args) #{:combine :axis :shape-fn} tag)
        mode (:combine opts)
        axis (get opts :axis -1)
        shape-fn (:shape-fn opts)
        children (rest args)]
    (when-not (and (or (#{:concat :add} mode) (fn? mode)) (seq children)
                   (integer? axis) (or (nil? shape-fn) (fn? shape-fn))
                   (or (not (contains? opts :axis)) (= mode :concat))
                   (or (nil? shape-fn) (fn? mode)))
      (invalid :nn/invalid-combination
               "Combination requires :add, :concat or a function and children; :axis is for concat"
               {:spec spec}))
    (FnBlock. (name tag)
              ^List (if (= tag :skip)
                      [(sequential! children) (Blocks/identityBlock)]
                      (mapv compile-block children))
              (if (fn? mode) mode (combine mode axis))
              (when (keyword? mode) (name mode)) (int axis)
              (when shape-fn
                (fn [child-shapes input-shapes]
                  (into-array Shape
                              (map (fn [dims]
                                     (if (instance? Shape dims) dims (Shape. (long-array dims))))
                                   (shape-fn
                                    (mapv (fn [shapes] (mapv (fn [^Shape s] (vec (.getShape s))) shapes))
                                          child-shapes)
                                    (mapv (fn [^Shape s] (vec (.getShape s))) input-shapes)))))))))

(defn- linear! [args spec]
  (let [arg (first args)
        opts (if (map? arg) (options arg #{:units :bias?} :linear) {:units arg})
        {:keys [units bias?] :or {bias? true}} opts]
    (when-not (and (= 1 (count args)) (positive-int? units) (boolean? bias?))
      (invalid :nn/invalid-linear "Linear requires positive integer units and boolean bias?" {:spec spec}))
    (-> (Linear/builder) (.setUnits (long units)) (.optBias (boolean bias?)) .build)))

(defn- dropout! [args spec]
  (let [{:keys [p] :or {p 0.5}} (options (first args) #{:p} :dropout)]
    (when-not (and (= 1 (count args)) (probability? p))
      (invalid :nn/invalid-dropout "Dropout p must be in [0, 1)" {:spec spec}))
    (-> (Dropout/builder) (.optRate (float p)) .build)))

(defn- convolution! [tag args]
  (let [transpose? (#{:conv1d-transpose :conv2d-transpose} tag)
        rank (case tag (:conv1d :conv1d-transpose) 1 (:conv2d :conv2d-transpose) 2 :conv3d 3)
        allowed (cond-> #{:filters :kernel :stride :padding :dilation :groups :bias?}
                  transpose? (conj :output-padding))
        opts (one-options args allowed tag)
        {:keys [filters kernel stride padding dilation groups bias? output-padding]} opts
        _ (when-not (positive-int? filters)
            (invalid :nn/invalid-spec "Convolution requires positive :filters" {:tag tag :options opts}))
        _ (validate-options opts {:groups positive-int? :bias? boolean?} tag)
        kernel-shape (dimensions kernel rank false tag)
        stride-shape (when stride (dimensions stride rank false tag))
        padding-shape (when padding (dimensions padding rank true tag))
        dilation-shape (when dilation (dimensions dilation rank false tag))]
    (if transpose?
      (let [^Deconvolution$DeconvolutionBuilder builder
            (case tag :conv1d-transpose (Conv1dTranspose/builder) :conv2d-transpose (Conv2dTranspose/builder))]
        (.setFilters builder (int filters))
        (.setKernelShape builder kernel-shape)
        (when stride-shape (.optStride builder stride-shape))
        (when padding-shape (.optPadding builder padding-shape))
        (when dilation-shape (.optDilation builder dilation-shape))
        (when groups (.optGroups builder (int groups)))
        (when (contains? opts :bias?) (.optBias builder (boolean bias?)))
        (when output-padding (.optOutPadding builder (dimensions output-padding rank true tag)))
        (TransposeConvolution. builder))
      (let [^Convolution$ConvolutionBuilder builder
            (case tag :conv1d (Conv1d/builder) :conv2d (Conv2d/builder) :conv3d (Conv3d/builder))]
        (.setFilters builder (int filters))
        (.setKernelShape builder kernel-shape)
        (when stride-shape (.optStride builder stride-shape))
        (when padding-shape (.optPadding builder padding-shape))
        (when dilation-shape (.optDilation builder dilation-shape))
        (when groups (.optGroups builder (int groups)))
        (when (contains? opts :bias?) (.optBias builder (boolean bias?)))
        (case tag
          :conv1d (.build ^Conv1d$Builder builder)
          :conv2d (.build ^Conv2d$Builder builder)
          :conv3d (.build ^Conv3d$Builder builder))))))

(defn- pooling! [tag args]
  (let [rank (case tag (:max-pool1d :avg-pool1d) 1 (:max-pool2d :avg-pool2d) 2 3)
        average? (#{:avg-pool1d :avg-pool2d :avg-pool3d} tag)
        opts (one-options args (cond-> #{:kernel :stride :padding :ceil?}
                                 average? (conj :count-include-pad?)) tag)
        _ (validate-options opts {:ceil? boolean? :count-include-pad? boolean?} tag)
        kernel (dimensions (:kernel opts) rank false tag)
        stride (dimensions (get opts :stride (:kernel opts)) rank false tag)
        padding (dimensions (get opts :padding (vec (repeat rank 0))) rank true tag)
        ceil? (boolean (:ceil? opts))
        include? (boolean (get opts :count-include-pad? true))]
    (case tag
      :max-pool1d (Pool/maxPool1dBlock kernel stride padding ceil?)
      :max-pool2d (Pool/maxPool2dBlock kernel stride padding ceil?)
      :max-pool3d (Pool/maxPool3dBlock kernel stride padding ceil?)
      :avg-pool1d (Pool/avgPool1dBlock kernel stride padding ceil? include?)
      :avg-pool2d (Pool/avgPool2dBlock kernel stride padding ceil? include?)
      :avg-pool3d (Pool/avgPool3dBlock kernel stride padding ceil? include?))))

(defn- ghost-batchnorm-block [block]
  ;; DJL leaves its virtual-batch axis in the forward result, unlike its declared shape.
  (FnBlock. "ghost-batchnorm" ^List [block]
            (fn [outputs ^NDList inputs]
              (let [^NDArray output (.singletonOrThrow ^NDList (first outputs))]
                (NDList. ^"[Lai.djl.ndarray.NDArray;"
                 (into-array NDArray [(.reshape output (.getShape (.singletonOrThrow inputs)))]))))
            nil (int -1) (fn [_ input-shapes] input-shapes)))

(defn- normalization! [tag args]
  (let [layer? (= tag :layernorm)
        opts (one-options args (cond-> #{:axis :epsilon :center? :scale?}
                                 (not layer?) (conj :momentum)
                                 (= tag :ghost-batchnorm) (conj :virtual-batch-size)) tag)
        _ (validate-options opts {:epsilon positive-number? :center? boolean? :scale? boolean?
                                  :momentum probability? :virtual-batch-size positive-int?} tag)]
    (if layer?
      (let [axis (get opts :axis [1])
            _ (when-not (and (vector? axis) (seq axis) (every? integer? axis))
                (invalid :nn/invalid-spec "Layer norm :axis must be an integer vector" {:options opts}))
            builder (-> (LayerNorm/builder) (.axis (int-array axis)))]
        (when (contains? opts :center?) (.optCenter builder (boolean (:center? opts))))
        (when (contains? opts :scale?) (.optScale builder (boolean (:scale? opts))))
        (when (:epsilon opts) (.optEpsilon builder (float (:epsilon opts))))
        (.build builder))
      (let [^BatchNorm$BaseBuilder builder
            (if (= tag :ghost-batchnorm) (GhostBatchNorm/builder) (BatchNorm/builder))]
        (when (contains? opts :axis)
          (when-not (integer? (:axis opts))
            (invalid :nn/invalid-spec "Batch norm :axis must be an integer" {:options opts}))
          (.optAxis builder (int (:axis opts))))
        (when (contains? opts :center?) (.optCenter builder (boolean (:center? opts))))
        (when (contains? opts :scale?) (.optScale builder (boolean (:scale? opts))))
        (when (:epsilon opts) (.optEpsilon builder (float (:epsilon opts))))
        (when (:momentum opts) (.optMomentum builder (float (:momentum opts))))
        (when (:virtual-batch-size opts)
          (.optVirtualBatchSize ^GhostBatchNorm$Builder builder (int (:virtual-batch-size opts))))
        (cond-> (.build builder) (= tag :ghost-batchnorm) ghost-batchnorm-block)))))

(defn- embedding! [tag args]
  (case tag
    :embedding
    (let [opts (one-options args #{:dictionary-size :embedding-size} tag)
          {:keys [dictionary-size embedding-size]} opts]
      (when-not (and (positive-int? dictionary-size) (positive-int? embedding-size))
        (invalid :nn/invalid-spec "Embedding requires positive dictionary and embedding sizes" {:options opts}))
      (-> (IdEmbedding$Builder.) (.setDictionarySize (int dictionary-size))
          (.setEmbeddingSize (int embedding-size)) .build))
    :word-embedding
    (let [opts (one-options args #{:tokens :vocabulary :embedding-size :unknown-token :sparse-format} tag)
          {:keys [tokens vocabulary embedding-size unknown-token sparse-format]} opts
          _ (when-not (and (positive-int? embedding-size)
                           (not (and tokens vocabulary))
                           (or (instance? Vocabulary vocabulary)
                               (and (vector? tokens) (seq tokens) (every? string? tokens)))
                           (or (nil? unknown-token) (string? unknown-token)))
              (invalid :nn/invalid-spec "Word embedding requires :tokens or native :vocabulary and :embedding-size"
                       {:options opts}))
          ^Vocabulary vocabulary (or vocabulary (DefaultVocabulary. ^List tokens))
          ^TrainableWordEmbedding$Builder builder
          (-> (TrainableWordEmbedding/builder) (.setVocabulary vocabulary)
              (.setEmbeddingSize (int embedding-size)))]
      (when unknown-token (.optUnknownToken builder unknown-token))
      (when sparse-format
        (.optSparseFormat builder
                          (case sparse-format :dense SparseFormat/DENSE :coo SparseFormat/COO
                                :csr SparseFormat/CSR :row-sparse SparseFormat/ROW_SPARSE
                                (invalid :nn/invalid-spec "Unknown sparse format" {:value sparse-format}))))
      (.build builder))
    :constant-embedding
    (do
      (when-not (and (= 1 (count args)) (instance? NDArray (first args)))
        (invalid :nn/invalid-spec "Constant embedding requires a caller-owned native NDArray" {:args args}))
      (ConstantEmbedding. ^NDArray (first args)))))

(defn- corrected-output-shapes! [block shape-fn]
  ;; DJL 0.36's LSTM omits cell-state shape. Keep the registered native child
  ;; and its real parameters while correcting only the shape metadata.
  (FnBlock. "output" ^List [block]
            (fn [outputs _input] (first outputs)) nil (int -1) shape-fn))

(defn- recurrent! [tag args]
  (let [opts (one-options args (cond-> #{:state-size :layers :dropout :batch-first? :bias? :bidirectional? :return-state?}
                                 (= tag :rnn) (conj :activation)) tag)
        {:keys [state-size layers dropout batch-first? bias? bidirectional? return-state?]
         :or {layers 1}} opts
        _ (when-not (and (positive-int? state-size) (positive-int? layers))
            (invalid :nn/invalid-spec "Recurrent block requires positive :state-size and :layers" {:options opts}))
        _ (validate-options opts {:dropout probability? :batch-first? boolean? :bias? boolean?
                                  :bidirectional? boolean? :return-state? boolean?} tag)
        ^RecurrentBlock$BaseBuilder builder
        (case tag :rnn (RNN/builder) :lstm (LSTM/builder) :gru (GRU/builder))]
    (.setStateSize builder (int state-size))
    (.setNumLayers builder (int layers))
    (when dropout (.optDropRate builder (float dropout)))
    (when (contains? opts :batch-first?) (.optBatchFirst builder (boolean batch-first?)))
    (when (contains? opts :bias?) (.optHasBiases builder (boolean bias?)))
    (when (contains? opts :bidirectional?) (.optBidirectional builder (boolean bidirectional?)))
    (when (contains? opts :return-state?) (.optReturnState builder (boolean return-state?)))
    (case tag
      :rnn (do
             (.setActivation ^RNN$Builder builder
                             (case (get opts :activation :tanh) :tanh RNN$Activation/TANH :relu RNN$Activation/RELU
                                   (invalid :nn/invalid-activation "RNN activation must be :tanh or :relu" {:options opts})))
             (.build ^RNN$Builder builder))
      :lstm (let [block (.build ^LSTM$Builder builder)]
              (if return-state?
                (corrected-output-shapes!
                 block
                 (fn [child-shapes _input-shapes]
                   (let [shapes (first child-shapes)]
                     (into-array Shape [(aget ^"[Lai.djl.ndarray.types.Shape;" shapes 0)
                                        (aget ^"[Lai.djl.ndarray.types.Shape;" shapes 1)
                                        (aget ^"[Lai.djl.ndarray.types.Shape;" shapes 1)]))))
                block))
      :gru (.build ^GRU$Builder builder))))

(defn- attention! [tag args]
  (let [encoder? (= tag :transformer-encoder)
        opts (one-options args (cond-> #{:embedding-size :heads :dropout}
                                 encoder? (conj :hidden-size :activation)) tag)
        {:keys [embedding-size heads hidden-size dropout activation]
         :or {dropout 0.0 activation :relu}} opts]
    (when-not (and (positive-int? embedding-size) (positive-int? heads)
                   (zero? (mod embedding-size heads)) (probability? dropout)
                   (or (not encoder?) (positive-int? hidden-size)))
      (invalid :nn/invalid-spec "Attention requires divisible :embedding-size/:heads and valid sizes/dropout" {:options opts}))
    (if encoder?
      (MaskedTransformerEncoder. (int embedding-size) (int heads) (int hidden-size) (float dropout)
                                 (sam (if (fn? activation) activation (activation-function activation))))
      (-> (ScaledDotProductAttentionBlock/builder) (.setEmbeddingSize (int embedding-size))
          (.setHeadCount (int heads)) (.optAttentionProbsDropoutProb (float dropout)) .build))))

(defn- bert! [tag args]
  (let [opts (one-options args #{:token-dictionary-size :type-dictionary-size :embedding-size
                                 :layers :heads :hidden-size :dropout :max-sequence-length :preset} tag)
        {:keys [token-dictionary-size type-dictionary-size embedding-size layers heads
                hidden-size dropout max-sequence-length preset]} opts
        _ (when-not (positive-int? token-dictionary-size)
            (invalid :nn/invalid-spec "BERT requires :token-dictionary-size" {:options opts}))
        _ (validate-options opts {:type-dictionary-size positive-int? :embedding-size positive-int?
                                  :layers positive-int? :heads positive-int? :hidden-size positive-int?
                                  :dropout probability? :max-sequence-length positive-int?} tag)
        ^BertBlock$Builder builder (BertBlock/builder)]
    (when preset
      (case preset :nano (.nano builder) :micro (.micro builder)
            :base (.base builder) :large (.large builder)
            (invalid :nn/invalid-spec "Unknown BERT preset" {:preset preset})))
    (.setTokenDictionarySize builder (int token-dictionary-size))
    (when type-dictionary-size (.optTypeDictionarySize builder (int type-dictionary-size)))
    (when embedding-size (.optEmbeddingSize builder (int embedding-size)))
    (when layers (.optTransformerBlockCount builder (int layers)))
    (when heads (.optAttentionHeadCount builder (int heads)))
    (when hidden-size (.optHiddenSize builder (int hidden-size)))
    (when dropout (.optHiddenDropoutProbability builder (float dropout)))
    (when max-sequence-length (.optMaxSequenceLength builder (int max-sequence-length)))
    (case tag
      :bert
      (let [block (.build builder)]
        ;; BertBlock.squeeze() drops batch=1 from its pooled output. Restore the
        ;; declared batch dimension with a view, not a tensor copy.
        (FnBlock. "bert" ^List [block]
                  (fn [outputs ^NDList inputs]
                    (let [^NDList output (first outputs)
                          ^NDArray sequence (.get output 0)
                          ^NDArray pooled (.get output 1)
                          batch (.get (.getShape (.head inputs)) 0)
                          normalized (if (= 2 (.dimension (.getShape pooled)))
                                       pooled
                                       (.reshape pooled (long-array [batch (.getEmbeddingSize block)])))]
                      (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray [sequence normalized]))))
                  nil (int -1) (fn [child-shapes _input-shapes] (first child-shapes))))
      :bert-pretraining
      (corrected-output-shapes!
       (BertPretrainingBlock. builder)
       (fn [child-shapes _input-shapes]
         ;; The native MLM forward returns [batch*masked, dictionary], not the
         ;; pretraining parent's advertised [batch, masked, dictionary].
         (let [shapes (first child-shapes)
               ^Shape next-sentence (aget ^"[Lai.djl.ndarray.types.Shape;" shapes 0)
               ^Shape masked (aget ^"[Lai.djl.ndarray.types.Shape;" shapes 1)]
           (into-array Shape [next-sentence
                              (Shape. (long-array [(* (.get masked 0) (.get masked 1)) (.get masked 2)]))])))))))

(defn- transform! [tag arguments]
  (let [options (one-options arguments
                             (case tag :reshape #{:shape} :transpose #{:axes}
                                   (:softmax :log-softmax :expand-dims :squeeze) #{:axis})
                             tag)]
    (case tag
      :reshape
      (let [dimensions (:shape options)
            _ (when-not (and (vector? dimensions)
                             (every? #(and (integer? %) (or (= -1 %) (pos? %))) dimensions)
                             (<= (count (filter #{-1} dimensions)) 1))
                (invalid :nn/invalid-spec "Reshape :shape requires positive dimensions and at most one -1"
                         {:options options}))
            shape (Shape. (long-array dimensions))]
        (LambdaBlock/singleton (sam (fn [^NDArray input] (.reshape input shape))) "reshape"))
      :transpose
      (let [axes (:axes options)
            _ (when-not (and (vector? axes) (every? #(and (integer? %) (<= 0 %)) axes)
                             (= (set axes) (set (range (count axes)))))
                (invalid :nn/invalid-spec "Transpose :axes must be a dimension permutation"
                         {:options options}))
            permutation (int-array axes)]
        (LambdaBlock/singleton (sam (fn [^NDArray input] (.transpose input permutation))) "transpose"))
      (:softmax :log-softmax :expand-dims :squeeze)
      (let [axis (get options :axis -1)
            _ (when-not (integer? axis)
                (invalid :nn/invalid-spec "Tensor operation :axis must be an integer"
                         {:options options}))
            dimension (int axis)]
        (LambdaBlock/singleton
         (sam (case tag
                :softmax (fn [^NDArray input] (.softmax input dimension))
                :log-softmax (fn [^NDArray input] (.logSoftmax input dimension))
                :expand-dims (fn [^NDArray input] (.expandDims input dimension))
                :squeeze (fn [^NDArray input] (.squeeze input dimension)))) (name tag))))))

(defn- index! [arguments]
  (let [^String index (first arguments)
        _ (when-not (and (= 1 (count arguments)) (string? index) (seq index)
                         (neg? (.indexOf index (int 0))))
            (invalid :nn/invalid-spec "Index requires one nonempty string without NUL"
                     {:tag :index :args arguments}))
        parameters (object-array 0)]
    ;; DJL 0.36 LambdaBlock infers shapes by evaluating the lambda on managed zeros.
    (LambdaBlock/singleton
     (sam (fn [^NDArray input] (.get input index parameters))) "index")))

(defn- native! [tag args]
  (case tag
    :block
    (do
      (when-not (and (= 1 (count args)) (instance? Block (first args)))
        (invalid :nn/invalid-spec "Native :block requires a DJL Block" {:args args}))
      (first args))
    :factory
    (do
      (when-not (and (= 1 (count args)) (fn? (first args)))
        (invalid :nn/invalid-spec "Factory requires a zero-argument function" {:args args}))
      (let [block ((first args))]
        (when-not (instance? Block block)
          (invalid :nn/invalid-spec "Block factory must return a DJL Block" {:returned block}))
        block))
    :lambda
    (let [[opts f] (if (= 1 (count args)) [{} (first args)] args)
          {:keys [singleton? name] :or {singleton? false name "lambda"}}
          (options opts #{:singleton? :name} :lambda)]
      (when-not (and (<= 1 (count args) 2) (fn? f) (boolean? singleton?) (string? name))
        (invalid :nn/invalid-spec "Lambda requires a Clojure function and optional :singleton?/:name" {:args args}))
      (let [function (sam (fn [input]
                            (let [result (f input)]
                              (when-not (if singleton? (instance? NDArray result) (instance? NDList result))
                                (invalid :nn/invalid-spec "Lambda returned the wrong DJL tensor type"
                                         {:singleton? singleton? :returned result}))
                              result)))]
        (if singleton? (LambdaBlock/singleton function name) (LambdaBlock. function name))))))

(def ^:private no-argument-tags
  #{:relu :relu6 :tanh :sigmoid :gelu :selu :mish :softplus :softsign :prelu :identity
    :flatten :global-max-pool1d :global-max-pool2d :global-max-pool3d
    :global-avg-pool1d :global-avg-pool2d :global-avg-pool3d :bert-next-sentence})

(def ^:private option-tags
  #{:linear :activation :dropout :conv1d :conv2d :conv3d :conv1d-transpose :conv2d-transpose
    :max-pool1d :max-pool2d :max-pool3d :avg-pool1d :avg-pool2d :avg-pool3d
    :batchnorm :layernorm :ghost-batchnorm :embedding :word-embedding
    :rnn :lstm :gru :attention :transformer-encoder :bert :bert-pretraining
    :reshape :transpose :index :softmax :log-softmax :expand-dims :squeeze :leaky-relu :elu :swish})

(defn compile-block
  "Compile Hiccup/native Block into a real DJL Block. No model/manager is created.
  :factory is invoked on each compilation and must return a fresh Block when used
  by distinct owners. Pool/convolution spatial vectors exclude batch/channels;
  reshape/transpose vectors describe the full tensor, including batch. Index uses
  DJL's NDIndex string grammar unchanged and borrows its singleton input."
  ^Block [spec]
  (if (instance? Block spec)
    spec
    (do
      (when-not (and (vector? spec) (keyword? (first spec)))
        (invalid :nn/invalid-spec "A network block must be a Hiccup vector or DJL Block" {:spec spec}))
      (let [[tag & arguments] spec]
        (when (and (no-argument-tags tag) (seq arguments))
          (invalid :nn/invalid-spec "This block accepts no options or children" {:spec spec}))
        (case tag
          :sequential (sequential! arguments)
          :linear (linear! arguments spec)
          :activation (do
                        (when-not (= 1 (count arguments))
                          (invalid :nn/invalid-spec "Activation requires one name" {:spec spec}))
                        (activation (first arguments)))
          :dropout (dropout! arguments spec)
          (:parallel :skip) (combination! tag arguments spec)
          (:relu :relu6 :tanh :sigmoid :gelu :selu :mish :softplus :softsign :prelu :identity)
          (activation tag)
          :flatten (Blocks/batchFlattenBlock)
          (:batchnorm :layernorm) (normalization! tag (if (seq arguments) arguments [{}]))
          :ghost-batchnorm (normalization! tag arguments)
          (:conv1d :conv2d :conv3d :conv1d-transpose :conv2d-transpose) (convolution! tag arguments)
          (:max-pool1d :max-pool2d :max-pool3d :avg-pool1d :avg-pool2d :avg-pool3d) (pooling! tag arguments)
          :global-max-pool1d (Pool/globalMaxPool1dBlock)
          :global-max-pool2d (Pool/globalMaxPool2dBlock)
          :global-max-pool3d (Pool/globalMaxPool3dBlock)
          :global-avg-pool1d (Pool/globalAvgPool1dBlock)
          :global-avg-pool2d (Pool/globalAvgPool2dBlock)
          :global-avg-pool3d (Pool/globalAvgPool3dBlock)
          (:embedding :word-embedding :constant-embedding) (embedding! tag arguments)
          (:rnn :lstm :gru) (recurrent! tag arguments)
          (:attention :transformer-encoder) (attention! tag arguments)
          (:bert :bert-pretraining) (bert! tag arguments)
          :bert-next-sentence (BertNextSentenceBlock.)
          :index (index! arguments)
          (:reshape :transpose :softmax :log-softmax :expand-dims :squeeze) (transform! tag arguments)
          (:leaky-relu :elu :swish)
          (let [key (if (= tag :swish) :beta :alpha)
                options (one-options arguments #{key} tag)
                value (get options key (if (= tag :leaky-relu) 0.01 1.0))]
            (when-not (finite-number? value)
              (invalid :nn/invalid-spec "Activation parameter must be finite" {:tag tag :options options}))
            (case tag :leaky-relu (Activation/leakyReluBlock (float value))
                  :elu (Activation/eluBlock (float value)) :swish (Activation/swishBlock (float value))))
          (:block :factory :lambda) (native! tag arguments)
          (invalid :nn/unknown-tag "Unknown neural network block" {:tag tag}))))))

(defn- plain-edn? [value]
  (cond
    (record? value) false
    (or (nil? value) (boolean? value) (string? value) (char? value) (keyword? value)
        (symbol? value) (number? value)) true
    (map? value) (every? (fn [[k v]] (and (plain-edn? k) (plain-edn? v))) value)
    (coll? value) (every? plain-edn? value)
    :else false))

(defn serializable-spec?
  "True only for known, reconstructible plain-EDN block trees. Native Blocks,
  lambda/factory/constant embeddings and function options are never serialized.
  This is a persistence-policy predicate, not block argument/shape validation."
  [spec]
  (and (vector? spec) (plain-edn? spec)
       (let [[tag & args] spec]
         (cond
           (= tag :sequential) (every? serializable-spec? args)
           (#{:parallel :skip} tag)
           (and (map? (first args)) (#{:add :concat} (:combine (first args)))
                (seq (rest args)) (every? serializable-spec? (rest args)))
           (no-argument-tags tag) (empty? args)
           (option-tags tag) true
           :else false))))
