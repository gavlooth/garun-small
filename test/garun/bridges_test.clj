(ns garun.bridges-test
  "Each Java bridge under java/ exists because stock DJL with the PyTorch engine
  cannot do one thing. Every test here shows that stock DJL still cannot. A test
  that fails after a DJL upgrade means DJL now can: delete that bridge, switch
  garun.blocks or garun.training.optimizers to the stock class, and delete the test.

  FnBlock and GraphBlock have no test here: they are block types Garun adds, not
  repairs of a stock one."
  (:require [clojure.test :refer [deftest is]]
            [garun.context :as context]
            [garun.session :as session]
            [garun.tensor :as tensor])
  (:import [ai.djl.ndarray NDArray NDList NDManager]
           [ai.djl.ndarray.types DataType Shape]
           [ai.djl.nn Activation Block]
           [ai.djl.nn.convolutional Conv2dTranspose Conv2dTranspose$Builder]
           [ai.djl.nn.recurrent GRU GRU$Builder OrderedRecurrent]
           [ai.djl.nn.transformer TransformerEncoderBlock]
           [ai.djl.training ParameterStore]
           [ai.djl.training.optimizer Optimizer]
           [ai.djl.training.tracker Tracker]
           [java.util.function Function]))

(set! *warn-on-reflection* true)

(defn- shapes ^"[Lai.djl.ndarray.types.Shape;" [& dimensions]
  (into-array Shape (map tensor/shape dimensions)))

(deftest tensor-optimizer-stock-nag-rmsprop-adagrad-and-adadelta-still-have-no-pytorch-kernel
  (with-open [current-session (session/open {:device :cpu})]
    (let [rate (Tracker/fixed (float 0.1))
          ^NDArray weight (tensor/array current-session [1.0 2.0] {:dtype :float32})
          ^NDArray gradient (tensor/array current-session [0.5 0.5] {:dtype :float32})]
      (doseq [[family ^Optimizer optimizer]
              [[:nag (-> (Optimizer/nag) (.setLearningRateTracker rate) (.setMomentum (float 0.9)) .build)]
               [:rmsprop (-> (Optimizer/rmsprop) (.optLearningRateTracker rate) .build)]
               [:adagrad (-> (Optimizer/adagrad) (.optLearningRateTracker rate) .build)]
               [:adadelta (.build (Optimizer/adadelta))]]]
        (is (thrown? UnsupportedOperationException (.update optimizer "parameter" weight gradient))
            (name family))))))

(defn- bidirectional-builder ^GRU$Builder []
  (-> (GRU/builder) (.setStateSize (int 2)) (.setNumLayers (int 1))
      (.optBidirectional true) (.optHasBiases true)))

(deftest ordered-recurrent-stock-bidirectional-parameters-are-still-ordered-by-type-before-direction
  (let [^Block stock (.build (bidirectional-builder))
        ^Block ordered (OrderedRecurrent/gru (bidirectional-builder))
        ;; PyTorch reads each direction's weights and biases together.
        by-direction ["l_0_i2h_WEIGHT" "l_0_h2h_WEIGHT" "l_0_i2h_BIAS" "l_0_h2h_BIAS"
                      "r_0_i2h_WEIGHT" "r_0_h2h_WEIGHT" "r_0_i2h_BIAS" "r_0_h2h_BIAS"]]
    (is (= by-direction (vec (.keys (.getParameters ordered)))))
    (is (= (set by-direction) (set (.keys (.getParameters stock)))))
    (is (not= by-direction (vec (.keys (.getParameters stock)))))))

(deftest masked-transformer-encoder-stock-encoder-still-cannot-initialize-with-an-attention-mask
  (with-open [current-session (session/open {:device :cpu})]
    (let [^NDManager manager (context/manager current-session)
          ^Block stock (TransformerEncoderBlock.
                        (int 8) (int 2) (int 16) (float 0.0)
                        (reify Function (apply [_ inputs] (Activation/relu ^NDList inputs))))]
      (is (thrown? IllegalArgumentException
                   (.initialize stock manager DataType/FLOAT32 (shapes [2 5 8] [2 5 5])))))))

(deftest transpose-convolution-stock-transposed-convolution-still-has-no-pytorch-forward
  (with-open [current-session (session/open {:device :cpu})]
    (let [^NDManager manager (context/manager current-session)
          ^Conv2dTranspose$Builder builder (-> (Conv2dTranspose/builder) (.setFilters (int 2))
                                               (.setKernelShape (tensor/shape [3 3])))
          ^Block stock (.build builder)
          ^NDArray image (tensor/array current-session (vec (repeat 16 1.0)) {:shape [1 1 4 4] :dtype :float32})
          inputs (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray [image]))]
      (.initialize stock manager DataType/FLOAT32 (shapes [1 1 4 4]))
      (is (thrown? UnsupportedOperationException
                   (.forward stock (ParameterStore. manager false) inputs false))))))
