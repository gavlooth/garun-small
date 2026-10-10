(ns garun.blocks-test
  (:require [clojure.test :refer [deftest is testing]]
            [garun.context :as context]
            [garun.blocks :as blocks]
            [garun.nn :as nn]
            [garun.session :as session])
  (:import [ai.djl Device]
           [ai.djl.ndarray NDArray NDList NDManager]
           [ai.djl.ndarray.types DataType Shape]
           [ai.djl.nn Block Blocks LambdaBlock Parameter Parameter$Type]
           [ai.djl.nn.core Linear]
           [ai.djl.training GradientCollector ParameterStore]
           [ai.djl.training.initializer XavierInitializer]
           [java.io ByteArrayInputStream ByteArrayOutputStream DataInputStream DataOutputStream File]
           [java.nio.file Files]
           [java.nio.file.attribute FileAttribute]))

(set! *warn-on-reflection* true)

(defn- shape-vectors [shapes]
  (mapv (fn [^Shape shape] (vec (.getShape shape))) shapes))

(defn- initialize! [spec ^NDManager manager ^NDList inputs]
  (let [^Block block (blocks/compile-block spec)]
    (.setInitializer block (XavierInitializer.) Parameter$Type/WEIGHT)
    ;; The parameter dtype is float32 even when embedding IDs are int64.
    (.initialize block manager DataType/FLOAT32 (.getShapes inputs))
    block))

(defn- exercise-gradients! [spec ^NDManager manager ^NDList inputs]
  (let [^Block block (initialize! spec manager inputs)
        store (ParameterStore. manager false)
        ^NDList outputs
        (with-open [^GradientCollector collector (.newGradientCollector (.getEngine manager))]
          (let [^NDList outputs (.forward block store inputs true)
                ^NDArray loss (reduce (fn [^NDArray a ^NDArray b] (.add a b))
                                      (map (fn [^NDArray output] (.sum (.square output))) outputs))]
            (.backward collector loss)
            outputs))
        gradients (mapcat (fn [^Parameter parameter]
                            (let [^NDArray array (.getArray parameter)]
                              (when (.hasGradient array)
                                (seq (.toFloatArray (.getGradient array))))))
                          (.values (.getParameters block)))]
    {:block block
     :actual (shape-vectors (.getShapes outputs))
     :declared (shape-vectors (.getOutputShapes block (.getShapes inputs)))
     :outputs outputs
     :gradients (vec gradients)}))

(defn- parameter-roundtrip! [spec ^NDManager manager ^NDList inputs ^Block block]
  (let [bytes (ByteArrayOutputStream.)
        _ (with-open [stream (DataOutputStream. bytes)] (.saveParameters block stream))
        ^Block restored (initialize! spec manager inputs)]
    (with-open [stream (DataInputStream. (ByteArrayInputStream. (.toByteArray bytes)))]
      (.loadParameters restored manager stream))
    (.forward restored (ParameterStore. manager false) inputs false)))

(defn- error-key [operation]
  (try
    (operation)
    nil
    (catch clojure.lang.ExceptionInfo error (:error (ex-data error)))))

(deftest convolution-families-forward-and-gradients
  (with-open [manager (NDManager/newBaseManager (Device/cpu) "PyTorch")]
    (doseq [[spec input-shape output-shape]
            [[[:conv1d {:filters 2 :kernel [3]}] [2 1 7] [2 2 5]]
             [[:conv2d {:filters 2 :kernel [3 3]}] [2 1 7 7] [2 2 5 5]]
             [[:conv3d {:filters 2 :kernel [3 3 3]}] [2 1 5 5 5] [2 2 3 3 3]]
             [[:conv1d-transpose {:filters 2 :kernel [3] :stride [2]}] [2 1 4] [2 2 9]]
             [[:conv2d-transpose {:filters 2 :kernel [3 3] :stride [2 2]}] [2 1 4 4] [2 2 9 9]]]]
      (testing (pr-str spec)
        (with-open [scope (.newSubManager manager)]
          (let [input (.ones scope (Shape. (long-array input-shape)))
                result (exercise-gradients! spec scope (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray [input])))]
            (is (= [output-shape] (:actual result) (:declared result)))
            (is (seq (:gradients result)))
            (is (every? #(Double/isFinite (double %)) (:gradients result)))
            (is (some #(not (zero? %)) (:gradients result)))))))))

(defn- coordinates [dimensions]
  (reduce (fn [prefixes size]
            (for [prefix prefixes index (range size)] (conj prefix index)))
          [[]] dimensions))

(defn- flat-index [dimensions indices]
  (reduce (fn [offset [size index]] (+ (* offset size) index))
          0 (map vector dimensions indices)))

(defn- transposed-reference
  "Independent scatter definition, with analytic gradients of sum(output)."
  [input-shape opts input-values weight-values bias-values]
  (let [{:keys [filters kernel stride padding dilation groups output-padding]} opts
        groups (or groups 1)
        rank (count kernel)
        stride (or stride (vec (repeat rank 1)))
        padding (or padding (vec (repeat rank 0)))
        dilation (or dilation (vec (repeat rank 1)))
        output-padding (or output-padding (vec (repeat rank 0)))
        [batch channels] input-shape
        spatial (subvec input-shape 2)
        output-spatial (mapv (fn [size step pad dilate width extra]
                               (+ (* (dec size) step) (- (* 2 pad))
                                  (* dilate (dec width)) extra 1))
                             spatial stride padding dilation kernel output-padding)
        output-shape (into [batch filters] output-spatial)
        weight-shape (into [filters (quot channels groups)] kernel)
        contributions
        (for [n (range batch)
              out (range filters)
              in (range (quot channels groups))
              position (coordinates spatial)
              tap (coordinates kernel)
              :let [target (mapv (fn [x step pad k dilate] (+ (* x step) (- pad) (* k dilate)))
                                 position stride padding tap dilation)]
              :when (every? true? (map (fn [x size] (<= 0 x (dec size))) target output-spatial))]
          [(flat-index output-shape (into [n out] target))
           (flat-index input-shape
                       (into [n (+ in (* (quot out (quot filters groups)) (quot channels groups)))]
                             position))
           (flat-index weight-shape (into [out in] tap))])
        initial-values (vec (for [_ (range batch) out (range filters)
                                  _ (range (reduce * output-spatial))]
                              (if bias-values (nth bias-values out) (float 0))))]
    (reduce (fn [result [output input weight]]
              (-> result
                  (update-in [:values output] + (* (nth input-values input) (nth weight-values weight)))
                  (update-in [:input-gradient input] + (nth weight-values weight))
                  (update-in [:weight-gradient weight] + (nth input-values input))))
            {:shape output-shape
             :values initial-values
             :input-gradient (vec (repeat (count input-values) (float 0)))
             :weight-gradient (vec (repeat (count weight-values) (float 0)))}
            contributions)))

(deftest transposed-convolution-values-all-options-gradients-and-persistence
  (with-open [manager (NDManager/newBaseManager (Device/cpu) "PyTorch")]
    (doseq [[tag input-shape opts]
            [[:conv1d-transpose [2 4 3]
              {:filters 6 :kernel [3] :stride [2] :padding [1] :dilation [2]
               :groups 2 :output-padding [1]}]
             [:conv2d-transpose [1 4 2 3]
              {:filters 6 :kernel [2 3] :stride [2 3] :padding [1 2] :dilation [2 1]
               :groups 2 :output-padding [1 2]}]
             ;; Cropping can remove every original input position while kernel tails
             ;; still contribute; crop the convolution result, not its input.
             [:conv1d-transpose [1 1 1]
              {:filters 1 :kernel [5] :padding [2] :bias? false}]
             ;; PyTorch permits output-padding >= stride when it is < dilation.
             [:conv1d-transpose [1 1 2]
              {:filters 1 :kernel [2] :stride [1] :dilation [3] :output-padding [2]}]]]
      (testing (pr-str [tag opts])
        (with-open [scope (.newSubManager manager)]
          (let [spec [tag opts]
                input-values (mapv #(float (inc (mod % 7))) (range (reduce * input-shape)))
                input (.create scope (float-array input-values) (Shape. (long-array input-shape)))
                inputs (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray [input]))
                ^Block block (initialize! spec scope inputs)
                ^Parameter weight (.get (.getDirectParameters block) "weight")
                ^Parameter bias (.get (.getDirectParameters block) "bias")
                weight-values (mapv #(float (- (mod % 9) 4)) (range (.size (.getArray weight))))
                bias-values (when bias (mapv #(float (inc %)) (range (:filters opts))))
                expected (transposed-reference input-shape opts input-values weight-values bias-values)
                _ (.set (.getArray weight) (float-array weight-values))
                _ (when bias (.set (.getArray bias) (float-array bias-values)))
                _ (.setRequiresGradient input true)
                ^NDList output
                (with-open [^GradientCollector collector (.newGradientCollector (.getEngine scope))]
                  (let [^NDList output (.forward block (ParameterStore. scope false) inputs true)]
                    (.backward collector (.sum (.head output)))
                    output))
                ^NDList restored (parameter-roundtrip! spec scope inputs block)]
            (is (= [(:shape expected)] (shape-vectors (.getShapes output))
                   (shape-vectors (.getOutputShapes block (.getShapes inputs)))))
            (is (= (:values expected) (vec (.toFloatArray (.head output)))))
            (is (= (:input-gradient expected) (vec (.toFloatArray (.getGradient input)))))
            (is (= (:weight-gradient expected)
                   (vec (.toFloatArray (.getGradient (.getArray weight))))))
            (when bias
              (is (= (vec (repeat (:filters opts)
                                  (float (* (first input-shape) (reduce * (drop 2 (:shape expected)))))))
                     (vec (.toFloatArray (.getGradient (.getArray bias)))))))
            (is (.allClose (.head output) (.head restored) 1.0e-6 1.0e-6 false))
            (is (not (.isReleased input)))))))))

(deftest cnn-composition-concat-axis-and-parameter-persistence
  (let [spec [:sequential
              [:parallel {:combine :concat :axis 1}
               [:conv2d {:filters 2 :kernel [3 3] :padding [1 1]}]
               [:conv2d {:filters 3 :kernel [1 1]}]]
              [:relu]
              [:max-pool2d {:kernel [2 2]}]
              [:global-avg-pool2d]
              [:flatten]
              [:linear 2]]]
    (with-open [manager (NDManager/newBaseManager (Device/cpu) "PyTorch")]
      (let [input (.ones manager (Shape. (long-array [2 1 8 8])))
            inputs (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray [input]))
            {:keys [block actual declared gradients outputs]} (exercise-gradients! spec manager inputs)
            ^NDList restored (parameter-roundtrip! spec manager inputs block)]
        (is (= [[2 2]] actual declared))
        (is (some #(not (zero? %)) gradients))
        (is (blocks/serializable-spec? spec))
        (is (.allClose (.head ^NDList outputs) (.head restored) 1.0e-6 1.0e-6 false))
        (is (not (.isReleased input)))))))

(deftest pooling-families-use-spatial-dimensions
  (with-open [manager (NDManager/newBaseManager (Device/cpu) "PyTorch")]
    (doseq [[rank suffix input-shape] [[1 "1d" [2 3 4]] [2 "2d" [2 3 4 4]] [3 "3d" [2 3 4 4 4]]]]
      (doseq [prefix ["max-pool" "avg-pool" "global-max-pool" "global-avg-pool"]]
        (let [global? (.startsWith ^String prefix "global")
              tag (keyword (str prefix suffix))
              spec (if global? [tag] [tag {:kernel (vec (repeat rank 2))}])
              input (.ones manager (Shape. (long-array input-shape)))
              inputs (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray [input]))
              ^Block block (initialize! spec manager inputs)
              ^NDArray output (.head (.forward block (ParameterStore. manager false) inputs false))
              expected (if global? [2 3] (into [2 3] (repeat rank 2)))]
          (is (= expected (vec (.getShape (.getShape output)))) (str tag))
          (is (every? #(= 1.0 (double %)) (.toFloatArray output)) (str tag)))))))

(deftest integer-embedding-inputs-and-variants
  (with-open [manager (NDManager/newBaseManager (Device/cpu) "PyTorch")]
    (doseq [spec [[:embedding {:dictionary-size 5 :embedding-size 3}]
                  [:word-embedding {:tokens ["zero" "one" "two" "three" "four"] :embedding-size 3}]]]
      (let [ids (.create manager (long-array [1 2 1 3]) (Shape. (long-array [2 2])))
            inputs (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray [ids]))
            {:keys [block actual declared gradients outputs]} (exercise-gradients! spec manager inputs)
            ^NDList restored (parameter-roundtrip! spec manager inputs block)
            ^NDArray output (.head ^NDList outputs)
            values (vec (.toFloatArray output))]
        (is (= [[2 2 3]] actual declared))
        (is (= (subvec values 0 3) (subvec values 6 9)))
        (is (= DataType/INT64 (.getDataType ids)))
        (is (= [1 2 1 3] (vec (.toLongArray ids))))
        (is (some #(not (zero? %)) gradients))
        (is (.allClose output (.head restored) 1.0e-6 1.0e-6 false))
        (is (blocks/serializable-spec? spec))))
    (let [weights (.create manager (float-array [0 1 2 3 4 5]) (Shape. (long-array [3 2])))
          inputs (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray [(.create manager (long-array [2 0]))]))
          spec [:constant-embedding weights]
          ^Block block (initialize! spec manager inputs)
          ^NDArray output (.head (.forward block (ParameterStore. manager false) inputs false))]
      (is (= [[2 3 2]] (shape-vectors (.getShapes (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray [output]))))))
      (is (= [0.0 1.0 2.0 3.0 4.0 5.0 0.0 1.0 2.0 3.0 4.0 5.0]
             (vec (.toFloatArray output))))
      (is (not (blocks/serializable-spec? spec)))
      (is (not (.isReleased weights))))))

(deftest recurrent-sequences-states-gradients-and-persistence
  (with-open [manager (NDManager/newBaseManager (Device/cpu) "PyTorch")]
    (doseq [[tag expected] [[:rnn [[2 3 4] [1 2 4]]]
                            [:gru [[2 3 4] [1 2 4]]]
                            [:lstm [[2 3 4] [1 2 4] [1 2 4]]]]]
      (with-open [scope (.newSubManager manager)]
        (let [spec [tag {:state-size 4 :return-state? true}]
              inputs (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray [(.ones scope (Shape. (long-array [2 3 2])))]))
              {:keys [block actual declared gradients outputs]} (exercise-gradients! spec scope inputs)
              ^NDList restored (parameter-roundtrip! spec scope inputs block)]
          (is (= expected actual declared) (str tag " forward outputs agree with shape inference"))
          (is (some #(not (zero? %)) gradients))
          (is (= expected (shape-vectors (.getShapes restored))))
          (is (every? true? (map (fn [^NDArray a ^NDArray b] (.allClose a b 1.0e-6 1.0e-6 false))
                                 outputs restored))))))))

(deftest attention-and-masked-encoder-gradients-shapes-and-persistence
  (with-open [manager (NDManager/newBaseManager (Device/cpu) "PyTorch")]
    (doseq [spec [[:attention {:embedding-size 4 :heads 2 :dropout 0.0}]
                  [:transformer-encoder {:embedding-size 4 :heads 2 :hidden-size 8 :dropout 0.0 :activation :gelu}]]]
      (with-open [scope (.newSubManager manager)]
        (let [embedding (.randomNormal scope (Shape. (long-array [2 3 4])))
              mask (.zeros scope (Shape. (long-array [2 1 3 3])))
              inputs (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray [embedding mask]))
              {:keys [block actual declared gradients]} (exercise-gradients! spec scope inputs)
              ^NDList before (.forward ^Block block (ParameterStore. scope false) inputs false)
              ^NDList restored (parameter-roundtrip! spec scope inputs block)]
          (is (= [[2 3 4]] actual declared) (str (first spec) " mask is not a second output"))
          (is (seq gradients))
          (is (every? #(Double/isFinite (double %)) gradients))
          (is (some #(not (zero? %)) gradients))
          (is (= [[2 3 4]] (shape-vectors (.getShapes restored))))
          (is (.allClose (.head before) (.head restored) 1.0e-6 1.0e-6 false))
          (let [restricted-mask (.create scope
                                         (float-array (take 18 (cycle [0 -100000 -100000])))
                                         (Shape. (long-array [2 1 3 3])))
                restricted-inputs (NDList. ^"[Lai.djl.ndarray.NDArray;"
                                   (into-array NDArray [embedding restricted-mask]))
                ^NDList restricted (.forward ^Block block (ParameterStore. scope false)
                                             restricted-inputs false)]
            (is (not (.allClose (.head before) (.head restricted) 1.0e-6 1.0e-6 false))
                (str (first spec) " mask changes attention rather than being ignored")))
          (is (not (.isReleased embedding)))
          (is (not (.isReleased mask))))))))

(deftest bert-builders-have-real-integer-inputs-multiple-outputs-and-persistence
  (with-open [manager (NDManager/newBaseManager (Device/cpu) "PyTorch")]
    (doseq [tag [:bert :bert-pretraining]]
      (with-open [scope (.newSubManager manager)]
        (let [spec [tag {:token-dictionary-size 7 :type-dictionary-size 2
                         :embedding-size 4 :layers 1 :heads 2 :hidden-size 8
                         :dropout 0.0 :max-sequence-length 3}]
              ids (.create scope (long-array [1 2 3 1 4 5]) (Shape. (long-array [2 3])))
              types (.zeros scope (Shape. (long-array [2 3])) DataType/INT64)
              masks (.ones scope (Shape. (long-array [2 3])))
              masked (.create scope (long-array [1 2]) (Shape. (long-array [2 1])))
              inputs (NDList. ^"[Lai.djl.ndarray.NDArray;"
                      (into-array NDArray (cond-> [ids types masks] (= tag :bert-pretraining) (conj masked))))
              {:keys [block actual declared gradients]} (exercise-gradients! spec scope inputs)
              expected (if (= tag :bert) [[2 3 4] [2 4]] [[2 2] [2 7]])
              ^NDList before (.forward ^Block block (ParameterStore. scope false) inputs false)
              ^NDList restored (parameter-roundtrip! spec scope inputs block)]
          (is (= expected actual declared) (str tag " actual multi-output shapes"))
          (is (some #(not (zero? %)) gradients))
          (is (every? #(Double/isFinite (double %)) gradients))
          (is (= expected (shape-vectors (.getShapes restored))))
          (is (every? true? (map (fn [^NDArray a ^NDArray b] (.allClose a b 1.0e-6 1.0e-6 false))
                                 before restored)))
          (is (blocks/serializable-spec? spec)))))))

(deftest bert-preserves-singleton-batch-in-pooled-output
  (with-open [manager (NDManager/newBaseManager (Device/cpu) "PyTorch")]
    (let [spec [:bert {:token-dictionary-size 7 :type-dictionary-size 2
                       :embedding-size 4 :layers 1 :heads 2 :hidden-size 8
                       :dropout 0.0 :max-sequence-length 3}]
          ids (.create manager (long-array [1 2 3]) (Shape. (long-array [1 3])))
          inputs (NDList. ^"[Lai.djl.ndarray.NDArray;"
                  (into-array NDArray [ids
                                       (.zeros manager (Shape. (long-array [1 3])) DataType/INT64)
                                       (.ones manager (Shape. (long-array [1 3])))]))
          ^Block block (initialize! spec manager inputs)
          ^NDList output (.forward block (ParameterStore. manager false) inputs false)]
      (is (= [[1 3 4] [1 4]]
             (shape-vectors (.getShapes output))
             (shape-vectors (.getOutputShapes block (.getShapes inputs))))))))

(deftest native-blocks-factories-and-clojure-lambda-sam
  (with-open [manager (NDManager/newBaseManager (Device/cpu) "PyTorch")]
    (let [native (-> (Linear/builder) (.setUnits (long 2)) .build)
          factory (fn [] (-> (Linear/builder) (.setUnits (long 2)) .build))
          twice (fn [^NDList input]
                  (NDList. ^"[Lai.djl.ndarray.NDArray;"
                   (into-array NDArray (map (fn [^NDArray x] (.mul x (Float/valueOf (float 2)))) input))))
          spec [:sequential [:block native] [:lambda twice]]
          inputs (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray [(.ones manager (Shape. (long-array [2 3])))]))
          result (exercise-gradients! spec manager inputs)
          factory-result (exercise-gradients! [:factory factory] manager inputs)]
      (is (identical? native (blocks/compile-block native)))
      (is (identical? native (blocks/compile-block [:block native])))
      (is (not (identical? (blocks/compile-block [:factory factory])
                           (blocks/compile-block [:factory factory]))))
      (is (instance? LambdaBlock (blocks/compile-block [:lambda twice])))
      (is (= [[2 2]] (:actual result) (:declared result)))
      (is (some #(not (zero? %)) (:gradients result)))
      (is (= [[2 2]] (:actual factory-result) (:declared factory-result)))
      (is (some #(not (zero? %)) (:gradients factory-result)))
      (is (not (blocks/serializable-spec? spec)))
      (is (= :nn/invalid-spec (error-key #(blocks/compile-block [:factory (fn [] :not-a-block)])))))
    (let [input (.create manager (float-array [1 2 3 4]) (Shape. (long-array [2 2])))
          inputs (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray [input]))
          ^Block block (initialize! [:lambda {:singleton? true :name "square"}
                                     (fn [^NDArray x] (.square x))] manager inputs)
          ^NDArray output (.head (.forward block (ParameterStore. manager false) inputs false))]
      (is (= [1.0 4.0 9.0 16.0] (vec (.toFloatArray output))))
      (is (= [1.0 2.0 3.0 4.0] (vec (.toFloatArray input)))))))

(deftest multiinput-multioutput-combinations-and-dtype-aware-shapes
  (with-open [manager (NDManager/newBaseManager (Device/cpu) "PyTorch")]
    (let [a (.create manager (long-array [1 2 3 4]) (Shape. (long-array [2 2])))
          b (.create manager (boolean-array [true false true false]) (Shape. (long-array [2 2])))
          inputs (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray [a b]))
          custom (fn [outputs ^NDList original]
                   (when-not (and (= DataType/INT64 (.getDataType ^NDArray (.get original 0)))
                                  (= DataType/BOOLEAN (.getDataType ^NDArray (.get original 1))))
                     (throw (IllegalArgumentException. "Shape probing must preserve input dtypes")))
                   (let [^NDList child (first outputs)]
                     (NDList. ^"[Lai.djl.ndarray.NDArray;"
                      (into-array NDArray [(.mul ^NDArray (.get child 0) (Long/valueOf (long 3))) (.get original 1)]))))
          spec [:parallel {:combine custom} [:identity]]
          ^Block block (initialize! spec manager inputs)
          ^NDList output (.forward block (ParameterStore. manager false) inputs false)
          declared (.getOutputShapes block (.getShapes inputs) (into-array DataType [DataType/INT64 DataType/BOOLEAN]))]
      (is (= [[2 2] [2 2]] (shape-vectors declared) (shape-vectors (.getShapes output))))
      (is (= [[2 2] [2 2]]
             (shape-vectors (.getOutputShapes block (.getShapes inputs) nil))))
      (is (= [3 6 9 12] (vec (.toLongArray ^NDArray (.get output 0)))))
      (is (= DataType/INT64 (.getDataType ^NDArray (.get output 0))))
      (is (= DataType/BOOLEAN (.getDataType ^NDArray (.get output 1))))
      (is (= [true false true false] (vec (.toBooleanArray b))))
      (is (not (blocks/serializable-spec? spec))))
    (doseq [mode [:add :concat]]
      (let [inputs (NDList. ^"[Lai.djl.ndarray.NDArray;"
                    (into-array NDArray [(.ones manager (Shape. (long-array [2 3])))
                                         (.ones manager (Shape. (long-array [2 1 4])))]))
            spec [:parallel {:combine mode} [:identity] [:identity]]
            ^Block block (initialize! spec manager inputs)
            ^NDList output (.forward block (ParameterStore. manager false) inputs false)
            expected (if (= mode :add) [[2 3] [2 1 4]] [[2 6] [2 1 8]])]
        (is (= expected (shape-vectors (.getShapes output))
               (shape-vectors (.getOutputShapes block (.getShapes inputs)))))
        ;; Repeated shape queries must not mutate the input Shape's backing array.
        (is (= [[2 3] [2 1 4]] (shape-vectors (.getShapes inputs))))
        (is (= expected (shape-vectors (.getOutputShapes block (.getShapes inputs)))))))))

(deftest shape-functions-can-avoid-data-dependent-probing
  (let [custom (fn [outputs ^NDList original]
                 (when (.contentEquals (.head original) (Float/valueOf (float 0)))
                   (throw (IllegalArgumentException. "This function cannot evaluate zero probes")))
                 (first outputs))
        shape-fn (fn [children _] (first children))
        spec [:parallel {:combine custom :shape-fn shape-fn} [:linear 3]]]
    (with-open [manager (NDManager/newBaseManager (Device/cpu) "PyTorch")]
      (let [inputs (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray [(.ones manager (Shape. (long-array [2 2])))]))
            result (exercise-gradients! spec manager inputs)]
        (is (= [[2 3]] (:actual result) (:declared result)))
        (is (some #(not (zero? %)) (:gradients result)))
        (is (not (blocks/serializable-spec? spec)))))))

(deftest custom-shape-probes-preserve-context-after-initialization-manager-closes
  (let [custom (fn [_outputs ^NDList original]
                 (let [^NDArray input (.head original)]
                   (when-not (and (= DataType/FLOAT64 (.getDataType input))
                                  (= "PyTorch" (.getEngineName (.getEngine (.getManager input)))))
                     (throw (IllegalArgumentException. "Shape probe lost dtype/engine context")))
                   original))
        ^Block block (blocks/compile-block [:parallel {:combine custom} [:identity]])
        shapes (into-array Shape [(Shape. (long-array [2 3]))])]
    (with-open [manager (NDManager/newBaseManager (Device/cpu) "PyTorch")]
      (with-open [scope (.newSubManager manager)]
        (.initialize block scope DataType/FLOAT64 shapes))
      (is (= [[2 3]] (shape-vectors (.getOutputShapes block shapes)))))))

(deftest skip-accepts-cnn-concat-axis-and-custom-functions
  (with-open [manager (NDManager/newBaseManager (Device/cpu) "PyTorch")]
    (let [custom (fn [outputs ^NDList original]
                   (let [^NDList branch (first outputs)]
                     (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray [(.add (.head branch) (.head original))]))))
          inputs (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray [(.ones manager (Shape. (long-array [2 1 6 6])))]))]
      (doseq [[spec expected] [[[:skip {:combine :concat :axis 1}
                                 [:conv2d {:filters 2 :kernel [1 1]}]] [[2 3 6 6]]]
                               [[:skip {:combine custom}
                                 [:conv2d {:filters 1 :kernel [1 1]}]] [[2 1 6 6]]]]]
        (let [result (exercise-gradients! spec manager inputs)]
          (is (= expected (:actual result) (:declared result)))
          (is (some #(not (zero? %)) (:gradients result))))))))

(deftest named-tensor-operations-and-norm-options
  (with-open [manager (NDManager/newBaseManager (Device/cpu) "PyTorch")]
    (let [spec [:sequential [:reshape {:shape [-1 2 2]}] [:transpose {:axes [0 2 1]}]
                [:layernorm {:axis [2] :epsilon 1.0e-5}]
                [:softmax {:axis 2}]]
          inputs (NDList. ^"[Lai.djl.ndarray.NDArray;"
                  (into-array NDArray [(.create manager (float-array (range 8))
                                                (Shape. (long-array [2 4])))]))
          ^Block block (initialize! spec manager inputs)
          ^NDArray output (.head (.forward block (ParameterStore. manager false) inputs false))
          values (vec (.toFloatArray output))]
      (is (= [[2 2 2]] (shape-vectors (.getOutputShapes block (.getShapes inputs)))))
      (is (every? #(< (Math/abs (- 1.0 (reduce + %))) 1.0e-6) (partition 2 values)))
      (is (blocks/serializable-spec? spec)))))

(deftest last-step-selects-the-last-entry-and-only-removes-its-axis
  (with-open [manager (NDManager/newBaseManager (Device/cpu) "PyTorch")]
    (let [input (.create manager (float-array (range 24)) (Shape. (long-array [2 3 4])))
          inputs (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray [input]))]
      (doseq [[spec expected-shape expected-values]
              [[[:last-step] [2 4] [8.0 9.0 10.0 11.0 20.0 21.0 22.0 23.0]]
               [[:last-step {:axis 1}] [2 4] [8.0 9.0 10.0 11.0 20.0 21.0 22.0 23.0]]
               [[:last-step {:axis -2}] [2 4] [8.0 9.0 10.0 11.0 20.0 21.0 22.0 23.0]]
               [[:last-step {:axis 0}] [3 4] (mapv double (range 12 24))]
               [[:last-step {:axis 2}] [2 3] [3.0 7.0 11.0 15.0 19.0 23.0]]
               [[:last-step {:axis -1}] [2 3] [3.0 7.0 11.0 15.0 19.0 23.0]]]]
        (testing (pr-str spec)
          (let [^Block block (initialize! spec manager inputs)
                ^NDList output (.forward block (ParameterStore. manager false) inputs false)]
            (is (= [expected-shape] (shape-vectors (.getShapes output))
                   (shape-vectors (.getOutputShapes block (.getShapes inputs)))))
            (is (= expected-values (vec (.toFloatArray (.singletonOrThrow output)))))
            (is (blocks/serializable-spec? spec))
            (is (not (.isReleased input)))))))
    (let [input (.create manager (float-array [3 7]) (Shape. (long-array [2 1 1])))
          inputs (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray [input]))
          ^Block block (initialize! [:last-step] manager inputs)
          ^NDList output (.forward block (ParameterStore. manager false) inputs false)]
      (is (= [[2 1]] (shape-vectors (.getShapes output))))
      (is (= [3.0 7.0] (vec (.toFloatArray (.singletonOrThrow output))))))))

(deftest last-step-backpropagates-only-through-the-selected-entry
  (with-open [manager (NDManager/newBaseManager (Device/cpu) "PyTorch")]
    (let [input (.create manager (float-array (range 12)) (Shape. (long-array [2 3 2])))
          _ (.setRequiresGradient input true)
          inputs (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray [input]))
          ^Block block (initialize! [:last-step] manager inputs)]
      (with-open [^GradientCollector collector (.newGradientCollector (.getEngine manager))]
        (let [^NDList output (.forward block (ParameterStore. manager false) inputs true)]
          (.backward collector (.sum (.singletonOrThrow output)))))
      (is (= [0.0 0.0 0.0 0.0 1.0 1.0 0.0 0.0 0.0 0.0 1.0 1.0]
             (vec (.toFloatArray (.getGradient input)))))
      (is (not (.isReleased input))))))

(deftest last-step-rejects-invalid-options-and-unselectable-axes
  (doseq [[spec expected-error]
          [[[:last-step {:axis nil}] :nn/invalid-spec]
           [[:last-step {:axis :time}] :nn/invalid-spec]
           [[:last-step {:axis 1.5}] :nn/invalid-spec]
           [[:last-step {:axis (inc (long Integer/MAX_VALUE))}] :nn/invalid-spec]
           [[:last-step {:axis (dec (long Integer/MIN_VALUE))}] :nn/invalid-spec]
           [[:last-step {:axis 1 :typo true}] :nn/unknown-option]
           [[:last-step nil] :nn/invalid-options]
           [[:last-step {} {}] :nn/invalid-spec]]]
    (is (= expected-error (error-key #(blocks/compile-block spec))) (pr-str spec)))
  (doseq [[spec dimensions]
          [[[:last-step {:axis 3}] [2 3 4]]
           [[:last-step {:axis -4}] [2 3 4]]
           [[:last-step] [2]]
           [[:last-step] [2 0 4]]
           [[:last-step {:axis 0}] []]]]
    (let [^Block block (blocks/compile-block spec)]
      (is (= :nn/invalid-spec
             (error-key #(.getOutputShapes block (into-array Shape [(Shape. (long-array dimensions))]))))
          (pr-str [spec dimensions])))))

(deftest last-step-network-spec-and-parameters-survive-save-and-load
  (let [directory (.toFile (Files/createTempDirectory "garun-last-step-" (make-array FileAttribute 0)))]
    (try
      (with-open [current-session (session/open)]
        (let [^NDManager manager (context/manager current-session)
              input (.create manager (float-array (range 24)) (Shape. (long-array [2 3 4])))
              inputs (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray [input]))]
          (doseq [last-step-spec [[:last-step] [:last-step {:axis 1}]]]
            (let [spec [:sequential last-step-spec [:linear 2]]
                  network (nn/build current-session spec {:input-shape [3 4]})]
              (nn/save! current-session network directory)
              (let [restored (nn/load current-session directory)]
                (with-open [^NDList before (nn/predict current-session network inputs {})
                            ^NDList after (nn/predict current-session restored inputs {})]
                  (is (= spec (:spec restored)))
                  (is (= [[2 2]] (shape-vectors (.getShapes before)) (shape-vectors (.getShapes after))))
                  (is (.allClose (.head before) (.head after) 1.0e-6 1.0e-6 false))))))
          (is (not (.isReleased input)))))
      (finally
        (doseq [^File file (reverse (file-seq directory))] (.delete file))))))

(deftest serialization-policy-rejects-native-values-before-metadata
  (doseq [spec [[:lambda identity] [:factory (fn [] (Blocks/identityBlock))]
                [:block (Blocks/identityBlock)]
                [:parallel {:combine (fn [children _] (first children))} [:identity]]
                [:word-embedding {:vocabulary (let [^java.util.List tokens ["a"]]
                                                (ai.djl.modality.nlp.DefaultVocabulary. tokens))
                                  :embedding-size 2}]
                [:not-a-real-block]]]
    (is (not (blocks/serializable-spec? spec))))
  (let [directory (.toFile (Files/createTempDirectory "garun-block-policy-" (make-array FileAttribute 0)))]
    (try
      (with-open [current-session (session/open)]
        (let [network (nn/build current-session
                                [:lambda {:singleton? true}
                                 (fn [^NDArray input] (.mul input (Float/valueOf (float 2))))]
                                {:input-shape [2]})]
          (is (some? (error-key #(nn/save! current-session network directory))))
          (is (empty? (.listFiles ^File directory)))))
      (finally
        (doseq [^File file (reverse (file-seq directory))] (.delete file))))))

(deftest cnn-network-edn-and-parameters-roundtrip-through-runtime
  (let [spec [:sequential
              [:parallel {:combine :concat :axis 1}
               [:conv2d {:filters 2 :kernel [1 1]}]
               [:conv2d {:filters 3 :kernel [1 1]}]]
              [:global-avg-pool2d]
              [:linear 2]]
        directory (.toFile (Files/createTempDirectory "garun-cnn-save-" (make-array FileAttribute 0)))]
    (try
      (with-open [current-session (session/open)]
        (let [^NDManager manager (context/manager current-session)
              network (nn/build current-session spec {:input-shape [1 4 4]})
              input (.ones manager (Shape. (long-array [2 1 4 4])))
              inputs (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray [input]))]
          (nn/save! current-session network directory)
          (let [restored (nn/load current-session directory)]
            (with-open [^NDList before (nn/predict current-session network inputs {})
                        ^NDList after (nn/predict current-session restored inputs {})]
              (is (= spec (:spec restored)))
              (is (= [[2 2]] (shape-vectors (.getShapes before))))
              (is (.allClose (.head before) (.head after) 1.0e-6 1.0e-6 false))))
          (is (not (.isReleased input)))))
      (finally
        (doseq [^File file (reverse (file-seq directory))] (.delete file))))))

(deftest ghost-batchnorm-restores-batch-shape-and-keeps-gradients
  (with-open [manager (NDManager/newBaseManager (Device/cpu) "PyTorch")]
    (let [inputs (NDList. ^"[Lai.djl.ndarray.NDArray;"
                  (into-array NDArray [(.create manager (float-array [1 3 5 7])
                                                (Shape. (long-array [4 1])))]))
          result (exercise-gradients! [:ghost-batchnorm {:virtual-batch-size 2}] manager inputs)
          ^NDArray output (.singletonOrThrow ^NDList (:outputs result))]
      (is (= [[4 1]] (:actual result) (:declared result)))
      (is (every? #(< (Math/abs (double %)) 1.0e-4)
                  (map - (.toFloatArray output) [-1 1 -1 1])))
      (is (some #(not (zero? %)) (:gradients result))))))
