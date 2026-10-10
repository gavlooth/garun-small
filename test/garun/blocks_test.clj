(ns garun.blocks-test
  (:require [clojure.test :refer [deftest is testing]]
            [garun.context :as context]
            [garun.blocks :as blocks]
            [garun.nn :as nn]
            [garun.ndarray :as nd]
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

(defn- directional-reference!
  "Run one direction independently; the reverse direction reads time backwards."
  [tag options ^NDManager manager ^NDArray input ^Block bidirectional direction time-axis]
  (let [reverse? (= direction "r")
        reference-input (if reverse? (nd/flip input time-axis) input)
        inputs (nd/ndlist [reference-input])
        ^Block reference (initialize! [tag (assoc options :bidirectional? false :return-state? false)]
                                      manager inputs)]
    (doseq [^Parameter parameter (.values (.getDirectParameters reference))]
      (let [name (str direction (subs (.getName parameter) 1))
            ^Parameter original (.get (.getDirectParameters bidirectional) name)]
        (.set (.getArray parameter) (.toFloatArray (.getArray original)))))
    (let [^NDArray output (.head (.forward reference (ParameterStore. manager false) inputs false))]
      (if reverse? (nd/flip output time-axis) output))))

(deftest bidirectional-recurrent-values-match-independent-directions
  (with-open [manager (NDManager/newBaseManager (Device/cpu) "PyTorch")]
    (doseq [tag [:rnn :gru :lstm]
            batch-first? [true false]
            bias? [true false]]
      (testing (pr-str [tag batch-first? bias?])
        (with-open [scope (.newSubManager manager)]
          (let [dimensions (if batch-first? [2 4 3] [4 2 3])
                input (.create scope (float-array (map #(/ % 20.0) (range 24)))
                               (Shape. (long-array dimensions)))
                options {:state-size 2 :batch-first? batch-first? :bidirectional? true :bias? bias?}
                inputs (nd/ndlist [input])
                ^Block block (initialize! [tag options] scope inputs)
                time-axis (if batch-first? 1 0)]
            ;; Distinct direction weights make an accidental branch swap observable.
            (doseq [^Parameter parameter (.values (.getDirectParameters block))]
              (let [base (if (.startsWith (.getName parameter) "l") 0.02 -0.04)]
                (.set (.getArray parameter)
                      (float-array (map #(+ base (* 0.002 %))
                                        (range (.size (.getArray parameter))))))))
            (let [left (directional-reference! tag options scope input block "l" time-axis)
                  right (directional-reference! tag options scope input block "r" time-axis)
                  expected (nd/concat [left right] 2)
                  ^NDArray actual (.head (.forward block (ParameterStore. scope false) inputs false))]
              (is (= (assoc dimensions 2 4) (nd/shape actual)))
              (is (.allClose actual expected 1.0e-6 1.0e-6 false))
              (is (not (.isReleased input))))))))))

(deftest bidirectional-recurrent-layers-states-gradients-and-serialization
  (with-open [manager (NDManager/newBaseManager (Device/cpu) "PyTorch")]
    (doseq [tag [:rnn :gru :lstm]
            layers [1 2]
            batch-first? [true false]
            bias? [true false]]
      (testing (pr-str [tag layers batch-first? bias?])
        (with-open [scope (.newSubManager manager)]
          (let [dimensions (if batch-first? [2 3 2] [3 2 2])
                spec [tag {:state-size 3 :layers layers :batch-first? batch-first?
                           :bidirectional? true :bias? bias? :return-state? true}]
                input (.ones scope (Shape. (long-array dimensions)))
                inputs (nd/ndlist [input])
                {:keys [block actual declared gradients]} (exercise-gradients! spec scope inputs)
                expected (into [(assoc dimensions 2 6)]
                               (repeat (if (= tag :lstm) 2 1) [(* 2 layers) 2 3]))
                restored (parameter-roundtrip! spec scope inputs block)]
            (is (= expected actual declared))
            (is (= expected (mapv nd/shape restored)))
            (is (seq gradients))
            (is (every? #(Double/isFinite (double %)) gradients))
            (is (some #(not (zero? %)) gradients))))))))

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
          twice (fn [arrays]
                  (nd/ndlist (mapv #(nd/mul % 2) arrays)))
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
                                     (fn [array] (nd/square array))] manager inputs)
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

(deftest index-supports-djl-grammar-and-infers-output-shapes
  (with-open [manager (NDManager/newBaseManager (Device/cpu) "PyTorch")]
    (let [input (.create manager (float-array (range 24)) (Shape. (long-array [2 3 4])))
          inputs (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray [input]))]
      (doseq [[index expected-shape expected-values]
              [[":, -1" [2 4] [8.0 9.0 10.0 11.0 20.0 21.0 22.0 23.0]]
               [":, ::2, :" [2 2 4] [0.0 1.0 2.0 3.0 8.0 9.0 10.0 11.0
                                     12.0 13.0 14.0 15.0 20.0 21.0 22.0 23.0]]
               ["..., 0" [2 3] [0.0 4.0 8.0 12.0 16.0 20.0]]
               [":, 1:3, 1:4:2" [2 2 2] [5.0 7.0 9.0 11.0 17.0 19.0 21.0 23.0]]
               ["-1" [3 4] (mapv double (range 12 24))]
               [":" [2 3 4] (mapv double (range 24))]]]
        (testing index
          (let [spec [:index index]
                ^Block block (blocks/compile-block spec)
                declared (shape-vectors (.getOutputShapes block (.getShapes inputs)))
                _ (.initialize block manager DataType/FLOAT32 (.getShapes inputs))
                ^NDList output (.forward block (ParameterStore. manager false) inputs false)]
            (is (= [expected-shape] declared (shape-vectors (.getShapes output))
                   (shape-vectors (.getOutputShapes block (.getShapes inputs)))))
            (is (= expected-values (vec (.toFloatArray (.singletonOrThrow output)))))
            (is (blocks/serializable-spec? spec))
            (is (not (.isReleased input)))))))
    (let [input (.create manager (float-array [3 7]) (Shape. (long-array [2 1 1])))
          inputs (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray [input]))
          ^Block block (initialize! [:index ":, -1"] manager inputs)
          ^NDList output (.forward block (ParameterStore. manager false) inputs false)]
      (is (= [[2 1]] (shape-vectors (.getShapes output))
             (shape-vectors (.getOutputShapes block (.getShapes inputs)))))
      (is (= [3.0 7.0] (vec (.toFloatArray (.singletonOrThrow output))))))))

(deftest index-backpropagates-only-through-selected-entries
  (with-open [manager (NDManager/newBaseManager (Device/cpu) "PyTorch")]
    (doseq [[index expected-gradient]
            [[":, -1" [0.0 0.0 0.0 0.0 1.0 1.0 0.0 0.0 0.0 0.0 1.0 1.0]]
             [":, ::2, :" [1.0 1.0 0.0 0.0 1.0 1.0 1.0 1.0 0.0 0.0 1.0 1.0]]
             ["..., 0" [1.0 0.0 1.0 0.0 1.0 0.0 1.0 0.0 1.0 0.0 1.0 0.0]]]]
      (testing index
        (with-open [scope (.newSubManager manager)]
          (let [input (.create scope (float-array (range 12)) (Shape. (long-array [2 3 2])))
                _ (.setRequiresGradient input true)
                inputs (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray [input]))
                ^Block block (initialize! [:index index] scope inputs)]
            (with-open [^GradientCollector collector (.newGradientCollector (.getEngine scope))]
              (let [^NDList output (.forward block (ParameterStore. scope false) inputs true)]
                (.backward collector (.sum (.singletonOrThrow output)))))
            (is (= expected-gradient (vec (.toFloatArray (.getGradient input)))))
            (is (not (.isReleased input)))))))))

(deftest index-rejects-invalid-specifications-and-nonsingleton-inputs
  (doseq [spec [[:index] [:index nil] [:index :time] [:index 1] [:index 1.5]
                [:index true] [:index \:] [:index {}] [:index [":, -1"]]
                [:index (StringBuilder. ":, -1")] [:index ""]
                [:index "\u0000"] [:index ":,\u0000-1"]
                [:index ":" nil] [:index ":" ":"]]]
    (is (= :nn/invalid-spec (error-key #(blocks/compile-block spec))) (pr-str spec)))
  (with-open [manager (NDManager/newBaseManager (Device/cpu) "PyTorch")]
    (doseq [input-count [0 2]]
      (let [inputs (NDList. ^java.util.Collection
                    (vec (repeat input-count (.ones manager (Shape. (long-array [2 3 4]))))))
            ^Block block (blocks/compile-block [:index ":, -1"])]
        (is (thrown? IndexOutOfBoundsException (.getOutputShapes block (.getShapes inputs))))
        (is (thrown? IndexOutOfBoundsException
                     (.forward block (ParameterStore. manager false) inputs false)))))))

(deftest index-network-spec-and-parameters-survive-save-and-load
  (let [directory (.toFile (Files/createTempDirectory "garun-index-" (make-array FileAttribute 0)))]
    (try
      (with-open [current-session (session/open)]
        (let [^NDManager manager (context/manager current-session)
              input (.create manager (float-array (range 24)) (Shape. (long-array [2 3 4])))
              inputs (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray [input]))
              spec [:sequential [:index ":, -1"] [:linear 2]]
              network (nn/build current-session spec {:input-shape [3 4]})]
          (nn/save! current-session network directory)
          (let [restored (nn/load current-session directory)]
            (with-open [^NDList before (nn/predict current-session network inputs {})
                        ^NDList after (nn/predict current-session restored inputs {})]
              (is (= spec (:spec restored)))
              (is (= [[2 2]] (shape-vectors (.getShapes before)) (shape-vectors (.getShapes after))))
              (is (.allClose (.head before) (.head after) 1.0e-6 1.0e-6 false))))
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
                                 (fn [array] (nd/mul array 2))]
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

(def ^:private image-measurements-graph
  [:graph {:inputs [:image :measurements] :outputs [:logits]}
   [:features [:image]
    [:sequential [:conv2d {:filters 8 :kernel [3 3]}] [:relu] [:global-avg-pool2d]]]
   [:joined [:features :measurements] [:concat {:axis 1}]]
   [:hidden [:joined] [:sequential [:linear 32] [:relu]]]
   [:logits [:hidden] [:linear 3]]])

(deftest graph-network-shapes-and-save-load
  (let [directory (.toFile (Files/createTempDirectory "garun-graph-" (make-array FileAttribute 0)))]
    (try
      (with-open [current-session (session/open)]
        (let [^NDManager manager (context/manager current-session)
              network (nn/build current-session image-measurements-graph
                                {:input-shapes [[1 1 8 8] [1 4]]})
              ^Block block (.getBlock ^ai.djl.Model (:model network))
              image (.ones manager (Shape. (long-array [2 1 8 8])))
              measurements (.ones manager (Shape. (long-array [2 4])))
              inputs (NDList. ^"[Lai.djl.ndarray.NDArray;"
                      (into-array NDArray [image measurements]))]
          (is (blocks/serializable-spec? image-measurements-graph))
          (is (= [[2 3]] (shape-vectors (.getOutputShapes block (.getShapes inputs)))))
          (nn/save! current-session network directory)
          (let [restored (nn/load current-session directory)]
            (with-open [^NDList before (nn/predict current-session network inputs {})
                        ^NDList after (nn/predict current-session restored inputs {})]
              (is (= image-measurements-graph (:spec restored)))
              (is (= [[1 1 8 8] [1 4]] (:input-shapes restored)))
              (is (= [[2 3]] (shape-vectors (.getShapes before))
                     (shape-vectors (.getShapes after))))
              (is (.allClose (.head before) (.head after) 0.0 0.0 false))))
          (is (not (.isReleased image)))
          (is (not (.isReleased measurements)))))
      (finally
        (doseq [^File file (reverse (file-seq directory))] (.delete file))))))

(deftest graph-out-of-order-skips-output-order-and-every-node-gradient
  (with-open [manager (NDManager/newBaseManager (Device/cpu) "PyTorch")]
    (let [spec [:graph {:inputs [:input] :outputs [:sum :second :input]}
                [:sum [:third :first] [:add]]
                [:third [:second] [:linear {:units 3 :bias? false}]]
                [:second [:first] [:linear {:units 3 :bias? false}]]
                [:first [:input] [:linear {:units 3 :bias? false}]]]
          input (.create manager (float-array [1 2 3 4]) (Shape. (long-array [2 2])))
          inputs (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray [input]))
          ^Block block (initialize! spec manager inputs)]
      (doseq [^Parameter parameter (.values (.getParameters block))]
        (.set (.getArray parameter) (float-array (repeat (.size (.getArray parameter)) 1))))
      (let [^NDList outputs
            (with-open [^GradientCollector collector (.newGradientCollector (.getEngine manager))]
              (let [^NDList outputs (.forward block (ParameterStore. manager false) inputs true)]
                (.backward collector (.sum (.head outputs)))
                outputs))
            ^NDList restored (parameter-roundtrip! spec manager inputs block)]
        (is (= [[2 3] [2 3] [2 2]] (shape-vectors (.getShapes outputs))
               (shape-vectors (.getOutputShapes block (.getShapes inputs)))))
        (is (= [30.0 30.0 30.0 70.0 70.0 70.0]
               (vec (.toFloatArray ^NDArray (.get outputs 0)))))
        (is (= [9.0 9.0 9.0 21.0 21.0 21.0]
               (vec (.toFloatArray ^NDArray (.get outputs 1)))))
        (is (identical? input (.get outputs 2)))
        (is (= 3 (.size (.getParameters block))))
        (doseq [^Parameter parameter (.values (.getParameters block))]
          (testing (.getName parameter)
            (is (.hasGradient (.getArray parameter)))
            (let [gradient (vec (.toFloatArray (.getGradient (.getArray parameter))))]
              (is (every? #(Double/isFinite (double %)) gradient))
              (is (every? pos? gradient)))))
        (doseq [index (range (.size outputs))]
          (is (.allClose ^NDArray (.get outputs (int index))
                         ^NDArray (.get restored (int index)) 0.0 0.0 false)))
        (is (blocks/serializable-spec? spec))
        (is (not (.isReleased input)))))))

(deftest graph-default-concat-add-broadcasting-and-input-only-outputs
  (with-open [manager (NDManager/newBaseManager (Device/cpu) "PyTorch")]
    (let [left (.ones manager (Shape. (long-array [2 3])))
          right (.ones manager (Shape. (long-array [2 1])))
          inputs (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray [left right]))]
      (doseq [[spec expected]
              [[[:graph {:inputs [:left :right] :outputs [:joined]}
                 [:joined [:left :right :right] [:concat]]] [[2 5]]]
               [[:graph {:inputs [:left :right] :outputs [:sum]}
                 [:sum [:left :right :right] [:add]]] [[2 3]]]
               [[:graph {:inputs [:left :right] :outputs [:right :left :right]}]
                [[2 1] [2 3] [2 1]]]]]
        (let [^Block block (initialize! spec manager inputs)
              ^NDList output (.forward block (ParameterStore. manager false) inputs false)]
          (is (= expected (shape-vectors (.getOutputShapes block (.getShapes inputs)))
                 (shape-vectors (.getShapes output))))
          (is (blocks/serializable-spec? spec))
          (is (not (.isReleased left)))
          (is (not (.isReleased right))))))))

(deftest graph-native-multiinput-nodes-receive-their-dependencies-in-order
  (with-open [manager (NDManager/newBaseManager (Device/cpu) "PyTorch")]
    (let [left (.ones manager (Shape. (long-array [2 3])))
          right (.ones manager (Shape. (long-array [2 1])))
          inputs (NDList. ^"[Lai.djl.ndarray.NDArray;" (into-array NDArray [left right]))
          ^Block native
          (blocks/compile-block
           [:lambda (fn [^NDList inputs]
                      (NDList. ^"[Lai.djl.ndarray.NDArray;"
                       (into-array NDArray [(.sub ^NDArray (.get inputs 0)
                                                  ^NDArray (.get inputs 1))])))])
          spec [:graph {:inputs [:left :right] :outputs [:difference]}
                [:difference [:right :left] native]]
          ^Block block (initialize! spec manager inputs)
          ^NDList output (.forward block (ParameterStore. manager false) inputs false)]
      (is (= [[2 1] [2 3]] (shape-vectors (.getInputShapes native))))
      (is (= [[2 3]] (shape-vectors (.getShapes output))
             (shape-vectors (.getOutputShapes block (.getShapes inputs)))))
      (is (every? zero? (.toFloatArray (.head output))))
      (is (not (blocks/serializable-spec? spec)))
      (is (not (blocks/serializable-spec?
                [:graph {:inputs [:input] :outputs [:output]}
                 [:output [:input] [:lambda identity]]]))))))

(deftest graph-rejects-invalid-options-names-dependencies-cycles-and-merges
  (doseq [spec
          [[:graph]
           [:graph nil]
           [:graph {:inputs [:input] :outputs [:input] :typo true}]
           [:graph {:inputs [] :outputs [:output]}]
           [:graph {:inputs [:input] :outputs []}]
           [:graph {:inputs '(:input) :outputs [:input]}]
           [:graph {:inputs [:input] :outputs '(:input)}]
           [:graph {:inputs ["input"] :outputs [:input]}]
           [:graph {:inputs [:input] :outputs ["input"]}]
           [:graph {:inputs [:input :input] :outputs [:input]}]
           [:graph {:inputs [:input] :outputs [:missing]}]
           [:graph {:inputs [:input] :outputs [:output]} [:output [:missing] [:identity]]]
           [:graph {:inputs [:input] :outputs [:input]} [:input [:input] [:identity]]]
           [:graph {:inputs [:input] :outputs [:output]}
            [:output [:input] [:identity]] [:output [:input] [:identity]]]
           [:graph {:inputs [:input] :outputs [:output]} [:output [:output] [:identity]]]
           [:graph {:inputs [:input] :outputs [:a]}
            [:a [:b] [:identity]] [:b [:a] [:identity]]]
           [:graph {:inputs [:input] :outputs [:input]} [:unused [:input] [:identity]]]
           [:graph {:inputs [:input] :outputs [:output]} ["output" [:input] [:identity]]]
           [:graph {:inputs [:input] :outputs [:output]} [:output [] [:identity]]]
           [:graph {:inputs [:input] :outputs [:output]} [:output '(:input) [:identity]]]
           [:graph {:inputs [:input] :outputs [:output]} [:output ["input"] [:identity]]]
           [:graph {:inputs [:input] :outputs [:output]} [:output [:input]]]
           [:graph {:inputs [:input] :outputs [:output]} '(:output [:input] [:identity])]
           [:graph {:inputs [:input] :outputs [:output]} [:output [:input] [:add]]]
           [:graph {:inputs [:input] :outputs [:output]} [:output [:input] [:concat]]]
           [:graph {:inputs [:input] :outputs [:output]}
            [:output [:input :input] [:add {}]]]
           [:graph {:inputs [:input] :outputs [:output]}
            [:output [:input :input] [:concat nil]]]
           [:graph {:inputs [:input] :outputs [:output]}
            [:output [:input :input] [:concat {:axis 1.5}]]]
           [:graph {:inputs [:input] :outputs [:output]}
            [:output [:input :input] [:concat {:axis 2147483648}]]]
           [:graph {:inputs [:input] :outputs [:output]}
            [:output [:input :input] [:concat {:typo 1}]]]
           [:graph {:inputs [:input] :outputs [:output]}
            [:output [:input :input] [:concat {} {}]]]]]
    (is (= :nn/invalid-graph (error-key #(blocks/compile-block spec))) (pr-str spec)))
  (doseq [spec [[:concat] [:add] [:sequential [:concat]]]]
    (is (= :nn/unknown-tag (error-key #(blocks/compile-block spec))) (pr-str spec))))
