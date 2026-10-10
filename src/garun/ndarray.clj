(ns garun.ndarray
  "Reflection-free functions over DJL NDArray and NDList values.

  get, abs, max, min, concat, flatten, sort and repeat shadow clojure.core
  names and are excluded from this namespace's core referrals. Use the nd alias.
  Operations retain DJL's dtype, broadcasting, gradient and manager semantics;
  they do not copy or transfer inputs unless the named DJL operation does.
  ndlist borrows its arrays: closing that NDList also closes its elements."
  (:refer-clojure :exclude [get abs max min concat flatten sort repeat])
  (:require [garun.context :as context]
            [garun.tensor :as tensor])
  (:import [ai.djl.ndarray NDArray NDArrays NDList NDManager]
           [ai.djl.ndarray.index NDIndex]
           [ai.djl.ndarray.types Shape]
           [garun.context Context]
           [java.util Collection]))

(set! *warn-on-reflection* true)

(def ^:private empty-index-arguments (object-array 0))

(def ^:private int-array-class (class (int-array 0)))

(def ^:private long-array-class (class (long-array 0)))

(defn- primitive-axes
  ^ints [axes]
  (if (instance? int-array-class axes) axes (int-array axes)))

(defmacro ^:private def-unary [function method & [documentation]]
  (let [array (with-meta 'array {:tag 'ai.djl.ndarray.NDArray})]
    `(defn ~(with-meta function {:tag 'ai.djl.ndarray.NDArray})
       ~(or documentation (str "Call DJL NDArray." method "()."))
       [~array]
       (. ~array ~(symbol method)))))

(def-unary neg "neg")

(def-unary abs "abs")

(def-unary square "square")

(def-unary sqrt "sqrt")

(def-unary exp "exp")

(def-unary log "log")

(def-unary sin "sin")

(def-unary cos "cos")

(def-unary tanh "tanh")

(def-unary sign "sign")

(def-unary floor "floor")

(def-unary ceil "ceil")

(def-unary round "round")

(def-unary flatten "flatten")

(def-unary duplicate "duplicate")

(def-unary zeros-like "zerosLike"
  "Call DJL NDArray.zerosLike(), preserving shape, dtype and device.
  Owned by the input manager, or its device-specific child if devices differ.")

(def-unary ones-like "onesLike"
  "Call DJL NDArray.onesLike(), preserving shape, dtype and device.
  Owned by the input manager, or its device-specific child if devices differ.")

(def-unary log10 "log10")

(def-unary log2 "log2")

(def-unary erf "erf")

(def-unary logical-not "logicalNot")

(def-unary is-nan "isNaN")

(def-unary is-infinite "isInfinite")

(def-unary stop-gradient "stopGradient"
  "Call DJL NDArray.stopGradient(), returning a detached array on the same manager/device.
  PyTorch shares its storage with the input, but stops tracking its gradients.")

(def-unary gradient "getGradient"
  "Call DJL NDArray.getGradient() on an array with gradient tracking enabled.
  The native gradient belongs to the input manager; it is not an independent copy.")

(defmacro ^:private def-binary [function method & [array-only?]]
  (let [array (with-meta 'array {:tag 'ai.djl.ndarray.NDArray})
        other (cond-> 'other array-only? (with-meta {:tag 'ai.djl.ndarray.NDArray}))]
    `(defn ~(with-meta function {:tag 'ai.djl.ndarray.NDArray})
       ~(str "Call DJL NDArray." method "(NDArray)"
             (if array-only? "." (str " or " method "(Number), preserving broadcasting.")))
       [~array ~other]
       ~(if array-only?
          `(. ~array ~(symbol method) ~other)
          `(if (instance? NDArray ~other)
             (. ~array ~(symbol method) ~(with-meta other {:tag 'ai.djl.ndarray.NDArray}))
             (. ~array ~(symbol method) ~(with-meta other {:tag 'java.lang.Number})))))))

(def-binary add "add")

(def-binary sub "sub")

(def-binary mul "mul")

(def-binary div "div")

(def-binary pow "pow")

(def-binary maximum "maximum")

(def-binary minimum "minimum")

(def-binary eq "eq")

(def-binary gt "gt")

(def-binary gte "gte")

(def-binary lt "lt")

(def-binary lte "lte")

(def-binary logical-and "logicalAnd" :array-only)

(def-binary logical-or "logicalOr" :array-only)

(def-binary atan2 "atan2" :array-only)

(def-binary batch-matmul "batchMatMul" :array-only)

(defmacro ^:private def-reduction [function method]
  (let [array (with-meta 'array {:tag 'ai.djl.ndarray.NDArray})
        axes 'axes
        keep-dims? 'keep-dims?]
    `(defn ~(with-meta function {:tag 'ai.djl.ndarray.NDArray})
       ~(str "Call DJL NDArray." method "(), " method "(int[]) or " method "(int[], boolean).\n"
             "  Axes are a vector or int-array; omitted/nil reduces all dimensions.\n"
             "  keep-dims? defaults to false.")
       ([~array] (. ~array ~(symbol method)))
       ([~array ~axes]
        (if (nil? ~axes)
          (~function ~array)
          (. ~array ~(symbol method) (primitive-axes ~axes))))
       ([~array ~axes ~keep-dims?]
        (if (nil? ~axes)
          ;; PyTorch max/min/prod reject multi-axis lists, so reduce every
          ;; element and restore size-one dimensions when requested.
          (let [^NDArray result# (. ~array ~(symbol method))]
            (if ~keep-dims?
              (.reshape result# (long-array (clojure.core/repeat (.dimension (.getShape ~array)) 1)))
              result#))
          (. ~array ~(symbol method) (primitive-axes ~axes) (boolean ~keep-dims?)))))))

(def-reduction sum "sum")

(def-reduction mean "mean")

(def-reduction max "max")

(def-reduction min "min")

(def-reduction prod "prod")

(defn- single-axis [axes]
  (cond
    (nil? axes) nil
    (integer? axes) (int axes)
    (= 1 (count axes)) (int (first axes))
    :else (throw (ex-info "DJL argMax/argMin accepts exactly one axis"
                          {:error :ndarray/invalid-axes :axes axes}))))

(defmacro ^:private def-index-reduction [function method]
  (let [array (with-meta 'array {:tag 'ai.djl.ndarray.NDArray})
        axes 'axes
        keep-dims? 'keep-dims?]
    `(defn ~(with-meta function {:tag 'ai.djl.ndarray.NDArray})
       ~(str "Call DJL NDArray." method "() or " method "(int).\n"
             "  DJL accepts one axis: an integer or one-element vector/int-array.\n"
             "  Omitted/nil axes returns a flat index over all elements.\n"
             "  keep-dims? restores reduced dimensions as size-one axes.")
       ([~array] (. ~array ~(symbol method)))
       ([~array ~axes] (~function ~array ~axes false))
       ([~array ~axes ~keep-dims?]
        (let [axis# (single-axis ~axes)
              ^NDArray result# (if (nil? axis#)
                                 (. ~array ~(symbol method))
                                 (. ~array ~(symbol method) (int axis#)))]
          (cond
            (not ~keep-dims?) result#
            (nil? axis#) (.reshape result# (long-array (clojure.core/repeat (.dimension (.getShape ~array)) 1)))
            :else (.expandDims result# (int axis#))))))))

(def-index-reduction argmax "argMax")

(def-index-reduction argmin "argMin")

(defn get
  "Call DJL NDArray.get with an index string, NDIndex, long-array/vector,
  or individual integer indices, e.g. (get array 1 2). String grammar is DJL's."
  (^NDArray [^NDArray array index]
   (cond
     (string? index) (.get array ^String index ^objects empty-index-arguments)
     (instance? NDIndex index) (.get array ^NDIndex index)
     (integer? index) (.get array (long-array [index]))
     (instance? long-array-class index) (.get array ^longs index)
     :else (.get array (long-array index))))
  (^NDArray [^NDArray array index other & indices]
   (.get array (long-array (cons index (cons other indices))))))

(defn shape
  "Return DJL NDArray.getShape().getShape() as a Clojure dimensions vector."
  [^NDArray array]
  (vec (.getShape (.getShape array))))

(defn reshape
  "Call DJL NDArray.reshape(long[]) with a dimensions vector; -1 infers a dimension."
  ^NDArray [^NDArray array dimensions]
  (.reshape array (long-array dimensions)))

(defn transpose
  "Call DJL NDArray.transpose(), or transpose(int[]) with an axes vector/int-array."
  (^NDArray [^NDArray array] (.transpose array))
  (^NDArray [^NDArray array axes] (.transpose array (primitive-axes axes))))

(defn expand-dims
  "Call DJL NDArray.expandDims(int) to insert a size-one axis."
  ^NDArray [^NDArray array axis]
  (.expandDims array (int axis)))

(defn squeeze
  "Call DJL NDArray.squeeze(), squeeze(int), or squeeze(int[]) with axes."
  (^NDArray [^NDArray array] (.squeeze array))
  (^NDArray [^NDArray array axes]
   (if (integer? axes)
     (.squeeze array (int axes))
     (.squeeze array (primitive-axes axes)))))

(defn ndlist
  "Construct DJL NDList from a vector/collection of borrowed arrays, without copying tensors.
  Closing the resulting list closes those arrays; it does not close their manager."
  (^NDList [] (NDList.))
  (^NDList [arrays] (NDList. ^Collection arrays)))

(defn head
  "Call DJL NDList.head() to return its first array."
  ^NDArray [^NDList arrays]
  (.head arrays))

(defn singleton
  "Call DJL NDList.singletonOrThrow(); fail unless the list contains exactly one array."
  ^NDArray [^NDList arrays]
  (.singletonOrThrow arrays))

(defn concat
  "Call DJL NDArrays.concat(NDList, int) on a vector of arrays; axis defaults to 0."
  (^NDArray [arrays] (concat arrays 0))
  (^NDArray [arrays axis] (NDArrays/concat (ndlist arrays) (int axis))))

(defn stack
  "Call DJL NDArrays.stack(NDList, int) on a vector of arrays; axis defaults to 0."
  (^NDArray [arrays] (stack arrays 0))
  (^NDArray [arrays axis] (NDArrays/stack (ndlist arrays) (int axis))))

(defn matmul
  "Call DJL NDArray.matMul(NDArray)."
  ^NDArray [^NDArray array ^NDArray other]
  (.matMul array other))

(defn dot
  "Call DJL NDArray.dot(NDArray)."
  ^NDArray [^NDArray array ^NDArray other]
  (.dot array other))

(defn softmax
  "Call DJL NDArray.softmax(int) along the given axis."
  ^NDArray [^NDArray array axis]
  (.softmax array (int axis)))

(defn log-softmax
  "Call DJL NDArray.logSoftmax(int) along the given axis."
  ^NDArray [^NDArray array axis]
  (.logSoftmax array (int axis)))

(defn to-type
  "Call DJL NDArray.toType(DataType, boolean), resolving dtype with tensor/dtype.
  copy? defaults to false; a matching dtype may return the original array.
  Unsupported backend dtypes fail before entering JNI."
  (^NDArray [^NDArray array dtype] (to-type array dtype false))
  (^NDArray [^NDArray array dtype copy?]
   (.toType array (tensor/dtype dtype (.getEngine (.getManager array))) (boolean copy?))))

(defn full-like
  "Call the input's NDManager.full(Shape, float, DataType, Device), preserving shape/dtype/device.
  DJL has no NDArray.fullLike method. The fill value is coerced to DJL's float argument.
  Owned by the input manager, or its device-specific child if devices differ."
  ^NDArray [^NDArray array value]
  (.full (.getManager array) (.getShape array) (float value) (.getDataType array) (.getDevice array)))

(defn- manager-create
  ^NDArray [^NDManager manager operation dimensions options]
  (let [^Shape dimensions (tensor/shape dimensions)
        dtype (tensor/dtype (:dtype options (if (= :arange operation) :int32 :float32))
                            (.getEngine manager))]
    (case operation
      :zeros (.zeros manager dimensions dtype)
      :ones (.ones manager dimensions dtype)
      :full (.full manager dimensions (float (:value options 0)) dtype)
      :random-normal (.randomNormal manager (float (:mean options 0))
                                    (float (:standard-deviation options 1)) dimensions dtype)
      :random-uniform (.randomUniform manager (float (:low options 0))
                                      (float (:high options 1)) dimensions dtype)
      :arange (let [start (:start options 0)
                    step (:step options 1)
                    end (+ start (* step (.size dimensions)))]
                (with-open [^NDArray flat (if (and (integer? start) (integer? step))
                                            (.arange manager (int start) (int end) (int step) dtype)
                                            (.arange manager (float start) (float end) (float step) dtype))]
                  (.reshape flat dimensions))))))

(defmacro ^:private def-creation [function method documentation]
  (let [owner (with-meta 'owner {:tag 'Object})
        dimensions 'dimensions
        options 'options]
    `(defn ~(with-meta function {:tag 'ai.djl.ndarray.NDArray})
       ~(str "Call DJL NDManager." method ", owned by the given NDManager or Context's manager.\n"
             "  The result uses that manager's device; dimensions is a shape vector.\n"
             "  :dtype is resolved by tensor/dtype against the owner's engine.\n"
             "  " documentation)
       ([~owner ~dimensions] (~function ~owner ~dimensions {}))
       ([~owner ~dimensions ~options]
        (if (instance? Context ~owner)
          (context/call ~owner #(manager-create (context/manager ~owner)
                                                ~(keyword function) ~dimensions ~options))
          (manager-create ~owner ~(keyword function) ~dimensions ~options))))))

(def-creation zeros "zeros(Shape, DataType)" "Options: :dtype (default :float32).")

(def-creation ones "ones(Shape, DataType)" "Options: :dtype (default :float32).")

(def-creation full "full(Shape, float, DataType)"
  "Options: :value (default 0, coerced to float), :dtype (default :float32).")

(def-creation arange "arange(int/float, int/float, int/float, DataType)"
  "Fill the shape in row-major order from :start (default 0) by :step (default 1).
  The exclusive end is start + step * element-count; :dtype defaults to :int32.
  Integral start/step use DJL's int endpoints; otherwise endpoints are floats.")

(def-creation random-normal "randomNormal(float, float, Shape, DataType)"
  "Options: :mean (default 0), :standard-deviation (default 1), :dtype (default :float32).
  Random distributions require a backend-supported floating dtype.")

(def-creation random-uniform "randomUniform(float, float, Shape, DataType)"
  "Options: :low (default 0, inclusive), :high (default 1, exclusive), :dtype (default :float32).
  Random distributions require a backend-supported floating dtype.")

(defn clip
  "Call DJL NDArray.clip(Number, Number), without modifying the input."
  ^NDArray [^NDArray array ^Number lower ^Number upper]
  (.clip array lower upper))

(defn where
  "Call DJL NDArrays.where(condition, when-true, when-false).
  PyTorch requires condition and when-true to have the same shape.
  The result belongs to when-true's manager/device, not the condition's."
  ^NDArray [^NDArray condition ^NDArray when-true ^NDArray when-false]
  (NDArrays/where condition when-true when-false))

(defn boolean-mask
  "Call DJL NDArray.booleanMask(NDArray[, int]); axis defaults to 0.
  PyTorch supports a same-shape mask (flat selection) or a mask matching the
  trailing dimensions starting at axis. A leading row-only mask is unsupported."
  (^NDArray [^NDArray array ^NDArray mask] (.booleanMask array mask))
  (^NDArray [^NDArray array ^NDArray mask axis] (.booleanMask array mask (int axis))))

(defn gather
  "Call DJL NDArray.gather(NDArray, int) with integer indices and an axis."
  ^NDArray [^NDArray array ^NDArray indices axis]
  (.gather array indices (int axis)))

(defn one-hot
  "Call DJL NDArray.oneHot(int[, DataType]); dtype defaults to :float32.
  Integer input indices become a trailing depth dimension of 0/1 values.
  Custom on/off values are not implemented by the pinned PyTorch engine."
  (^NDArray [^NDArray array depth] (.oneHot array (int depth)))
  (^NDArray [^NDArray array depth dtype]
   (.oneHot array (int depth) (tensor/dtype dtype (.getEngine (.getManager array))))))

(defn sort
  "Call DJL NDArray.sort() or sort(int), ascending along the last or given axis."
  (^NDArray [^NDArray array] (.sort array))
  (^NDArray [^NDArray array axis] (.sort array (int axis))))

(defn arg-sort
  "Call DJL NDArray.argSort(), argSort(int), or argSort(int, boolean).
  Defaults to the last axis, ascending; false selects descending indices."
  (^NDArray [^NDArray array] (.argSort array))
  (^NDArray [^NDArray array axis] (.argSort array (int axis)))
  (^NDArray [^NDArray array axis ascending?] (.argSort array (int axis) (boolean ascending?))))

(defn top-k
  "Call DJL NDArray.topK(int, int[, boolean, boolean]).
  Returns an NDList [values indices] owned by the input manager/device.
  largest? and sorted? default to true; closing the list closes its result arrays."
  (^NDList [^NDArray array k axis] (.topK array (int k) (int axis)))
  (^NDList [^NDArray array k axis largest? sorted?]
   (.topK array (int k) (int axis) (boolean largest?) (boolean sorted?))))

(defn cum-sum
  "Call DJL NDArray.cumSum() or cumSum(int); PyTorch's default axis is 0."
  (^NDArray [^NDArray array] (.cumSum array))
  (^NDArray [^NDArray array axis] (.cumSum array (int axis))))

(defn flip
  "Call DJL NDArray.flip(int[]) with an axis integer or axes vector/int-array."
  ^NDArray [^NDArray array axes]
  (.flip array (if (integer? axes) (int-array [axes]) (primitive-axes axes))))

(defn norm
  "Call DJL NDArray.norm(), norm(int[][, boolean]), or norm(int, int[], boolean).
  Default order is 2; omitted/nil axes reduces all dimensions.
  Axes are a vector/int-array; keep-dims? defaults to false."
  (^NDArray [^NDArray array] (.norm array))
  (^NDArray [^NDArray array axes] (norm array axes false))
  (^NDArray [^NDArray array axes keep-dims?]
   (if (nil? axes)
     (.norm array (boolean keep-dims?))
     (.norm array (primitive-axes axes) (boolean keep-dims?))))
  (^NDArray [^NDArray array order axes keep-dims?]
   (.norm array (int order) (primitive-axes (or axes [])) (boolean keep-dims?))))

(defn broadcast
  "Call DJL NDArray.broadcast(Shape) with a target shape vector."
  ^NDArray [^NDArray array dimensions]
  (.broadcast array (tensor/shape dimensions)))

(defn repeat
  "Call DJL NDArray.repeat(long), repeat(long[]), or repeat(int, long).
  A scalar count repeats elements on every axis; a vector/long-array supplies
  counts per axis. The three-argument form is (repeat array axis count)."
  (^NDArray [^NDArray array counts]
   (cond
     (integer? counts) (.repeat array (long counts))
     (instance? long-array-class counts) (.repeat array ^longs counts)
     :else (.repeat array (long-array counts))))
  (^NDArray [^NDArray array axis count] (.repeat array (int axis) (long count))))

(defn tile
  "Call DJL NDArray.tile(long) or tile(long[]) to tile whole dimensions.
  A scalar count tiles every axis; a vector/long-array supplies counts per axis.
  PyTorch's tile(int, long) and tile(Shape) overloads are not implemented."
  ^NDArray [^NDArray array counts]
  (cond
    (integer? counts) (.tile array (long counts))
    (instance? long-array-class counts) (.tile array ^longs counts)
    :else (.tile array (long-array counts))))

(defn requires-gradient!
  "Call DJL NDArray.setRequiresGradient(boolean) and return the original array.
  Only changes gradient tracking; it does not mutate tensor values."
  ^NDArray [^NDArray array requires-gradient?]
  (.setRequiresGradient array (boolean requires-gradient?))
  array)

(defn to-device
  "Call DJL NDArray.toDevice(Device, boolean), resolving specs through tensor/device.
  copy? defaults to false; a matching device may return the original array.
  The result uses the requested device. PyTorch keeps the input manager when its
  device matches, otherwise the result belongs to a device-specific child manager."
  (^NDArray [^NDArray array device] (to-device array device false))
  (^NDArray [^NDArray array device copy?]
   (.toDevice array (tensor/device device (.getEngine (.getManager array))) (boolean copy?))))

(defn to-vector
  "Return flat host numbers using DJL's dtype-specific to*Array method, preserving int64.
  Boolean arrays become numeric 1/0; nonnumeric/unsupported host dtypes fail.
  This copies device data to host; use it at an output or metrics boundary."
  [^NDArray array]
  (case (.name (.getDataType array))
    ("FLOAT16" "FLOAT32") (vec (.toFloatArray array))
    "FLOAT64" (vec (.toDoubleArray array))
    "INT8" (vec (.toByteArray array))
    "UINT8" (vec (.toUint8Array array))
    "INT16" (vec (.toShortArray array))
    "UINT16" (vec (.toUnsignedShortArray array))
    "INT32" (vec (.toIntArray array))
    "UINT32" (vec (.toUnsignedIntArray array))
    "INT64" (vec (.toLongArray array))
    "UINT64" (mapv #(if (neg? %) (+ 18446744073709551616N %) %) (.toLongArray array))
    "BOOLEAN" (mapv #(if % 1 0) (.toBooleanArray array))
    (throw (ex-info "DJL dtype has no supported numeric host conversion"
                    {:error :ndarray/unsupported-dtype :dtype (.getDataType array)}))))
