(ns garun.ndarray
  "Reflection-free functions over DJL NDArray and NDList values.

  get, abs, max, min, concat and flatten shadow clojure.core names and are
  excluded from this namespace's core referrals. Use the nd alias at callsites.
  Operations retain DJL's dtype, broadcasting, gradient and manager semantics;
  they do not copy or transfer inputs unless the named DJL operation does.
  ndlist borrows its arrays: closing that NDList also closes its elements."
  (:refer-clojure :exclude [get abs max min concat flatten])
  (:require [garun.tensor :as tensor])
  (:import [ai.djl.ndarray NDArray NDArrays NDList]
           [ai.djl.ndarray.index NDIndex]
           [java.util Collection]))

(set! *warn-on-reflection* true)

(def ^:private empty-index-arguments (object-array 0))

(def ^:private int-array-class (class (int-array 0)))

(def ^:private long-array-class (class (long-array 0)))

(defn- primitive-axes
  ^ints [axes]
  (if (instance? int-array-class axes) axes (int-array axes)))

(defmacro ^:private def-unary [function method]
  (let [array (with-meta 'array {:tag 'ai.djl.ndarray.NDArray})]
    `(defn ~(with-meta function {:tag 'ai.djl.ndarray.NDArray})
       ~(str "Call DJL NDArray." method "().")
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

(defmacro ^:private def-binary [function method]
  (let [array (with-meta 'array {:tag 'ai.djl.ndarray.NDArray})
        other 'other]
    `(defn ~(with-meta function {:tag 'ai.djl.ndarray.NDArray})
       ~(str "Call DJL NDArray." method "(NDArray) or " method "(Number), preserving broadcasting.")
       [~array ~other]
       (if (instance? NDArray ~other)
         (. ~array ~(symbol method) ~(with-meta other {:tag 'ai.djl.ndarray.NDArray}))
         (. ~array ~(symbol method) ~(with-meta other {:tag 'java.lang.Number}))))))

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
              (.reshape result# (long-array (repeat (.dimension (.getShape ~array)) 1)))
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
            (nil? axis#) (.reshape result# (long-array (repeat (.dimension (.getShape ~array)) 1)))
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
