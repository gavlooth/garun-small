(ns garun.tensor
  "Safe DJL tensor construction and aligned float32 table extraction.

  Every array/ndlist result belongs to the given NDManager or context's manager,
  including independent copies of native inputs. Close caller-owned managers
  explicitly; session contexts own their managers. Inputs are borrowed only
  during construction, never attached, closed, or mutated. Closing a result
  NDList closes its elements, not its manager.
  Use garun.ndarray functions or native NDArray methods for tensor operations."
  (:require [clojure.string :as str]
            [garun.context :as context])
  (:import [ai.djl Device]
           [ai.djl.engine Engine]
           [ai.djl.ndarray NDArray NDList NDManager]
           [ai.djl.ndarray.types DataType Shape]
           [garun.context Context]
           [java.nio Buffer ByteBuffer DoubleBuffer FloatBuffer IntBuffer LongBuffer ShortBuffer]
           [java.nio.charset StandardCharsets]))

(set! *warn-on-reflection* true)

(defn- invalid! [error message value]
  (throw (ex-info message {:error error :value value})))

(defn- options! [options allowed]
  (when-not (map? options)
    (invalid! :tensor/invalid-options "Tensor options must be a map" options))
  (when-let [unknown (seq (remove allowed (keys options)))]
    (invalid! :tensor/unknown-option "Unknown tensor option" (vec unknown))))

(defn device
  "Resolve :cpu, :auto, [:gpu index], or Device. Only :auto chooses a device.
  Explicit GPU requests fail if the selected engine cannot use that GPU.
  The optional engine is a DJL Engine or engine name; omitted/nil uses default."
  (^Device [spec] (device spec (Engine/getInstance)))
  (^Device [spec engine]
   (let [^Engine engine (cond
                          (nil? engine) (Engine/getInstance)
                          (instance? Engine engine) engine
                          (string? engine) (Engine/getEngine engine)
                          :else (invalid! :tensor/invalid-engine "Expected Engine or engine name" engine))
         ^Device resolved (cond
                            (instance? Device spec) spec
                            (= :cpu spec) (Device/cpu)
                            (= :auto spec) (.defaultDevice engine)
                            (and (vector? spec) (= 2 (count spec)) (= :gpu (first spec))
                                 (integer? (second spec)) (<= 0 (second spec) Integer/MAX_VALUE))
                            (Device/gpu (int (second spec)))
                            :else (invalid! :tensor/invalid-device "Expected :cpu, :auto, [:gpu index], or Device" spec))]
     (when (and (.isGpu resolved)
                (not (< -1 (.getDeviceId resolved) (.getGpuCount engine))))
       (throw (ex-info "Requested GPU is unavailable for the selected engine"
                       {:error :tensor/unavailable-device :device resolved
                        :engine (.getEngineName engine)})))
     resolved)))

(def ^:private dtypes
  (assoc (into {} (map (fn [^DataType value]
                         [(keyword (str/lower-case (.name value))) value]))
               (DataType/values))
         :bool DataType/BOOLEAN))

(defn dtype
  "Resolve a native DataType or its lower-case keyword, including :bool.
  One argument recognizes all DJL enum values. The engine-aware arity rejects
  unmapped native dtypes before calling JNI. PyTorch 0.36 does not map int16,
  bfloat16, wider unsigned integers, unknown, or native string conversions."
  (^DataType [spec]
   (cond
     (instance? DataType spec) spec
     (contains? dtypes spec) (get dtypes spec)
     :else (invalid! :tensor/invalid-dtype "Unknown DJL dtype" spec)))
  (^DataType [spec engine]
   (let [^DataType resolved (dtype spec)
         ^Engine engine (cond
                          (nil? engine) (Engine/getInstance)
                          (instance? Engine engine) engine
                          (string? engine) (Engine/getEngine engine)
                          :else (invalid! :tensor/invalid-engine "Expected Engine or engine name" engine))]
     ;; The installed JNI mapper accepts ordinals 0..8; its default is a bare
     ;; C++ throw, not a recoverable Java exception. Buffer changes cannot fix it.
     (when (and (= "PyTorch" (.getEngineName engine))
                (> (.ordinal resolved) (.ordinal DataType/COMPLEX64)))
       (throw (ex-info "Requested dtype is not implemented by the selected engine"
                       {:error :tensor/unsupported-dtype :dtype resolved
                        :engine (.getEngineName engine)})))
     resolved)))

(defn shape
  "Resolve concrete dimensions or Shape. Scalars use []; zero dimensions are
  allowed. Negative/unknown, nonintegral, and overflowing dimensions fail."
  ^Shape [dimensions]
  (let [values (if (instance? Shape dimensions) (vec (.getShape ^Shape dimensions)) dimensions)]
    (when-not (and (sequential? values)
                   (every? #(and (integer? %) (<= 0 % Long/MAX_VALUE)) values))
      (invalid! :tensor/invalid-shape "Shape dimensions must be nonnegative int64 integers" dimensions))
    (when (> (reduce *' 1 values) Long/MAX_VALUE)
      (invalid! :tensor/invalid-shape "Shape cardinality exceeds int64" dimensions))
    (if (instance? Shape dimensions) dimensions (Shape. (long-array values)))))

(defn- open-manager! [^NDManager manager]
  (when-not (and manager (.isOpen manager))
    (invalid! :tensor/closed-manager "Tensor manager must be open" manager)))

(defn- primitive-dtype [data]
  (when data
    (case (.getName (class data))
      "[F" DataType/FLOAT32
      "[D" DataType/FLOAT64
      "[J" DataType/INT64
      "[I" DataType/INT32
      "[S" DataType/INT16
      "[B" DataType/INT8
      "[Z" DataType/BOOLEAN
      nil)))

(defn- container? [data]
  (or (sequential? data) (and data (.isArray (class data)))))

(defn- layout
  "Describe rectangular host values without coercing integer leaves."
  [data]
  (cond
    (primitive-dtype data)
    {:dims [(count data)] :values data :types #{(primitive-dtype data)}}

    (container? data)
    (let [children (mapv layout data)
          dimensions (mapv :dims children)]
      (when-not (apply = (cons (first dimensions) dimensions))
        (invalid! :tensor/ragged-data "Tensor data must be rectangular" data))
      {:dims (into [(count children)] (:dims (first children)))
       :values (into [] (mapcat :values) children)
       :types (into #{} (mapcat :types) children)})

    :else
    (let [type (cond
                 (boolean? data) DataType/BOOLEAN
                 (string? data) DataType/STRING
                 (integer? data) DataType/INT64
                 (instance? Float data) DataType/FLOAT32
                 (number? data) DataType/FLOAT64
                 :else (invalid! :tensor/invalid-data "Expected numeric, boolean, or string tensor values" data))]
      (when (and (integer? data) (not (<= Long/MIN_VALUE data Long/MAX_VALUE)))
        (invalid! :tensor/invalid-data "Integer tensor values must fit int64" data))
      {:dims [] :values [data] :types #{type}})))

(defn- inferred-dtype [types]
  (cond
    (empty? types) DataType/FLOAT32
    (= 1 (count types)) (first types)
    (or (contains? types DataType/BOOLEAN) (contains? types DataType/STRING))
    (invalid! :tensor/invalid-data "Tensor scalar types cannot be mixed" types)
    (contains? types DataType/FLOAT64) DataType/FLOAT64
    (contains? types DataType/FLOAT32) DataType/FLOAT32
    :else DataType/INT64))

(defn- host-buffer
  "Wrap primitive storage directly; flattening never routes integers through float."
  ^Buffer [data ^DataType type]
  (cond
    (and (= type DataType/FLOAT32) (= type (primitive-dtype data))) (FloatBuffer/wrap ^floats data)
    (and (= type DataType/FLOAT64) (= type (primitive-dtype data))) (DoubleBuffer/wrap ^doubles data)
    (and (= type DataType/INT64) (= type (primitive-dtype data))) (LongBuffer/wrap ^longs data)
    (and (= type DataType/INT32) (= type (primitive-dtype data))) (IntBuffer/wrap ^ints data)
    (and (= type DataType/INT16) (= type (primitive-dtype data))) (ShortBuffer/wrap ^shorts data)
    (and (= type DataType/INT8) (= type (primitive-dtype data))) (ByteBuffer/wrap ^bytes data)
    (= type DataType/FLOAT32) (FloatBuffer/wrap (float-array data))
    (= type DataType/FLOAT64) (DoubleBuffer/wrap (double-array data))
    (= type DataType/INT32) (IntBuffer/wrap (int-array data))
    (= type DataType/INT16) (ShortBuffer/wrap (short-array data))
    (= type DataType/INT8) (ByteBuffer/wrap (byte-array data))
    (= type DataType/BOOLEAN) (ByteBuffer/wrap (byte-array (map #(if % 1 0) data)))
    :else (LongBuffer/wrap (long-array data))))

(defn- cardinality! [^Shape dimensions size]
  (when-not (= (.size dimensions) size)
    (throw (ex-info "Shape cardinality does not match tensor data"
                    {:error :tensor/shape-mismatch :shape (vec (.getShape dimensions))
                     :expected (.size dimensions) :actual size}))))

(defn- copy-native! [^NDManager work ^NDArray data ^Shape dimensions]
  (when (.isReleased data)
    (invalid! :tensor/closed-array "Input NDArray is closed" data))
  (cardinality! dimensions (.size data))
  (let [^NDArray copy (if (= (.getDataType data) DataType/STRING)
                        (.create work ^"[Ljava.lang.String;" (aclone (.toStringArray data))
                                 StandardCharsets/UTF_8 dimensions)
                        (.duplicate data))]
    ;; PyTorch duplicate invokes tensorClone on the source device. copyTo uses
    ;; NDArray's default host-buffer path, causing a GPU -> CPU -> GPU detour.
    ;; Attach only the independent clone before any device transfer; toDevice
    ;; may share its storage, but never storage belonging to the caller input.
    (.attach copy work)
    (let [^NDArray result (if (= (.getDataType data) DataType/STRING)
                            copy (.toDevice copy (.getDevice work) false))]
      (.attach result work)
      (when-let [name (.getName data)] (.setName result name))
      result)))

(defn- manager-array
  ^NDArray [^NDManager manager data options]
  (open-manager! manager)
  (options! options #{:shape :dtype})
  (when (and (instance? NDArray data) (.isReleased ^NDArray data))
    (invalid! :tensor/closed-array "Input NDArray is closed" data))
  (let [native? (instance? NDArray data)
        primitive-type (primitive-dtype data)
        description (when-not (or native? primitive-type) (layout data))
        inferred-shape (cond native? (.getShape ^NDArray data)
                             primitive-type [(count data)]
                             :else (:dims description))
        ^Shape dimensions (shape (get options :shape inferred-shape))
        ^DataType source-type (cond native? (.getDataType ^NDArray data)
                                    primitive-type primitive-type
                                    :else (inferred-dtype (:types description)))
        ^DataType target-type (if (contains? options :dtype) (dtype (:dtype options)) source-type)
        ;; A short's numeric value is not a FLOAT16 bit pattern. Widen directly
        ;; into supported host storage when the caller explicitly requests it.
        ^DataType storage-type (if (and (= source-type DataType/INT16)
                                        (contains? options :dtype) (not native?))
                                 (if (contains? #{DataType/FLOAT32 DataType/FLOAT64 DataType/INT32
                                                  DataType/INT64} target-type)
                                   target-type DataType/INT32)
                                 source-type)
        values (if primitive-type data (:values description))
        ^Engine engine (.getEngine manager)]
    (when (and description (> (count (:dims description)) 1)
               (not= (:dims description) (vec (.getShape dimensions))))
      (invalid! :tensor/shape-mismatch "Explicit shape disagrees with nested data dimensions" (:shape options)))
    (when-not native? (cardinality! dimensions (count values)))
    (when (and (or (= source-type DataType/STRING) (= target-type DataType/STRING))
               (not= source-type target-type))
      (throw (ex-info "String tensors cannot be converted to or from numeric dtypes"
                      {:error :tensor/unsupported-dtype :dtype target-type
                       :source-dtype source-type :engine (.getEngineName engine)})))
    ;; String arrays use DJL's Java-backed constructor, not native dtype JNI.
    (when-not (= target-type DataType/STRING)
      (dtype target-type engine)
      (dtype storage-type engine))
    ;; The work manager bounds intermediate copies, casts, and reshape views.
    ;; Only the final independent result is promoted into the owner's manager.
    (with-open [work (.newSubManager manager)]
      (let [^NDArray created (cond
                               native? (copy-native! work data dimensions)
                               (= source-type DataType/STRING)
                               (.create work ^"[Ljava.lang.String;" (into-array String values)
                                        StandardCharsets/UTF_8 dimensions)
                               :else (.create work (host-buffer values storage-type) dimensions storage-type))
            ^NDArray typed (if (= target-type (.getDataType created))
                             created (.toType created target-type false))
            ^NDArray result (if (= dimensions (.getShape typed)) typed (.reshape typed dimensions))]
        (.attach result manager)
        result))))

(defn array
  "Create an independent NDArray owned by the given manager or context's manager.
  The result uses that manager's device. Options: :shape (optional concrete
  dimensions), :dtype (optional DJL dtype).
  Primitive arrays preserve backend-supported dtypes; nested integers default
  to int64, floating values to float64 (Float leaves to float32), booleans to
  boolean. Unsupported inferred/requested dtypes fail before JNI. PyTorch
  short arrays require an explicit supported :dtype such as :int32/:int64;
  that host conversion never creates an unsupported int16 intermediate.
  Flat storage may be reshaped; explicit shapes of nested storage must match
  its rectangular layout. Native inputs retain their dtype unless overridden
  and use device-native cloning/transfer without closing caller resources."
  (^NDArray [owner data] (array owner data {}))
  (^NDArray [owner data options]
   (if (instance? Context owner)
     (context/call ^Context owner #(manager-array (context/manager ^Context owner) data options))
     (manager-array owner data options))))

(defn- input-descriptor? [input]
  (or (instance? NDArray input) (map? input)))

(defn- manager-ndlist
  ^NDList [^NDManager manager inputs]
  (open-manager! manager)
  (let [items (cond
                (instance? NDList inputs) (vec inputs)
                (input-descriptor? inputs) [inputs]
                (and (sequential? inputs) (every? input-descriptor? inputs)) inputs
                :else [inputs])]
    (with-open [work (.newSubManager manager)]
      (let [arrays (mapv (fn [input]
                           (if (map? input)
                             (do
                               (options! input #{:data :shape :dtype})
                               (when-not (contains? input :data)
                                 (invalid! :tensor/invalid-data "Tensor descriptor requires :data" input))
                               (manager-array work (:data input) (dissoc input :data)))
                             (manager-array work input {})))
                         items)
            result (NDList. ^java.util.Collection arrays)]
        (.attach result manager)
        result))))

(defn ndlist
  "Build an independent NDList owned by the given manager or context's manager.
  Inputs may be native NDList/NDArray, one descriptor, a sequence of descriptors
  or native arrays, or plain host data.
  Descriptor: {:data values-or-NDArray, :shape optional, :dtype optional}.
  Plain nested numeric/boolean/string data denotes ONE tensor, not many inputs.
  All native inputs are borrowed for this call and copied; closing the returned
  list or owning manager cannot close caller inputs. Empty input sequence means
  no inputs."
  ^NDList [owner inputs]
  (if (instance? Context owner)
    (context/call ^Context owner #(manager-ndlist (context/manager ^Context owner) inputs))
    (manager-ndlist owner inputs)))

