# garun-small

A small JVM Clojure version of the Garun Hiccup DSL. It has two backends:

- **DuckDB** (plain JDBC) for tables. Garun clauses become HoneySQL query data;
  HoneySQL renders SQL and bound parameters at execution time.
- **Deep Java Library 0.36 / PyTorch 2.7.1** for tensors, neural networks,
  training and typed model inference. CPU is the default; CUDA is opt-in.

The DSL names common DJL components and accepts native DJL objects for the rest.
It does not reimplement DJL's entire Java API. Rust, classical estimators
(k-means, SVM, XGBoost, …), visualization and session services remain out of scope.

## Build and test

```sh
clojure -T:build javac   # compile the small DJL Java bridge classes
clojure -M:test          # DuckDB, blocks, tensors, training, inference and metrics
```

All dependencies resolve from Maven. The default PyTorch native library comes
from the `pytorch-native-cpu` linux-x86_64 jar: no native download at CPU runtime.
The `:cuda` profile instead lets DJL download/cache the pinned CUDA library.

## Namespaces

| Namespace | Surface |
|---|---|
| `garun.table` | Pure plans: `from`, `query`, `->sql`, `table?`. Execution: `open`, `rows`, `columns`, `schema`, `row-count`, `matrix`, `write-csv!`, `write-parquet!` |
| `garun.blocks` | `compile-block`, `serializable-spec?`; Hiccup or native DJL `Block` |
| `garun.tensor` | `manager`, `device`, `dtype`, `shape`, `array`, `ndlist`, `table-matrices`; caller-owned tensor resources |
| `garun.training` | `loss`, `optimizer`, `tracker`, `initializer`, `config`; specs or native DJL components |
| `garun.nn` | `build`, `train!`, `predict`, `predict-with`, `save!`, `load`; a `Network` implements `AutoCloseable` |
| `garun.model` | `criteria`, `load!`, `predict`, `batch-predict`; native translators and caller-owned `ZooModel` |
| `garun.metrics` | `accuracy mse mae r2 precision recall f1 confusion-matrix log-loss`, called as `(metric y-pred y-true)` |

**Table sources:** a column map, a vector of row maps, `[:csv {:path :header? :delimiter}]`,
`[:parquet {:path}]`, `[:json {:path}]`, or `[:duckdb {:query "SELECT …"}]`.
The `:duckdb` query is trusted SQL.

**Clauses:** `:select :rename :where :derive :group-by :order-by :limit :join :sample :split :honeysql`.
`:split` must be the last clause. It returns a vector of tables that are
deterministic, do not overlap and together cover every input row.

**Expressions:** Garun's operators, aggregates and window functions (`[:over expr {...}]`).

Use `[:honeysql clause-map]` for native HoneySQL SELECT transformations. Garun
supplies `FROM` from the previous table and defaults to `SELECT *` when no
projection is supplied. Supported clauses include projections (including
DISTINCT), WHERE, grouping/HAVING, windows/QUALIFY, ordering, limits/offsets,
joins and CTEs. Source replacement and DDL/DML clause keys are rejected.
HoneySQL helpers can build the same clause maps.

```clojure
(table/query data
  [[:where [:not-null :amount]]
   [:honeysql {:select [:category [[:sum :amount] :total]]
               :group-by [:category]
               :having [:> [:sum :amount] 10]}]
   [:order-by [:total :desc]]])
```

Inside `:honeysql`, expressions follow HoneySQL's identifier and NULL rules:
`[:= :x nil]` means `IS NULL`, unlike Garun's bound NULL equality. `:raw` forms
are trusted SQL, not sanitized input. Arbitrary projection names are resolved
by DuckDB; a later Garun `:select` can explicitly name the output columns.

## Neural Hiccup

| Family | Tags |
|---|---|
| Composition | `:sequential`, `:parallel`, `:skip`, `:block`, `:factory`, `:lambda` |
| Dense | `:linear`, `:dropout`, `:flatten` |
| Activations | `:activation`, `:relu`, `:relu6`, `:tanh`, `:sigmoid`, `:gelu`, `:selu`, `:mish`, `:softplus`, `:softsign`, `:prelu`, `:identity`, `:leaky-relu`, `:elu`, `:swish` |
| Convolution | `:conv1d`, `:conv2d`, `:conv3d`, `:conv1d-transpose`, `:conv2d-transpose` |
| Pooling | `:max-pool1d/2d/3d`, `:avg-pool1d/2d/3d`, `:global-max-pool1d/2d/3d`, `:global-avg-pool1d/2d/3d` |
| Normalization | `:batchnorm`, `:layernorm`, `:ghost-batchnorm` |
| Embeddings | `:embedding`, `:word-embedding`, `:constant-embedding` |
| Recurrent | `:rnn`, `:lstm`, `:gru` |
| Attention | `:attention`, `:transformer-encoder`, `:bert`, `:bert-pretraining`, `:bert-next-sentence` |
| Tensor transforms | `:reshape`, `:transpose`, `:softmax`, `:log-softmax`, `:expand-dims`, `:squeeze` |

The `/1d/2d/3d` notation above abbreviates three separate tags.
Options are closed maps: misspelled or unsupported keys fail rather than vanish.
The per-tag builders in `src/garun/blocks.clj` define the full option contracts.

- `[:linear 16]` or `[:linear {:units 16 :bias? false}]`.
- Convolutions require `:filters` and `:kernel`; support stride, padding,
  dilation, groups and bias. Transposed convolutions also accept output padding.
- Recurrent blocks require `:state-size`; attention requires `:embedding-size`
  and `:heads`; transformer encoders also require `:hidden-size`.
- `:parallel` runs every child on the same input. `:skip` combines the inner
  result with the original input. Both accept `{:combine :add}` or
  `{:combine :concat :axis -1}`, or a combining function with an optional
  `:shape-fn`. Multi-output addition requires matching output counts.
- A native `Block` works directly or as `[:block block]`. `[:factory make-block]`
  calls a zero-argument constructor. `[:lambda f]` takes a Clojure function
  from `NDList` to `NDList`; `[:lambda {:singleton? true} f]` uses `NDArray`.

`garun.djl.FnBlock` supplies DJL's protected child initialization/forwarding
boundary. Small Java leaves also implement differentiable transposed
convolution, masked transformer initialization and tensor-native optimizers
where the pinned PyTorch backend lacks or misimplements the public operation.
Transposed convolution retains DJL's weight layout
`[filters, input-channels/groups, kernel…]`, not PyTorch's ConvTranspose layout.

`nn/build` accepts either `:input-shape` (one sample; batch dimension 1 is
prepended) or `:input-shapes` (full concrete DJL shapes, including multiple
inputs, scalars and empty tensors). They are mutually exclusive.
`:dtype` defaults to `:float32`, `:device` to `:cpu`; `:engine` names a DJL engine.

### Tabular training example

```clojure
(require '[garun.table :as table] '[garun.nn :as nn] '[garun.metrics :as metrics])

(with-open [db (table/open)]
  (let [base (table/query (table/from [:csv {:path "data.csv"}])
                          [[:derive {:r [:sqrt [:+ [:* :x1 :x1] [:* :x2 :x2]]]}]
                           [:where [:not-null :label]]])
        [train test] (table/query base [[:split {:fractions [0.8 0.2] :seed 7 :key :id}]])
        spec [:sequential
              [:skip {:combine :concat} [:linear 16] [:relu]]
              [:parallel {:combine :add} [:linear 16] [:sequential [:linear 16] [:tanh]]]
              [:relu]
              [:linear 1]]]
    (with-open [net (nn/build spec {:input-shape [3] :seed 3})]
      (nn/train! net db train {:x [:x1 :x2 :r] :y :label :loss :bce :epochs 60
                               :batch 32 :optim [:adam {:lr 0.01}] :validation test})
      (metrics/accuracy (nn/predict net db test {:x [:x1 :x2 :r] :output :class})
                        (:label (table/columns db test))))))
```

## Training components

| Component | Named specs |
|---|---|
| Loss | `:mse`, `:mae`, `:bce`, `:ce`, `:quantile`, `:hinge`, `:masked-ce`, `:composite`, `:l1-decay`, `:l2-decay`, `:elastic-net` |
| Optimizer | `:sgd`, `:nag`, `:adam`, `:adamw`, `:rmsprop`, `:adagrad`, `:adadelta` |
| Learning-rate tracker | `:fixed`, `:factor`, `:multi-factor`, `:cosine`, `:linear`, `:polynomial`, `:warm-up` |
| Initializer | `:zeros`, `:ones`, `:constant`, `:normal`, `:truncated-normal`, `:uniform`, `:xavier` |

Constructors also accept the corresponding native DJL instance unchanged.
Losses use DJL shape semantics; BCE and CE consume logits by default.
`[:bce {:from-sigmoid? true}]` consumes probabilities. Masked CE handles sparse
or dense labels, sequence lengths and a configurable class axis.

Optimizers take `[tag options]`, e.g. `[:adam {:lr 0.01}]`. Common controls are
`:weight-decay`, `:clip-grad`, `:rescale-grad`, `:begin-update`; family-specific
controls include momentum, beta values, rho, epsilon and centered RMSProp.
`:lr` accepts a number, tracker spec or native `ParameterTracker`; Adadelta has
no learning-rate option. AdamW uses decoupled decay. Rescaling and symmetric
gradient clipping precede other optimizers' coupled L2 decay.
Create fresh stateful optimizers/trackers for independent training runs.

The table overload `(nn/train! net db source opts)` returns
`{:history [...]}` and requires one rank-2 float32 input/output. Feature and label
columns are extracted together in one DuckDB query, preserving row alignment
even for random transformations.

The native overload `(nn/train! net dataset opts)` prepares a DJL `Dataset`,
uses `EasyTrain.fit` and returns a native `TrainingResult`. Use it for images,
sequences, multiple inputs, precise labels and non-tabular shapes.
Options include `:epochs`, `:validation` (another Dataset), `:seed`, and either
`:config` (native `TrainingConfig`) or training component options.
`training/config` accepts listeners, evaluators, initializers, devices and a
caller-owned executor. Explicit listeners replace the defaults.
Datasets and their source arrays remain caller-owned.

## Tensors and device placement

DuckDB remains the ingestion/transformation boundary. `table/matrix` returns
row-major float32 host storage; `tensor/table-matrices` extracts multiple
column groups from one query. Construct DJL arrays once, then keep matrix
arithmetic and training on native arrays. Use DJL's `NDArray` methods directly;
there is no duplicate tensor-operation DSL.

```clojure
(require '[garun.tensor :as tensor] '[garun.nn :as nn])

(with-open [manager (tensor/manager {:device :cpu})
            net (nn/build [:sequential
                           [:conv2d {:filters 4 :kernel [3 3]}]
                           [:global-avg-pool2d]
                           [:linear 1]]
                          {:input-shape [1 8 8]})]
  (let [images (tensor/array manager (float-array (repeat 128 0.5))
                            {:shape [2 1 8 8]})]
    (with-open [outputs (nn/predict net images {:manager manager})]
      (.toFloatArray (.singletonOrThrow outputs)))))
```

Managers are caller-owned; use `with-open`. `tensor/array` and `tensor/ndlist`
copy borrowed native inputs independently onto the destination manager's device.
They do not steal, mutate or close inputs. Native copies/placement use device
operations, not a GPU → host → GPU serialization path.

Primitive arrays preserve supported dtypes. Nested Clojure integers default to
int64 and floating numbers to float64; request `:float32` for ordinary networks.
Descriptors are `{:data values :shape dimensions :dtype dtype}`; an NDList can
contain several descriptors. Shapes include scalars `[]` and zero dimensions.
Rectangular layout and cardinality are checked; int64 input avoids a floating
intermediate.

The pinned DJL PyTorch JNI mapping lacks int16, bfloat16 and wider unsigned
types. The safe constructors reject these before entering JNI, which otherwise
can abort the process. For Java `short[]`, explicitly request a supported dtype
such as `:int32` or `:int64`. Direct native calls remain subject to DJL's backend
limits.

### CUDA

```sh
clojure -M:cuda
```

This excludes the CPU native jar and selects PyTorch 2.7.1's `cu128` library.
DJL downloads/caches it on first use; a compatible NVIDIA driver is required.
Start a fresh JVM when changing native profiles. Pass `:device [:gpu 0]` to
both `tensor/manager` and `nn/build`; `:auto` lets the engine choose.
Explicit unavailable GPU requests fail instead of silently falling back to CPU.
GPU execution has not been verified in this workspace: the NVIDIA driver
cannot be contacted.

## Translators and model loading

`nn/predict`'s three-argument overload requires `{:manager manager}` and returns
an independent caller-manager-owned `NDList`, valid after predictor/network
closure. The four-argument table overload still returns Clojure vectors.
`nn/predict-with` accepts a native DJL `Translator` and a typed input.

`model/criteria` accepts a native `Criteria` or a map requiring Java
`:input-class` and `:output-class`. Optional keys select a translator/factory,
block, local path, URL, model zoo, application, group/artifact, engine, device,
filters, arguments and progress. Native Criteria exposes more detailed DJL
settings without more DSL machinery.

`model/load!` calls DJL's real model loader and returns a caller-owned `ZooModel`.
`model/predict` and `model/batch-predict` use its packaged translator, or accept
an explicit translator with any DJL Model. Predictors close after each call;
the model remains open. Direct NDArray/NDList outputs are independent detached
copies: callers must close or attach them to their own manager. Other outputs
retain translator-defined ownership; custom objects containing tensors must
manage their nested tensors in `processOutput`.

```sh
clojure -M:zoo          # optional DJL + PyTorch model-zoo providers
clojure -M:cuda:zoo     # combine providers with CUDA
```

The base API supports explicit local/URL Criteria without these provider jars.
Model paths, URLs, native translators and native blocks are trusted caller
inputs, not a sandbox. Local Criteria loading and real PNG translator inference
have been exercised; remote pretrained artifact downloads have not.

## Persistence

`nn/save!` writes named Hiccup specs as EDN plus DJL parameters.
Native blocks, factories, functions and constant embedding tensors require an
explicit `{:spec reconstruction-spec}`. A non-EDN reconstruction spec is not
serialized: `nn/load` then requires `:spec` from the caller.
Missing reconstruction information fails before writing files.

Load options can override `:spec`, `:device`, `:dtype` and `:engine`.
For example, `(nn/load directory {:device :cpu})` restores a GPU save on CPU.
Models, managers and native output resources must be closed by their owner.
