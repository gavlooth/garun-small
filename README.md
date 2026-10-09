# garun-small

A small JVM Clojure version of the Garun Hiccup DSL. It has two backends:

- **DuckDB** (plain JDBC) for tables. Garun clauses become HoneySQL query data;
  HoneySQL renders SQL and bound parameters at execution time.
- **Deep Java Library 0.36 / PyTorch 2.7.1** for tensors, neural networks,
  training and typed model inference. CPU is the default; CUDA is opt-in.

One `garun.session` object owns all state: the DuckDB connection, tensor memory,
defaults and named networks, models, trainers and tables. The DSL names common
DJL components and accepts native DJL objects for the rest; it does not
reimplement DJL's entire Java API. Rust, classical estimators (k-means, SVM,
XGBoost, …) and visualization remain out of scope.

## Build and test

```sh
clojure -T:build javac   # compile the Java leaves (session core, DJL bridges)
clojure -M:test          # sessions, DuckDB, blocks, tensors, training, inference, metrics
```

All dependencies resolve from Maven. The default PyTorch native library comes
from the `pytorch-native-cpu` linux-x86_64 jar: no native download at CPU runtime.
The `:cuda` profile instead lets DJL download/cache the pinned CUDA library.

## Namespaces

| Namespace | Surface |
|---|---|
| `garun.session` | `open`, `worker`, `defaults`, `register!`, `resource`, `value`, `shared`, `exclusive`, `release!`, `describe` |
| `garun.table` | Pure plans: `from`, `query`, `->sql`, `table?`. Through a context: `rows`, `columns`, `schema`, `row-count`, `matrix`, `materialize!`, `write-csv!`, `write-parquet!` |
| `garun.blocks` | `compile-block`, `serializable-spec?`; Hiccup or native DJL `Block` |
| `garun.tensor` | `device`, `dtype`, `shape`, `array`, `ndlist`, `table-matrices` |
| `garun.training` | `loss`, `optimizer`, `tracker`, `initializer`, `config`; specs or native DJL components |
| `garun.nn` | `build`, `train!`, `trainer`, `epoch!`, `validate`, `restore-best!`, `evaluate`, `predict`, `predict-with`, `save!`, `load` |
| `garun.model` | `criteria`, `load!`, `predict`, `batch-predict`; native translators |
| `garun.metrics` | `accuracy mse mae r2 precision recall f1 confusion-matrix log-loss`, called as `(metric y-pred y-true)` |

## Sessions

```clojure
(require '[garun.session :as session])

(with-open [current-session (session/open {:database :memory :device :cpu :dtype :float32 :seed 7})]
  ...)
```

Every resource-using function takes a **context** first: the session itself or
a worker. Pure data (table plans, Hiccup specs, training component specs)
never needs one. Functions accept objects; given a context they also accept the
**keyword name** of a registered resource (`(nn/predict current-session :gru test {})`).

| Owned by the session | Notes |
|---|---|
| DuckDB connection | `:database` is `:memory` or a file path |
| Root tensor manager | Tensors created or returned through a context belong to it |
| Defaults | `:device`, `:dtype`, `:engine`, `:seed`; per-call options override |
| Registry | Networks, zoo models, trainers and materialized tables, optionally named |

Closing the session closes open workers, then owned resources in reverse
registration order, drops its private DuckDB schema, closes the tensor manager
and finally the connection. Resources passed in with `{:owned? false}` are
borrowed and never closed. `session/release!` closes one resource early.
Reusing a name fails unless `{:replace? true}` is given.

**Inspection:** `prn`, the REPL and `clojure.datafy/datafy` show plain data.

```clojure
(prn current-session)
;; #garun/session {:id "…" :status :open :database ":memory" :device :cpu :dtype :float32
;;                 :engine "PyTorch" :seed 7 :workers 0
;;                 :resources [{:id 1 :name :batteries :kind :table :rows 3200 :columns […]}
;;                             {:id 2 :name :gru :kind :network :spec […] :parameters 2401}]}
```

Networks print as `#garun/network {...}`, trainers as `#garun/trainer {...}`.

**Threads:** use one thread per context. `(session/worker context)` returns a
closeable worker with its own duplicated DuckDB connection and tensor
sub-manager; it sees the session's registry and materialized tables.
Predictions on one network may run concurrently; training takes the network's
exclusive lock. Named tables are locked while a query reads them, so a
release or replacement waits for running queries. The Java leaf
(`java/garun/session/`) holds these locks and the registry; Clojure code holds no
mutable state.

**Table sources:** a column map, a vector of row maps, `[:csv {:path :header? :delimiter}]`,
`[:parquet {:path}]`, `[:json {:path}]`, `[:duckdb {:query "SELECT …"}]`, or the
keyword name of a materialized session table. The `:duckdb` query is trusted SQL.

`(table/materialize! context :name source)` runs a plan once into a table in the
session's private schema and returns `(table/from :name)`. Later reads are
fast and stable even for random sources, and workers see the same table.
Pure `table/->sql` rejects plans naming session tables; `(table/->sql context plan)`
resolves them.

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

## Training

The training loop is one call. DuckDB does the data work: splits, ordering,
windowing boundaries and standardization statistics. This is a battery
remaining-useful-life GRU, from CSV to evaluated model:

```clojure
(require '[garun.session :as session] '[garun.table :as table] '[garun.nn :as nn])

(with-open [current-session (session/open {:seed 2026})]
  (let [batteries (table/materialize! current-session :batteries [:csv {:path "data/batteries.csv"}])
        [train validation test] (table/query batteries [[:split {:fractions [0.7 0.15 0.15]
                                                                 :seed 2026 :key :battery_id}]])]
    (nn/build current-session [:sequential [:gru {:state-size 24 :batch-first? true}]
                                           [:last-step]
                                           [:linear 1]]
              {:input-shape [30 7] :name :gru})
    (nn/train! current-session :gru train
               {:x [:IR :QC :QD :Tavg :Tmin :Tmax :chargetime] :y :RUL
                :sequence {:length 30 :partition-by :battery_id :order-by :cycle}
                :standardize [:x :y]
                :validation validation :loss :mse :optim [:adam {:lr 0.001}]
                :epochs 25 :batch 128
                :early-stopping {:patience 5 :restore-best? true}
                :observe tap>})
    (nn/evaluate current-session :gru test {:metrics [:mae]})))
```

| Option | Meaning |
|---|---|
| `:x`, `:y` | Feature and label columns |
| `:sequence` | Full windows of `:length` rows per `:partition-by`, ordered by `:order-by`; the label is the last row's `:y`. Input is `[windows length features]` |
| `:standardize` | `[:x]`, `[:y]` or both. Mean and population standard deviation come from **the training source only**, in one DuckDB aggregate |
| `:validation` | Another table source, or a DJL `Dataset` for native sources |
| `:loss`, `:optim`, `:epochs`, `:batch`, `:shuffle?`, `:seed` | As in `garun.training`; `:loss` may also be `(fn [labels predictions] NDArray)` |
| `:early-stopping` | `{:patience n :min-delta d :restore-best? b}`; monitors validation loss, else training loss |
| `:metrics` | `garun.metrics` keywords or `(fn [predictions truth] number)`, computed on validation every epoch in original units |
| `:observe` | Function receiving one event map per epoch, then a `:finished` event. Exceptions abort training |
| `:stop?` | Function of the epoch state; returning true stops training |
| `:forward` | `(fn [trainer inputs training?] NDList)` replacing the forward pass |
| `:config` | A native `TrainingConfig`, instead of the component options |

`train!` returns plain data:

```clojure
{:history [{:epoch 1 :training-loss 0.067 :validation-loss 0.041 :best? true
            :elapsed-milliseconds 312.0 :metrics {:mae 3.1}} …]
 :best {:epoch 24 :loss 2.8e-5 :monitor :validation-loss}
 :stopped :completed}   ; or :early-stopping
```

The input specification (`:x :y :sequence :standardize`) and its statistics are
stored with the network and saved by `nn/save!`. `nn/predict` and
`nn/evaluate` reuse them: features are scaled, outputs return in original
units, and evaluation aligns predictions with truth itself, including
windows. Pass `:x` (and friends) to override.

**Custom loops.** `train!` is a short `reduce` over a lower layer that users can
drive directly for schedules, curricula or their own stopping rules:

```clojure
(let [handle (nn/trainer current-session :gru train {:x features :y :RUL :validation validation})]
  (try
    (loop [state {}]
      (let [state (nn/epoch! handle state)]
        (if (my-stop? state) (nn/restore-best! handle state) (recur state))))
    (finally (session/release! current-session handle))))
```

`epoch!` returns the next immutable state (`:epoch :history :best`);
`validate` returns `{:loss :metrics}`. A trainer is a session resource: release
it with `session/release!` or leave it to session close.

Table and native DJL `Dataset` sources go through the same loop
(`Trainer.iterateDataset`, gradient collector, optimizer step). Table sources
become `ArrayDataset`s with seeded shuffling; native datasets and their arrays
remain borrowed. Best-epoch parameters are snapshotted and restored on the
device, not through host memory.

Tabular networks without sequences use the same options without `:sequence`.
Classification uses `:loss :bce` or `:ce` on logits; `nn/predict` with
`:output :class` or `:probability` converts them.

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

Constructors are used by `nn/train!` and `garun.training/config`, which also
accepts listeners, evaluators, initializers, devices and a caller-owned
executor. Explicit listeners replace the defaults.

## Tensors and device placement

DuckDB remains the ingestion/transformation boundary. `table/matrix` returns
row-major float32 host storage; `tensor/table-matrices` extracts multiple
column groups from one query. Construct DJL arrays once, then keep matrix
arithmetic and training on native arrays. Use DJL's `NDArray` methods directly;
there is no duplicate tensor-operation DSL.

```clojure
(require '[garun.session :as session] '[garun.tensor :as tensor] '[garun.nn :as nn])

(with-open [current-session (session/open {:device :cpu})]
  (let [network (nn/build current-session [:sequential
                                           [:conv2d {:filters 4 :kernel [3 3]}]
                                           [:global-avg-pool2d]
                                           [:linear 1]]
                          {:input-shape [1 8 8]})
        images (tensor/array current-session (float-array (repeat 128 0.5)) {:shape [2 1 8 8]})]
    (.toFloatArray (.singletonOrThrow (nn/predict current-session network images {})))))
```

`tensor/array` and `tensor/ndlist` take a context or an `NDManager` as owner.
They copy borrowed native inputs independently onto the owner's device and
never steal, mutate or close inputs. Native copies/placement use device
operations, not a GPU → host → GPU path. Native `nn/predict` returns an
`NDList` owned by the context's manager, or by `{:manager manager}` when given.

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
`session/open` (or per call); `:auto` lets the engine choose.
Explicit unavailable GPU requests fail instead of silently falling back to CPU.
GPU execution has not been verified in this workspace: the NVIDIA driver
cannot be contacted.

## Translators and model loading

`nn/predict` with `:x` (or a stored input specification) predicts from a table
and returns Clojure vectors; otherwise it predicts native tensors.
`nn/predict-with` accepts a native DJL `Translator` and a typed input.

`model/criteria` accepts a native `Criteria` or a map requiring Java
`:input-class` and `:output-class`. Optional keys select a translator/factory,
block, local path, URL, model zoo, application, group/artifact, engine, device,
filters, arguments and progress. Native Criteria exposes more detailed DJL
settings without more DSL machinery.

`(model/load! context criteria {:name :resnet})` calls DJL's real model loader
and registers the `ZooModel` in the session; criteria maps without `:device`
use the session default. `model/predict` and `model/batch-predict` use its
packaged translator, or accept an explicit translator with any DJL Model.
Predictors close after each call; the model stays open. Direct NDArray/NDList
outputs of `model/predict`, `nn/predict-with` and native `nn/predict` belong to
the context's manager and are freed when the session (or worker) closes. Other
outputs keep translator-defined ownership; custom objects containing tensors
must manage their nested tensors in `processOutput`.

```sh
clojure -M:zoo          # optional DJL + PyTorch model-zoo providers
clojure -M:cuda:zoo     # combine providers with CUDA
```

The base API supports explicit local/URL Criteria without these provider jars.
Model paths, URLs, native translators and native blocks are trusted caller
inputs, not a sandbox. Local Criteria loading and real PNG translator inference
have been exercised; remote pretrained artifact downloads have not.

## Persistence

`(nn/save! context network directory)` writes named Hiccup specs as EDN plus DJL
parameters and the stored input specification.
Native blocks, factories, functions and constant embedding tensors require an
explicit `{:spec reconstruction-spec}`. A non-EDN reconstruction spec is not
serialized: `nn/load` then requires `:spec` from the caller.
Missing reconstruction information fails before writing files.

`(nn/load context directory {:name :gru})` registers the restored network.
Options can override `:spec`, `:device`, `:dtype` and `:engine`; for example
`{:device :cpu}` restores a GPU save on CPU.
