# garun-small

**garun-small is a small experiment: a Clojure (JVM) data and deep-learning DSL
in Hiccup form.** Tables are pure data plans—vectors and maps compiled through
HoneySQL into SQL executed by DuckDB. Neural networks are Hiccup specs compiled
into Deep Java Library (DJL) blocks using the PyTorch engine. The experiment's question was
whether the whole data-to-model workflow could be ordinary Clojure data. In
practice the Hiccup DSL turned out very readable and easy to write:

```clojure
(require '[garun.session :as session]
         '[garun.table :as table]
         '[garun.nn :as nn])

;; Supply data/readings.csv with numeric sample_id,x,target columns and
;; enough rows for nonempty training, validation and test partitions.
(with-open [current-session (session/open {:seed 7 :device :cpu})]
  (let [readings (table/query [:csv {:path "data/readings.csv" :header? true}]
                             [[:select :sample_id :x :target]
                              [:where [:not-null :target]]])
        [train validation test]
        (table/query readings [:split {:fractions [0.7 0.15 0.15]
                                       :seed 7 :key :sample_id}])
        network [:sequential [:linear 8] [:relu] [:linear 1]]]
    (nn/build current-session network {:input-shape [1] :name :regressor})
    (nn/train! current-session :regressor train
               {:x [:x] :y :target :standardize [:x :y]
                :validation validation :loss :mse
                :optim [:adam {:lr 0.01}] :epochs 25 :batch 128
                :early-stopping {:patience 5 :restore-best? true}
                :metrics [:mae] :observe tap>})
    (nn/evaluate current-session :regressor test {:metrics [:mae :mse]})))
```

One **Session** owns the DuckDB connection, tensor memory and named resources
(tables, networks, models and trainers), and closes them together. Training is
one `nn/train!` call, including partition-aware sequence windows, training-only
standardization, early stopping and epoch observation. The example evaluates a
separate test partition; validation is used to choose when to stop.

This is a small layer, not a replacement engine. The full **DJL** API—blocks,
training, tensors, model zoo and translators—and the full **DuckDB** SQL
engine—file formats, windows, joins and more—remain available underneath.
The DSL names common operations, and DJL-facing APIs accept the corresponding
native DJL objects wherever specs are accepted as escape hatches. HoneySQL and
trusted SQL expose DuckDB operations beyond the named table clauses. Rust,
classical estimators (k-means, SVM, XGBoost, …) and visualization are outside
this project's scope.

## Why tables instead of a dataframe

A dataframe is essentially a table. Here, a table is a **pure, inspectable
plan**, not an in-memory object containing the result. Plans compose as data;
you can inspect them at the REPL, reason about transformations before running
them, and print the SQL and bound parameters with `(table/->sql plan)`.
Building or composing a plan does not open a connection or read a file.

That gives you DuckDB's capabilities without a dataframe library dependency:

- A vectorized, columnar SQL engine that reads **CSV, Parquet and JSON directly**.
- Out-of-core SQL execution for data larger than RAM, subject to DuckDB's
  operator, memory and temporary-storage limits.
- Joins, window functions and grouped aggregates instead of host-side
  dataframe transformations.
- Seeded, keyed deterministic splits: the same key stays in one partition,
  including all rows belonging to one battery, customer or other entity.
  Fractions are hash-bucket proportions, not exact row-count quotas.
- Materialized session tables that are shared by worker threads, each using
  its own DuckDB connection and tensor sub-manager.
- Standardization means and population standard deviations computed
  **in-engine from training rows only**, then reused for validation and inference.

For file-backed training, selected feature/label data crosses into JVM host
storage once per prepared dataset as **float32 matrices**, then becomes DJL
tensors; the epoch loop does not re-extract table rows. This is not zero-copy
or streaming training: the current table training path eagerly holds matrices,
sequence windows and tensors in memory. DuckDB's out-of-core preprocessing
does not imply that an arbitrarily large training dataset fits in tensor
memory. Explicit calls such as `table/rows`, predictions and metrics also
return host-side results.

Plans are not a sandbox. `[:duckdb {:query "..."}]` and HoneySQL `:raw` forms
are **trusted SQL**; do not pass untrusted SQL to them. Host-side matrix
extraction is float32, so it is not a lossless route for arbitrary numeric
values or large integer identifiers.

## Status and limits

CPU execution is verified by the test suite and runnable gallery. CPU is the
default; the opt-in **CUDA profile is unverified**. Dependencies are pinned to
**DJL 0.36.0 / PyTorch 2.7.1**. Small Java bridge classes address missing,
inaccessible or defective DJL operations in that pinned stack:

| Bridge | Reason |
|---|---|
| [`FnBlock`](java/garun/djl/FnBlock.java) | Exposes protected child initialization/forwarding hooks for Clojure parallel and residual combinations. |
| [`TransposeConvolution`](java/garun/djl/TransposeConvolution.java) | Implements differentiable transposed convolution because PyTorch JNI has no transposed-convolution entry point. |
| [`MaskedTransformerEncoder`](java/garun/djl/MaskedTransformerEncoder.java) | Initializes encoder children from the embedding shape, not the optional attention-mask shape. |
| [`TensorOptimizer`](java/garun/djl/TensorOptimizer.java) | Supplies device-native optimizer updates where the pinned PyTorch optimizer kernels are incomplete. |
| [`OrderedRecurrent`](java/ai/djl/nn/recurrent/OrderedRecurrent.java) | Fixes biased bidirectional RNN/GRU/LSTM parameter ordering to match PyTorch's native contract. |

## Requirements and quick start

- **Clojure CLI** and **JDK 17+** (JDK 21 is recommended); Java bridges compile
  with `--release 17`. This is JVM Clojure, not Babashka.
- The default CPU native dependency targets **Linux x86-64**. Other platforms
  need a matching PyTorch native artifact instead of the bundled one.
- Maven access for dependency resolution on the first build/run.

From the repository root:

```sh
clojure -T:build javac   # compile Java bridges into target/classes
clojure -M:test          # sessions, DuckDB, blocks, tensors, training, inference, metrics
clojure -Sdeps '{:paths ["src" "examples" "target/classes"]}' -M -e "(require 'garun.examples.training) (garun.examples.training/run!)"
```

The gallery uses synthetic data and needs no downloaded dataset. All default
dependencies resolve from Maven, including the `pytorch-native-cpu`
Linux x86-64 jar, so CPU startup needs no separate native download. The `:cuda`
profile instead lets DJL download/cache the pinned CUDA library; see
[CUDA](#cuda) for device selection and driver requirements.

## License

MIT; see [`LICENSE`](LICENSE).

## Gallery

Copy a spec, change its dimensions, and run it. These namespaces are short,
top-to-bottom CPU stories with synthetic data, real native predictions and
small training runs; they need no datasets or extra dependencies:

- [Tables](examples/garun/examples/tables.clj): expressions, HoneySQL CASE,
  aggregates, running windows, left/semi/anti joins, keyed splits, seeded
  sampling, named materialization and a temporary parquet round trip.
- [Networks](examples/garun/examples/networks.clj): **14 runnable specs**:
  MLP, residual, inception, CNN, sequence convolution, convolutional autoencoder,
  GRU, LSTM, bidirectional LSTM, embedding/text classifier, transformer
  classifier, attention, NDArray lambda and custom parallel combination.
- [Training](examples/garun/examples/training.clj): BCE and CE classification
  and regression with deterministic held-out validation, standardization,
  battery sequences, early stopping, epoch observers, a custom trainer loop
  and model persistence.

Each file exposes `(run!)`, returning a plain-data summary, and `-main`,
printing and returning it. `garun.examples-test` runs all three and checks
shapes, finite/decreasing losses, disjoint keyed training/validation partitions
and persistence.

```sh
clojure -Sdeps '{:paths ["src" "examples" "target/classes"]}' -M -m garun.examples.tables
clojure -Sdeps '{:paths ["src" "examples" "target/classes"]}' -M -m garun.examples.networks
clojure -Sdeps '{:paths ["src" "examples" "target/classes"]}' -M -m garun.examples.training
```

For the following snippets, require the DSL and the gallery's synthetic sources:

```clojure
(require '[garun.session :as session] '[garun.table :as table] '[garun.nn :as nn]
         '[garun.examples.tables :as tables] '[garun.examples.training :as training])
```

**1. Expressions, CASE and a running window.** CASE uses native HoneySQL;
the surrounding clauses use Garun expressions.

```clojure
(def enriched
  (table/query tables/orders
    [[:derive {:gross [:* :amount :quantity]}]
     [:honeysql {:select [:* [[:case [:>= :gross 60] "large"
                                    :else "small"] :size]]}]
     [:derive {:running_total
               [:cast
                [:over [:sum :gross] {:partition-by [:customer_id]
                                     :order-by [:order_id]
                                     :rows [:unbounded :current]}]
                :int64]}]
     [:order-by :order_id]]))
```

DuckDB's integer `SUM` produces a 128-bit HUGEINT, while parquet has no 128-bit
integer type. This small-data example explicitly chooses `:int64` for its
running total before the parquet round trip.

**2. Grouped aggregates are ordinary Hiccup.**

```clojure
(table/query enriched
  [[:group-by [:customer_id] {:orders [:count] :revenue [:sum :gross]}]
   [:order-by [:revenue :desc]]])
```

**3. Join customers, then keep each customer's orders in one split.**
Use `:semi` to retain matched orders only, or `:anti` for unmatched orders.

```clojure
(table/query enriched
  [[:join tables/customers {:on [:customer_id] :how :left}]
   [:split {:fractions [0.75 0.25] :seed 7 :key :customer_id}]])
```

**4. A tabular MLP with batch normalization and dropout.**

```clojure
[:sequential
 [:linear 8] [:batchnorm] [:relu] [:dropout {:p 0.2}]
 [:linear 2]]
;; :input-shape [4], native batch [2 4] -> [2 2]
```

**5. Inception-style parallel branches concatenate image channels.**

```clojure
[:sequential
 [:parallel {:combine :concat :axis 1}
  [:sequential [:conv2d {:filters 2 :kernel [1 1]}] [:relu]]
  [:sequential [:conv2d {:filters 3 :kernel [3 3] :padding [1 1]}] [:relu]]]
 [:global-avg-pool2d] [:linear 3]]
;; :input-shape [1 8 8], native batch [2 1 8 8] -> [2 3]
```

**6. A convolutional autoencoder restores the original image size.**

```clojure
[:sequential
 [:conv2d {:filters 2 :kernel [3 3] :stride [2 2] :padding [1 1]}]
 [:relu]
 [:conv2d-transpose {:filters 1 :kernel [3 3] :stride [2 2]
                     :padding [1 1] :output-padding [1 1]}]
 [:sigmoid]]
;; :input-shape [1 8 8], native batch [2 1 8 8] -> [2 1 8 8]
```

**7. Last-timestep sequence regression needs no custom indexing function.**

```clojure
(def battery-network
  [:sequential
   [:gru {:state-size 4 :batch-first? true}]
   [:index ":, -1"] [:linear 1]])
;; :input-shape [3 2], native batch [N 3 2] -> [N 1]
```

**8. Train that GRU on held-out batteries with scaling and early stopping.**

```clojure
(with-open [current-session (session/open {:seed 7 :device :cpu})]
  (let [train (table/query training/battery-data [:where [:< :battery_id 6]])
        validation (table/query training/battery-data [:where [:>= :battery_id 6]])]
    (nn/build current-session battery-network {:input-shape [3 2] :name :battery})
    (nn/train! current-session :battery train
      {:x [:voltage :temperature] :y :remaining_life
       :sequence {:length 3 :partition-by :battery_id :order-by :cycle}
       :standardize [:x :y] :validation validation :loss :mse
       :optim [:adam {:lr 0.05}] :epochs 8 :batch 48 :shuffle? false
       :early-stopping {:patience 2 :min-delta 0.0001 :restore-best? true}
       :metrics [:mae]})))
```

Network specs are values, not model instances. The network gallery shows
`nn/build` plus an actual native `nn/predict` for every spec, including int64
embedding inputs. The training gallery uses `:observe tap>` for BCE metrics;
add a tap at the REPL to inspect epoch and finished events.

## Namespaces

| Namespace | Surface |
|---|---|
| `garun.context` | Borrowed native access: `connection`, `manager`, `owner`; lifecycle admission: `call` |
| `garun.session` | `open`, `worker`, `defaults`, `register!`, `resource`, `value`, `shared`, `exclusive`, `release!`, `describe` |
| `garun.table` | Pure plans: `from`, `query`, `->sql`, `table?`. Through a context: `rows`, `columns`, `schema`, `row-count`, `matrix`, `matrices`, `materialize!`, `write-csv!`, `write-parquet!` |
| `garun.blocks` | `compile-block`, `serializable-spec?`; Hiccup or native DJL `Block` |
| `garun.tensor` | `device`, `dtype`, `shape`, `array`, `ndlist` |
| `garun.ndarray` (`nd`) | Reflection-free NDArray math, indexing, reductions, shape transforms and NDList access |
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
;; #garun/session {:identifier "…" :status :open :database ":memory" :device :cpu :dtype :float32
;;                 :engine "PyTorch" :seed 7 :workers 0
;;                 :resources [{:identifier 1 :name :batteries :kind :table :rows 3200 :columns […]}
;;                             {:identifier 2 :name :gru :kind :network :spec […] :parameters 2401}]}
```

Networks print as `#garun/network {...}`, trainers as `#garun/trainer {...}`.

**Threads:** use one thread per context. `(session/worker context)` returns a
closeable worker with its own duplicated DuckDB connection and tensor
sub-manager; it sees the session's registry and materialized tables.
Predictions on one network may run concurrently; training takes the network's
exclusive lock. Named tables are locked while a query reads them, so a
release or replacement waits for running queries. Clojure Session, Worker and
Resource types own lifecycle/resource locks and immutable registry snapshots.
Atoms hold registry, identifier, worker and closed-state updates; native cleanup
runs outside atom updates. Workers duplicate their parent context's connection.

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
| Tensor transforms | `:reshape`, `:transpose`, `:index`, `:softmax`, `:log-softmax`, `:expand-dims`, `:squeeze` |

The `/1d/2d/3d` notation above abbreviates three separate tags.
Options are closed maps: misspelled or unsupported keys fail rather than vanish.
The per-tag builders in `src/garun/blocks.clj` define the full option contracts.

- `[:linear 16]` or `[:linear {:units 16 :bias? false}]`.
- Convolutions require `:filters` and `:kernel`; support stride, padding,
  dilation, groups and bias. Transposed convolutions also accept output padding.
- Recurrent blocks require `:state-size`; attention requires `:embedding-size`
  and `:heads`; transformer encoders also require `:hidden-size`.
- `[:index "<DJL NDIndex string>"]` indexes one input tensor using DJL's grammar
  unchanged: slices, steps, negative indices, `...` and `:`. The argument must
  be a nonempty string without NUL. Output shapes are inferred automatically.
  `[:index ":, -1"]` selects the last time step, mapping `[N,T,H]` to `[N,H]`;
  `[:index ":, ::2, :"]` strides over time, and `[:index "..., 0"]` drops the
  last axis by selecting its first entry. These specs serialize without `:spec`.
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

**Pinned recurrent backend limit:** DJL 0.36 orders biased bidirectional
RNN/GRU/LSTM parameters differently from PyTorch's native contract. Garun's
`ai.djl.nn.recurrent.OrderedRecurrent` bridge reorders the existing parameter
objects once at construction, preserving native gradients and persistence.
It lives in the DJL package solely to access package-private constructors.
Bidirectional outputs are checked against independent forward/reversed-time
unidirectional runs, with multi-layer, layout, state and save/load regressions.

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
                                           [:index ":, -1"]
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
All seven run as tensor operations on the weight's device (no host round trip).
For hand-written steps, `(training/update! optimizer "w" weight gradient)` applies
one update in place without caller type hints; reuse the same optimizer and
identifier so momentum/moment state carries across steps.

Constructors are used by `nn/train!` and `garun.training/config`, which also
accepts listeners, evaluators, initializers, devices and a caller-owned
executor. Explicit listeners replace the defaults.

## NDArray operations without type hints

Require `[garun.ndarray :as nd]` for Clojure functions over DJL arrays and
lists. User lambdas, functional losses and metrics need no Java type hints or
reflective method calls:

```clojure
(require '[garun.ndarray :as nd])

[:lambda {:singleton? true} (fn [array] (nd/get array ":, -1"))]
[:lambda (fn [arrays] (nd/ndlist (mapv #(nd/mul % 2) arrays)))]

;; A functional :loss receives labels and predictions as NDLists.
(fn [labels predictions]
  (nd/mean (nd/square (nd/sub (nd/singleton predictions)
                             (nd/singleton labels)))))
```

The CPU-tested wrapper groups are:

| Group | Functions |
|---|---|
| Creation | `zeros-like`, `ones-like`, `full-like`; owner-based `zeros`, `ones`, `full`, `arange`, `random-normal`, `random-uniform` |
| Selection and masking | `clip`, `where`, `boolean-mask`, `gather`, `one-hot`, `logical-and`, `logical-or`, `logical-not`, `is-nan`, `is-infinite` |
| Ordering | `sort`, `arg-sort`, `top-k`, `cum-sum`, `flip` |
| Additional math | `log10`, `log2`, `erf`, `atan2`, `norm`, `batch-matmul`, `broadcast`, `repeat`, `tile` |
| Gradients and devices | `stop-gradient`, `gradient`, `requires-gradient!`, `to-device` |

Owner-based creation takes `(owner dimensions options)`, where the owner is
an `NDManager` or a `garun.context/Context` (session or worker), dimensions is a
shape vector, and options defaults to `{}`. Results belong to that manager on
its device. `:dtype` uses `tensor/dtype`; the default is `:float32`, except
`:int32` for `arange`. `full` accepts `:value` (default 0). `arange` fills the
shape in row-major order from `:start` (default 0) by `:step` (default 1),
using DJL's int endpoints for integers or float endpoints otherwise.
`random-normal` accepts `:mean` and `:standard-deviation` (defaults 0 and 1);
`random-uniform` accepts inclusive `:low` and exclusive `:high` (defaults 0
and 1). Random distributions require backend-supported floating dtypes.
`full-like` uses the input's manager because DJL 0.36 has no `NDArray.fullLike`.
Fill values pass through DJL's float argument. Like-creation preserves shape,
dtype and device; a device unlike the input manager's uses its device-specific
child manager.

`where` takes three arrays `(condition when-true when-false)` and belongs to
`when-true`'s manager/device. PyTorch requires condition and `when-true` to have
the same shape. `boolean-mask` supports a same-shape mask (flat selection) or
a mask matching trailing dimensions beginning at its optional axis; a leading
row-only mask is unsupported. `gather` requires integer indices and an axis.
`one-hot` takes depth and an optional dtype, using 0/1 values; custom on/off
values are not implemented by this engine.

`sort` and `arg-sort` default to the last axis; `arg-sort` also accepts an
ascending boolean. `top-k` returns an `NDList` containing `[values indices]`,
owned by the input manager; closing it closes those two result arrays.
Its arguments are `(array k axis)` or `(array k axis largest? sorted?)`,
with both flags defaulting to true. `cum-sum` defaults to axis 0.
`flip` accepts an axis or axis vector, and `norm` accepts axes, `keep-dims?`,
and optionally an integer norm order. `repeat` repeats individual elements;
`tile` tiles entire dimensions. Both accept a scalar count for every axis or
a vector/long-array of per-axis counts. Only `repeat` exposes an axis/count
arity: PyTorch does not implement `tile(int, long)` or `tile(Shape)`.

`requires-gradient!` changes gradient tracking and returns the same array,
without changing its values. `gradient` reads the native gradient after
tracking/backpropagation; the result belongs to the input manager.
`stop-gradient` returns a detached array (shared storage on PyTorch) without
changing the original's tracking. `to-device` resolves `tensor/device` specs
and accepts optional `copy?` (default false): a matching device may return
the input. A result on the manager's device retains that manager; PyTorch
places a result on another device in a device-specific child manager.
No in-place value operations are wrapped.

Arithmetic and comparisons accept either another NDArray or a number.
`nd/sum`, `nd/mean`, `nd/max`, `nd/min` and `nd/prod` take optional axes
(a vector or `int-array`) and `keep-dims?`, e.g. `(nd/sum array [1] true)`.
`nd/argmax` and `nd/argmin` follow DJL's one-axis limit; omitted/nil axes
returns a flat index, and `keep-dims?` also works. Shape transforms use
dimension/axis vectors, `nd/shape` returns a vector, and `nd/concat`/`nd/stack`
accept a vector of arrays plus an optional axis. `nd/softmax` and
`nd/log-softmax` require an axis.

`nd/to-vector` copies flat numeric values to the host without routing int64
through floating point; boolean values become 1/0. Use it at output or metrics
boundaries, not inside differentiable tensor computations. `nd/to-type` resolves
dtype keywords through `tensor/dtype`; `nd/duplicate` explicitly copies an array.
All operations keep DJL's native gradients and manager ownership. `nd/ndlist`
wraps existing arrays without copying them; closing that list closes its
elements. In contrast, `tensor/ndlist` constructs independently owned copies.

## Tensors and device placement

DuckDB remains the ingestion/transformation boundary. `table/matrix` returns
row-major float32 host storage; `table/matrices` extracts multiple
column groups from one query. Construct DJL arrays once, then keep matrix
arithmetic and training on native arrays. `garun.ndarray` supplies type-hinted
functions over DJL operations; native methods remain available for the rest.

```clojure
(require '[garun.session :as session] '[garun.tensor :as tensor]
         '[garun.ndarray :as nd] '[garun.nn :as nn])

(with-open [current-session (session/open {:device :cpu})]
  (let [network (nn/build current-session [:sequential
                                           [:conv2d {:filters 4 :kernel [3 3]}]
                                           [:global-avg-pool2d]
                                           [:linear 1]]
                          {:input-shape [1 8 8]})
        images (tensor/array current-session (float-array (repeat 128 0.5)) {:shape [2 1 8 8]})]
    (nd/to-vector (nd/singleton (nn/predict current-session network images {})))))
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
The CUDA profile is available, but GPU execution remains unverified.

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
