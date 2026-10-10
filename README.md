# garun-small

garun-small is an experimental Clojure library for data preparation and deep learning on the JVM. Tables are query plans compiled through HoneySQL into SQL executed by DuckDB. Neural networks are Hiccup data structures compiled into Deep Java Library (DJL) blocks using the PyTorch engine. Both the query plan and the network in the example below are ordinary Clojure vectors and maps:

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

## A larger example: two inputs, a graph and sequences

The [battery-hybrid story](examples/garun/examples/training.clj) generates 32 synthetic batteries with 12 cycles each, per-cycle measurements and battery attributes repeated on every row. A keyed split keeps all rows for each battery in one of the training, validation or test partitions. Sequence preparation creates history windows with shape `[N,T,F] = [N,4,3]` and static windows `[N,4,2]`; the GRU branch and the static branch each use `[:index ":, -1"]` to select the last step. The graph concatenates a 16-value history encoding and an 8-value static context, then adds two `[N,24]` tensors through the residual edge. Training learns scaling from training rows, restores the best validation epoch and evaluates test MAE in remaining-life units.

```clojure
(require '[garun.examples.training :as training])

(with-open [current-session (session/open {:seed 7 :device :cpu})]
  (let [readings (table/from training/battery-hybrid-data)
        [training-batteries validation-batteries test-batteries]
        (table/query readings
                     [:split {:fractions [0.7 0.15 0.15] :seed 7 :key :battery_id}])
        network
        [:graph {:inputs [:history :static] :outputs [:remaining]}
         [:encoded [:history]
          [:sequential [:gru {:state-size 16 :batch-first? true}] [:index ":, -1"]]]
         [:context [:static]
          [:sequential [:index ":, -1"] [:linear 8] [:relu]]]
         [:joined [:encoded :context] [:concat {:axis 1}]]
         [:hidden [:joined] [:sequential [:linear 24] [:relu]]]
         [:residual [:hidden :joined] [:add]]
         [:remaining [:residual] [:linear 1]]]]
    (nn/build current-session network
              {:input-shapes [[1 4 3] [1 4 2]] :name :battery-hybrid})
    (nn/train! current-session :battery-hybrid training-batteries
               {:x {:history [:voltage :temperature :capacity]
                    :static [:nominal_capacity :resistance]}
                :y :remaining_life
                :sequence {:length 4 :partition-by :battery_id :order-by :cycle}
                :standardize [:x :y] :loss :mse
                :optim [:adam {:lr 0.03}] :epochs 8 :batch 256 :shuffle? false
                :early-stopping {:patience 3 :min-delta 0.0001 :restore-best? true}
                :metrics [:mae] :validation validation-batteries})
    {:test (nn/evaluate current-session :battery-hybrid test-batteries {:metrics [:mae]})
     :predictions (nn/predict current-session :battery-hybrid test-batteries {})}))
```

A session manages a DuckDB connection, a root tensor manager, and registered resources (tables, networks, models, and trainers). Closing the session closes these resources together. `nn/train!` executes the training loop, including sequence windowing, training-set feature standardization, early stopping, and epoch metrics observation.

Garun is a thin layer over DJL and DuckDB. The full DJL API (blocks, trainers, tensors, model zoo, and translators) and DuckDB SQL engine (file formats, window functions, and joins) remain directly accessible. Functions accepting Hiccup specifications also accept native DJL objects. HoneySQL maps and raw SQL strings expose DuckDB operations beyond built-in query clauses.

Classical estimators (such as k-means, SVM, or XGBoost), Rust bindings, and visualization tools are outside the scope of this project.

## Why table plans and DuckDB

`table/from` and `table/query` return a description of a query as a Clojure map. Building a plan does not open a database connection or read a file. You can inspect plans at the REPL and print the generated SQL and parameters with `(table/->sql plan)`. DuckDB runs the query when you pass the plan to an execution function such as `table/rows`, `table/materialize!`, or `nn/train!`.

Using table plans and DuckDB provides:
- Direct file reads for CSV, Parquet, and JSON files.
- Out-of-core SQL execution for data larger than RAM, subject to DuckDB memory and disk limits.
- In-engine joins, window functions, and grouped aggregations.
- Deterministic keyed data splits: rows with the same key (such as an entity or customer ID) stay in one partition. Fractions define hash-bucket proportions for row assignment.
- Materialized session tables in DuckDB, accessible across worker threads with per-worker connections and tensor sub-managers.
- In-engine standardization: DuckDB computes mean and population standard deviation across training rows in one query; garun reuses these values for validation and inference.

Costs and limits:
- Float32 matrix extraction: For model training, selected feature and label columns cross into JVM host memory once as float32 matrices before conversion into DJL tensors.
- In-memory training data: The training path holds feature matrices, sequence windows, and tensors in memory. DuckDB executes queries out of core, but the prepared training dataset must fit in memory.
- Trusted SQL: Query clauses using `[:duckdb {:query "..."}]` and HoneySQL `:raw` execute as trusted SQL. Untrusted input must not be passed to them.
- Numeric precision: Matrix extraction converts numeric data to float32, which truncates integers exceeding 24 bits of precision.
- Host copies: Functions such as `table/rows`, `nn/predict`, and evaluation metrics copy results to JVM host data structures.

## Status and limits

CPU execution is verified by the test suite and example gallery. CPU is the default device. The opt-in CUDA profile is unverified.

Dependencies are pinned to DJL 0.36.0 and PyTorch 2.7.1. Java bridge classes in this repository provide workarounds for limitations in this pinned stack:

| Bridge | Reason |
|---|---|
| [`FnBlock`](java/garun/djl/FnBlock.java) | Exposes protected child initialization/forwarding hooks for Clojure parallel and residual combinations. |
| [`TransposeConvolution`](java/garun/djl/TransposeConvolution.java) | Implements differentiable transposed convolution because PyTorch JNI has no transposed-convolution entry point. |
| [`MaskedTransformerEncoder`](java/garun/djl/MaskedTransformerEncoder.java) | Initializes encoder children from the embedding shape, not the optional attention-mask shape. |
| [`TensorOptimizer`](java/garun/djl/TensorOptimizer.java) | Supplies device-native optimizer updates where the pinned PyTorch optimizer kernels are incomplete. |
| [`OrderedRecurrent`](java/ai/djl/nn/recurrent/OrderedRecurrent.java) | Fixes biased bidirectional RNN/GRU/LSTM parameter ordering to match PyTorch's native contract. |

## Requirements and quick start

- Clojure CLI and JDK 17+ (JDK 21 is recommended). Java bridges compile with `--release 17`. This project runs on JVM Clojure.
- The default CPU native dependency targets Linux x86-64. Other platforms require a matching PyTorch native dependency.
- Network access to Maven Central for dependency resolution on first build.

From the repository root:

```sh
clojure -T:build javac   # compile Java bridges into target/classes
clojure -M:test          # sessions, DuckDB, blocks, tensors, training, inference, metrics
clojure -Sdeps '{:paths ["src" "examples" "target/classes"]}' -M -e "(require 'garun.examples.training) (garun.examples.training/run!)"
```

The gallery uses synthetic data and requires no downloaded dataset. Default dependencies resolve from Maven Central, including the `pytorch-native-cpu` Linux x86-64 jar. The `:cuda` profile downloads the pinned CUDA library on first use; see [CUDA](#cuda) for driver requirements.

## License

MIT; see [`LICENSE`](LICENSE).

## Gallery

The example namespaces demonstrate query and model specifications using synthetic data. They run on CPU without external datasets:

- [Tables](examples/garun/examples/tables.clj): expressions, HoneySQL CASE, aggregations, running window functions, left/semi/anti joins, keyed splits, seeded sampling, table materialization, and a Parquet round trip.
- [Networks](examples/garun/examples/networks.clj): 16 runnable specs: MLP, residual, Inception, CNN, image-and-measurements graph classifier, long-skip graph, sequence convolution, convolutional autoencoder, GRU, LSTM, bidirectional LSTM, embedding/text classifier, transformer classifier, attention, NDArray lambda, and custom parallel layers.
- [Training](examples/garun/examples/training.clj): BCE and CE classification, regression, multi-input graph training and a two-input battery graph with held-out validation and test batteries, feature standardization, sequence windows, early stopping, epoch observers, custom training loops, and model persistence.

Each file exposes `(run!)`, returning a data summary, and `-main`, printing and returning it. `garun.examples-test` runs all three and verifies output shapes, decreasing losses, keyed split boundaries, and persistence.

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

1. Expressions, CASE, and a running window. CASE uses native HoneySQL; surrounding clauses use Garun expressions.

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

DuckDB integer aggregations produce 128-bit HUGEINT values. Parquet does not support 128-bit integers. This query explicitly casts the running total to `:int64` before saving to Parquet.

2. Grouped aggregations in Hiccup syntax.

```clojure
(table/query enriched
  [[:group-by [:customer_id] {:orders [:count] :revenue [:sum :gross]}]
   [:order-by [:revenue :desc]]])
```

3. Joins and keyed splits. Use `:semi` to retain matched orders only, or `:anti` for unmatched orders.

```clojure
(table/query enriched
  [[:join tables/customers {:on [:customer_id] :how :left}]
   [:split {:fractions [0.75 0.25] :seed 7 :key :customer_id}]])
```

4. Tabular MLP with batch normalization and dropout.

```clojure
[:sequential
 [:linear 8] [:batchnorm] [:relu] [:dropout {:p 0.2}]
 [:linear 2]]
;; :input-shape [4], native batch [2 4] -> [2 2]
```

5. Inception-style parallel branches concatenating image channels.

```clojure
[:sequential
 [:parallel {:combine :concat :axis 1}
  [:sequential [:conv2d {:filters 2 :kernel [1 1]}] [:relu]]
  [:sequential [:conv2d {:filters 3 :kernel [3 3] :padding [1 1]}] [:relu]]]
 [:global-avg-pool2d] [:linear 3]]
;; :input-shape [1 8 8], native batch [2 1 8 8] -> [2 3]
```

6. Convolutional autoencoder restoring the original image size.

```clojure
[:sequential
 [:conv2d {:filters 2 :kernel [3 3] :stride [2 2] :padding [1 1]}]
 [:relu]
 [:conv2d-transpose {:filters 1 :kernel [3 3] :stride [2 2]
                     :padding [1 1] :output-padding [1 1]}]
 [:sigmoid]]
;; :input-shape [1 8 8], native batch [2 1 8 8] -> [2 1 8 8]
```

7. Sequence regression selecting the final time step.

```clojure
(def battery-network
  [:sequential
   [:gru {:state-size 4 :batch-first? true}]
   [:index ":, -1"] [:linear 1]])
;; :input-shape [3 2], native batch [N 3 2] -> [N 1]
```

8. Train a GRU on held-out battery partitions with scaling and early stopping.

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

9. A graph classifier with separate image and measurement inputs.

```clojure
[:graph {:inputs [:image :measurements] :outputs [:logits]}
 [:features [:image]
  [:sequential [:conv2d {:filters 8 :kernel [3 3]}] [:relu] [:global-avg-pool2d]]]
 [:joined [:features :measurements] [:concat {:axis 1}]]
 [:hidden [:joined] [:sequential [:linear 32] [:relu]]]
 [:logits [:hidden] [:linear 3]]]
;; :input-shapes [[1 1 8 8] [1 4]], native batches [2 1 8 8] and [2 4] -> [2 3]
```

Network specifications are data structures. `nn/build` compiles a specification into a DJL model instance. The network gallery demonstrates `nn/build` and `nn/predict` for each spec, including int64 embedding inputs. The training gallery uses `:observe tap>` to send epoch and completion maps to Clojure's tap queue.

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

Functions managing stateful resources take a context as their first argument (a session or a worker). Functions operating on data definitions (table plans or Hiccup network specs) do not require a context. When a context is supplied, functions accept either native objects or the keyword names of registered resources (`(nn/predict current-session :gru test {})`).

| Owned by the session | Notes |
|---|---|
| DuckDB connection | `:database` is `:memory` or a file path |
| Root tensor manager | Tensors created or returned through a context belong to it |
| Defaults | `:device`, `:dtype`, `:engine`, `:seed`; per-call options override |
| Registry | Networks, zoo models, trainers and materialized tables, optionally named |

Closing a session closes open workers, closes owned resources in reverse registration order, drops the session's private DuckDB schema, closes the tensor manager, and closes the DuckDB connection. Resources registered with `{:owned? false}` remain open. `session/release!` closes an individual resource before session close. Registering an existing name throws an exception unless `{:replace? true}` is passed.

Printing a session, network, or trainer at the REPL shows its record fields. `clojure.datafy/datafy` converts sessions into Clojure maps.

```clojure
(prn current-session)
;; #garun/session {:identifier "…" :status :open :database ":memory" :device :cpu :dtype :float32
;;                 :engine "PyTorch" :seed 7 :workers 0
;;                 :resources [{:identifier 1 :name :batteries :kind :table :rows 3200 :columns […]}
;;                             {:identifier 2 :name :gru :kind :network :spec […] :parameters 2401}]}
```

Networks print as `#garun/network {...}`, trainers as `#garun/trainer {...}`.

Contexts are thread-confined: use one thread per context. `(session/worker context)` returns a worker with its own DuckDB connection and child tensor manager. Workers share access to the parent session's registry and materialized tables. Predictions on a network can run concurrently across threads. Training acquires an exclusive lock on the network. Queries acquire read locks on named tables, so release or replacement waits for active queries to complete.

Table sources include a column map, a vector of row maps, `[:csv {:path :header? :delimiter}]`, `[:parquet {:path}]`, `[:json {:path}]`, `[:duckdb {:query "SELECT …"}]`, or the keyword name of a materialized session table. DuckDB query strings execute as trusted SQL.

`(table/materialize! context :name source)` runs a plan once into a table in the session's private schema and returns `(table/from :name)`. Materialized tables avoid re-executing data ingestion queries, and workers see the same table. Pure `table/->sql` rejects plans naming session tables; `(table/->sql context plan)` resolves them.

Supported query clauses: `:select`, `:rename`, `:where`, `:derive`, `:group-by`, `:order-by`, `:limit`, `:join`, `:sample`, `:split`, `:honeysql`.
`:split` must be the final clause in a query pipeline. It returns a vector of query plans that partition input rows deterministically.

Expressions use Garun operators, aggregates, and window functions (`[:over expr {...}]`).

Use `[:honeysql clause-map]` for HoneySQL SELECT queries. Garun supplies `FROM` from the preceding table and defaults to `SELECT *` when no projection is supplied. Supported clauses include projections (including DISTINCT), WHERE, grouping/HAVING, windows/QUALIFY, ordering, limits/offsets, joins, and CTEs. Source replacement and DDL/DML clauses are rejected.

```clojure
(table/query data
  [[:where [:not-null :amount]]
   [:honeysql {:select [:category [[:sum :amount] :total]]
               :group-by [:category]
               :having [:> [:sum :amount] 10]}]
   [:order-by [:total :desc]]])
```

Expressions inside `:honeysql` follow HoneySQL rules: `[:= :x nil]` compiles to `IS NULL`. HoneySQL `:raw` forms execute as trusted SQL strings. DuckDB resolves generated column names; a later Garun `:select` clause can explicitly name the output columns.

## Neural Hiccup

| Family | Tags |
|---|---|
| Composition | `:sequential`, `:parallel`, `:skip`, `:graph`, `:block`, `:factory`, `:lambda` |
| Dense | `:linear`, `:dropout`, `:flatten` |
| Activations | `:activation`, `:relu`, `:relu6`, `:tanh`, `:sigmoid`, `:gelu`, `:selu`, `:mish`, `:softplus`, `:softsign`, `:prelu`, `:identity`, `:leaky-relu`, `:elu`, `:swish` |
| Convolution | `:conv1d`, `:conv2d`, `:conv3d`, `:conv1d-transpose`, `:conv2d-transpose` |
| Pooling | `:max-pool1d/2d/3d`, `:avg-pool1d/2d/3d`, `:global-max-pool1d/2d/3d`, `:global-avg-pool1d/2d/3d` |
| Normalization | `:batchnorm`, `:layernorm`, `:ghost-batchnorm` |
| Embeddings | `:embedding`, `:word-embedding`, `:constant-embedding` |
| Recurrent | `:rnn`, `:lstm`, `:gru` |
| Attention | `:attention`, `:transformer-encoder`, `:bert`, `:bert-pretraining`, `:bert-next-sentence` |
| Tensor transforms | `:reshape`, `:transpose`, `:index`, `:softmax`, `:log-softmax`, `:expand-dims`, `:squeeze` |

The `/1d/2d/3d` notation above indicates three separate tags.
Options are closed maps: unrecognized keys throw an exception.
The per-tag builders in `src/garun/blocks.clj` define the full option contracts.

- `[:linear 16]` or `[:linear {:units 16 :bias? false}]`.
- Convolutions require `:filters` and `:kernel`; they support stride, padding, dilation, groups, and bias. Transposed convolutions also accept output padding.
- Recurrent blocks require `:state-size`; attention requires `:embedding-size` and `:heads`; transformer encoders also require `:hidden-size`.
- `[:index "<DJL NDIndex string>"]` indexes one input tensor using DJL's index grammar: slices, steps, negative indices, `...`, and `:`. The argument must be a nonempty string without NUL characters. Output shapes are inferred automatically. `[:index ":, -1"]` selects the last time step, mapping `[N,T,H]` to `[N,H]`; `[:index ":, ::2, :"]` strides over time, and `[:index "..., 0"]` drops the last axis by selecting its first entry. These specs serialize directly as EDN.
- `:parallel` passes the same input to every child block. `:skip` adds or concatenates the block output with the original input. Both accept `{:combine :add}` or `{:combine :concat :axis -1}`, or a custom combining function with an optional `:shape-fn`. Multi-output addition requires matching output counts.
- A native `Block` works directly or wrapped as `[:block block]`. `[:factory make-block]` calls a zero-argument constructor function. `[:lambda f]` accepts a Clojure function from `NDList` to `NDList`; `[:lambda {:singleton? true} f]` accepts a function from `NDArray` to `NDArray`.

`garun.djl.FnBlock` supplies DJL's protected child initialization and forwarding methods. Java bridge classes also implement differentiable transposed convolution, masked transformer initialization, and tensor-native optimizers where the pinned PyTorch backend lacks the public operation. Transposed convolution uses DJL's weight layout `[filters, input-channels/groups, kernel…]`.

DJL 0.36 orders biased bidirectional RNN/GRU/LSTM parameters differently from PyTorch's native contract. The `ai.djl.nn.recurrent.OrderedRecurrent` bridge reorders parameter objects during block construction to match PyTorch parameter layout, preserving native gradients and serialization compatibility.

`nn/build` accepts either `:input-shape` (one sample shape; batch dimension 1 is prepended automatically) or `:input-shapes` (concrete DJL shapes for multi-input architectures). These options are mutually exclusive. `:dtype` defaults to `:float32`, `:device` defaults to `:cpu`, and `:engine` selects a DJL engine.

### Graphs

A graph declares its input and output names and routes tensors through named nodes:

```clojure
[:graph {:inputs [:left :right] :outputs [:output]}
 [:joined [:left :right] [:concat {:axis 1}]]
 [:output [:joined] [:linear 1]]]
```

`:inputs` and `:outputs` are nonempty keyword vectors. Input names are unique. Each node is `[name [dependencies] block]`, with a keyword name and a nonempty keyword dependency vector. Node names are unique and distinct from input names. Names identify single tensors, and each node block must produce exactly one tensor. Dependencies may refer to inputs or other nodes; nodes may appear in any order. Every node must contribute to a declared output. Output names may refer to inputs or nodes, and output tensors follow the declared order.

Graph node blocks `[:concat {:axis n}]` and `[:add]` merge at least two dependencies. `[:concat]` defaults to axis `-1`; concatenation requires matching dimensions outside the chosen axis. `[:add]` performs elementwise addition with broadcasting. These merge tags are valid as graph node blocks. Referencing an earlier tensor alongside a later node creates a long skip connection.

Malformed graph options or nodes, duplicate names, unknown references, dependency cycles, unused nodes, and invalid merge options throw an exception with `:error :nn/invalid-graph`. Tensor counts and incompatible shapes fail during initialization or forwarding.

Graphs composed of serializable EDN blocks are serializable specifications; `nn/save!` and `nn/load` preserve their topology and parameters. Graphs containing native blocks or functions use the existing explicit reconstruction-spec policy.

## Training

`nn/train!` executes a complete training and validation loop. DuckDB computes dataset splits, sample ordering, sequence window boundaries, and standardization statistics. The following example trains a GRU model on battery data from CSV:

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
| `:x` | One keyword vector for a single input, a vector of keyword vectors in network input order, or a map from graph input names to keyword vectors |
| `:y` | Label column keyword or vector of label column keywords |
| `:sequence` | Full windows of `:length` rows per `:partition-by`, ordered by `:order-by`; the label is the last row's `:y`. Input is `[windows length features]` |
| `:standardize` | `[:x]`, `[:y]` or both. Mean and population standard deviation come from the training source only, in one DuckDB aggregate |
| `:validation` | Another table source, or a DJL `Dataset` for native sources |
| `:loss`, `:optim`, `:epochs`, `:batch`, `:shuffle?`, `:seed` | As in `garun.training`; `:loss` may also be `(fn [labels predictions] NDArray)` |
| `:early-stopping` | `{:patience n :min-delta d :restore-best? b}`; monitors validation loss, else training loss |
| `:metrics` | `garun.metrics` keywords or `(fn [predictions truth] number)`, computed on validation every epoch in original units |
| `:observe` | Function receiving one event map per epoch, then a `:finished` event. Exceptions abort training |
| `:stop?` | Function of the epoch state; returning true stops training |
| `:forward` | `(fn [trainer inputs training?] NDList)` replacing the forward pass |
| `:config` | A native `TrainingConfig` overriding component options |

For multi-input table training, `:x [[:horizontal :vertical] [:temperature]]` supplies two tensors in network input order. A graph with `:inputs [:position :measurements]` also accepts `:x {:position [:horizontal :vertical] :measurements [:temperature]}`. The map must contain exactly the graph's input names; map iteration order has no effect. Each group is a nonempty keyword vector and must match its input shape.

All column groups and labels come from one table extraction, preserving row alignment. Tabular groups produce `[rows features]` tensors; with `:sequence`, every group uses the same windows and produces `[windows length features]`. `:standardize [:x]` computes training-only statistics for every feature group. The [multi-input training example](examples/garun/examples/training.clj) trains a graph on a sample-keyed held-out split, then evaluates and predicts using its stored column groups.

`train!` returns a map of training metrics:

```clojure
{:history [{:epoch 1 :training-loss 0.067 :validation-loss 0.041 :best? true
            :elapsed-milliseconds 312.0 :metrics {:mae 3.1}} …]
 :best {:epoch 24 :loss 2.8e-5 :monitor :validation-loss}
 :stopped :completed}   ; or :early-stopping
```

The input specification (`:x`, `:y`, `:sequence`, `:standardize`) and its computed statistics are stored with the network and serialized by `nn/save!`. Named graph groups are stored in declared network input order. `nn/predict` and `nn/evaluate` reuse these settings, including all input groups when `:x` is omitted: features are scaled using training statistics, outputs return in original units, and evaluation aligns sequence windows with ground truth labels. Pass explicit `:x` or `:y` options to override them.

`train!` is implemented as a loop over `nn/epoch!`. You can drive this loop directly for custom schedules or stopping rules:

```clojure
(let [handle (nn/trainer current-session :gru train {:x features :y :RUL :validation validation})]
  (try
    (loop [state {}]
      (let [state (nn/epoch! handle state)]
        (if (my-stop? state) (nn/restore-best! handle state) (recur state))))
    (finally (session/release! current-session handle))))
```

`epoch!` returns the next state map (`:epoch :history :best`); `validate` returns `{:loss :metrics}`. A trainer is a session resource: release it with `session/release!` or at session close.

Table sources and native DJL `Dataset` sources use the same training loop (`Trainer.iterateDataset`, gradient collector, and optimizer step). Table sources convert to `ArrayDataset` instances with seeded shuffling; native datasets remain borrowed. Best-epoch parameters are saved and restored directly on the device.

For tabular networks without sequence data, omit `:sequence`. Classification uses `:loss :bce` or `:loss :ce` on logits; `nn/predict` with `:output :class` or `:output :probability` converts them.

## Training components

| Component | Named specs |
|---|---|
| Loss | `:mse`, `:mae`, `:bce`, `:ce`, `:quantile`, `:hinge`, `:masked-ce`, `:composite`, `:l1-decay`, `:l2-decay`, `:elastic-net` |
| Optimizer | `:sgd`, `:nag`, `:adam`, `:adamw`, `:rmsprop`, `:adagrad`, `:adadelta` |
| Learning-rate tracker | `:fixed`, `:factor`, `:multi-factor`, `:cosine`, `:linear`, `:polynomial`, `:warm-up` |
| Initializer | `:zeros`, `:ones`, `:constant`, `:normal`, `:truncated-normal`, `:uniform`, `:xavier` |

Functions accept either keyword specifications or native DJL instances. Losses follow DJL shape semantics; BCE and CE take logits by default. `[:bce {:from-sigmoid? true}]` accepts probabilities. Masked CE supports sparse or dense labels, sequence lengths, and a configurable class axis.

Optimizers take `[tag options]`, for example `[:adam {:lr 0.01}]`. Common options include `:weight-decay`, `:clip-grad`, `:rescale-grad`, and `:begin-update`. Family-specific options include momentum, beta values, rho, epsilon, and centered RMSProp. `:lr` accepts a number, tracker spec, or native `ParameterTracker`; Adadelta does not take a learning-rate parameter. AdamW uses decoupled weight decay. Rescaling and symmetric gradient clipping precede coupled L2 decay in other optimizers. Create fresh optimizer and tracker instances for independent training runs. All seven optimizers execute updates directly on the device hosting the weight tensor.

For manual update steps, `(training/update! optimizer "w" weight gradient)` applies an in-place update. Reusing the optimizer instance and parameter identifier preserves momentum and accumulator states across steps.

`garun.training/config` accepts training options, listeners, evaluators, initializers, devices, and a caller-owned executor. Explicit listeners replace the defaults.

## NDArray operations without type hints

The `garun.ndarray` namespace (`nd`) provides Clojure functions for DJL `NDArray` and `NDList` instances without requiring reflection or Java type hints in user code:

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

Owner-based creation takes `(owner dimensions options)`, where the owner is an `NDManager` or a `Context` (session or worker), dimensions is a shape vector, and options defaults to `{}`. Created arrays belong to that manager on its device. `:dtype` defaults to `:float32`, except `:int32` for `arange`. `full` accepts `:value` (default 0). `arange` fills the shape in row-major order from `:start` (default 0) by `:step` (default 1), using integer endpoints for integers and float endpoints otherwise. `random-normal` accepts `:mean` and `:standard-deviation` (defaults 0 and 1); `random-uniform` accepts inclusive `:low` and exclusive `:high` (defaults 0 and 1). Random distributions require floating-point dtypes supported by the backend. `full-like` allocates through the input array's manager. Like-creation functions preserve shape, dtype, and device; requesting a different device creates the result in a child manager on that device.

`where` takes three arrays `(condition when-true when-false)` and assigns ownership to `when-true`'s manager. PyTorch requires `condition` and `when-true` to share the same shape. `boolean-mask` supports a same-shape mask (flat selection) or a mask matching trailing dimensions beginning at its optional axis; leading row-only masks are not supported by PyTorch. `gather` requires integer index tensors and an axis. `one-hot` takes depth and an optional dtype, producing 0/1 values.

`sort` and `arg-sort` sort along the last axis by default; `arg-sort` accepts an ascending boolean flag. `top-k` returns an `NDList` containing `[values indices]`, owned by the input manager. Its arguments are `(array k axis)` or `(array k axis largest? sorted?)`, with both boolean flags defaulting to true. `cum-sum` defaults to axis 0. `flip` accepts an axis or axis vector. `norm` accepts axes, `keep-dims?`, and an optional integer norm order. `repeat` repeats individual elements; `tile` tiles entire dimensions. Both accept a scalar count for all axes or a vector of per-axis counts. Only `repeat` supports a single axis argument: PyTorch does not implement single-axis `tile` overloads.

`requires-gradient!` enables or disables gradient tracking on an array in place and returns the array. `gradient` reads the native gradient tensor after backpropagation; the result belongs to the input manager. `stop-gradient` returns a detached array sharing underlying storage without modifying tracking on the original. `to-device` resolves device specifications and accepts an optional `copy?` flag (default false): an array already on the target device is returned directly unless copy is requested. Results on another device belong to a child manager on that device. All wrapped math operations return new `NDArray` instances; in-place value mutations are not exposed.

Arithmetic and comparisons accept either an NDArray or a scalar number. `nd/sum`, `nd/mean`, `nd/max`, `nd/min`, and `nd/prod` accept optional axis vectors and a `keep-dims?` boolean flag. `nd/argmax` and `nd/argmin` follow DJL's single-axis constraint; omitting the axis returns a flattened index. Shape transforms use dimension vectors, `nd/shape` returns a vector, and `nd/concat`/`nd/stack` accept a vector of arrays and an axis. `nd/softmax` and `nd/log-softmax` require an explicit axis argument.

`nd/to-vector` copies flat numeric values to JVM vectors, preserving int64 integer representations. Boolean values become 1/0 integers. Use it at output and metric boundaries; it is non-differentiable. `nd/to-type` converts dtypes using `tensor/dtype`; `nd/duplicate` explicitly copies an array. Operations retain DJL native gradients and manager ownership. `nd/ndlist` wraps existing arrays without copying them; closing that list closes its constituent arrays. `tensor/ndlist` constructs independently owned copies.

## Tensors and device placement

DuckDB handles data filtering, transformations, and sequence windowing. `table/matrix` extracts query results into row-major float32 JVM arrays; `table/matrices` extracts multiple column groups in a single query. Converting these arrays into DJL tensors creates native memory structures for model execution and training.

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

`tensor/array` and `tensor/ndlist` take a context or an `NDManager` as owner. They create copies of input data on the target device, leaving input arrays unmodified and open. Native transfers and copies execute directly on the target device. Native `nn/predict` returns an `NDList` owned by the context manager, or by `{:manager manager}` when specified.

Primitive Java arrays preserve supported dtypes. Nested Clojure integers default to int64 and floating numbers to float64; pass `:float32` for model inputs. Descriptors use the map form `{:data values :shape dimensions :dtype dtype}`; an NDList can contain multiple descriptors. Supported shapes include scalars `[]` and zero-dimension tensors. Dimension cardinality and rectangular layout are validated; int64 inputs avoid intermediate float conversion.

The pinned DJL PyTorch JNI bindings do not support int16, bfloat16, or unsigned integer types larger than uint8. Garun constructors reject these types before calling JNI to prevent native process aborts. For Java `short[]` data, request an explicit supported dtype such as `:int32` or `:int64`.

### CUDA

```sh
clojure -M:cuda
```

The `:cuda` alias excludes the CPU native archive and selects PyTorch 2.7.1's `cu128` library. DJL downloads and caches the CUDA runtime on first use; an NVIDIA driver supporting CUDA 12.8 is required. Restart the JVM when switching dependency profiles. Pass `:device [:gpu 0]` to `session/open` or to individual function calls; `:auto` delegates device selection to the engine. Requesting an unavailable GPU throws an exception. GPU execution under the `:cuda` profile is unverified.

## Translators and model loading

`nn/predict` with `:x` (or a stored input specification) predicts from a table source and returns Clojure vectors. When passed an explicit tensor, `nn/predict` returns native tensor outputs. `nn/predict-with` accepts a native DJL `Translator` and typed Java inputs.

`model/criteria` accepts a native `Criteria` instance or a map specifying Java `:input-class` and `:output-class`. Optional keys configure translator factories, blocks, local file paths, URLs, model zoo names, application tags, group/artifact identifiers, engine, device, filters, arguments, and progress listeners. Native `Criteria` objects can be passed directly for settings not covered by the map syntax.

`(model/load! context criteria {:name :resnet})` loads a model through DJL and registers the `ZooModel` in the session. Criteria maps without `:device` inherit the session default. `model/predict` and `model/batch-predict` run inference using the model's packaged translator, or an explicit translator passed by the caller. Predictors close after each call while the model remains open. Direct NDArray and NDList outputs belong to the context manager and close when the session or worker closes. Other return types follow translator-defined lifecycle rules.

```sh
clojure -M:zoo          # optional DJL + PyTorch model-zoo providers
clojure -M:cuda:zoo     # combine providers with CUDA
```

The base API supports local file and URL `Criteria` directly. Model paths, URLs, custom translators, and native blocks execute with application permissions. Model loading from local files and PNG image inference with built-in translators are tested; remote pretrained artifact downloads are untested.

## Persistence

`(nn/save! context network directory)` serializes a network specification as EDN, along with PyTorch parameter weights and input preprocessing metadata.

Custom blocks, factory functions, lambdas, and constant embedding tensors require an explicit `{:spec reconstruction-spec}` option at save time. If a reconstruction spec cannot be serialized to EDN, `nn/load` requires `:spec` to be provided explicitly. `nn/save!` validates reconstruction requirements before writing to disk.

`(nn/load context directory {:name :gru})` restores and registers a saved network. Caller options can override `:spec`, `:device`, `:dtype`, and `:engine`. For example, passing `{:device :cpu}` loads a model saved on a GPU onto the CPU.
