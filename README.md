# garun-small

A small JVM Clojure version of the Garun Hiccup DSL. It has two backends:

- **DuckDB** (plain JDBC) for tables. It uses Garun's table clause DSL, which
  `garun.table` renders to SQL with bound parameters (no SQL-builder library).
- **Deep Java Library 0.36 / PyTorch 2.7.1 CPU** for dense neural networks.

Left out on purpose: the Rust engine, classical estimators
(k-means, SVM, XGBoost, …), conv/RNN/attention/transformer layers, ONNX,
visualization and sessions.

## Build and test

```sh
clojure -T:build javac   # compile the Java leaf (garun.djl.FnBlock) into target/classes
clojure -M:test          # table, nn and metrics suites
```

All dependencies resolve from Maven. The PyTorch native library comes from the
`pytorch-native-cpu` linux-x86_64 jar, so the engine does not download anything
at runtime.

## Namespaces

| Namespace | Surface |
|---|---|
| `garun.table` | Pure plans: `from`, `query`, `->sql`, `table?`. Execution: `open`, `rows`, `columns`, `schema`, `row-count`, `matrix`, `write-csv!`, `write-parquet!` |
| `garun.nn` | `build`, `train!`, `predict`, `save!`, `load`. A `Network` implements `AutoCloseable` |
| `garun.metrics` | `accuracy mse mae r2 precision recall f1 confusion-matrix log-loss`, called as `(metric y-pred y-true)` |

**Table sources:** a column map, a vector of row maps, `[:csv {:path :header? :delimiter}]`,
`[:parquet {:path}]`, `[:json {:path}]`, or `[:duckdb {:query "SELECT …"}]`.
The `:duckdb` query is trusted SQL.

**Clauses:** `:select :rename :where :derive :group-by :order-by :limit :join :sample :split`.
`:split` must be the last clause. It returns a vector of tables that are
deterministic, do not overlap and together cover every input row.

**Expressions:** Garun's operators, aggregates and window functions (`[:over expr {...}]`).

**Layers:** `:sequential :linear :activation :relu :tanh :sigmoid :gelu :identity
:dropout :batchnorm :layernorm :flatten :parallel :skip`.

- `:parallel {:combine :concat|:add}` runs each child on the same input and combines the results.
- `:skip` computes `combine(inner(x), x)`.
- Both are built on `garun.djl.FnBlock`. This is a small Java subclass of DJL's
  `AbstractBlock`. It is needed because `addChildBlock`, `forwardInternal`,
  `initializeChildBlocks` and `getOutputShapes` are protected.

**Training:** losses are `:mse :mae :bce :ce`; `:bce` and `:ce` take logits.
Optimizers are `[:sgd {:lr :momentum}]`, `[:adam {:lr}]` and
`[:adamw {:lr :weight-decay}]`. Runs with the same seed produce the same history.

## Example

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
