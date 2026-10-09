# Session reports

## 2026-10-09 — garun-small: JVM Clojure Garun subset on DuckDB + DJL

- **Objective:** Build a small Garun DSL in traditional JVM Clojure, limited to what DuckDB and Deep Java Library can back.
- **Workspace:** /home/heefoo/Documents/code/garun-small.
- **Changes:**
  - `deps.edn`: Clojure 1.12.4, HoneySQL 2.7.1392, duckdb_jdbc 1.5.5.1, DJL api/pytorch-engine 0.36.0, pytorch-jni 2.7.1-0.36.0 and pytorch-native-cpu linux-x86_64 2.7.1. Aliases `:test` and `:build`.
  - `src/garun/table.clj`: pure plans built with Garun's clause and expression grammar. They are lowered to quoted HoneySQL and run through JDBC.
  - `src/garun/nn.clj` with `java/garun/djl/FnBlock.java`: Hiccup specs become DJL blocks. Also deterministic manual training, predict, and save/load.
  - `src/garun/metrics.clj`: pure metric functions.
  - `build.clj`: tools.build javac.
  - Tests under `test/garun/`, plus `README.md`.
- **Coverage:** about 100% of the planned subset. Basis: every contract function exists, has tests, and passes the integrated suite. Scoped down on purpose.
- **Commands:** `clojure -T:build javac`; `clojure -M:test`; an end-to-end smoke script (CSV → derive/where → split → skip+parallel BCE net → accuracy/f1 → save/load).
- **Results:**
  - Test suite: 26 tests, 321 assertions, 0 failures and 0 errors.
  - Smoke: test accuracy 1.0. Loss fell from 0.787 to 0.0013. Predictions after reload matched exactly.
  - Code review (code-review agent): no findings.
- **Negative memory:**
  - DJL's BCE boolean option is `fromSigmoid`, not `fromLogit`. Pass `false` when the inputs are logits.
  - DuckDB has no single "add or replace column" projection for an unknown schema. `:derive` handles that case with `struct_update`/`unnest`.
  - COPY output is rendered inline, because DuckDB rejects prepared parameters in COPY.
  - The Java leaf is required. AbstractBlock's protected child and forward hooks can't be cleanly reached from proxy.
- **Excluded:** classical estimators, conv/RNN/attention/transformer, ONNX, viz, sessions and the Rust engine.
- **Next actions:** add layers or table features only on request, for example a `:graph` form built on FnBlock or embeddings for integer inputs.
- **Dependencies:** run `clojure -T:build javac` before tests or the REPL so that `target/classes` exists.

Signature: omp/claude-opus-5-5

## 2026-10-09 — garun-small: HoneySQL removal and readability pass

- **Objective:** Remove HoneySQL and make the code readable by humans, following the Code Style rules in `../parallax-relay/AGENTS.md` and `.agents/READABILITY_AUDIT_PLAYBOOK.md` §3/§7 plus `~/.agents/CLOJURE.md`. The rules were condensed into a brief for the subagents.
- **Why HoneySQL went:** it only formatted the outer query (`->sql`) and inlined literals for COPY; expressions were already 50 hand-written `:raw` fragments. That mixed two SQL conventions. `CLOJURE.md` exempts Garun's own DSL from the HoneySQL rule.
- **Changes:**
  - `table.clj`: one `{:sql :params}` renderer, combined in SQL-text order. Expression and clause dispatch moved to multimethods and operator tables. `join-plan` and `split-plan` are split into named steps. Namespace docstring explains the plan model. `deps.edn` no longer lists HoneySQL.
  - COPY now writes the plan into a UUID-named temporary table (`CREATE TEMP TABLE … AS`, bound params), COPYs that, and drops it in `finally`. This replaced a parameter-span/inline-literal mechanism. Verified first: DuckDB keeps the exact BIGINT, DOUBLE and string values through bound params in CTAS.
  - `nn.clj`: spec→block uses a compiler table; `train!` is split into named steps; float-array conversions are named functions. FnBlock has Javadoc explaining why it exists.
  - `metrics.clj`: formulas read directly; tests are case tables.
- **Commands:** `clojure -T:build javac && clojure -M:test`; a reflection check with `*warn-on-reflection*`; the end-to-end smoke (CSV → derive/where → split → write-parquet → skip/parallel BCE net).
- **Results:** 34 tests, 392 assertions, 0 failures, 0 errors, 0 reflection warnings. Smoke matches the first session exactly: loss 0.787 → 0.0013, test accuracy 1.0, 66 parquet rows.
- **Negative memory:** a Java `MatrixOps` helper for `table/matrix` was proposed and rejected. The Clojure version was readable, there was no measured hotspot, and the table namespace would have needed javac. FnBlock is the only Java.
- **Next actions:** none pending. Possible later work: a `:graph` form on FnBlock, embeddings.

Signature: omp/claude-opus-5-5
