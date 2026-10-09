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

## 2026-10-09 — HoneySQL AST cutover and native table transformations

- **Objective/workspace:** In `/home/heefoo/Documents/code/garun-small`, save the existing implementation, then restore HoneySQL to remove bespoke SQL construction. User additionally requested native HoneySQL transformations.
- **Checkpoint:** Git was already initialized and colocated with jj. Reinitialized Git idempotently; preserved existing history and created `pre-honeysql` at `0fe5bcd3` before source changes. Ignored `.clj-kondo/.cache/` and `.lsp/.cache/`; untracked generated cache files accidentally captured during the initial snapshot.
- **Changes:** HoneySQL 2.7.1392 now owns standard SQL formatting and ordered parameters. Plans hold `:query` AST maps; removed `fragment`, `combine-sql` and manual parameter accumulation. Retained four small extensions for literal identifiers, DuckDB file options, struct fields and COPY. JDBC still executes through bound statements, and COPY still uses bound CTAS followed by cleanup.
- **Native transformations:** `[:honeysql clause-map]` composes SELECT-compatible HoneySQL clauses with Garun transformations. The prior table supplies FROM; arbitrary source replacement and DDL/DML keys are rejected. Native expressions retain HoneySQL conventions, including NULL comparisons and explicit trusted `:raw` forms. README documents usage and the boundary.
- **Coverage:** 100% of this cutover and native-transformation scope. Basis: existing public DSL regression cases pass; added executable identifier, scalar-type, NULL, joined-parameter, finite-window and native CASE/DISTINCT/HAVING/composition/error cases.
- **Size:** `table.clj` changed from 709 to 684 lines for the renderer cutover, then to 699 with native HoneySQL transformations. Net reduction is only 10 lines (1.4%); the substantive simplification is deleting custom SQL/parameter composition, not a large total code reduction.
- **Commands:** `git init`; `jj file untrack .clj-kondo/.cache`; `jj describe`; `jj bookmark create pre-honeysql -r @`; `jj new`; `clojure -M:test`; end-to-end throwaway smoke using `clojure -M /tmp/garun-honeysql-smoke.clj`; `cljfmt fix/check`; `clj-kondo --lint src test build.clj deps.edn`; Eastwood 1.4.3 via `clojure -Sdeps` with src/test/target/classes paths; Kibit 0.1.11 via temporary CLI dependencies; reflection reload check.
- **Results:** 40 tests, 426 assertions, zero failures/errors. Formatting clean; clj-kondo zero warnings/errors; Eastwood zero warnings/exceptions; zero reflection warnings. Smoke: 200 generated CSV rows → Garun derive → native HoneySQL projection/filter → keyed split → parquet → residual/parallel BCE network → save/load. Train/test rows 162/38; loss 0.6066505725 → 0.00018357975; test accuracy 1.0; exact parquet and saved-model prediction round-trips.
- **Lint detail:** Corrected preexisting Network return-hint placement and a local `compile` shadow; formatting-only NN/test changes. Kibit initially suggested four changes; applied the relevant nested-assoc → assoc-in simplification. Other suggestions concern unchanged metric/test formulas and are advisory, not defects.
- **Negative memory:** A generated temporary-table name as a string works in CTAS but HoneySQL DROP rejects it and prevents cleanup; use one generated keyword consistently. Targeted tests and Eastwood also need `test` explicitly on the classpath, not merely in scanner paths. HoneySQL ordinary identifier formatting treats `*` specially, so literal Garun names require the small quoted-identifier extension.
- **Unresolved issues:** None in the exercised cutover. Native HoneySQL `:raw` is intentionally trusted, not a SQL security sandbox. No performance improvement was measured or claimed.
- **Recommendation/next action:** Keep the AST cutover as a separately saved milestone. The user's next request is autonomous expansion across DJL capabilities, preserving minimal dependencies and readable code.

Signature: omp/gpt-6.1-sol
