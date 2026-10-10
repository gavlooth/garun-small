(ns garun.test-runner
  "Run every Garun suite; failures set the process exit code."
  (:require [clojure.test :as test]
            [garun.session-test]
            [garun.table-test]
            [garun.nn-test]
            [garun.metrics-test]
            [garun.blocks-test]
            [garun.tensor-test]
            [garun.ndarray-test]
            [garun.training-test]
            [garun.model-test]
            [garun.losses-test]
            [garun.optimizers-test]))

(set! *warn-on-reflection* true)

(defn -main
  "Run every Garun test namespace and exit nonzero on test failures or errors."
  [& _]
  (let [{:keys [fail error]} (test/run-tests 'garun.session-test
                                             'garun.table-test
                                             'garun.nn-test
                                             'garun.metrics-test
                                             'garun.blocks-test
                                             'garun.tensor-test
                                             'garun.ndarray-test
                                             'garun.training-test
                                             'garun.model-test
                                             'garun.losses-test
                                             'garun.optimizers-test)
        failed? (pos? (+ fail error))]
    (System/exit (if failed? 1 0))))
