(ns garun.test-runner
  "Run the table, neural-network and metric suites; failures set the process exit code."
  (:require [clojure.test :as test]
            [garun.table-test]
            [garun.nn-test]
            [garun.metrics-test]))

(set! *warn-on-reflection* true)

(defn -main
  "Run every Garun test namespace and exit nonzero on test failures or errors."
  [& _]
  (let [{:keys [fail error]} (test/run-tests 'garun.table-test
                                             'garun.nn-test
                                             'garun.metrics-test)
        failed? (pos? (+ fail error))]
    (System/exit (if failed? 1 0))))
