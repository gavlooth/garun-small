(ns build
  "Compile the remaining Java DJL bridges: clojure -T:build javac."
  (:require [clojure.tools.build.api :as b]))

(set! *warn-on-reflection* true)

(defn javac
  "Compile java/ into target/classes using the project's dependency basis.
  Takes the tools.build task options map (unused); throws on compiler failure."
  [_]
  (b/javac {:src-dirs ["java"]
            :class-dir "target/classes"
            :basis (b/create-basis {:project "deps.edn"})
            :javac-opts ["--release" "17" "-Xlint:all"]}))
