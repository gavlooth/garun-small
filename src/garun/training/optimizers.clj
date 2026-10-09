(ns garun.training.optimizers
  "Device-native named optimizer updates without engine-specific optimizer kernels.
  Common controls rescale, symmetrically clip, then add L2 decay; AdamW instead
  decays weights separately. State is reused and owned by each weight's manager.
  SGD/NAG retain learning-rate-scaled velocity. Adam uses DJL's bias-corrected
  rate and uncorrected sqrt(v)+epsilon denominator. RMSProp and Adadelta put
  epsilon inside the square root, Adagrad outside. RMSProp supports momentum
  in either variant (default .9 when centered, zero otherwise). Independent
  supplied gradients are unchanged; attached parameter gradients are cleared
  after an update, matching DJL's trainer contract."
  (:import [ai.djl.training.optimizer Optimizer$OptimizerBuilder]
           [ai.djl.training.tracker ParameterTracker]
           [garun.djl TensorOptimizer TensorOptimizer$Family]))

(set! *warn-on-reflection* true)

(defn build
  "Construct an Optimizer from a validated family/options pair, resolved native
  ParameterTracker (nil for Adadelta), and DJL builder with common flags set.
  Native Optimizers supplied directly by users never pass through this helper."
  [tag opts ^ParameterTracker rate ^Optimizer$OptimizerBuilder builder]
  (let [family (case tag
                 :sgd TensorOptimizer$Family/SGD
                 :nag TensorOptimizer$Family/NAG
                 :adam TensorOptimizer$Family/ADAM
                 :adamw TensorOptimizer$Family/ADAMW
                 :rmsprop TensorOptimizer$Family/RMSPROP
                 :adagrad TensorOptimizer$Family/ADAGRAD
                 :adadelta TensorOptimizer$Family/ADADELTA)
        centered? (get opts :centered? false)
        momentum (get opts :momentum (case tag
                                       :nag 0.9
                                       :rmsprop (if centered? 0.9 0)
                                       0))]
    (TensorOptimizer. builder family rate (float momentum)
                      (float (get opts :beta1 0.9)) (float (get opts :beta2 0.999))
                      (float (get opts :rho 0.9)) (float (get opts :epsilon 1.0e-8))
                      (boolean centered?))))
