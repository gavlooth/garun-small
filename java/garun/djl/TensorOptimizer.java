package garun.djl;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.NDScope;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.ParameterTracker;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Dense, device-native updates for DJL optimizer families whose PyTorch kernels are incomplete.
 * The native builder supplies common controls and DJL's update counters. Optimizer state belongs
 * to each parameter's manager, never the system manager. Only weights and their attached gradients
 * are changed: independent gradient arguments remain caller-owned and unchanged.
 */
public final class TensorOptimizer extends Optimizer {
    public enum Family { SGD, NAG, ADAM, ADAMW, RMSPROP, ADAGRAD, ADADELTA }

    private final Family family;
    private final ParameterTracker rate;
    private final float momentum;
    private final float beta1;
    private final float beta2;
    private final float rho;
    private final float epsilon;
    private final boolean centered;
    private final int stateSize;
    private final Map<String, Map<Device, NDArray[]>> states = new ConcurrentHashMap<>();

    public TensorOptimizer(OptimizerBuilder<?> builder, Family family, ParameterTracker rate,
                           float momentum, float beta1, float beta2, float rho,
                           float epsilon, boolean centered) {
        super(builder);
        this.family = family;
        this.rate = rate;
        this.momentum = momentum;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.rho = rho;
        this.epsilon = epsilon;
        this.centered = centered;
        switch (family) {
            case SGD:
            case NAG:
                stateSize = momentum == 0 ? 0 : 1;
                break;
            case ADAM:
            case ADAMW:
            case ADADELTA:
                stateSize = 2;
                break;
            case RMSPROP:
                stateSize = centered ? 3 : (momentum == 0 ? 1 : 2);
                break;
            case ADAGRAD:
                stateSize = 1;
                break;
            default:
                throw new IllegalArgumentException("Unknown optimizer family: " + family);
        }
    }

    private NDArray[] state(String id, NDArray weight) {
        if (stateSize == 0) {
            return null;
        }
        Map<Device, NDArray[]> devices = states.get(id);
        if (devices == null) {
            devices = new ConcurrentHashMap<>();
            Map<Device, NDArray[]> existing = states.putIfAbsent(id, devices);
            if (existing != null) {
                devices = existing;
            }
        }
        Device device = weight.getDevice();
        NDManager owner = weight.getManager();
        NDArray[] arrays = devices.get(device);
        if (arrays != null && arrays[0].getManager() == owner) {
            return arrays;
        }
        // Reusing an optimizer with a different model must not retain the former model's state.
        if (arrays != null) {
            for (NDArray array : arrays) {
                array.close();
            }
        }
        arrays = new NDArray[stateSize];
        for (int i = 0; i < stateSize; i++) {
            arrays[i] = owner.zeros(weight.getShape(), weight.getDataType(), device);
            NDScope.unregister(arrays[i]);
        }
        devices.put(device, arrays);
        return arrays;
    }

    private NDArray prepareGradient(NDArray weight, NDArray gradient) {
        NDArray prepared = gradient;
        try {
            if (rescaleGrad != 1) {
                prepared = gradient.mul(rescaleGrad);
            }
            if (clipGrad >= 0) {
                NDArray clipped = prepared.clip(-clipGrad, clipGrad);
                if (prepared != gradient) {
                    prepared.close();
                }
                prepared = clipped;
            }
            if (family != Family.ADAMW && getWeightDecay() != 0) {
                try (NDArray decay = weight.mul(getWeightDecay())) {
                    if (prepared == gradient) {
                        prepared = gradient.add(decay);
                    } else {
                        prepared.addi(decay);
                    }
                }
            }
            return prepared;
        } catch (RuntimeException error) {
            if (prepared != gradient) {
                prepared.close();
            }
            throw error;
        }
    }

    @Override
    public void update(String parameterId, NDArray weight, NDArray gradient) {
        int update = updateCount(parameterId);
        float learningRate = family == Family.ADADELTA ? 1 : rate.getNewValue(parameterId, update);
        if (!Float.isFinite(learningRate) || !Float.isFinite(getWeightDecay())) {
            throw new IllegalArgumentException("learning rate or weight decay is nan or infinite");
        }
        NDArray[] history = state(parameterId, weight);
        // Detached views share storage, but do not record updates in an active autograd graph.
        // The original parameter retains its requires-gradient flag for the next forward pass.
        try (NDArray value = weight.stopGradient(); NDArray input = gradient.stopGradient()) {
            NDArray grad = prepareGradient(value, input);
            try {
                switch (family) {
                    case SGD:
                    case NAG:
                        updateMomentum(value, grad, history, learningRate);
                        break;
                    case ADAM:
                    case ADAMW:
                        updateAdam(value, grad, history, learningRate, update);
                        break;
                    case RMSPROP:
                        updateRmsProp(value, grad, history, learningRate);
                        break;
                    case ADAGRAD:
                        updateAdagrad(value, grad, history, learningRate);
                        break;
                    case ADADELTA:
                        updateAdadelta(value, grad, history);
                        break;
                    default:
                        throw new IllegalArgumentException("Unknown optimizer family: " + family);
                }
            } finally {
                if (grad != input) {
                    grad.close();
                }
            }
        }
        // DJL/PyTorch accumulates backward results; native optimizers also clear parameter grads.
        if (weight.hasGradient()) {
            try (NDArray attachedGradient = weight.getGradient()) {
                attachedGradient.muli(0);
            }
        }
    }

    private void updateMomentum(NDArray weight, NDArray grad, NDArray[] history, float lr) {
        try (NDArray step = grad.mul(lr)) {
            if (momentum == 0) {
                weight.subi(step);
                return;
            }
            NDArray velocity = history[0];
            velocity.muli(momentum).addi(step);
            if (family == Family.NAG) {
                // v = momentum*v + lr*g; w -= momentum*v + lr*g (DJL's NAG convention).
                try (NDArray lookahead = velocity.mul(momentum)) {
                    weight.subi(lookahead.addi(step));
                }
            } else {
                weight.subi(velocity);
            }
        }
    }

    private static void average(NDArray history, NDArray observation, float decay) {
        try (NDArray contribution = observation.mul(1 - decay)) {
            history.muli(decay).addi(contribution);
        }
    }

    private void updateAdam(NDArray weight, NDArray grad, NDArray[] history, float lr, int t) {
        NDArray mean = history[0];
        NDArray variance = history[1];
        average(mean, grad, beta1);
        try (NDArray squared = grad.square()) {
            variance.muli(beta2).addi(squared.muli(1 - beta2));
        }
        if (family == Family.ADAMW && getWeightDecay() != 0) {
            weight.muli(1 - lr * getWeightDecay());
        }
        double correctedRate = lr * Math.sqrt(1 - Math.pow(beta2, t)) / (1 - Math.pow(beta1, t));
        // DJL applies bias correction to the rate, leaving epsilon beside uncorrected sqrt(v).
        try (NDArray denominator = variance.sqrt(); NDArray step = mean.mul(correctedRate)) {
            weight.subi(step.divi(denominator.addi(epsilon)));
        }
    }

    private void updateRmsProp(NDArray weight, NDArray grad, NDArray[] history, float lr) {
        NDArray variance = history[0];
        try (NDArray squared = grad.square()) {
            variance.muli(rho).addi(squared.muli(1 - rho));
        }
        if (centered) {
            NDArray mean = history[1];
            average(mean, grad, rho);
            try (NDArray meanSquared = mean.square();
                 NDArray denominator = variance.sub(meanSquared)) {
                applyRmsStep(weight, grad, history[2], denominator.addi(epsilon).powi(0.5), lr);
            }
        } else {
            try (NDArray denominator = variance.add(epsilon)) {
                applyRmsStep(weight, grad, momentum == 0 ? null : history[1], denominator.powi(0.5), lr);
            }
        }
    }

    private void applyRmsStep(NDArray weight, NDArray grad, NDArray velocity, NDArray denominator, float lr) {
        try (NDArray step = grad.mul(lr)) {
            step.divi(denominator);
            if (velocity == null || momentum == 0) {
                weight.subi(step);
            } else {
                weight.subi(velocity.muli(momentum).addi(step));
            }
        }
    }

    private void updateAdagrad(NDArray weight, NDArray grad, NDArray[] history, float lr) {
        NDArray sumSquares = history[0];
        try (NDArray squared = grad.square()) {
            sumSquares.addi(squared);
        }
        try (NDArray denominator = sumSquares.sqrt(); NDArray step = grad.mul(lr)) {
            weight.subi(step.divi(denominator.addi(epsilon)));
        }
    }

    private void updateAdadelta(NDArray weight, NDArray grad, NDArray[] history) {
        NDArray meanSquareGradient = history[0];
        NDArray meanSquareDelta = history[1];
        try (NDArray squared = grad.square()) {
            meanSquareGradient.muli(rho).addi(squared.muli(1 - rho));
        }
        try (NDArray denominator = meanSquareGradient.add(epsilon);
             NDArray step = meanSquareDelta.add(epsilon)) {
            step.divi(denominator).powi(0.5).muli(grad);
            try (NDArray squaredStep = step.square()) {
                meanSquareDelta.muli(rho).addi(squaredStep.muli(1 - rho));
            }
            weight.subi(step);
        }
    }
}
