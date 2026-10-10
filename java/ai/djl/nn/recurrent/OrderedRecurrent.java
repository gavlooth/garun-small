package ai.djl.nn.recurrent;

import ai.djl.nn.Parameter;
import java.util.LinkedHashMap;

/**
 * DJL 0.36 inserts recurrent parameters in type/direction order, but PyTorch
 * consumes direction/type order. With biases, bidirectional RNN/GRU/LSTM pass
 * reverse weights where forward biases belong. Reorder the existing parameters
 * once; native forwarding, gradients and parameter serialization stay unchanged.
 *
 * This bridge lives in DJL's package because all three native constructors are
 * package-private. It is deliberately pinned to DJL 0.36's parameter names and
 * constructor contracts, not a replacement recurrent implementation.
 */
public final class OrderedRecurrent {
    private static final String[] DIRECTIONS = {"l", "r"};
    private static final String[] SUFFIXES = {
        "i2h_WEIGHT", "h2h_WEIGHT", "i2h_BIAS", "h2h_BIAS"
    };

    private OrderedRecurrent() {}

    public static RNN rnn(RNN.Builder builder) {
        return new RNN(builder) {
            {
                order(parameters, numLayers, bidirectional, hasBiases);
            }
        };
    }

    public static GRU gru(GRU.Builder builder) {
        return new GRU(builder) {
            {
                order(parameters, numLayers, bidirectional, hasBiases);
            }
        };
    }

    public static LSTM lstm(LSTM.Builder builder) {
        return new LSTM(builder) {
            {
                order(parameters, numLayers, bidirectional, hasBiases);
            }
        };
    }

    private static void order(LinkedHashMap<String, Parameter> parameters,
                              int layers, boolean bidirectional, boolean biases) {
        if (!bidirectional || !biases) {
            return;
        }
        for (int layer = 0; layer < layers; ++layer) {
            for (String direction : DIRECTIONS) {
                String prefix = direction + '_' + layer + '_';
                for (String suffix : SUFFIXES) {
                    String name = prefix + suffix;
                    Parameter parameter = parameters.remove(name);
                    parameters.put(name, parameter);
                }
            }
        }
    }
}
