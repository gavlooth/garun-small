package garun.djl;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Block;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.util.List;

/**
 * Executes a validated, topologically ordered graph compiled by garun.blocks.
 *
 * <p>Each name denotes one array. Slots start with graph inputs, followed by node outputs in
 * topological order. Routing borrows array references without copying, attaching, or closing them;
 * ordinary DJL child registration owns initialization, gradients, and parameter persistence.
 */
public final class GraphBlock extends AbstractBlock {
    private final int inputCount;
    private final int[][] nodeInputs;
    private final Block[] nodes;
    private final int[] outputs;
    private DataType[] observedInputTypes;

    public GraphBlock(int inputCount, List<String> names, List<int[]> nodeInputs,
                      List<Block> nodes, int[] outputs) {
        super((byte) 1);
        this.inputCount = inputCount;
        this.nodeInputs = nodeInputs.toArray(new int[nodeInputs.size()][]);
        this.nodes = nodes.toArray(new Block[nodes.size()]);
        this.outputs = outputs;
        for (int node = 0; node < this.nodes.length; node++) {
            addChildBlock(names.get(node), this.nodes[node]);
        }
    }

    /** Compiles the graph-only merge nodes without a Clojure callback or shape-probe tensors. */
    public static Block merge(String mode, int axis) {
        return new MergeBlock("concat".equals(mode), axis);
    }

    @Override
    protected NDList forwardInternal(ParameterStore store, NDList inputs, boolean training,
                                     PairList<String, Object> params) {
        requireInputCount(inputs.size());
        NDArray[] values = new NDArray[inputCount + nodes.length];
        inputs.toArray(values);
        for (int node = 0; node < nodes.length; node++) {
            int[] indices = nodeInputs[node];
            NDList routed = new NDList(indices.length);
            for (int index : indices) {
                routed.add(values[index]);
            }
            values[inputCount + node] = nodes[node].forward(store, routed, training, params)
                    .singletonOrThrow();
        }
        NDList result = new NDList(outputs.length);
        for (int index : outputs) {
            result.add(values[index]);
        }
        observedInputTypes = dataTypes(inputs, observedInputTypes);
        outputDataTypes = dataTypes(result, outputDataTypes);
        return result;
    }

    @Override
    protected void initializeChildBlocks(NDManager manager, DataType type, Shape... shapes) {
        routeShapes(manager, type, shapes, null);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] shapes) {
        return routeShapes(null, null, shapes, observedInputTypes);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] shapes, DataType[] types) {
        if (types != null) {
            requireInputCount(types.length);
            observedInputTypes = types.clone();
        }
        return getOutputShapes(shapes);
    }

    private Shape[] routeShapes(NDManager manager, DataType type, Shape[] shapes,
                                DataType[] inputTypes) {
        requireInputCount(shapes.length);
        Shape[] values = new Shape[inputCount + nodes.length];
        System.arraycopy(shapes, 0, values, 0, inputCount);
        DataType[] types = inputTypes == null ? null : new DataType[values.length];
        if (types != null) {
            System.arraycopy(inputTypes, 0, types, 0, inputCount);
        }
        for (int node = 0; node < nodes.length; node++) {
            int[] indices = nodeInputs[node];
            Shape[] routed = new Shape[indices.length];
            DataType[] routedTypes = types == null ? null : new DataType[indices.length];
            boolean knownTypes = types != null;
            for (int input = 0; input < indices.length; input++) {
                routed[input] = values[indices[input]];
                if (routedTypes != null) {
                    routedTypes[input] = types[indices[input]];
                    knownTypes &= routedTypes[input] != null;
                }
            }
            Block child = nodes[node];
            if (manager != null) {
                child.initialize(manager, type, routed);
            }
            Shape[] childShapes = knownTypes ? child.getOutputShapes(routed, routedTypes)
                    : child.getOutputShapes(routed);
            if (childShapes.length != 1) {
                throw new IllegalArgumentException("A named graph node must output exactly one array");
            }
            values[inputCount + node] = childShapes[0];
            if (types != null) {
                DataType[] childTypes = child.getOutputDataTypes();
                types[inputCount + node] = childTypes != null && childTypes.length == 1
                        ? childTypes[0] : null;
            }
        }
        Shape[] result = new Shape[outputs.length];
        for (int output = 0; output < outputs.length; output++) {
            result[output] = values[outputs[output]];
        }
        return result;
    }

    private void requireInputCount(int count) {
        if (count != inputCount) {
            throw new IllegalArgumentException("Graph requires " + inputCount + " inputs, received " + count);
        }
    }

    private static DataType[] dataTypes(NDList arrays, DataType[] previous) {
        DataType[] types = previous == null ? new DataType[arrays.size()] : previous;
        for (int index = 0; index < types.length; index++) {
            types[index] = arrays.get(index).getDataType();
        }
        return types;
    }

    private static final class MergeBlock extends AbstractBlock {
        private final boolean concat;
        private final int axis;

        private MergeBlock(boolean concat, int axis) {
            super((byte) 1);
            this.concat = concat;
            this.axis = axis;
        }

        @Override
        protected NDList forwardInternal(ParameterStore store, NDList inputs, boolean training,
                                         PairList<String, Object> params) {
            if (concat) {
                return new NDList(NDArrays.concat(inputs, axis));
            }
            NDArray result = inputs.head();
            for (int input = 1; input < inputs.size(); input++) {
                result = result.add(inputs.get(input));
            }
            return new NDList(result);
        }

        @Override
        public Shape[] getOutputShapes(Shape[] shapes) {
            Shape result = shapes[0];
            for (int input = 1; input < shapes.length; input++) {
                result = concat ? FnBlock.concatShape(result, shapes[input], axis)
                        : FnBlock.broadcastShape(result, shapes[input]);
            }
            return new Shape[] {result};
        }
    }
}
