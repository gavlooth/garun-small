package garun.djl;

import ai.djl.Device;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Block;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import clojure.lang.IFn;
import java.util.ArrayList;
import java.util.List;

/**
 * Protected DJL child hooks for Clojure parallel/residual combinations.
 *
 * <p>Children receive the same NDList, without copying or changing its ownership. Named add/concat
 * combinations infer shapes without allocating tensors. Custom combinations may supply a shape
 * function; otherwise their value-independent shape is probed on temporary zero tensors using the
 * initialization manager's engine/device and the observed input/output dtypes when available.
 */
public final class FnBlock extends AbstractBlock {
    private final List<Block> branches;
    private final IFn combine;
    private final String mode;
    private final int axis;
    private final IFn shapeFunction;
    private NDManager initializationManager;
    private String initializationEngine;
    private Device initializationDevice;
    private DataType initializationType = DataType.FLOAT32;
    private DataType[] inputTypes;
    private final DataType[][] branchTypes;

    public FnBlock(String name, List<Block> branches, IFn combine,
                   String mode, int axis, IFn shapeFunction) {
        super((byte) 1);
        this.branches = List.copyOf(branches);
        this.combine = combine;
        this.mode = mode;
        this.axis = axis;
        this.shapeFunction = shapeFunction;
        this.branchTypes = new DataType[branches.size()][];
        for (int i = 0; i < branches.size(); i++) {
            addChildBlock(name + "-" + i, branches.get(i));
        }
    }

    @Override
    protected NDList forwardInternal(ParameterStore store, NDList input, boolean training,
                                     PairList<String, Object> params) {
        List<NDList> outputs = new ArrayList<>(branches.size());
        for (int i = 0; i < branches.size(); i++) {
            NDList output = branches.get(i).forward(store, input, training, params);
            outputs.add(output);
            if (mode == null) {
                branchTypes[i] = dataTypes(output, branchTypes[i]);
            }
        }
        if (mode == null) {
            inputTypes = dataTypes(input, inputTypes);
        }
        NDList result = combined(outputs, input);
        outputDataTypes = dataTypes(result, outputDataTypes);
        return result;
    }

    private NDList combined(List<NDList> outputs, NDList input) {
        Object result = combine.invoke(outputs, input);
        if (!(result instanceof NDList)) {
            throw new IllegalArgumentException("Combination function must return an NDList");
        }
        return (NDList) result;
    }

    private static DataType[] dataTypes(NDList arrays, DataType[] previous) {
        DataType[] types = previous != null && previous.length == arrays.size()
                ? previous : new DataType[arrays.size()];
        for (int i = 0; i < types.length; i++) {
            types[i] = arrays.get(i).getDataType();
        }
        return types;
    }

    @Override
    protected void initializeChildBlocks(NDManager manager, DataType type, Shape... shapes) {
        initializationManager = manager;
        initializationType = type;
        initializationEngine = manager.getEngine().getEngineName();
        initializationDevice = manager.getDevice();
        for (Block branch : branches) {
            branch.initialize(manager, type, shapes);
        }
    }

    @Override
    public Shape[] getOutputShapes(Shape[] shapes, DataType[] types) {
        // SequentialBlock passes null when it has no dtype metadata. That is
        // absence of new information, not a request to discard observed dtypes.
        if (types != null) {
            if (shapes.length != types.length) {
                throw new IllegalArgumentException("Input shapes and dtypes must have equal counts");
            }
            inputTypes = types.clone();
        }
        return getOutputShapes(shapes);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] shapes) {
        List<Shape[]> childShapes = new ArrayList<>(branches.size());
        for (Block branch : branches) {
            childShapes.add(inputTypes == null ? branch.getOutputShapes(shapes)
                    : branch.getOutputShapes(shapes, inputTypes));
        }
        if (shapeFunction != null) {
            return (Shape[]) shapeFunction.invoke(childShapes, shapes);
        }
        if (mode != null) {
            return namedOutputShapes(childShapes);
        }
        // Never close child outputs or caller inputs: only these synthetic probe tensors belong
        // to the short-lived manager. No real child forward or parameters are touched by probing.
        try (NDManager manager = newShapeManager()) {
            NDList input = zerosForShapes(manager, shapes, inputTypes);
            List<NDList> outputs = new ArrayList<>(branches.size());
            for (int i = 0; i < branches.size(); i++) {
                DataType[] types = branchTypes[i] == null
                        ? branches.get(i).getOutputDataTypes() : branchTypes[i];
                outputs.add(zerosForShapes(manager, childShapes.get(i), types));
            }
            NDList result = combined(outputs, input);
            outputDataTypes = dataTypes(result, outputDataTypes);
            return result.getShapes();
        }
    }

    private NDManager newShapeManager() {
        if (initializationManager != null && initializationManager.isOpen()) {
            return initializationManager.newSubManager();
        }
        if (initializationEngine != null) {
            return NDManager.newBaseManager(initializationDevice, initializationEngine);
        }
        return NDManager.newBaseManager();
    }

    private Shape[] namedOutputShapes(List<Shape[]> children) {
        int count = children.get(0).length;
        Shape[] result = children.get(0).clone();
        for (int branch = 1; branch < children.size(); branch++) {
            Shape[] next = children.get(branch);
            if (next.length != count) {
                throw new IllegalArgumentException("Combination branches must have equal output counts");
            }
            for (int output = 0; output < count; output++) {
                result[output] = "concat".equals(mode)
                        ? concatShape(result[output], next[output])
                        : broadcastShape(result[output], next[output]);
            }
        }
        return result;
    }

    private Shape concatShape(Shape left, Shape right) {
        long[] dims = left.getShape().clone();
        long[] other = right.getShape();
        int dimension = axis < 0 ? axis + dims.length : axis;
        if (dims.length != other.length || dimension < 0 || dimension >= dims.length) {
            throw new IllegalArgumentException("Invalid concat rank or axis");
        }
        for (int i = 0; i < dims.length; i++) {
            if (i != dimension && dims[i] != other[i]) {
                throw new IllegalArgumentException("Concat dimensions must match outside its axis");
            }
        }
        dims[dimension] += other[dimension];
        return new Shape(dims);
    }

    private static Shape broadcastShape(Shape left, Shape right) {
        long[] a = left.getShape();
        long[] b = right.getShape();
        long[] result = new long[Math.max(a.length, b.length)];
        for (int i = 1; i <= result.length; i++) {
            long x = i <= a.length ? a[a.length - i] : 1;
            long y = i <= b.length ? b[b.length - i] : 1;
            if (x != y && x != 1 && y != 1) {
                throw new IllegalArgumentException("Add dimensions must be broadcast-compatible");
            }
            result[result.length - i] = x == 1 ? y : x;
        }
        return new Shape(result);
    }

    private NDList zerosForShapes(NDManager manager, Shape[] shapes, DataType[] types) {
        NDList arrays = new NDList();
        for (int i = 0; i < shapes.length; i++) {
            DataType type = types != null && i < types.length && types[i] != null
                    ? types[i] : initializationType;
            arrays.add(manager.zeros(shapes[i], type));
        }
        return arrays;
    }
}
