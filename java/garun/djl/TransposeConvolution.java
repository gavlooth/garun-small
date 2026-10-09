package garun.djl;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.LayoutType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.convolutional.Deconvolution;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.util.Arrays;

/**
 * Transposed convolution using differentiable tensor views/padding and native convolution.
 * DJL 0.36's PyTorch JNI has convolution, but no transposed-convolution entry point.
 * Parameters retain DJL's [outputChannels, inputChannels/groups, spatial...] layout.
 * Each input value contributes weight[k] at inputIndex*stride - padding + k*dilation.
 */
public final class TransposeConvolution extends Deconvolution {
    private final int rank;
    private final LayoutType[] layout;
    private final int[] spatialAxes;
    private final Shape unitStride;
    private final Shape fullPadding;
    private final Shape rightPadding;
    private final Shape[] insertionPadding;
    private final boolean hasOutputPadding;
    private final boolean hasCropping;

    public TransposeConvolution(DeconvolutionBuilder<?> builder) {
        super(builder);
        rank = kernelShape.dimension();
        if (rank != 1 && rank != 2) {
            throw new IllegalArgumentException("Transposed convolution requires 1 or 2 spatial axes");
        }
        layout = LayoutType.fromValue(rank == 1 ? "NCW" : "NCHW");
        spatialAxes = new int[rank];
        insertionPadding = new Shape[rank];
        long[] ones = new long[rank];
        long[] full = new long[rank];
        long[] right = new long[2 * rank];
        boolean extra = false;
        boolean crop = false;
        for (int i = 0; i < rank; i++) {
            if (outPadding.get(i) >= stride.get(i) && outPadding.get(i) >= dilation.get(i)) {
                throw new IllegalArgumentException("Output padding must be less than stride or dilation");
            }
            spatialAxes[i] = i + 2;
            ones[i] = 1;
            full[i] = dilation.get(i) * (kernelShape.get(i) - 1);
            right[2 * (rank - i - 1) + 1] = outPadding.get(i);
            long[] insertion = new long[2 * (rank - i)];
            insertion[insertion.length - 1] = stride.get(i) - 1;
            insertionPadding[i] = new Shape(insertion);
            extra |= outPadding.get(i) != 0;
            crop |= padding.get(i) != 0;
        }
        unitStride = new Shape(ones);
        fullPadding = new Shape(full);
        rightPadding = new Shape(right);
        hasOutputPadding = extra;
        hasCropping = crop;
    }

    @Override
    protected LayoutType[] getExpectedLayout() {
        return layout;
    }

    @Override
    protected String getStringLayout() {
        return rank == 1 ? "NCW" : "NCHW";
    }

    @Override
    protected int numDimensions() {
        return rank + 2;
    }

    @Override
    protected void prepare(Shape[] shapes) {
        validateInput(shapes);
        super.prepare(shapes);
    }

    private void validateInput(Shape[] shapes) {
        if (shapes.length != 1 || shapes[0].dimension() != rank + 2) {
            throw new IllegalArgumentException("Transposed convolution requires one channels-first tensor");
        }
        if (shapes[0].get(1) % groups != 0 || filters % groups != 0) {
            throw new IllegalArgumentException("Input channels and filters must be divisible by groups");
        }
    }

    @Override
    public Shape[] getOutputShapes(Shape[] shapes) {
        validateInput(shapes);
        Shape[] result = super.getOutputShapes(shapes);
        for (int i = 2; i < rank + 2; i++) {
            if (result[0].get(i) <= 0) {
                throw new IllegalArgumentException("Transposed convolution output dimensions must be positive");
            }
        }
        return result;
    }

    @Override
    protected NDList forwardInternal(ParameterStore store, NDList inputs, boolean training,
                                     PairList<String, Object> params) {
        NDArray input = inputs.singletonOrThrow();
        Shape outputShape = getOutputShapes(inputs.getShapes())[0];
        NDArray expanded = input;
        for (int i = 0; i < rank; i++) {
            long step = stride.get(i);
            if (step == 1) {
                continue;
            }
            int axis = spatialAxes[i];
            long[] dimensions = Arrays.copyOf(expanded.getShape().getShape(), rank + 2);
            long length = dimensions[axis];
            dimensions[axis] *= step;
            expanded = expanded.expandDims(axis + 1).pad(insertionPadding[i], 0)
                    .reshape(dimensions)
                    .get(NDIndex.sliceAxis(axis, 0, (length - 1) * step + 1));
        }
        if (hasOutputPadding) {
            expanded = expanded.pad(rightPadding, 0);
        }
        NDArray kernel = store.getValue(weight, input.getDevice(), training).flip(spatialAxes);
        NDArray offset = includeBias ? store.getValue(bias, input.getDevice(), training) : null;
        NDList convolved = expanded.getNDArrayInternal().convolution(
                expanded, kernel, offset, unitStride, fullPadding, dilation, groups);
        if (!hasCropping) {
            return convolved;
        }
        NDIndex crop = new NDIndex().addAllDim(2);
        for (int i = 0; i < rank; i++) {
            crop.addSliceDim(padding.get(i), padding.get(i) + outputShape.get(i + 2));
        }
        return new NDList(convolved.singletonOrThrow().get(crop));
    }
}
