package garun.djl;

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
 * Bridges Clojure combination functions to DJL's protected {@link AbstractBlock} hooks.
 *
 * <p>DJL needs an AbstractBlock subclass to forward, initialize, and infer the shapes of child
 * blocks; those protected hooks cannot be implemented by a Clojure function alone. Each child
 * receives the same input. The combine function receives two arguments: a {@code List<NDList>}
 * of child outputs in declaration order, and the original input {@code NDList}. It must return
 * an {@code NDList} and use differentiable NDArray operations so gradients reach the children.
 */
public final class FnBlock extends AbstractBlock {
    private final List<Block> branches;
    private final IFn combine;

    /**
     * Registers the branches as DJL children so their parameters are trained and saved.
     *
     * @param name prefix for the registered child names
     * @param branches child blocks in combination order
     * @param combine function of (child output lists, original input list) returning an NDList
     */
    public FnBlock(String name, List<Block> branches, IFn combine) {
        super((byte) 1);
        this.branches = List.copyOf(branches);
        this.combine = combine;
        for (int i = 0; i < branches.size(); i++) {
            addChildBlock(name + "-" + i, branches.get(i));
        }
    }

    @Override
    protected NDList forwardInternal(ParameterStore store, NDList input, boolean training,
                                     PairList<String, Object> params) {
        List<NDList> outputs = new ArrayList<>(branches.size());
        for (Block branch : branches) {
            outputs.add(branch.forward(store, input, training, params));
        }
        return (NDList) combine.invoke(outputs, input);
    }

    @Override
    protected void initializeChildBlocks(NDManager manager, DataType type, Shape... shapes) {
        for (Block branch : branches) {
            branch.initialize(manager, type, shapes);
        }
    }

    @Override
    public Shape[] getOutputShapes(Shape[] shapes) {
        try (NDManager manager = NDManager.newBaseManager()) {
            // Run the same combination on zero tensors to infer its output shapes.
            NDList input = zerosForShapes(manager, shapes);
            List<NDList> outputs = new ArrayList<>(branches.size());
            for (Block branch : branches) {
                outputs.add(zerosForShapes(manager, branch.getOutputShapes(shapes)));
            }
            return ((NDList) combine.invoke(outputs, input)).getShapes();
        }
    }

    private static NDList zerosForShapes(NDManager manager, Shape[] shapes) {
        NDList arrays = new NDList();
        for (Shape shape : shapes) {
            arrays.add(manager.zeros(shape));
        }
        return arrays;
    }
}
