package garun.djl;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.transformer.TransformerEncoderBlock;
import java.util.function.Function;

/** DJL's encoder forwards an optional attention mask, but its other children take one tensor. */
public final class MaskedTransformerEncoder extends TransformerEncoderBlock {
    public MaskedTransformerEncoder(int embeddingSize, int heads, int hiddenSize, float dropout,
                                    Function<NDList, NDList> activation) {
        super(embeddingSize, heads, hiddenSize, dropout, activation);
    }

    @Override
    public void initializeChildBlocks(NDManager manager, DataType type, Shape... shapes) {
        // Only the embedding determines parameter shapes, including attention projections.
        // The inherited forward still receives the complete NDList, including the mask.
        super.initializeChildBlocks(manager, type, shapes[0]);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] shapes) {
        return new Shape[] {shapes[0]};
    }
}
