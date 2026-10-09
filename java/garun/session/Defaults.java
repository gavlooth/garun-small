package garun.session;

import ai.djl.Device;
import ai.djl.ndarray.types.DataType;

/** Immutable native defaults shared by the session and its workers. */
public final class Defaults {
    public final Device device;
    public final DataType dataType;
    public final String engine;
    public final long seed;
    public final String database;

    public Defaults(Device device, DataType dataType, String engine, long seed, String database) {
        this.device = device;
        this.dataType = dataType;
        this.engine = engine;
        this.seed = seed;
        this.database = database;
    }
}
