package garun.session;

import clojure.lang.Keyword;
import clojure.lang.Util;
import java.sql.Connection;
import java.sql.Statement;

/** Immutable captured metadata for a table owned by a session's generated schema. */
public final class MaterializedTable implements AutoCloseable {
    public final Session session;
    public final String schema;
    public final String table;
    public final Object columns;
    public final long rows;

    public MaterializedTable(Session session, String schema, String table, Object columns, long rows) {
        if (!session.schema().equals(schema) || !isGeneratedTableName(table)) {
            throw Session.error("invalid-table-name", "Materialized tables require session-generated identifiers",
                    Keyword.intern(null, "schema"), schema,
                    Keyword.intern(null, "table"), table);
        }
        this.session = session;
        this.schema = schema;
        this.table = table;
        this.columns = columns;
        this.rows = rows;
    }

    private static boolean isGeneratedTableName(String table) {
        if (table == null || !table.startsWith("t_") || table.length() == 2) {
            return false;
        }
        for (int index = 2; index < table.length(); index++) {
            char digit = table.charAt(index);
            if (digit < '0' || digit > '9') {
                return false;
            }
        }
        return true;
    }

    /** Drop on the releasing or replacing context's own connection. */
    public void drop(Connection connection) {
        // Both identifiers were checked against the session's generated names at construction.
        try (Statement statement = connection.createStatement()) {
            statement.execute("DROP TABLE IF EXISTS \"" + schema + "\".\"" + table + "\"");
        } catch (Throwable failure) {
            throw Util.sneakyThrow(failure);
        }
    }

    /** Session shutdown only: its lifecycle write lock excludes every context operation. */
    @Override
    public void close() {
        drop(session.connection());
    }
}
