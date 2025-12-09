import lancedb
import pyarrow as pa

data = pa.table({"a": [2, 1, 3], "b": ["a", "b", "c"]})
db = lancedb.connect("./.lancedbignore")

db.drop_all_tables()

table = db.create_table("my_table", data)
new_data = pa.table({"a": [2, 3, 4], "b": ["x", "y", "z"]})

# Perform a "upsert" operation
res = (
    table.merge_insert("a")
    .when_matched_update_all()
    .when_not_matched_insert_all()
    .execute(new_data)
)

print(res.num_updated_rows)

# The order of new rows is non-deterministic since we use
# a hash-join as part of this operation and so we sort here
print(table.to_arrow().sort_by("a").to_pandas())
