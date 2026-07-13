import duckdb

db_path = r"F:/PROJECT/Database/Duckdb/analytics.duckdb"

print("Opening:", db_path)

conn = duckdb.connect(db_path)

print("Connected!")

conn.close()