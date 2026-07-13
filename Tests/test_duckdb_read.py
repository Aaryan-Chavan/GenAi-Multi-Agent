import duckdb

conn = duckdb.connect("Database/Duckdb/analytics.duckdb")

# show tables
print(conn.execute("SHOW TABLES").fetchall())

# preview table
df = conn.execute("SELECT * FROM structured_data LIMIT 10").fetchdf()
print(df)

conn.close()