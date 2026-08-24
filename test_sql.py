import sqlite3

conn = sqlite3.connect("crm_data.db")
cursor = conn.cursor()

cursor.execute("""
SELECT COUNT(DISTINCT ru.Entity_Type_ID) AS accountant_bench_count
FROM relay_user ru
JOIN entity_placement ep
    ON ru.Entity_Type_ID = ep.Entity_Type_ID
WHERE (ru.[Job Title]) = 'accountant'
  AND LOWER(TRIM(ep.PATHWAY)) IN ('bench');
""")

for row in cursor.fetchall():
    print(row)

conn.close()