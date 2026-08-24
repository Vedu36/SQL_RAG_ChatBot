import sqlite3

conn = sqlite3.connect("crm_data.db")
cursor = conn.cursor()

cursor.execute("""
SELECT e.DEPARTMENT, COUNT(e.PLACEMENT_NAME) AS placement_count
FROM entity_placement e
GROUP BY e.DEPARTMENT;
""")

for row in cursor.fetchall():
    print(row)

conn.close()