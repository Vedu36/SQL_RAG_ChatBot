import sqlite3
import csv
import json
import os
from collections import defaultdict

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
DATA_DIR = "Data - Customer Wiki"
DB_PATH = "crm_data.db"

# ------------------------------------------------------------
# STEP 1: Read all CSVs and generate metadata
# ------------------------------------------------------------
def infer_metadata_from_csvs(data_folder):
    """Scan all CSVs, infer PKs, types, and related tables."""
    table_meta = []
    column_meta = []
    all_tables = set()

    csv_files = [f for f in os.listdir(data_folder) if f.lower().endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_folder}")

    print(f"📁 Found {len(csv_files)} CSV files. Reading headers...")

    table_info = {}

    for filename in csv_files:
        table_name = filename[:-4]
        all_tables.add(table_name)
        filepath = os.path.join(data_folder, filename)

        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            if not headers:
                print(f"⚠️ {filename} is empty or has no headers – skipping.")
                continue

            sample_rows = []
            row_count = 0
            for i, row in enumerate(reader):
                row_count += 1
                if i < 2:
                    sample_rows.append(row)

        print(f"  📄 {filename}: {row_count} rows, {len(headers)} columns")

        unique_counts = {col: set() for col in headers}
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                for col in headers:
                    val = row[col]
                    if val:
                        unique_counts[col].add(val)

        pk_candidates = [col for col in headers if len(unique_counts[col]) == row_count]

        table_info[table_name] = {
            "row_count": row_count,
            "columns": headers,
            "pk_candidates": pk_candidates,
            "sample_rows": sample_rows,
            "unique_counts": unique_counts,
        }

    related_map = defaultdict(set)
    for table_name, info in table_info.items():
        for col in info["columns"]:
            if col.lower().endswith('_id'):
                prefix = col[:-3]
                for other in all_tables:
                    if other.lower().startswith(prefix.lower()) and other != table_name:
                        related_map[table_name].add(other)

    for table_name, info in table_info.items():
        entry = {
            "table_name": table_name,
            "description": "",
            "labels": [],
            "intent_tags": [],
            "primary_key": info["pk_candidates"],
            "related_tables": list(related_map.get(table_name, []))
        }
        table_meta.append(entry)

    for table_name, info in table_info.items():
        for col in info["columns"]:
            unique_set = info["unique_counts"][col]
            unique_count = len(unique_set)
            row_count = info["row_count"]

            sample_val = None
            for row in info["sample_rows"]:
                if row.get(col):
                    sample_val = row[col]
                    break

            if sample_val:
                try:
                    int(sample_val)
                    dtype = "integer"
                except ValueError:
                    try:
                        float(sample_val)
                        dtype = "float"
                    except ValueError:
                        if '/' in sample_val or '-' in sample_val:
                            dtype = "datetime"
                        else:
                            dtype = "string"
            else:
                dtype = "string"

            null_count = row_count - len(unique_set)
            null_pct = null_count / row_count if row_count > 0 else 0
            is_filterable = True
            if null_pct > 0.8:
                is_filterable = False
            if dtype == "string" and unique_count > 0.5 * row_count:
                is_filterable = False

            is_aggregatable = dtype in ("integer", "float", "datetime")

            col_entry = {
                "table_name": table_name,
                "column_name": col,
                "description": "",
                "labels": [],
                "aliases": [],
                "data_type": dtype,
                "is_filterable": is_filterable,
                "is_aggregatable": is_aggregatable
            }
            column_meta.append(col_entry)

    with open(os.path.join(data_folder, "table_temp.json"), "w", encoding="utf-8") as f:
        json.dump(table_meta, f, indent=2, ensure_ascii=False)

    with open(os.path.join(data_folder, "column_temp.json"), "w", encoding="utf-8") as f:
        json.dump(column_meta, f, indent=2, ensure_ascii=False)

    print(f"✅ Generated table_temp.json and column_temp.json in {data_folder}")
    return table_meta, column_meta

# ------------------------------------------------------------
# STEP 2: Build SQLite DB from the metadata and CSVs
# ------------------------------------------------------------
def build_database(data_folder, db_path, table_meta, column_meta):
    table_cols = defaultdict(list)
    for col_entry in column_meta:
        table_cols[col_entry["table_name"]].append(col_entry)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")

    # Create tables
    for tbl in table_meta:
        table_name = tbl["table_name"]
        pk_list = tbl.get("primary_key", [])
        cols = table_cols.get(table_name, [])
        if not cols:
            print(f"⚠️ No columns for {table_name}, skipping.")
            continue

        col_defs = []
        for col in cols:
            col_name = col["column_name"]
            dtype = col["data_type"]
            if dtype == "integer":
                sql_type = "INTEGER"
            elif dtype == "float":
                sql_type = "REAL"
            elif dtype == "datetime":
                sql_type = "TIMESTAMP"
            else:
                sql_type = "TEXT"
            col_defs.append(f'"{col_name}" {sql_type}')

        if pk_list:
            pk_str = ", ".join([f'"{pk}"' for pk in pk_list])
            col_defs.append(f"PRIMARY KEY ({pk_str})")

        create_sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" (\n  ' + ",\n  ".join(col_defs) + "\n);"
        print(f"Creating table: {table_name}")
        cursor.execute(create_sql)

    # Import CSVs
    import_stats = {}
    for tbl in table_meta:
        table_name = tbl["table_name"]
        csv_file = os.path.join(data_folder, f"{table_name}.csv")
        if not os.path.exists(csv_file):
            print(f"⚠️ CSV not found: {csv_file}, skipping import.")
            continue

        print(f"Importing {csv_file} -> {table_name}")
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames
            if not columns:
                continue
            placeholders = ", ".join(["?"] * len(columns))
            col_names = ", ".join([f'"{col}"' for col in columns])
            insert_sql = f'INSERT OR REPLACE INTO "{table_name}" ({col_names}) VALUES ({placeholders})'

            batch = []
            batch_size = 1000
            total_rows = 0
            for row in reader:
                values = [row[col] for col in columns]
                batch.append(values)
                total_rows += 1
                if len(batch) >= batch_size:
                    cursor.executemany(insert_sql, batch)
                    batch = []
            if batch:
                cursor.executemany(insert_sql, batch)
            conn.commit()
            import_stats[table_name] = total_rows
            print(f"  ✅ Imported {total_rows} rows")

        # Index PKs
        if tbl.get("primary_key"):
            pk_list = tbl["primary_key"]
            index_name = f"idx_{table_name}_pk"
            pk_cols = ", ".join([f'"{pk}"' for pk in pk_list])
            cursor.execute(f'CREATE INDEX IF NOT EXISTS "{index_name}" ON "{table_name}" ({pk_cols});')
            print(f"  🏷️ Indexed PK on {table_name}")

    conn.commit()
    conn.close()
    print(f"✅ Database built at {db_path}")

    # ----- VERIFICATION: Check that data was actually inserted -----
    print("\n" + "=" * 60)
    print("🔍 VERIFYING DATABASE CONTENTS")
    print("=" * 60)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get list of tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    for table_name in tables:
        cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
        count = cursor.fetchone()[0]
        print(f"  📊 Table '{table_name}': {count} rows")

        # Show first 3 rows as sample
        if count > 0:
            cursor.execute(f'SELECT * FROM "{table_name}" LIMIT 3')
            rows = cursor.fetchall()
            # Get column names
            cursor.execute(f'PRAGMA table_info("{table_name}")')
            cols = [info[1] for info in cursor.fetchall()]
            print(f"     Sample rows (first 3):")
            for row in rows:
                print(f"       {dict(zip(cols, row))}")
        else:
            print(f"     ⚠️  Table '{table_name}' is EMPTY!")

    conn.close()

    # Compare with import stats
    print("\n" + "=" * 60)
    print("📊 IMPORT SUMMARY")
    print("=" * 60)
    for table_name, expected_count in import_stats.items():
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
        actual_count = cursor.fetchone()[0]
        conn.close()
        status = "✅" if actual_count == expected_count else "⚠️"
        print(f"  {status} {table_name}: expected {expected_count}, found {actual_count}")

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    # Generate JSONs
    table_meta, column_meta = infer_metadata_from_csvs(DATA_DIR)

    # Build DB
    build_database(DATA_DIR, DB_PATH, table_meta, column_meta)

    print("\n🎉 All done! You now have:")
    print(f"   - Metadata JSONs in {DATA_DIR}")
    print(f"   - SQLite database: {DB_PATH}")