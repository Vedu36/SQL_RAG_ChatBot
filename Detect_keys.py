"""
detect_keys.py

Identifies primary keys and *verified* foreign keys in crm_data.db and
writes them to relationships.json, in a shape ready to merge into your
table_meta / sql_context.json.

Why this is different from build_db.py's current approach:
- build_db.py only guesses FKs by name (col ends in "_id" -> table name
  starts with that prefix). It never checks whether the values line up.
- This script does an actual referential-integrity check: for a
  candidate FK column, what fraction of its non-null values actually
  exist in the candidate referenced table's PK column? Only keeps
  matches above a confidence threshold, and picks the best match when
  more than one table matches by name.

Run this AFTER build_db.py has created crm_data.db.
"""

import sqlite3
import json
from itertools import product

DB_PATH = "crm_data.db"
OUT_PATH = "relationships.json"
MIN_CONFIDENCE = 0.95  # fraction of FK values that must exist in the PK column


def get_tables(conn):
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    return [r[0] for r in cur.fetchall()]


def get_columns(conn, table):
    cur = conn.cursor()
    cur.execute(f'PRAGMA table_info("{table}")')
    # (cid, name, type, notnull, dflt_value, pk)
    return cur.fetchall()


def get_primary_keys(conn, table):
    cols = get_columns(conn, table)
    pk_cols = [c[1] for c in cols if c[5] > 0]
    if pk_cols:
        return pk_cols
    # Fallback: PRAGMA table_info won't show composite/PK created via
    # a separate PRIMARY KEY(...) clause reliably in older sqlite - so
    # also check uniqueness directly as a backstop.
    return infer_pk_by_uniqueness(conn, table, cols)


def infer_pk_by_uniqueness(conn, table, cols):
    cur = conn.cursor()
    cur.execute(f'SELECT COUNT(*) FROM "{table}"')
    row_count = cur.fetchone()[0]
    if row_count == 0:
        return []
    candidates = []
    for c in cols:
        col_name = c[1]
        cur.execute(f'SELECT COUNT(DISTINCT "{col_name}") FROM "{table}" WHERE "{col_name}" IS NOT NULL')
        distinct_count = cur.fetchone()[0]
        if distinct_count == row_count:
            candidates.append(col_name)
    return candidates


def column_value_set(conn, table, col):
    cur = conn.cursor()
    cur.execute(f'SELECT DISTINCT "{col}" FROM "{table}" WHERE "{col}" IS NOT NULL')
    return {r[0] for r in cur.fetchall()}


def candidate_referenced_tables(col_name, all_tables, current_table):
    """
    Generates plausible referenced table names from a column name,
    e.g. customer_id -> ['customer', 'customers'], acct_id -> ['acct', 'accts'].
    Only used to shortlist candidates - every candidate is still verified
    against real data before being accepted.
    """
    if not col_name.lower().endswith("_id"):
        return []
    prefix = col_name[:-3].lower()
    shortlist = []
    for t in all_tables:
        if t == current_table:
            continue
        tl = t.lower()
        if tl == prefix or tl == prefix + "s" or tl.rstrip("s") == prefix:
            shortlist.append(t)
    return shortlist


def detect_foreign_keys(conn):
    all_tables = get_tables(conn)
    pk_map = {t: get_primary_keys(conn, t) for t in all_tables}

    relationships = {t: {"primary_key": pk_map[t], "foreign_keys": []} for t in all_tables}

    for table in all_tables:
        cols = [c[1] for c in get_columns(conn, table)]
        for col in cols:
            # Don't treat a table's own PK as an FK candidate
            if col in pk_map[table]:
                continue

            shortlisted = candidate_referenced_tables(col, all_tables, table)
            if not shortlisted:
                continue

            fk_values = column_value_set(conn, table, col)
            if not fk_values:
                continue

            best_match = None
            best_confidence = 0.0

            for ref_table in shortlisted:
                for ref_pk in pk_map.get(ref_table, []):
                    ref_values = column_value_set(conn, ref_table, ref_pk)
                    if not ref_values:
                        continue
                    matched = fk_values & ref_values
                    confidence = len(matched) / len(fk_values)
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = (ref_table, ref_pk)

            if best_match and best_confidence >= MIN_CONFIDENCE:
                relationships[table]["foreign_keys"].append({
                    "column": col,
                    "references_table": best_match[0],
                    "references_column": best_match[1],
                    "confidence": round(best_confidence, 3)
                })

    return relationships


def main():
    conn = sqlite3.connect(DB_PATH)
    relationships = detect_foreign_keys(conn)
    conn.close()

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(relationships, f, indent=2)

    print(f"Wrote {OUT_PATH}\n")
    for table, info in relationships.items():
        if info["foreign_keys"]:
            print(f"{table}  (PK: {info['primary_key']})")
            for fk in info["foreign_keys"]:
                print(f"   {fk['column']} -> {fk['references_table']}.{fk['references_column']} "
                      f"(confidence {fk['confidence']})")


if __name__ == "__main__":
    main()