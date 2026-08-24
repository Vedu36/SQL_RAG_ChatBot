import os
import json
import sqlite3
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# ---------- Load .env ----------
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# GPT-5 mini credentials
CHAT_ENDPOINT = "https://rhc-foundry-sandbox.cognitiveservices.azure.com/"
CHAT_MODEL = "gpt-5-mini"
API_VERSION = "2024-12-01-preview"
CHAT_KEY = os.environ.get("AZURE_OPENAI_CHAT_KEY")

# ---------- Client setup ----------
def get_chat_client():
    if CHAT_KEY:
        return AzureOpenAI(
            azure_endpoint=CHAT_ENDPOINT,
            api_key=CHAT_KEY,
            api_version=API_VERSION,
        )
    else:
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://ai.azure.com/.default"
        )
        return AzureOpenAI(
            azure_endpoint=CHAT_ENDPOINT,
            azure_ad_token_provider=token_provider,
            api_version=API_VERSION,
        )

chat_client = get_chat_client()
print("✅ GPT-5 client ready.")

# ---------- SQL generation (temperature removed) ----------
def generate_sql(user_query: str, context: dict) -> str:
    schema_text = json.dumps(context, indent=2)

    prompt = f"""
You are an expert SQL assistant. Your task is to translate a natural‑language question into a **single SQLite SELECT query**.

### Schema context (JSON):
{schema_text}

### User question:
{user_query}

### Important instructions:
1. **Only use tables and columns that are directly relevant** to answering the question.  
   - If a table or column’s description, labels, or sample values do not match the query’s intent, **ignore it**.
   - Prefer columns that are marked `is_filterable` or `is_aggregatable` when they are needed for filtering or counting.
2. If the information needed is **not present** in the schema, return the text:  
   `"I don't have enough information to answer that query."`  
   (Do NOT guess or fabricate columns.)
3. Use **exact column names** as provided in the JSON – they are case‑sensitive.
4. Output **only** the SQL query (or the "I don't have..." message).  
   No extra text, no markdown, no explanation.

SQL:
"""
    response = chat_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=16384,
        # temperature omitted (defaults to 1)
    )
    sql = response.choices[0].message.content.strip()

    # Remove markdown fences if present
    if sql.startswith("```sql"):
        sql = sql[6:]
    if sql.startswith("```"):
        sql = sql[3:]
    if sql.endswith("```"):
        sql = sql[:-3]

    return sql.strip()

# ---------- Execute SQL ----------
def execute_sql(sql: str, db_path: str = "crm_data.db") -> tuple[list, list]:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        rows = cursor.fetchall()
        column_names = [description[0] for description in cursor.description] if cursor.description else []
        conn.close()
        return rows, column_names
    except Exception as e:
        conn.close()
        raise RuntimeError(f"SQL execution failed: {e}")

# ---------- Format results (temperature removed) ----------
def format_results(user_query: str, sql: str, rows: list, column_names: list) -> str:
    print("\n📊 Full result (all rows):")
    if rows:
        print("  " + " | ".join(column_names))
        for row in rows:
            print("  " + " | ".join(str(val) for val in row))
    else:
        print("  (No rows returned)")

    limited_rows = rows[:10]
    if len(rows) > 10:
        note = f"\n(Note: The query returned {len(rows)} rows; only the first 10 are shown here.)"
    else:
        note = ""

    rows_text = "\n".join(" | ".join(str(v) for v in row) for row in limited_rows)
    if not rows_text:
        rows_text = "(No rows)"

    prompt = f"""
You are a helpful assistant that presents database query results in a clear, concise, and user‑friendly way.

The user asked: "{user_query}"

The SQL query that was executed:
{sql}

The result set (first {len(limited_rows)} row{'s' if len(limited_rows)!=1 else ''}){note}:
Columns: {', '.join(column_names)}
Rows:
{rows_text}

Based on the result, write a natural‑language answer to the user.
- If the result is a single number (like a count), just say "The answer is X."
- If there are rows, present them in a readable list or table, but keep it concise.
- Do not mention SQL or the technical details unless necessary.

Answer:
"""
    response = chat_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=1024,
        # temperature omitted (defaults to 1)
    )
    return response.choices[0].message.content.strip()

# ---------- Main pipeline ----------
def answer_query(user_query: str, context_path: str = "sql_context.json", db_path: str = "crm_data.db"):
    with open(context_path, "r") as f:
        context = json.load(f)

    print(f"🔍 Generating SQL for query: '{user_query}'")
    sql = generate_sql(user_query, context)
    print(f"\n🗄️ Generated SQL:\n{sql}")

    try:
        rows, column_names = execute_sql(sql, db_path)
        print(f"\n✅ Query executed. Returned {len(rows)} rows.")
    except Exception as e:
        print(f"\n❌ Error executing SQL: {e}")
        return

    answer = format_results(user_query, sql, rows, column_names)
    print("\n🤖 Answer to user:")
    print(answer)

if __name__ == "__main__":
    user_query = "how many accountants do we have on the bench?"
    answer_query(user_query)