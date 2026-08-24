import os
import json
import sqlite3
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# ---------- Load .env (chat + embedding if needed but we only need chat here) ----------
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# Required chat variables
CHAT_ENDPOINT = os.environ.get("AZURE_OPENAI_CHAT_ENDPOINT")
CHAT_MODEL = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")
CHAT_API_VERSION_OVERRIDE = os.environ.get("AZURE_OPENAI_CHAT_API_VERSION")
CHAT_KEY = os.environ.get("AZURE_OPENAI_CHAT_KEY")

if not CHAT_ENDPOINT or not CHAT_MODEL:
    raise RuntimeError("Missing AZURE_OPENAI_CHAT_ENDPOINT or CHAT_DEPLOYMENT in .env")

# ---------- Auth / chat client ----------
token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://ai.azure.com/.default")

def is_foundry_v1_endpoint(endpoint: str) -> bool:
    return "services.ai.azure.com" in endpoint

def normalize_v1_endpoint(endpoint: str) -> str:
    e = endpoint.rstrip("/")
    if not e.endswith("/openai/v1"):
        e = e + "/openai/v1"
    return e

def make_client(endpoint: str, api_version: str | None, api_key: str | None):
    if is_foundry_v1_endpoint(endpoint):
        base_url = normalize_v1_endpoint(endpoint)
        key = api_key if api_key else token_provider()
        return OpenAI(base_url=base_url, api_key=key)
    if api_key:
        return AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)
    return AzureOpenAI(azure_endpoint=endpoint, azure_ad_token_provider=token_provider, api_version=api_version)

CANDIDATE_API_VERSIONS = [
    "2024-10-21", "2024-08-01-preview", "2024-06-01",
    "2024-02-15-preview", "2023-12-01-preview", "2023-05-15",
]

def find_working_chat_client(endpoint, model, api_key, override):
    if is_foundry_v1_endpoint(endpoint):
        c = make_client(endpoint, None, api_key)
        c.chat.completions.create(model=model, messages=[{"role": "user", "content": "Hi"}], max_tokens=5)
        print("✔️ Chat client OK (foundry v1)")
        return c, "v1"
    versions = [override] if override else CANDIDATE_API_VERSIONS
    for v in versions:
        try:
            c = make_client(endpoint, v, api_key)
            c.chat.completions.create(model=model, messages=[{"role": "user", "content": "Hi"}], max_tokens=5)
            print(f"✔️ Chat client OK (api_version={v})")
            return c, v
        except Exception as e:
            print(f"❌ Chat api_version {v} failed: {e}")
    raise RuntimeError("No working chat API version found.")

chat_client, _ = find_working_chat_client(CHAT_ENDPOINT, CHAT_MODEL, CHAT_KEY, CHAT_API_VERSION_OVERRIDE)

# ---------- SQL generation prompt ----------
def generate_sql(user_query: str, context: dict) -> str:
    """
    Given the user query and the schema context (from phase 1),
    ask the LLM to produce a SQLite SELECT query.
    """
    schema_text = json.dumps(context, indent=2)

    prompt = f"""
You are a helpful assistant that translates natural language questions into SQLite queries.

Given the following database schema (tables, columns, descriptions, data types, and filterable/aggregatable flags), generate a **single** SQLite SELECT query that answers the user's question.

Schema context (JSON):
{schema_text}

User question: {user_query}

Instructions:
- Only generate a SELECT query – no DDL, no updates, no deletes.
- Use the table names and column names exactly as provided.
- Output **only** the SQL query, no extra text, no code fences, no explanation.

SQL:
"""

    response = chat_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=500
    )
    sql = response.choices[0].message.content.strip()
    # Remove markdown code fences if present
    if sql.startswith("```sql"):
        sql = sql[6:]
    if sql.startswith("```"):
        sql = sql[3:]
    if sql.endswith("```"):
        sql = sql[:-3]
    return sql.strip()

# ---------- Execute SQL on SQLite ----------
def execute_sql(sql: str, db_path: str = "crm_data.db") -> tuple[list, list]:
    """
    Executes the SQL query on the given SQLite database.
    Returns (rows, column_names).
    """
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

# ---------- Format results using LLM ----------
def format_results(user_query: str, sql: str, rows: list, column_names: list) -> str:
    """
    Sends the first 10 rows (if more than 10) to the LLM to generate a
    natural-language response to the user. Also prints the full result set.
    """
    # Print full result to console for debugging / user transparency
    print("\n📊 Full result (all rows):")
    if rows:
        print("  " + " | ".join(column_names))
        for row in rows:
            print("  " + " | ".join(str(val) for val in row))
    else:
        print("  (No rows returned)")

    # Prepare data for LLM: only first 10 rows to avoid token explosion
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
        temperature=0.3,
        max_tokens=500
    )
    return response.choices[0].message.content.strip()

# ---------- Main pipeline ----------
def answer_query(user_query: str, context_path: str = "sql_context.json", db_path: str = "crm_data.db"):
    # 1. Load context
    with open(context_path, "r") as f:
        context = json.load(f)

    print(f"🔍 Generating SQL for query: '{user_query}'")
    # 2. Generate SQL
    sql = generate_sql(user_query, context)
    print(f"\n🗄️ Generated SQL:\n{sql}")

    # 3. Execute
    try:
        rows, column_names = execute_sql(sql, db_path)
        print(f"\n✅ Query executed. Returned {len(rows)} rows.")
    except Exception as e:
        print(f"\n❌ Error executing SQL: {e}")
        return

    # 4. Format results
    answer = format_results(user_query, sql, rows, column_names)
    print("\n🤖 Answer to user:")
    print(answer)

if __name__ == "__main__":
    # Example usage
    user_query = "how many accountants do we have on the bench?"
    answer_query(user_query)