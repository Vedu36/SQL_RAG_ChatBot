import os
import json
import uuid
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# ---------- Token counting ----------
# tiktoken gives accurate counts; falls back to a rough char/4 estimate if not
# installed. Install with: pip install tiktoken --break-system-packages
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except ImportError:
    print("⚠️  tiktoken not installed - using rough token estimate (chars/4). "
          "Run: pip install tiktoken  for accurate batching.")

    def count_tokens(text: str) -> int:
        return max(1, len(text) // 4)

# ---------- Batching configuration ----------
# These bound how many columns go into a single LLM call. Batches are ALWAYS
# scoped to one table - columns from different tables are never mixed, even
# if that leaves budget unused on a small table.
MAX_INPUT_TOKENS_PER_CALL = 20000    # prompt (column list) budget per call
MAX_OUTPUT_TOKENS_PER_CALL = 8000    # response (JSON) budget per call
EST_OUTPUT_TOKENS_PER_COLUMN = 90    # rough tokens for one column's
                                      # {description, labels, aliases} reply -
                                      # tune this up if your columns produce
                                      # longer descriptions/more labels

# ---------- Load .env ----------
ENV_PATH = Path(__file__).resolve().parent / ".env"
if not ENV_PATH.exists():
    print(f"⚠️  No .env file found at {ENV_PATH}")
load_dotenv(dotenv_path=ENV_PATH)

REQUIRED_VARS = [
    "AZURE_OPENAI_CHAT_ENDPOINT",
    "AZURE_OPENAI_CHAT_DEPLOYMENT",
    "AZURE_OPENAI_EMBED_ENDPOINT",
    "AZURE_OPENAI_EMBED_DEPLOYMENT",
]
missing = [v for v in REQUIRED_VARS if not os.environ.get(v)]
if missing:
    raise RuntimeError(f"Missing env vars: {', '.join(missing)}")

CHAT_ENDPOINT = os.environ["AZURE_OPENAI_CHAT_ENDPOINT"]
CHAT_MODEL = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]
CHAT_API_VERSION_OVERRIDE = os.environ.get("AZURE_OPENAI_CHAT_API_VERSION")
CHAT_KEY = os.environ.get("AZURE_OPENAI_CHAT_KEY")

EMBED_ENDPOINT = os.environ["AZURE_OPENAI_EMBED_ENDPOINT"]
EMBED_MODEL = os.environ["AZURE_OPENAI_EMBED_DEPLOYMENT"]
EMBED_API_VERSION_OVERRIDE = os.environ.get("AZURE_OPENAI_EMBED_API_VERSION")
EMBED_KEY = os.environ.get("AZURE_OPENAI_EMBED_KEY")

# ---------- Auth / clients (unchanged) ----------
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
        print(f"✔️ Chat client OK (foundry v1)")
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

def find_working_embed_client(endpoint, model, api_key, override):
    if is_foundry_v1_endpoint(endpoint):
        c = make_client(endpoint, None, api_key)
        c.embeddings.create(input="test", model=model)
        print(f"✔️ Embedding client OK (foundry v1)")
        return c, "v1"
    versions = [override] if override else CANDIDATE_API_VERSIONS
    for v in versions:
        try:
            c = make_client(endpoint, v, api_key)
            c.embeddings.create(input="test", model=model)
            print(f"✔️ Embedding client OK (api_version={v})")
            return c, v
        except Exception as e:
            print(f"❌ Embedding api_version {v} failed: {e}")
    raise RuntimeError("No working embedding API version found.")

chat_client, _ = find_working_chat_client(CHAT_ENDPOINT, CHAT_MODEL, CHAT_KEY, CHAT_API_VERSION_OVERRIDE)
embed_client, _ = find_working_embed_client(EMBED_ENDPOINT, EMBED_MODEL, EMBED_KEY, EMBED_API_VERSION_OVERRIDE)

def get_embedding(text: str) -> list[float]:
    response = embed_client.embeddings.create(input=text, model=EMBED_MODEL)
    return response.data[0].embedding

# ---------- Token-aware batching (per table, never mixed) ----------
MAX_SAMPLES_PER_COLUMN = 2  # critical: keep this low (1-2) - sample values are
                             # the biggest driver of prompt size when batches
                             # have 80-100+ columns, so a couple extra samples
                             # per column can blow the input token budget fast

def format_column_line(col: str, dtype: str, samples: list) -> str:
    sample_str = ", ".join(str(v)[:30] for v in samples[:MAX_SAMPLES_PER_COLUMN]) if samples else "none"
    return f"Name: {col}, Type: {dtype}, Samples: {sample_str}"

def chunk_table_columns(all_cols_info: list) -> list:
    """
    Greedily packs a single table's columns into the fewest possible batches,
    each batch bounded by:
      - MAX_INPUT_TOKENS_PER_CALL  (the column list we send)
      - MAX_OUTPUT_TOKENS_PER_CALL (the JSON the model has to return)
    A batch never contains columns from more than one table, because this is
    called once per table with only that table's columns.
    """
    batches = []
    current_batch = []
    current_input_tokens = 0

    for col_info in all_cols_info:
        col, dtype, samples = col_info
        line = format_column_line(col, dtype, samples)
        line_tokens = count_tokens(line)

        projected_output_tokens = (len(current_batch) + 1) * EST_OUTPUT_TOKENS_PER_COLUMN
        would_exceed_input = current_input_tokens + line_tokens > MAX_INPUT_TOKENS_PER_CALL
        would_exceed_output = projected_output_tokens > MAX_OUTPUT_TOKENS_PER_CALL

        if current_batch and (would_exceed_input or would_exceed_output):
            batches.append(current_batch)
            current_batch = []
            current_input_tokens = 0

        current_batch.append(col_info)
        current_input_tokens += line_tokens

    if current_batch:
        batches.append(current_batch)

    return batches

# ---------- Batch metadata generation (one call per batch) ----------
def generate_column_metadata_batch(table_name: str, batch_cols: list) -> dict:
    """
    batch_cols: list of tuples (column_name, data_type, sample_values_list)
    Returns: dict {column_name: {"description": str, "labels": list, "aliases": list}}
    """
    if not batch_cols:
        return {}

    cols_text = [
        f"{idx}. {format_column_line(col, dtype, samples)}"
        for idx, (col, dtype, samples) in enumerate(batch_cols, 1)
    ]

    prompt = f"""
Table name: {table_name}

For each of the following columns, generate:
- A concise description (max 15 words)
- Up to 5 relevant labels (as a JSON array of strings)
- Up to 5 potential aliases (as a JSON array of strings)

Return **ONLY** a single valid JSON object where keys are the exact column names,
and values are objects with keys "description", "labels", "aliases".

Example format:
{{
  "employee_id": {{"description": "Unique employee identifier", "labels": ["id", "key"], "aliases": ["emp_id", "staff_id"]}},
  "salary": {{"description": "Annual salary in USD", "labels": ["pay", "compensation"], "aliases": ["comp", "wage"]}}
}}

Columns:
{chr(10).join(cols_text)}
"""
    # Size the response budget to the batch so a big batch isn't cut off
    # mid-JSON, but never ask for more than the model will honor.
    max_tokens = min(MAX_OUTPUT_TOKENS_PER_CALL + 500, len(batch_cols) * 150 + 500)

    try:
        response = chat_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}  # works on recent API versions
        )
        content = response.choices[0].message.content
        if response.choices[0].finish_reason == "length":
            print(f"⚠️  Response for '{table_name}' batch of {len(batch_cols)} columns was "
                  f"cut off (hit max_tokens={max_tokens}). Lower MAX_OUTPUT_TOKENS_PER_CALL "
                  f"or raise EST_OUTPUT_TOKENS_PER_COLUMN to shrink batches.")

        # Be defensive: extract the JSON object even if the model added stray text.
        start = content.find('{')
        end = content.rfind('}') + 1
        data = json.loads(content[start:end])

        result = {}
        for col, _, _ in batch_cols:
            if col in data:
                entry = data[col]
                result[col] = {
                    "description": entry.get("description", f"Column {col}"),
                    "labels": entry.get("labels", [col]) if isinstance(entry.get("labels"), list) else [col],
                    "aliases": entry.get("aliases", []) if isinstance(entry.get("aliases"), list) else [],
                }
            else:
                result[col] = {
                    "description": f"Column '{col}' from table '{table_name}'",
                    "labels": [col],
                    "aliases": [],
                }
        return result
    except Exception as e:
        print(f"⚠️  Batch metadata generation failed for '{table_name}' "
              f"({len(batch_cols)} columns): {e}")
        fallback = {}
        for col, dtype, samples in batch_cols:
            fallback[col] = {
                "description": f"Column '{col}' from table '{table_name}'",
                "labels": [col],
                "aliases": [],
            }
        return fallback

# ---------- Build index (batched, per-table, schema-locked) ----------
def build_index(folder_path: str) -> dict:
    documents = []
    embeddings = []

    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    print(f"📂 Found {len(csv_files)} CSV files.")

    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        table_name = os.path.splitext(file)[0]

        all_cols_info = []
        for col in df.columns:
            dtype = df[col].dtype
            data_type = str(dtype)
            unique_vals = df[col].dropna().unique()
            sample_values = [str(v)[:50] for v in unique_vals[:MAX_SAMPLES_PER_COLUMN]]
            all_cols_info.append((col, data_type, sample_values))

        batches = chunk_table_columns(all_cols_info)
        print(f"  📄 Table '{table_name}': {len(all_cols_info)} columns "
              f"-> {len(batches)} batch(es) [{', '.join(str(len(b)) for b in batches)}]")

        for batch_num, batch in enumerate(batches, 1):
            print(f"    🔄 Batch {batch_num}/{len(batches)} ({len(batch)} columns)...")
            meta_dict = generate_column_metadata_batch(table_name, batch)

            for col, data_type, sample_values in batch:
                meta = meta_dict.get(col, {
                    "description": f"Column '{col}' from table '{table_name}'",
                    "labels": [col],
                    "aliases": [],
                })

                dtype = df[col].dtype
                is_aggregatable = bool(pd.api.types.is_numeric_dtype(dtype))
                unique_vals = df[col].dropna().unique()
                is_filterable = bool(
                    pd.api.types.is_object_dtype(dtype)
                    or isinstance(dtype, pd.CategoricalDtype)
                    or len(unique_vals) < 100
                )

                # Exact schema requested: table_name, column_name, description,
                # labels, aliases, data_type, is_filterable, is_aggregatable.
                doc = {
                    "table_name": table_name,
                    "column_name": col,
                    "description": meta.get("description", ""),
                    "labels": meta.get("labels", [col]),
                    "aliases": meta.get("aliases", []),
                    "data_type": data_type,
                    "is_filterable": is_filterable,
                    "is_aggregatable": is_aggregatable,
                }
                documents.append(doc)

                embed_text = f"{doc['description']} {' '.join(doc['labels'])} {' '.join(doc['aliases'])} {col}"
                emb = get_embedding(embed_text)
                embeddings.append(emb)

        print(f"    ✅ Table '{table_name}' done.")

    print(f"✅ Index built: {len(documents)} columns total.")
    return {
        "columns": documents,
        "embeddings": np.array(embeddings)
    }

# ---------- Query (unchanged) ----------
def query_index(index: dict, user_query: str, top_k: int = 5) -> list[dict]:
    q_emb = np.array(get_embedding(user_query)).reshape(1, -1)
    sims = np.dot(q_emb, index["embeddings"].T)[0]
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [
        {"column": index["columns"][i], "score": float(sims[i])}
        for i in top_indices
    ]

# ---------- Main ----------
if __name__ == "__main__":
    index = build_index("Data - Customer Wiki/")

    import pickle
    with open("column_index.pkl", "wb") as f:
        pickle.dump(index, f)
    print("💾 Index saved to column_index.pkl")

    results = query_index(index, "how many accountants do we have on the bench?")
    print("\n🔍 Top matching columns:")
    for r in results:
        col = r["column"]
        print(f"  📌 {col['table_name']}.{col['column_name']} (score: {r['score']:.3f})")
        print(f"     Description: {col['description']}")
        print(f"     Labels: {', '.join(col['labels'])}")
        print(f"     Filterable: {col['is_filterable']} | Aggregatable: {col['is_aggregatable']}\n")