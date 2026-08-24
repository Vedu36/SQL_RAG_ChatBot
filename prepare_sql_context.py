import os
import json
import pickle
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# ---------- Load .env (embedding credentials only) ----------
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

EMBED_ENDPOINT = os.environ.get("AZURE_OPENAI_EMBED_ENDPOINT")
EMBED_MODEL = os.environ.get("AZURE_OPENAI_EMBED_DEPLOYMENT")
EMBED_KEY = os.environ.get("AZURE_OPENAI_EMBED_KEY")
EMBED_API_VERSION_OVERRIDE = os.environ.get("AZURE_OPENAI_EMBED_API_VERSION")

if not EMBED_ENDPOINT or not EMBED_MODEL:
    raise RuntimeError("Missing AZURE_OPENAI_EMBED_ENDPOINT or EMBED_DEPLOYMENT in .env")

# ---------- Auth / embedding client ----------
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

embed_client, _ = find_working_embed_client(
    EMBED_ENDPOINT, EMBED_MODEL, EMBED_KEY, EMBED_API_VERSION_OVERRIDE
)

def get_embedding(text: str) -> list[float]:
    response = embed_client.embeddings.create(input=text, model=EMBED_MODEL)
    return response.data[0].embedding

def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    norms[norms == 0] = 1e-10
    return mat / norms

# ---------- Configuration for output ----------
MAX_ACTUAL_VALUES_TO_SHOW = 20   # cap the list of actual values to avoid bloated JSON

# ---------- Workflow ----------
def prepare_sql_context(
    user_query: str,
    top_k_tables: int = 3,
    top_k_cols_per_table: int = 5,
    min_table_score: float = 0.0,
    debug: bool = False
) -> dict:
    # 1. Load both indexes
    with open("table_index.pkl", "rb") as f:
        table_index = pickle.load(f)
    with open("column_index.pkl", "rb") as f:
        column_index = pickle.load(f)

    # 2. Build mapping: table_name -> list of column indices
    table_to_col_indices = {}
    for idx, doc in enumerate(column_index["columns"]):
        table_name = doc["table_name"]
        table_to_col_indices.setdefault(table_name, []).append(idx)

    # 3. Embed the user query
    q_emb = _normalize(np.array(get_embedding(user_query)).reshape(1, -1))

    # 4. Table-level retrieval
    table_embs = _normalize(table_index["embeddings"])
    table_sims = np.dot(q_emb, table_embs.T)[0]
    top_table_indices = np.argsort(table_sims)[::-1][:top_k_tables]

    if debug:
        print("\n🔎 Table similarity scores:")
        ranked = np.argsort(table_sims)[::-1]
        for i in ranked[:10]:
            print(f"  {table_index['tables'][i]['table_name']:30s} {table_sims[i]:.4f}")

    # 5. Column-level similarity scores
    col_embs = _normalize(column_index["embeddings"])
    col_sims = np.dot(q_emb, col_embs.T)[0]

    # 6. Build the result JSON
    result_json = {
        "user_query": user_query,
        "relevant_schema": []
    }

    for idx in top_table_indices:
        score = float(table_sims[idx])
        if score < min_table_score:
            continue

        table_doc = table_index["tables"][idx]
        table_name = table_doc["table_name"]

        col_indices = table_to_col_indices.get(table_name, [])
        if not col_indices:
            continue

        # Rank columns for this table
        table_col_scores = [(ci, col_sims[ci]) for ci in col_indices]
        table_col_scores.sort(key=lambda x: x[1], reverse=True)
        top_cols = table_col_scores[:top_k_cols_per_table]

        # after loading table_index/column_index, also load once:
        with open("relationships.json") as f:
            relationships = json.load(f)

        # inside the loop building table_entry:
        table_entry = {
            "table_name": table_name,
            "table_score": score,
            "table_description": table_doc.get("description", ""),
            "table_labels": table_doc.get("labels", ""),
            "primary_key": relationships.get(table_name, {}).get("primary_key", []),
            "foreign_keys": relationships.get(table_name, {}).get("foreign_keys", []),
            "columns": []
        }

        for col_idx, col_score in top_cols:
            doc = column_index["columns"][col_idx]

            # ----- Extract actual_values, cap at MAX_ACTUAL_VALUES_TO_SHOW -----
            actual_vals = doc.get("actual_values", [])
            if len(actual_vals) > MAX_ACTUAL_VALUES_TO_SHOW:
                actual_vals = actual_vals[:MAX_ACTUAL_VALUES_TO_SHOW]

            table_entry["columns"].append({
                "column_name": doc["column_name"],
                "description": doc["description"],
                "data_type": doc["data_type"],
                "is_filterable": doc["is_filterable"],
                "is_aggregatable": doc["is_aggregatable"],
                "actual_values": actual_vals,          # only this, not sample_values
                "score": float(col_score)
            })

        result_json["relevant_schema"].append(table_entry)

    return result_json


if __name__ == "__main__":
    query = "Placements by relay department"
    print(f"🔍 Processing query: '{query}'")
    context = prepare_sql_context(query, debug=True)

    print("\n📋 Generated SQL Context:")
    print(json.dumps(context, indent=2))

    with open("sql_context.json", "w") as f:
        json.dump(context, f, indent=2)
    print("\n💾 Saved to sql_context.json")