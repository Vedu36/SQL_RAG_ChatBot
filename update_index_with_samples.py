import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# ---------- Configuration ----------
SAMPLE_VALUES_PER_COLUMN = 10   # number of sample values to include
CSV_FOLDER = "Data - Customer Wiki/"   # same folder used during indexing

# ---------- Load .env (embedding credentials only) ----------
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

EMBED_ENDPOINT = os.environ.get("AZURE_OPENAI_EMBED_ENDPOINT")
EMBED_MODEL = os.environ.get("AZURE_OPENAI_EMBED_DEPLOYMENT")
EMBED_KEY = os.environ.get("AZURE_OPENAI_EMBED_KEY")
EMBED_API_VERSION_OVERRIDE = os.environ.get("AZURE_OPENAI_EMBED_API_VERSION")

if not EMBED_ENDPOINT or not EMBED_MODEL:
    raise RuntimeError("Missing embedding credentials in .env")

# ---------- Auth / embedding client (same as before) ----------
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
        print("✔️ Embedding client OK (foundry v1)")
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

# ---------- Main update ----------
def update_index_with_samples():
    # 1. Load existing column index
    with open("column_index.pkl", "rb") as f:
        col_index = pickle.load(f)

    columns = col_index["columns"]  # list of dicts
    print(f"Loaded {len(columns)} columns from column_index.pkl")

    # 2. Build a lookup: table_name -> DataFrame (read CSVs)
    table_dfs = {}
    for file in os.listdir(CSV_FOLDER):
        if file.endswith(".csv"):
            table_name = os.path.splitext(file)[0]
            df = pd.read_csv(os.path.join(CSV_FOLDER, file))
            table_dfs[table_name] = df
            print(f"Loaded CSV: {table_name} ({len(df.columns)} columns)")

    # 3. For each column, get sample values and re-embed
    new_embeddings = []
    for i, doc in enumerate(columns):
        table_name = doc["table_name"]
        col_name = doc["column_name"]

        # Get sample values from the DataFrame
        df = table_dfs.get(table_name)
        if df is None or col_name not in df.columns:
            print(f"⚠️  Skipping {table_name}.{col_name} – table/column not found in CSV.")
            # Keep old embedding? We'll just re-use old embedding.
            new_embeddings.append(col_index["embeddings"][i])
            continue

        # Get up to SAMPLE_VALUES_PER_COLUMN unique values (strings)
        unique_vals = df[col_name].dropna().unique()[:SAMPLE_VALUES_PER_COLUMN]
        sample_text = " ".join(str(v)[:50] for v in unique_vals)   # truncate long values to avoid tokens

        # Build the new embedding text
        description = doc.get("description", "")
        labels = " ".join(doc.get("labels", []))
        aliases = " ".join(doc.get("aliases", []))
        embed_text = f"{description} {labels} {aliases} {col_name} {sample_text}"

        # Re-embed
        try:
            emb = get_embedding(embed_text)
            new_embeddings.append(emb)
        except Exception as e:
            print(f"❌ Failed to embed {table_name}.{col_name}: {e}")
            # Fallback: keep old embedding
            new_embeddings.append(col_index["embeddings"][i])

        if (i+1) % 50 == 0:
            print(f"  Processed {i+1}/{len(columns)} columns...")

    # 4. Update the index
    col_index["embeddings"] = np.array(new_embeddings)

    # 5. Save as a new file
    with open("column_index_with_samples.pkl", "wb") as f:
        pickle.dump(col_index, f)
    print("✅ Saved new index to column_index_with_samples.pkl")

if __name__ == "__main__":
    update_index_with_samples()
    