import os
import json
import uuid
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# ---------- Load .env reliably, regardless of current working directory ----------
ENV_PATH = Path(__file__).resolve().parent / ".env"
if not ENV_PATH.exists():
    print(f"⚠️  No .env file found at {ENV_PATH}")
load_dotenv(dotenv_path=ENV_PATH)

# ---------- Required environment variables ----------
# Chat (text) resource
#   AZURE_OPENAI_CHAT_ENDPOINT
#   AZURE_OPENAI_CHAT_DEPLOYMENT
#   AZURE_OPENAI_CHAT_API_VERSION
#   AZURE_OPENAI_CHAT_KEY        (optional - only if using key auth instead of AAD)
#
# Embedding resource
#   AZURE_OPENAI_EMBED_ENDPOINT
#   AZURE_OPENAI_EMBED_DEPLOYMENT
#   AZURE_OPENAI_EMBED_API_VERSION
#   AZURE_OPENAI_EMBED_KEY       (optional - only if using key auth instead of AAD)

REQUIRED_VARS = [
    "AZURE_OPENAI_CHAT_ENDPOINT",
    "AZURE_OPENAI_CHAT_DEPLOYMENT",
    "AZURE_OPENAI_EMBED_ENDPOINT",
    "AZURE_OPENAI_EMBED_DEPLOYMENT",
]

missing = [v for v in REQUIRED_VARS if not os.environ.get(v)]
if missing:
    raise RuntimeError(
        "Missing required environment variable(s): "
        + ", ".join(missing)
        + f"\n\nExpected them to be defined in: {ENV_PATH}"
        + "\n\nYou need separate settings for the chat resource and the "
          "embedding resource, e.g.:\n"
        "AZURE_OPENAI_CHAT_ENDPOINT=https://<chat-resource>.openai.azure.com/\n"
        "AZURE_OPENAI_CHAT_DEPLOYMENT=<chat-deployment-name>\n"
        "AZURE_OPENAI_CHAT_KEY=<optional-api-key>\n\n"
        "AZURE_OPENAI_EMBED_ENDPOINT=https://<embed-resource>.openai.azure.com/\n"
        "AZURE_OPENAI_EMBED_DEPLOYMENT=<embed-deployment-name>\n"
        "AZURE_OPENAI_EMBED_KEY=<optional-api-key>\n"
        "(no quotes, no trailing spaces around the '=')\n\n"
        "API versions are auto-detected, so you don't need to set them "
        "manually. If you DO know the exact version each resource requires, "
        "you can optionally set AZURE_OPENAI_CHAT_API_VERSION / "
        "AZURE_OPENAI_EMBED_API_VERSION to skip discovery and use it directly."
    )

CHAT_ENDPOINT = os.environ["AZURE_OPENAI_CHAT_ENDPOINT"]
CHAT_MODEL = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]
CHAT_API_VERSION_OVERRIDE = os.environ.get("AZURE_OPENAI_CHAT_API_VERSION")  # optional
CHAT_KEY = os.environ.get("AZURE_OPENAI_CHAT_KEY")  # optional

EMBED_ENDPOINT = os.environ["AZURE_OPENAI_EMBED_ENDPOINT"]
EMBED_MODEL = os.environ["AZURE_OPENAI_EMBED_DEPLOYMENT"]
EMBED_API_VERSION_OVERRIDE = os.environ.get("AZURE_OPENAI_EMBED_API_VERSION")  # optional
EMBED_KEY = os.environ.get("AZURE_OPENAI_EMBED_KEY")  # optional

# ---------- Auth ----------
# If a key is provided for a resource, use key auth for that client.
# Otherwise fall back to Azure AD (DefaultAzureCredential) bearer token.
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://ai.azure.com/.default"
)

def is_foundry_v1_endpoint(endpoint: str) -> bool:
    """
    Detects the newer Azure AI Foundry unified endpoint style
    (https://<resource>.services.ai.azure.com/openai/v1), as opposed to a
    classic Azure OpenAI resource (https://<resource>.openai.azure.com).
    The v1 endpoint is OpenAI-compatible and does NOT take an api_version -
    that's why passing api_version to it returns "API version not supported"
    no matter which value you try.
    """
    return "services.ai.azure.com" in endpoint

def normalize_v1_endpoint(endpoint: str) -> str:
    """Ensure the Foundry v1 base_url ends with /openai/v1 (no trailing slash)."""
    e = endpoint.rstrip("/")
    if not e.endswith("/openai/v1"):
        e = e + "/openai/v1"
    return e

def make_client(endpoint: str, api_version: str | None, api_key: str | None):
    """
    Builds the right client type for the given endpoint:
      - Foundry v1 endpoint -> plain OpenAI client with base_url, no api_version
      - Classic Azure OpenAI resource -> AzureOpenAI client with api_version
    """
    if is_foundry_v1_endpoint(endpoint):
        base_url = normalize_v1_endpoint(endpoint)
        key = api_key if api_key else token_provider()  # resolve AAD token to a string once
        return OpenAI(base_url=base_url, api_key=key)

    if api_key:
        return AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version
        )
    return AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        api_version=api_version
    )

# Candidate API versions to try, newest-first, for classic Azure OpenAI
# resources only. Foundry v1 endpoints skip this entirely - they don't use
# api_version.
CANDIDATE_API_VERSIONS = [
    "2024-10-21",
    "2024-08-01-preview",
    "2024-06-01",
    "2024-02-15-preview",
    "2023-12-01-preview",
    "2023-05-15",
]

def find_working_chat_client(endpoint: str, model: str, api_key: str | None, override: str | None):
    if is_foundry_v1_endpoint(endpoint):
        c = make_client(endpoint, None, api_key)
        c.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5
        )
        print(f"✔️ Chat client OK (foundry v1 endpoint={normalize_v1_endpoint(endpoint)})")
        return c, "v1"

    versions = [override] if override else CANDIDATE_API_VERSIONS
    last_error = None
    for version in versions:
        try:
            c = make_client(endpoint, version, api_key)
            c.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5
            )
            print(f"✔️ Chat client OK (endpoint={endpoint}, api_version={version})")
            return c, version
        except Exception as e:
            last_error = e
            print(f"❌ Chat api_version {version} failed: {e}")
    raise RuntimeError(
        f"Chat client failed to connect with any known API version. "
        f"Check endpoint/deployment/key. Last error: {last_error}"
    )

def find_working_embed_client(endpoint: str, model: str, api_key: str | None, override: str | None):
    if is_foundry_v1_endpoint(endpoint):
        c = make_client(endpoint, None, api_key)
        c.embeddings.create(input="test", model=model)
        print(f"✔️ Embedding client OK (foundry v1 endpoint={normalize_v1_endpoint(endpoint)})")
        return c, "v1"

    versions = [override] if override else CANDIDATE_API_VERSIONS
    last_error = None
    for version in versions:
        try:
            c = make_client(endpoint, version, api_key)
            c.embeddings.create(input="test", model=model)
            print(f"✔️ Embedding client OK (endpoint={endpoint}, api_version={version})")
            return c, version
        except Exception as e:
            last_error = e
            print(f"❌ Embedding api_version {version} failed: {e}")
    raise RuntimeError(
        f"Embedding client failed to connect with any known API version. "
        f"Check endpoint/deployment/key. Last error: {last_error}"
    )

chat_client, CHAT_API_VERSION = find_working_chat_client(CHAT_ENDPOINT, CHAT_MODEL, CHAT_KEY, CHAT_API_VERSION_OVERRIDE)
embed_client, EMBED_API_VERSION = find_working_embed_client(EMBED_ENDPOINT, EMBED_MODEL, EMBED_KEY, EMBED_API_VERSION_OVERRIDE)

def get_embedding(text: str) -> list[float]:
    response = embed_client.embeddings.create(input=text, model=EMBED_MODEL)
    return response.data[0].embedding

# ---------- Metadata Generation ----------
def generate_metadata(table_name: str, columns: list[str]) -> dict:
    prompt = f"""
    Table name: {table_name}
    Columns: {', '.join(columns)}
    
    Generate a concise description (max 30 words) and 5-10 comma-separated labels/tags.
    Respond **only** with valid JSON having keys "description" and "labels".
    Example: {{"description": "Stores employee details...", "labels": "employees, staff, role"}}
    """
    try:
        response = chat_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        content = response.choices[0].message.content
        start = content.find('{')
        end = content.rfind('}') + 1
        return json.loads(content[start:end])
    except Exception as e:
        print(f"Error generating metadata for {table_name}: {e}")
        return {
            "description": f"Table '{table_name}' with columns: {', '.join(columns)}",
            "labels": ", ".join(columns)
        }

# ---------- Build Index ----------
def build_index(folder_path: str) -> dict:
    docs, embeddings = [], []
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder_path, file))
            table_name = os.path.splitext(file)[0]
            columns = df.columns.tolist()
            meta = generate_metadata(table_name, columns)
            text = f"{meta['description']} {meta['labels']}"
            emb = get_embedding(text)
            docs.append({
                "id": str(uuid.uuid4()),
                "table_name": table_name,
                "schema": "dbo",
                "description": meta["description"],
                "labels": meta["labels"]
            })
            embeddings.append(emb)
    return {"tables": docs, "embeddings": np.array(embeddings)}

# ---------- Query ----------
def query_index(index: dict, user_query: str, top_k: int = 3) -> list[dict]:
    q_emb = np.array(get_embedding(user_query)).reshape(1, -1)
    sims = np.dot(q_emb, index["embeddings"].T)[0]
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [{"table": index["tables"][i], "score": float(sims[i])} for i in top_indices]

# ---------- Main ----------
if __name__ == "__main__":
    index = build_index("Data - Customer Wiki/")
    import pickle
    with open("table_index.pkl", "wb") as f:
        pickle.dump(index, f)
    
    results = query_index(index, "how many accountants do we have on the bench?")
    for r in results:
        t = r["table"]
        print(f"Table: {t['table_name']} (score: {r['score']:.3f})")
        print(f"  Description: {t['description']}")
        print(f"  Labels: {t['labels']}\n")