import os, json, re
from collections import defaultdict
from typing import Dict, Any
from pinecone import Pinecone
from openai import OpenAI
from agents.tool import FunctionTool


# ───────────────────────── helper ─────────────────────────
def connect_index():
    host = os.getenv("RESUME_INDEX_HOST")
    pc   = Pinecone(api_key=os.getenv("PINECONE_API_KEY", ""))
    return pc.Index("genairesumesbiggerchunks", host=host)


# ───────────────────────── tool class ─────────────────────
class SearchResumesTool(FunctionTool):
    """
    Smart résumé search tool:
      • Skill/title query  → similarity search (top_k, default 10) with up to
        3 chunks / résumé.
      • Name / filename OR full_resume=True → similarity search top-10k, then
        select only chunks whose metadata-filename contains that token; this
        yields every chunk for that résumé without using unsupported operators.
    """

    def __init__(self):
        schema = {
            "type": "object",
            "properties": {
                "query":       {"type": "string",  "description": "Skill, title, or candidate name / filename"},
                "top_k":       {"type": "integer", "description": "Top-k similar chunks (skill search)"},
                "full_resume": {"type": "boolean", "description": "If true, fetch every chunk for that résumé"}
            },
            # All keys must be listed per OpenAI validation
            "required": ["query", "top_k", "full_resume"],
            "additionalProperties": False
        }
        super().__init__(
            name="search_resumes",
            description="Retrieve résumé data from Pinecone (smart or full).",
            params_json_schema=schema,
            on_invoke_tool=self.on_invoke_tool
        )

    # ------------------------------------------------------------------ #
    async def on_invoke_tool(self, _ctx, args: str) -> str:
        p: Dict[str, Any] = json.loads(args)
        query       = p["query"].strip()
        top_k       = p.get("top_k") or 50
        full_flag   = p.get("full_resume", False)

        looks_like_name = len(query.split()) >= 2 or query.lower().endswith(".pdf")
        want_full       = full_flag or looks_like_name

        idx = connect_index()
        dim = idx.describe_index_stats()["dimension"]
        files: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"filename": None,
                                                                "file_id": "",
                                                                "chunks": []})

        # ------------- embed once (works for both modes) ----------------
        embed = OpenAI().embeddings.create(
            input=query, model="text-embedding-3-large"
        ).data[0].embedding

        if want_full:
            # big top_k to get every chunk that contains similar embedding
            res = idx.query(vector=embed, top_k=10_000, include_metadata=True)
            token = re.sub(r"[^A-Za-z0-9]", "", query.split()[0]).lower()

            # keep only chunks whose filename contains the token
            for m in res.get("matches", []):
                meta   = m["metadata"]
                fname  = meta.get("metadata-filename", "").lower()
                if token in fname:
                    files[fname]["filename"] = meta.get("metadata-filename", "Unknown")
                    files[fname]["file_id"]  = meta.get("file_id", "")
                    files[fname]["chunks"].append(meta.get("text", ""))
        else:
            res = idx.query(vector=embed, top_k=top_k, include_metadata=True)
            per_file_cap = 3
            for m in res.get("matches", []):
                meta   = m["metadata"]
                fname  = meta.get("metadata-filename", "Unknown")
                if len(files[fname]["chunks"]) >= per_file_cap:
                    continue
                files[fname]["filename"] = fname
                files[fname]["file_id"]  = meta.get("file_id", "")
                files[fname]["chunks"].append(meta.get("text", ""))

        return json.dumps(list(files.values()))
