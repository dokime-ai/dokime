"""FastAPI server for the Dokime Explorer web UI."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Dokime Explorer", docs_url="/docs")

# ---------------------------------------------------------------------------
# Module-level state (populated by `launch()`)
# ---------------------------------------------------------------------------
_state: dict[str, Any] = {
    "input_path": None,
    "documents": [],
    "embeddings": None,  # np.ndarray or None
    "index": None,  # EmbeddingIndex or None
    "model": None,  # EmbeddingModel or None
    "umap_coords": None,  # np.ndarray (N,2) or None
}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the single-page HTML frontend."""
    html_path = STATIC_DIR / "index.html"
    return FileResponse(html_path, media_type="text/html")


@app.get("/api/data")
async def get_data(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    text_field: str = Query("text"),
):
    """Return a paginated slice of the dataset."""
    docs = _state["documents"]
    total = len(docs)
    start = (page - 1) * page_size
    end = min(start + page_size, total)
    page_docs = docs[start:end]

    # Build rows: index + truncated text + all fields
    rows = []
    for i, doc in enumerate(page_docs, start=start):
        text = doc.get(text_field, "")
        rows.append(
            {
                "idx": i,
                "text_preview": text[:300],
                "fields": {k: str(v)[:200] for k, v in doc.items()},
            }
        )

    return {
        "rows": rows,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": math.ceil(total / page_size) if total else 0,
    }


@app.get("/api/document/{idx}")
async def get_document(idx: int):
    """Return full document by index."""
    docs = _state["documents"]
    if 0 <= idx < len(docs):
        return {"idx": idx, "document": docs[idx]}
    return JSONResponse({"error": "Index out of range"}, status_code=404)


@app.get("/api/stats")
async def get_stats():
    """Return dataset statistics."""
    docs = _state["documents"]
    total = len(docs)

    if total == 0:
        return {"total": 0}

    lengths = [len(doc.get("text", "")) for doc in docs]
    total_chars = sum(lengths)
    avg_len = total_chars / total
    min_len = min(lengths)
    max_len = max(lengths)

    # Field names from first doc
    fields = list(docs[0].keys()) if docs else []

    # Length distribution buckets
    buckets = [0, 100, 500, 1000, 5000, 10000, 50000, float("inf")]
    bucket_labels = []
    bucket_counts = []
    for i in range(len(buckets) - 1):
        lo, hi = buckets[i], buckets[i + 1]
        label = f"{lo}-{hi}" if hi != float("inf") else f"{lo}+"
        count = sum(1 for length in lengths if lo <= length < hi)
        bucket_labels.append(label)
        bucket_counts.append(count)

    has_embeddings = _state["embeddings"] is not None

    return {
        "total": total,
        "total_chars": total_chars,
        "avg_length": round(avg_len, 1),
        "min_length": min_len,
        "max_length": max_len,
        "median_length": round(sorted(lengths)[total // 2], 1),
        "fields": fields,
        "has_embeddings": has_embeddings,
        "embedding_dim": int(_state["embeddings"].shape[1]) if has_embeddings else None,
        "length_distribution": {"labels": bucket_labels, "counts": bucket_counts},
        "input_path": _state["input_path"],
    }


@app.get("/api/search")
async def search_docs(
    q: str = Query(..., min_length=1),
    k: int = Query(20, ge=1, le=200),
):
    """Semantic search (requires embeddings)."""
    if _state["embeddings"] is None:
        return JSONResponse({"error": "No embeddings loaded. Compute them first."}, status_code=400)

    index = _state["index"]
    model = _state["model"]

    if index is None or model is None:
        return JSONResponse({"error": "Embedding index not ready."}, status_code=500)

    results = index.search(q, model, k=k)

    return {
        "query": q,
        "results": [
            {
                "idx": r.index,
                "score": round(r.score, 4),
                "text_preview": r.document.get("text", "")[:300],
                "fields": {k_: str(v)[:200] for k_, v in r.document.items()},
            }
            for r in results
        ],
    }


@app.post("/api/embed")
async def trigger_embed(
    model_name: str = Query("all-MiniLM-L6-v2"),
):
    """Compute embeddings for the loaded dataset (blocking)."""
    from dokime.embeddings.compute import EmbeddingModel, compute_embeddings
    from dokime.embeddings.search import EmbeddingIndex

    docs = _state["documents"]
    if not docs:
        return JSONResponse({"error": "No documents loaded"}, status_code=400)

    _, embeddings = compute_embeddings(
        data=iter(docs),
        model_name=model_name,
        quiet=False,
    )

    emb_model = EmbeddingModel(model_name)
    idx = EmbeddingIndex(embeddings, docs)

    _state["embeddings"] = embeddings
    _state["model"] = emb_model
    _state["index"] = idx
    _state["umap_coords"] = None  # reset UMAP cache

    return {"status": "ok", "shape": list(embeddings.shape)}


@app.get("/api/embeddings/umap")
async def get_umap(
    n_neighbors: int = Query(15, ge=2, le=200),
    min_dist: float = Query(0.1, ge=0.0, le=1.0),
    sample_limit: int = Query(5000, ge=100, le=50000),
):
    """Return 2D UMAP coordinates for visualization."""
    import numpy as np

    if _state["embeddings"] is None:
        return JSONResponse({"error": "No embeddings available."}, status_code=400)

    embeddings = _state["embeddings"]
    docs = _state["documents"]
    n = len(docs)

    # Sample if dataset is too large
    if n > sample_limit:
        rng = np.random.default_rng(42)
        indices = rng.choice(n, sample_limit, replace=False)
        indices.sort()
        emb_subset = embeddings[indices]
    else:
        indices = np.arange(n)
        emb_subset = embeddings

    # Use cached coords if available and same size
    if _state["umap_coords"] is not None and len(_state["umap_coords"]) == len(indices):
        coords = _state["umap_coords"]
    else:
        try:
            from umap import UMAP
        except ImportError:
            # Fall back to PCA if umap not installed
            from sklearn.decomposition import PCA

            reducer = PCA(n_components=2)
            coords = reducer.fit_transform(emb_subset)
            _state["umap_coords"] = coords
        else:
            reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=42)
            coords = reducer.fit_transform(emb_subset)
            _state["umap_coords"] = coords

    # Build response
    points = []
    for i, global_idx in enumerate(indices):
        doc = docs[global_idx]
        text = doc.get("text", "")
        points.append(
            {
                "x": round(float(coords[i, 0]), 4),
                "y": round(float(coords[i, 1]), 4),
                "idx": int(global_idx),
                "text_preview": text[:120],
            }
        )

    return {"points": points, "total_sampled": len(points), "total_docs": n}


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------


def launch(
    input_path: str,
    host: str = "127.0.0.1",
    port: int = 8765,
    embeddings_path: str | None = None,
) -> None:
    """Load data, optionally load embeddings, and start the server."""
    import numpy as np
    import uvicorn
    from rich.console import Console

    from dokime.io.readers import auto_read

    console = Console()
    console.print(f"[bold blue]Dokime Explorer[/] loading [bold]{input_path}[/] ...")

    # Load documents
    documents = list(auto_read(input_path))
    _state["documents"] = documents
    _state["input_path"] = input_path
    console.print(f"  Loaded {len(documents):,} documents")

    # Load embeddings if path provided or auto-detect
    emb_path: Path | None = None
    if embeddings_path:
        emb_path = Path(embeddings_path)
    else:
        # Auto-detect: look for <input_stem>.embeddings.npy next to the input
        input_p = Path(input_path)
        auto_path = input_p.parent / f"{input_p.stem}.embeddings.npy"
        if auto_path.exists():
            emb_path = auto_path

    if emb_path and emb_path.exists():
        console.print(f"  Loading embeddings from [bold]{emb_path}[/]")
        embeddings = np.load(str(emb_path))
        _state["embeddings"] = embeddings

        from dokime.embeddings.compute import EmbeddingModel
        from dokime.embeddings.search import EmbeddingIndex

        model = EmbeddingModel()
        _state["model"] = model
        _state["index"] = EmbeddingIndex(embeddings, documents)
        console.print(f"  Embeddings: {embeddings.shape}")
    else:
        console.print("  No embeddings found. Use the UI to compute them, or pass --embeddings.")

    console.print(f"\n  [bold green]>>> Open http://{host}:{port} in your browser[/]\n")

    uvicorn.run(app, host=host, port=port, log_level="warning")
