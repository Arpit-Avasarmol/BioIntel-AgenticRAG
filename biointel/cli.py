"""BioIntel command-line interface.

Heavy subsystems (embeddings, agent, stores) are imported lazily inside each
command so that ``biointel --help`` and lightweight commands work without the
full ML stack installed.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from biointel.common.config import settings
from biointel.common.logging import setup_logging

app = typer.Typer(
    add_completion=False,
    help="BioIntel Agent — ingest, index, and query biomedical + patent knowledge.",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def _root() -> None:
    setup_logging()


# --------------------------------------------------------------------------- info
@app.command()
def info() -> None:
    """Show the active configuration (models, stores, infra endpoints)."""
    table = Table(title="BioIntel configuration", show_header=True, header_style="bold")
    table.add_column("Setting")
    table.add_column("Value", overflow="fold")
    rows = {
        "model_profile": settings.model_profile,
        "llm_model": settings.llm_model,
        "ollama_base_url": settings.ollama_base_url,
        "embedding_model": f"{settings.embedding_model} ({settings.embedding_dim}d)",
        "reranker_model": settings.reranker_model,
        "vector_backend": settings.vector_backend,
        "qdrant_url": settings.qdrant_url,
        "opensearch_url": settings.opensearch_url,
        "database": settings.resolved_database_url,
        "redis_url": settings.redis_url,
        "minio_endpoint": settings.minio_endpoint,
        "google_patents_enabled": str(settings.google_patents_enabled),
        "obs_enabled": str(settings.obs_enabled),
    }
    for k, v in rows.items():
        table.add_row(k, str(v))
    console.print(table)


# ------------------------------------------------------------------------- ingest
@app.command()
def ingest(
    source: str | None = typer.Option(
        None,
        help="Single source to ingest (pubmed, pmc, clinicaltrials, chembl, "
        "opentargets, patentsview, google_patents).",
    ),
    query: str | None = typer.Option(None, help="Free-text query for the source."),
    max_records: int = typer.Option(200, help="Max records to fetch."),
    config: Path | None = typer.Option(
        None, exists=True, help="YAML corpus config (overrides --source/--query)."
    ),
) -> None:
    """Fetch documents from official APIs into MinIO + Postgres (no scraping)."""
    from biointel.ingestion.runner import ingest_from_config, ingest_single

    if config:
        n = ingest_from_config(config)
    elif source:
        n = ingest_single(source, query=query, max_records=max_records)
    else:
        raise typer.BadParameter("Provide either --config or --source.")
    console.print(f"[green]✓ Ingested {n} documents.[/green]")


# -------------------------------------------------------------------------- index
@app.command()
def index(
    all_docs: bool = typer.Option(False, "--all", help="Index every not-yet-indexed document."),
    source: str | None = typer.Option(None, help="Index only this source."),
    reindex: bool = typer.Option(False, help="Re-chunk and re-embed everything."),
) -> None:
    """Chunk, embed, and upsert documents into Qdrant + OpenSearch."""
    from biointel.indexing.runner import run_indexing

    n_chunks = run_indexing(all_docs=all_docs, source=source, reindex=reindex)
    console.print(f"[green]✓ Indexed {n_chunks} chunks.[/green]")


# -------------------------------------------------------------------------- query
@app.command()
def query(
    question: str = typer.Argument(..., help="The question to ask the agent."),
    top_k: int = typer.Option(settings.retrieval_final_top_k, help="Chunks to use."),
    show_sources: bool = typer.Option(True, help="Print citations."),
    auto_ingest: bool | None = typer.Option(
        None,
        "--auto-ingest/--no-auto-ingest",
        help="Fetch + index live sources for the question before answering.",
    ),
) -> None:
    """Run the full agent pipeline on a single question and print the answer."""
    from biointel.agent.graph import run_agent

    answer = run_agent(question, top_k=top_k, auto_ingest=auto_ingest)
    console.rule("[bold]Answer")
    console.print(answer.answer)
    if answer.contradictions:
        console.rule("[bold yellow]Contradictions")
        for c in answer.contradictions:
            console.print(f"- {c.explanation} ({c.source_a} vs {c.source_b})")
    if answer.warnings:
        console.rule("[bold yellow]Notes")
        for w in answer.warnings:
            console.print(f"- {w}")
    if show_sources and answer.citations:
        console.rule("[bold]Sources")
        for cit in answer.citations:
            console.print(f"{cit.marker} {cit.label} — {cit.source_url}")
    console.print(
        f"\n[dim]verified={answer.verified} · model={answer.model} · "
        f"chunks_used={len(answer.used_chunks)}[/dim]"
    )


# ------------------------------------------------------------------------ stores
@app.command("init-stores")
def init_stores() -> None:
    """Create the Qdrant collection and OpenSearch index if missing."""
    from biointel.retrieval.factory import get_keyword_store, get_vector_store

    vs = get_vector_store()
    vs.ensure_collection()
    ks = get_keyword_store()
    ks.ensure_index()
    console.print("[green]✓ Vector + keyword stores ready.[/green]")


if __name__ == "__main__":
    app()
