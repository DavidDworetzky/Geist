#!/usr/bin/env python3
"""
Enumerate open models from Hugging Face Hub with optional filters.

This script queries the Hugging Face Hub for models and prints a concise list
including model id, downloads, likes, task tags, and license. By default it
targets text-generation models usable with Transformers and excludes gated or
private models. You can customize filters and optionally save JSON output.
"""

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from huggingface_hub import list_models, ModelFilter, ModelInfo


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


# A conservative set of permissive/open licenses commonly used for open models
OPEN_LICENSES: Tuple[str, ...] = (
    "apache-2.0",
    "mit",
    "bsd-2-clause",
    "bsd-3-clause",
    "mpl-2.0",
    "lgpl-2.1",
    "lgpl-3.0",
    "unlicense",
    "cc0-1.0",
    "cc-by-4.0",
    "cc-by-sa-4.0",
)


@dataclass
class ModelSummary:
    """Serializable summary of a model entry from Hugging Face Hub."""
    model_id: str
    likes: Optional[int]
    downloads: Optional[int]
    license: Optional[str]
    tasks: List[str]
    library_name: Optional[str]
    gated: Optional[bool]
    private: Optional[bool]


def _extract_tags(info: ModelInfo) -> Tuple[List[str], Optional[str]]:
    """
    Extract task tags and primary library from a ModelInfo's tags.

    Args:
        info: The ModelInfo object returned by the hub.

    Returns:
        A tuple of (tasks, library_name)
    """
    tags: Sequence[str] = getattr(info, "tags", []) or []
    tasks = [t.replace("task:", "") for t in tags if t.startswith("task:")]
    libs = [t.replace("library:", "") for t in tags if t.startswith("library:")]
    library_name = libs[0] if libs else None
    return list(tasks), library_name


def enumerate_open_models(
    task: Optional[str] = "text-generation",
    library: Optional[str] = "vllm",
    limit: int = 200,
    open_licenses_only: bool = True,
    include_gated: bool = False,
    include_private: bool = False,
) -> List[ModelSummary]:
    """
    Query Hugging Face Hub for models and return filtered summaries.

    Args:
        task: Optional task filter (e.g., "text-generation"). If None, no task filter is applied.
        library: Optional library filter (e.g., "transformers"). If None, no library filter.
        limit: Maximum number of models to fetch.
        open_licenses_only: If True, restrict to a set of permissive/open licenses.
        include_gated: If True, include gated models in results.
        include_private: If True, include private models in results.

    Returns:
        A list of ModelSummary entries.
    """
    filters: List[Any] = []

    if task:
        filters.append(ModelFilter(task=task))
    if library:
        filters.append(ModelFilter(library=library))
    if open_licenses_only:
        # ModelFilter supports license filtering; apply as a union of acceptable licenses
        filters.append(ModelFilter(license=list(OPEN_LICENSES)))

    logger.info(
        "Querying Hugging Face Hub (task=%s, library=%s, open_licenses_only=%s, limit=%d)",
        task, library, open_licenses_only, limit
    )

    # full=True to pull expanded fields like tags, downloads, likes, private/gated
    results: Iterable[ModelInfo] = list_models(
        filter=filters if filters else None,
        full=True,
        sort="downloads",
        direction=-1,
        limit=limit,
    )

    summaries: List[ModelSummary] = []
    for info in results:
        tasks, primary_lib = _extract_tags(info)
        license_str = getattr(info, "license", None)
        likes = getattr(info, "likes", None)
        downloads = getattr(info, "downloads", None)
        gated = getattr(info, "gated", None)
        private = getattr(info, "private", None)

        if not include_gated and gated:
            continue
        if not include_private and private:
            continue
        if open_licenses_only and license_str and license_str.lower() not in OPEN_LICENSES:
            # Skip models with non-open licenses if the flag is set.
            continue

        summaries.append(
            ModelSummary(
                model_id=getattr(info, "modelId", getattr(info, "id", "")),
                likes=likes,
                downloads=downloads,
                license=license_str,
                tasks=tasks,
                library_name=primary_lib,
                gated=gated,
                private=private,
            )
        )

    return summaries


def main() -> None:
    """CLI entry point to list open models from the Hugging Face Hub."""
    parser = argparse.ArgumentParser(
        description=(
            "List open models from Hugging Face Hub with optional filters."
        )
    )
    parser.add_argument(
        "--task",
        type=str,
        default="text-generation",
        help="Task filter, e.g., 'text-generation'. Use empty string to disable.",
    )
    parser.add_argument(
        "--library",
        type=str,
        default="transformers",
        help="Library filter, e.g., 'transformers'. Use empty string to disable.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Max number of models to fetch (default: 200).",
    )
    parser.add_argument(
        "--all-licenses",
        action="store_true",
        help="Include all licenses (not only permissive/open).",
    )
    parser.add_argument(
        "--include-gated",
        action="store_true",
        help="Include gated models in the output.",
    )
    parser.add_argument(
        "--include-private",
        action="store_true",
        help="Include private models in the output.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Optional path to write JSON output with the model list.",
    )

    args = parser.parse_args()

    task = args.task if args.task.strip() else None
    library = args.library if args.library.strip() else None

    summaries = enumerate_open_models(
        task=task,
        library=library,
        limit=args.limit,
        open_licenses_only=not args.all_licenses,
        include_gated=args.include_gated,
        include_private=args.include_private,
    )

    if not summaries:
        logger.info("No models matched the provided filters.")
        return

    # Pretty print to stdout
    for s in summaries:
        license_part = f" | license={s.license}" if s.license else ""
        lib_part = f" | lib={s.library_name}" if s.library_name else ""
        task_part = f" | tasks={','.join(s.tasks)}" if s.tasks else ""
        gates = []
        if s.gated:
            gates.append("gated")
        if s.private:
            gates.append("private")
        gate_part = f" | {'/'.join(gates)}" if gates else ""
        print(f"{s.model_id} | downloads={s.downloads} | likes={s.likes}{license_part}{lib_part}{task_part}{gate_part}")

    # Optional JSON output
    if args.json_out:
        try:
            payload: List[Dict[str, Any]] = [asdict(s) for s in summaries]
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            logger.info("Wrote JSON output to %s", args.json_out)
        except Exception as e:  # pragma: no cover - best effort
            logger.error("Failed to write JSON output: %s", e)


if __name__ == "__main__":
    main()


