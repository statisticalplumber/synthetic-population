"""Parquet / JSONL storage with partitioning and run manifests.

Layout convention:
    data/generated/<stage>/<run_id>/part-00000.parquet
    data/generated/<stage>/<run_id>/records.jsonl   (debug/interop sidecar)
    data/generated/<stage>/<run_id>/manifest.json   (provenance + counts)

Large datasets are partitioned by the given fields (e.g. country, shard).
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def make_run_id(country_code: str, config_version: str, n: int, seed: int) -> str:
    return f"{country_code.lower()}_{config_version}_n{n}_s{seed}"


def write_partitioned_parquet(
    records: list[dict[str, Any]],
    out_dir: Path,
    partition_by: list[str] | None = None,
    prefix: str = "part",
) -> list[Path]:
    """Write records to one Parquet file per partition. Returns written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    partition_by = partition_by or []

    groups: dict[tuple, list[dict]] = defaultdict(list)
    if partition_by:
        for r in records:
            key = tuple(str(r.get(f)) for f in partition_by)
            groups[key].append(r)
    else:
        groups[()] = records

    written: list[Path] = []
    for i, (key, group) in enumerate(groups.items()):
        suffix = "_".join(key) if key else "all"
        path = out_dir / f"{prefix}-{i:05d}_{suffix}.parquet"
        table = pa.Table.from_pylist(group)
        pq.write_table(table, path)
        written.append(path)
    return written


def write_jsonl(records: list[dict[str, Any]], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def write_manifest(
    out_dir: Path,
    stage: str,
    run_id: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage": stage,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "extra": extra or {},
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def read_parquet_dir(out_dir: Path) -> list[dict[str, Any]]:
    """Read all parquet files under a run directory back into plain dicts."""
    out_dir = Path(out_dir)
    records: list[dict[str, Any]] = []
    for path in sorted(out_dir.glob("*.parquet")):
        records.extend(pq.read_table(path).to_pylist())
    return records


def load_skeletons(records: list[dict[str, Any]]) -> list:
    """Re-validate stored dicts into PersonaSkeleton objects."""
    from ..models.persona import PersonaSkeleton

    return [PersonaSkeleton.model_validate(r) for r in records]
