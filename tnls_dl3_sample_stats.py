"""TNLS DL3 sample statistics for prompt concept configs.

This tool scans image folders, matches GT label maps by relative path/stem, and
stores per-image label counts in a cache JSON. The cache can later answer
queries such as {"tnls_dl3": [20, 21, 22]} without reading the GT images again.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping as MappingABC
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


CACHE_VERSION = 1
DEFAULT_IMAGE_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
DEFAULT_GT_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json_arg(value: str) -> Any:
    stripped = value.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return json.loads(stripped)
    return load_json(Path(value))


def normalize_suffixes(suffixes: Sequence[str]) -> Tuple[str, ...]:
    normalized = []
    for suffix in suffixes:
        if not suffix:
            continue
        suffix = suffix.lower()
        if not suffix.startswith("."):
            suffix = "." + suffix
        normalized.append(suffix)
    return tuple(dict.fromkeys(normalized))


def posix_rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def load_concepts(prompt_config: Path, skip_deleted: bool = False) -> List[Dict[str, Any]]:
    data = load_json(prompt_config)
    concepts: List[Dict[str, Any]] = []
    used_keys = set()

    for group_idx, group in enumerate(data.get("groups", [])):
        group_id = group.get("group_id", str(group_idx))
        for concept_idx, concept in enumerate(group.get("concepts", [])):
            if skip_deleted and concept.get("isdeleted"):
                continue
            concept_id = str(concept.get("concept_id") or f"{group_id}_{concept_idx}")
            concept_key = concept_id
            if concept_key in used_keys:
                concept_key = f"{concept_id}#{concept_idx}"
            used_keys.add(concept_key)

            labels = [int(value) for value in concept.get("tnls_dl3", [])]
            concepts.append(
                {
                    "key": concept_key,
                    "concept_id": concept_id,
                    "concept_name": concept.get("concept_name") or concept_id,
                    "group_id": group_id,
                    "group_name": group.get("group_name"),
                    "group_name_en": group.get("group_name_en"),
                    "tnls_dl3": labels,
                    "prompts": concept.get("prompts", []),
                    "isdeleted": bool(concept.get("isdeleted", False)),
                }
            )
    return concepts


def iter_files(root: Path, suffixes: Sequence[str], recursive: bool) -> Iterable[Path]:
    suffix_set = set(normalize_suffixes(suffixes))
    iterator = root.rglob("*") if recursive else root.glob("*")
    for path in iterator:
        if path.is_file() and path.suffix.lower() in suffix_set:
            yield path


def build_gt_index(
    gt_dir: Path, gt_suffixes: Sequence[str], recursive: bool
) -> Tuple[Dict[str, List[Path]], Dict[str, List[Path]]]:
    by_rel_stem: Dict[str, List[Path]] = {}
    by_name_stem: Dict[str, List[Path]] = {}
    for path in iter_files(gt_dir, gt_suffixes, recursive):
        rel_stem = path.relative_to(gt_dir).with_suffix("").as_posix()
        by_rel_stem.setdefault(rel_stem, []).append(path)
        by_name_stem.setdefault(path.stem, []).append(path)
    return by_rel_stem, by_name_stem


def unique_path(paths: Optional[List[Path]]) -> Optional[Path]:
    if paths and len(paths) == 1:
        return paths[0]
    return None


def locate_gt(
    image_path: Path,
    data_dir: Path,
    gt_dir: Path,
    gt_suffixes: Sequence[str],
    gt_by_rel_stem: Mapping[str, List[Path]],
    gt_by_name_stem: Mapping[str, List[Path]],
) -> Optional[Path]:
    rel = image_path.relative_to(data_dir)
    suffixes = normalize_suffixes((image_path.suffix, *gt_suffixes))

    for suffix in suffixes:
        candidate = gt_dir / rel.with_suffix(suffix)
        if candidate.is_file():
            return candidate

    rel_stem = rel.with_suffix("").as_posix()
    matched = unique_path(gt_by_rel_stem.get(rel_stem))
    if matched is not None:
        return matched

    return unique_path(gt_by_name_stem.get(image_path.stem))


def read_label_map(path: Path) -> np.ndarray:
    try:
        from PIL import Image

        with Image.open(path) as img:
            label = np.asarray(img)
    except ImportError:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "Install Pillow or opencv-python to read label maps."
            ) from exc
        label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if label is None:
            raise RuntimeError(f"Failed to read label map: {path}")

    if label.ndim == 3:
        if label.shape[2] == 1:
            label = label[:, :, 0]
        elif np.all(label == label[:, :, :1]):
            label = label[:, :, 0]
        else:
            raise ValueError(
                f"GT appears to be multi-channel/color-coded, not label ids: {path}"
            )
    return label


def label_counts_for_image(
    gt_path: Path,
    tracked_label_ids: Optional[Sequence[int]],
    ignore_index: Optional[int],
) -> Tuple[int, Dict[str, int]]:
    label = read_label_map(gt_path)
    if ignore_index is None:
        valid_values = label.reshape(-1)
    else:
        valid_values = label[label != ignore_index]
    valid_pixel_count = int(valid_values.size)
    if valid_pixel_count == 0:
        return 0, {}

    values, counts = np.unique(valid_values, return_counts=True)
    all_counts = {int(value): int(count) for value, count in zip(values, counts)}
    if tracked_label_ids is None:
        kept = all_counts
    else:
        tracked = set(int(value) for value in tracked_label_ids)
        kept = {label_id: count for label_id, count in all_counts.items() if label_id in tracked}
    return valid_pixel_count, {str(label_id): count for label_id, count in sorted(kept.items())}


def empty_accumulator() -> Dict[str, Any]:
    return {
        "positive_samples": 0,
        "negative_samples": 0,
        "positive_pixel_count": 0,
        "positive_valid_pixel_count": 0,
        "_positive_ratio_sum": 0.0,
    }


def add_image_to_accumulator(
    accumulator: Dict[str, Any], positive_pixels: int, valid_pixels: int
) -> None:
    if valid_pixels <= 0:
        return
    if positive_pixels > 0:
        accumulator["positive_samples"] += 1
        accumulator["positive_pixel_count"] += int(positive_pixels)
        accumulator["positive_valid_pixel_count"] += int(valid_pixels)
        accumulator["_positive_ratio_sum"] += float(positive_pixels) / float(valid_pixels)
    else:
        accumulator["negative_samples"] += 1


def finalize_accumulator(accumulator: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(accumulator)
    ratio_sum = float(result.pop("_positive_ratio_sum", 0.0))
    positive_samples = int(result["positive_samples"])
    positive_valid_pixels = int(result["positive_valid_pixel_count"])
    if positive_valid_pixels > 0:
        result["positive_pixel_ratio"] = (
            float(result["positive_pixel_count"]) / float(positive_valid_pixels)
        )
    else:
        result["positive_pixel_ratio"] = None
    if positive_samples > 0:
        result["mean_positive_pixel_ratio"] = ratio_sum / float(positive_samples)
    else:
        result["mean_positive_pixel_ratio"] = None
    result["total_samples"] = int(result["positive_samples"]) + int(result["negative_samples"])
    return result


def stats_for_label_set(
    image_records: Sequence[Mapping[str, Any]], label_ids: Sequence[int]
) -> Dict[str, Any]:
    label_keys = [str(int(label_id)) for label_id in label_ids]
    accumulator = empty_accumulator()
    for record in image_records:
        valid_pixels = int(record.get("valid_pixels", 0))
        label_counts = record.get("label_counts", {})
        positive_pixels = sum(int(label_counts.get(label_key, 0)) for label_key in label_keys)
        add_image_to_accumulator(accumulator, positive_pixels, valid_pixels)
    return finalize_accumulator(accumulator)


def summarize_images(
    image_records: Sequence[Mapping[str, Any]],
    concepts: Sequence[Mapping[str, Any]],
    tracked_label_ids: Sequence[int],
) -> Dict[str, Any]:
    per_label = {
        str(label_id): stats_for_label_set(image_records, [label_id])
        for label_id in sorted(int(value) for value in tracked_label_ids)
    }
    per_concept = {}
    for concept in concepts:
        stats = stats_for_label_set(image_records, concept.get("tnls_dl3", []))
        stats.update(
            {
                "concept_id": concept["concept_id"],
                "concept_name": concept["concept_name"],
                "tnls_dl3": concept.get("tnls_dl3", []),
                "isdeleted": concept.get("isdeleted", False),
            }
        )
        per_concept[concept["key"]] = stats
    return {"tnls_dl3": per_label, "concepts": per_concept}


def enrich_concepts_meta(
    concepts: Sequence[Mapping[str, Any]],
    concept_stats: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    enriched = []
    for concept in concepts:
        item = dict(concept)
        stats = concept_stats.get(str(concept.get("key")), {})
        item.update(
            {
                "positive_samples": int(stats.get("positive_samples", 0)),
                "negative_samples": int(stats.get("negative_samples", 0)),
                "positive_pixel_ratio": stats.get("positive_pixel_ratio"),
                "positive_pixel_count": int(stats.get("positive_pixel_count", 0)),
                "positive_valid_pixel_count": int(
                    stats.get("positive_valid_pixel_count", 0)
                ),
                "mean_positive_pixel_ratio": stats.get(
                    "mean_positive_pixel_ratio"
                ),
                "total_samples": int(stats.get("total_samples", 0)),
            }
        )
        enriched.append(item)
    return enriched


def scan_data_dir(
    data_dir: Path,
    gt_dir: Path,
    concepts: Sequence[Mapping[str, Any]],
    tracked_label_ids: Optional[Sequence[int]],
    image_suffixes: Sequence[str],
    gt_suffixes: Sequence[str],
    recursive: bool,
    ignore_index: Optional[int],
    strict_missing_gt: bool,
    gt_by_rel_stem: Mapping[str, List[Path]],
    gt_by_name_stem: Mapping[str, List[Path]],
) -> Dict[str, Any]:
    image_paths = sorted(iter_files(data_dir, image_suffixes, recursive))
    image_records: List[Dict[str, Any]] = []
    missing_gt: List[str] = []
    errors: List[Dict[str, str]] = []
    discovered_labels = set(int(value) for value in tracked_label_ids or [])

    for image_path in image_paths:
        gt_path = locate_gt(
            image_path,
            data_dir,
            gt_dir,
            gt_suffixes,
            gt_by_rel_stem,
            gt_by_name_stem,
        )
        image_rel = posix_rel(image_path, data_dir)
        if gt_path is None:
            missing_gt.append(image_rel)
            continue
        try:
            valid_pixels, label_counts = label_counts_for_image(
                gt_path, tracked_label_ids, ignore_index
            )
        except Exception as exc:
            errors.append(
                {
                    "image": image_rel,
                    "gt": str(gt_path),
                    "error": str(exc),
                }
            )
            continue
        discovered_labels.update(int(label_id) for label_id in label_counts)
        image_records.append(
            {
                "image": image_rel,
                "gt": posix_rel(gt_path, gt_dir),
                "valid_pixels": valid_pixels,
                "label_counts": label_counts,
            }
        )

    if strict_missing_gt and missing_gt:
        raise FileNotFoundError(
            f"{len(missing_gt)} images in {data_dir} have no matched GT, "
            f"first missing: {missing_gt[0]}"
        )

    effective_labels = sorted(discovered_labels)
    summary = summarize_images(image_records, concepts, effective_labels)
    valid_image_count = sum(1 for record in image_records if int(record["valid_pixels"]) > 0)

    result = {
        "data_dir": str(data_dir),
        "image_count": len(image_paths),
        "matched_gt_count": len(image_records),
        "missing_gt_count": len(missing_gt),
        "error_count": len(errors),
        "valid_image_count": valid_image_count,
        "zero_valid_pixel_count": len(image_records) - valid_image_count,
        "missing_gt": missing_gt,
        "errors": errors,
        "images": image_records,
    }
    result.update(summary)
    return result


def combine_dir_records(per_data_dir: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    combined: List[Mapping[str, Any]] = []
    for data_dir_stats in per_data_dir:
        prefix = data_dir_stats.get("data_dir", "")
        for record in data_dir_stats.get("images", []):
            enriched = dict(record)
            enriched["data_dir"] = prefix
            combined.append(enriched)
    return combined


def label_ids_in_data_dir(data_dir_stats: Mapping[str, Any]) -> List[int]:
    return sorted(
        {
            int(label_id)
            for record in data_dir_stats.get("images", [])
            for label_id in record.get("label_counts", {})
        }
    )


def count_summary(per_data_dir: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    names = (
        "image_count",
        "matched_gt_count",
        "missing_gt_count",
        "error_count",
        "valid_image_count",
        "zero_valid_pixel_count",
    )
    return {
        name: sum(int(stats.get(name, 0)) for stats in per_data_dir)
        for name in names
    }


def build_data_dir_result(
    data_dir_stats: Mapping[str, Any],
    concepts: Sequence[Mapping[str, Any]],
    label_ids: Sequence[int],
    count_all_labels: bool,
) -> Dict[str, Any]:
    per_data_dir = [dict(data_dir_stats)]
    combined_images = combine_dir_records(per_data_dir)
    combined = summarize_images(combined_images, concepts, label_ids)
    combined.update(count_summary(per_data_dir))
    concepts_meta = enrich_concepts_meta(concepts, combined.get("concepts", {}))
    return {
        "data_dir": data_dir_stats.get("data_dir"),
        "tracked_label_ids": list(label_ids),
        "count_all_labels": bool(count_all_labels),
        "concepts_meta": concepts_meta,
        "combined": combined,
        "per_data_dir": per_data_dir,
    }


def build_cache(args: argparse.Namespace) -> Dict[str, Any]:
    prompt_config = Path(args.prompt_config).resolve()
    gt_dir = Path(args.gt_dir).resolve()
    data_dirs = [Path(path).resolve() for path in args.data_dirs]
    image_suffixes = normalize_suffixes(args.image_suffixes)
    gt_suffixes = normalize_suffixes(args.gt_suffixes)

    if not prompt_config.is_file():
        raise FileNotFoundError(f"Prompt config not found: {prompt_config}")
    if not gt_dir.is_dir():
        raise NotADirectoryError(f"GT dir not found: {gt_dir}")
    for data_dir in data_dirs:
        if not data_dir.is_dir():
            raise NotADirectoryError(f"Data dir not found: {data_dir}")

    concepts = load_concepts(prompt_config, skip_deleted=args.skip_deleted)
    concept_label_ids = sorted(
        {
            int(label_id)
            for concept in concepts
            for label_id in concept.get("tnls_dl3", [])
        }
    )
    tracked_label_ids: Optional[List[int]] = None if args.count_all_labels else concept_label_ids
    gt_by_rel_stem, gt_by_name_stem = build_gt_index(gt_dir, gt_suffixes, args.recursive)

    data_dir_results = []
    for data_dir in data_dirs:
        data_dir_stats = scan_data_dir(
            data_dir=data_dir,
            gt_dir=gt_dir,
            concepts=concepts,
            tracked_label_ids=tracked_label_ids,
            image_suffixes=image_suffixes,
            gt_suffixes=gt_suffixes,
            recursive=args.recursive,
            ignore_index=args.ignore_index,
            strict_missing_gt=args.strict_missing_gt,
            gt_by_rel_stem=gt_by_rel_stem,
            gt_by_name_stem=gt_by_name_stem,
        )
        available_label_ids = (
            label_ids_in_data_dir(data_dir_stats)
            if args.count_all_labels
            else concept_label_ids
        )
        data_dir_results.append(
            build_data_dir_result(
                data_dir_stats=data_dir_stats,
                concepts=concepts,
                label_ids=available_label_ids,
                count_all_labels=args.count_all_labels,
            )
        )

    return {
        "cache_version": CACHE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt_config": str(prompt_config),
        "gt_dir": str(gt_dir),
        "data_dir_paths": [str(path) for path in data_dirs],
        "image_suffixes": list(image_suffixes),
        "gt_suffixes": list(gt_suffixes),
        "recursive": bool(args.recursive),
        "ignore_index": args.ignore_index,
        "skip_deleted": bool(args.skip_deleted),
        "count_all_labels": bool(args.count_all_labels),
        "data_dirs": data_dir_results,
    }


def cache_data_dir_results(cache: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    data_dirs = cache.get("data_dirs", [])
    if (
        isinstance(data_dirs, list)
        and data_dirs
        and isinstance(data_dirs[0], MappingABC)
        and "per_data_dir" in data_dirs[0]
    ):
        return data_dirs
    if "per_data_dir" in cache:
        return [cache]
    return []


def concepts_meta_for_result(
    data_dir_result: Mapping[str, Any],
    parent_cache: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    concepts = data_dir_result.get("concepts_meta")
    if concepts is None and parent_cache is not None:
        concepts = parent_cache.get("concepts_meta", [])
    concepts = list(concepts or [])

    concept_stats = data_dir_result.get("combined", {}).get("concepts", {})
    if not concept_stats and parent_cache is not None:
        concept_stats = parent_cache.get("combined", {}).get("concepts", {})
    return enrich_concepts_meta(concepts, concept_stats)


def ensure_query_labels_available(
    data_dir_result: Mapping[str, Any],
    label_ids: Sequence[int],
    parent_cache: Optional[Mapping[str, Any]] = None,
) -> None:
    count_all_labels = data_dir_result.get("count_all_labels")
    if count_all_labels is None and parent_cache is not None:
        count_all_labels = parent_cache.get("count_all_labels")
    if count_all_labels:
        return

    tracked_values = data_dir_result.get("tracked_label_ids")
    if tracked_values is None and parent_cache is not None:
        tracked_values = parent_cache.get("tracked_label_ids", [])
    tracked = {int(value) for value in tracked_values or []}
    missing = sorted(set(int(value) for value in label_ids) - tracked)
    if missing:
        raise ValueError(
            "Query labels are not in the cache: "
            f"{missing}. Rebuild with a prompt config containing them or use --count-all-labels."
        )


def query_cache(cache: Mapping[str, Any], label_ids: Sequence[int]) -> Dict[str, Any]:
    labels = sorted(dict.fromkeys(int(value) for value in label_ids))
    data_dir_results = cache_data_dir_results(cache)
    if not data_dir_results:
        raise ValueError("Cache has no data_dirs/per_data_dir results to query.")

    query_results = []
    for data_dir_result in data_dir_results:
        ensure_query_labels_available(data_dir_result, labels, cache)
        per_data_dir = []
        all_records: List[Mapping[str, Any]] = []

        for data_dir_stats in data_dir_result.get("per_data_dir", []):
            records = data_dir_stats.get("images", [])
            all_records.extend(records)
            per_label = {
                str(label_id): stats_for_label_set(records, [label_id])
                for label_id in labels
            }
            per_data_dir.append(
                {
                    "data_dir": data_dir_stats.get("data_dir"),
                    "summary": stats_for_label_set(records, labels),
                    "tnls_dl3": per_label,
                }
            )

        query_results.append(
            {
                "data_dir": data_dir_result.get("data_dir"),
                "concepts_meta": concepts_meta_for_result(data_dir_result, cache),
                "combined": {
                    "summary": stats_for_label_set(all_records, labels),
                    "tnls_dl3": {
                        str(label_id): stats_for_label_set(all_records, [label_id])
                        for label_id in labels
                    },
                },
                "per_data_dir": per_data_dir,
            }
        )

    return {"query": {"tnls_dl3": labels}, "data_dirs": query_results}


def format_ratio(value: Optional[float]) -> str:
    if value is None:
        return "nan"
    return f"{value:.6f}"


def print_stat_line(label: str, stats: Mapping[str, Any]) -> None:
    print(
        f"{label}: pos={stats['positive_samples']} "
        f"neg={stats['negative_samples']} "
        f"pos_px_ratio={format_ratio(stats.get('positive_pixel_ratio'))} "
        f"mean_pos_px_ratio={format_ratio(stats.get('mean_positive_pixel_ratio'))}"
    )


def print_build_summary(cache: Mapping[str, Any]) -> None:
    data_dir_results = cache_data_dir_results(cache)
    print("Built TNLS DL3 stats cache")
    print(f"data_dirs={len(data_dir_results)}")
    for idx, data_dir_result in enumerate(data_dir_results, start=1):
        combined = data_dir_result["combined"]
        print(f"[{idx}/{len(data_dir_results)}] {data_dir_result.get('data_dir')}")
        print(
            f"images={combined['image_count']} matched_gt={combined['matched_gt_count']} "
            f"missing_gt={combined['missing_gt_count']} errors={combined['error_count']} "
            f"valid_images={combined['valid_image_count']}"
        )
        print("Concept summary:")
        for concept in data_dir_result.get("concepts_meta", []):
            labels = ",".join(str(value) for value in concept.get("tnls_dl3", []))
            deleted = " deleted" if concept.get("isdeleted") else ""
            print_stat_line(
                f"  {concept.get('key', concept.get('concept_id'))}[{labels}]{deleted}",
                concept,
            )


def print_query_summary(query: Mapping[str, Any]) -> None:
    labels = query["query"]["tnls_dl3"]
    print(f"Query tnls_dl3={labels}")
    data_dir_results = query.get("data_dirs", [])
    for idx, data_dir_result in enumerate(data_dir_results, start=1):
        print(f"[{idx}/{len(data_dir_results)}] {data_dir_result.get('data_dir')}")
        print_stat_line("combined", data_dir_result["combined"]["summary"])
        print("Per id:")
        for label_id, stats in data_dir_result["combined"].get("tnls_dl3", {}).items():
            print_stat_line(f"  {label_id}", stats)
        if len(data_dir_result.get("per_data_dir", [])) > 1:
            print("Per data dir:")
            for data_dir_stats in data_dir_result["per_data_dir"]:
                print_stat_line(
                    f"  {data_dir_stats['data_dir']}",
                    data_dir_stats["summary"],
                )


def labels_from_query_args(args: argparse.Namespace) -> Optional[List[int]]:
    labels: Optional[List[int]] = None
    if args.query_tnls_dl3:
        labels = [int(value) for value in args.query_tnls_dl3]
    if args.query_json:
        loaded = load_json_arg(args.query_json)
        if not isinstance(loaded, dict) or "tnls_dl3" not in loaded:
            raise ValueError("--query-json must contain {'tnls_dl3': [...]}")
        query_labels = [int(value) for value in loaded["tnls_dl3"]]
        labels = query_labels if labels is None else labels + query_labels
    if labels is None:
        return None
    return sorted(dict.fromkeys(labels))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count image-level TNLS DL3 positives/negatives and positive pixel ratios."
    )
    parser.add_argument("--prompt-config", help="Prompt concept JSON, e.g. configs/cls_*.json")
    parser.add_argument(
        "--data-dirs",
        nargs="*",
        default=[],
        metavar="DATA_DIR",
        help=(
            "Image folders to scan. Accepts multiple string args after "
            "--data-dirs until the next option."
        ),
    )
    parser.add_argument("--gt-dir", help="Folder containing GT label maps.")
    parser.add_argument(
        "--output-json",
        help="Where to write the built intermediate cache JSON.",
    )
    parser.add_argument(
        "--cache-json",
        help="Existing cache for query mode, or output cache path when building.",
    )
    parser.add_argument(
        "--query-tnls-dl3",
        nargs="+",
        type=int,
        help="Query label ids from a cache or newly built stats, e.g. 20 21 22.",
    )
    parser.add_argument(
        "--query-json",
        help="Inline JSON or JSON file containing {'tnls_dl3': [20, 21, 22]}.",
    )
    parser.add_argument(
        "--query-output-json",
        help="Optional path to save query results.",
    )
    parser.add_argument(
        "--image-suffixes",
        nargs="+",
        default=list(DEFAULT_IMAGE_SUFFIXES),
        help="Image suffixes to scan.",
    )
    parser.add_argument(
        "--gt-suffixes",
        nargs="+",
        default=list(DEFAULT_GT_SUFFIXES),
        help="GT suffixes to match.",
    )
    parser.add_argument(
        "--ignore-index",
        type=int,
        default=255,
        help="Ignore label id. Use --ignore-index -1 if there is no ignore id.",
    )
    parser.add_argument(
        "--count-all-labels",
        action="store_true",
        help="Cache every label id found in GT, not only ids from the prompt config.",
    )
    parser.add_argument(
        "--skip-deleted",
        action="store_true",
        help="Skip concepts whose JSON entry has isdeleted=true.",
    )
    parser.add_argument(
        "--strict-missing-gt",
        action="store_true",
        help="Fail if any image has no matched GT.",
    )
    parser.add_argument(
        "--recursive",
        dest="recursive",
        action="store_true",
        default=True,
        help="Scan image and GT folders recursively (default).",
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Only scan direct children of image and GT folders.",
    )

    args = parser.parse_args(argv)

    query_labels = labels_from_query_args(args)
    args.query_labels = query_labels

    wants_build = bool(args.data_dirs or args.prompt_config or args.gt_dir)
    if wants_build:
        missing = [
            name
            for name, value in (
                ("--prompt-config", args.prompt_config),
                ("--data-dirs", args.data_dirs),
                ("--gt-dir", args.gt_dir),
            )
            if not value
        ]
        if missing:
            parser.error(f"Build mode requires: {', '.join(missing)}")
    elif not args.cache_json:
        parser.error("Provide build inputs or --cache-json for query mode.")

    if not wants_build and query_labels is None:
        parser.error("Query mode requires --query-tnls-dl3 or --query-json.")

    if args.ignore_index < 0:
        args.ignore_index = None
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.data_dirs:
            cache = build_cache(args)
            output_json = args.output_json or args.cache_json
            if output_json:
                dump_json(cache, Path(output_json))
            print_build_summary(cache)
            if output_json:
                print(f"Saved cache: {output_json}")
        else:
            cache = load_json(Path(args.cache_json))

        if args.query_labels is not None:
            query = query_cache(cache, args.query_labels)
            print_query_summary(query)
            if args.query_output_json:
                dump_json(query, Path(args.query_output_json))
                print(f"Saved query: {args.query_output_json}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
