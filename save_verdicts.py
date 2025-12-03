import os
import json
import csv
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List

def _ensure_dir_for_file(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def select_best_evidence(evidence_list: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Heuristics to pick the best evidence from evidence_list (as returned by your pipeline).
    - Prefer items with highest retrieval_score.
    - Prefer items whose nli_label is not 'NEI'.
    - Tie-breaker: higher nli_confidence.
    - Returns the chosen evidence dict or None if evidence_list empty.
    """
    if not evidence_list:
        return None

    # Normalize fields that may have slightly different names in different versions
    normalized = []
    for ev in evidence_list:
        evc = dict(ev)  # copy
        evc.setdefault('retrieval_score', evc.get('score', evc.get('retrieval_score', 0.0)))
        evc.setdefault('nli_label', evc.get('nli_label', evc.get('label', None)))
        evc.setdefault('nli_confidence', float(evc.get('nli_confidence', evc.get('confidence', 0.0) or 0.0)))
        normalized.append(evc)

    # Sort: prefer nli_label != 'NEI', then retrieval_score desc, then nli_conf desc
    def key_fn(e):
        is_nei = 1 if (e.get('nli_label') in (None, 'NEI')) else 0
        return (is_nei, -float(e.get('retrieval_score', 0.0)), -float(e.get('nli_confidence', 0.0)))

    sorted_list = sorted(normalized, key=key_fn)
    return sorted_list[0]

def build_record(claim: str, label: str, confidence: float, best_ev: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a standardized record dict to persist.
    """
    record = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "claim": claim,
        "label": label,
        "label_confidence": float(confidence or 0.0),
    }
    if best_ev:
        record.update({
            "evidence_title": best_ev.get("title", ""),
            "evidence_url": best_ev.get("url", ""),
            "evidence_source": best_ev.get("source", best_ev.get("domain", "")),
            "evidence_excerpt": best_ev.get("excerpt", best_ev.get("snippet", "")),
            "evidence_retrieval_score": float(best_ev.get("retrieval_score", best_ev.get("score", 0.0) or 0.0)),
            "evidence_nli_label": best_ev.get("nli_label", best_ev.get("label", "")),
            "evidence_nli_confidence": float(best_ev.get("nli_confidence", best_ev.get("confidence", 0.0) or 0.0))
        })
    else:
        record.update({
            "evidence_title": "",
            "evidence_url": "",
            "evidence_source": "",
            "evidence_excerpt": "",
            "evidence_retrieval_score": 0.0,
            "evidence_nli_label": "",
            "evidence_nli_confidence": 0.0
        })
    return record

def save_record_jsonl(record: Dict[str, Any], path: str = "data/verdicts.jsonl"):
    """
    Append a record as a JSON line.
    """
    _ensure_dir_for_file(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def save_record_csv(record: Dict[str, Any], path: str = "data/verdicts.csv"):
    """
    Append a record as a row in CSV. If file doesn't exist, write header.
    """
    _ensure_dir_for_file(path)
    fieldnames = [
        "id", "timestamp", "claim", "label", "label_confidence",
        "evidence_title", "evidence_url", "evidence_source", "evidence_excerpt",
        "evidence_retrieval_score", "evidence_nli_label", "evidence_nli_confidence"
    ]
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        row = {k: record.get(k, "") for k in fieldnames}
        writer.writerow(row)

def save_verdict(claim: str, adaptive_result: Dict[str, Any],
                 jsonl_path: str = "data/verdicts.jsonl",
                 csv_path: str = "data/verdicts.csv"):
    """
    High-level helper: pick best evidence and save to both JSONL and CSV.
    adaptive_result is the dict returned by adaptive_predict (with keys 'label','confidence','evidence' ...)
    """
    label = adaptive_result.get("label", "")
    confidence = float(adaptive_result.get("confidence", 0.0) or 0.0)
    evidence_list = adaptive_result.get("evidence", []) or []

    best_ev = select_best_evidence(evidence_list)
    rec = build_record(claim, label, confidence, best_ev)

    save_record_jsonl(rec, jsonl_path)
    save_record_csv(rec, csv_path)
    return rec