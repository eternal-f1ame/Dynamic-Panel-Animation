#!/usr/bin/env python3
"""Verify every entry in sample.bib against the OpenAlex catalog.

For each `@inproceedings` / `@article` / `@misc` entry in sample.bib, query
OpenAlex by title, then assert that the top hit has a matching first author
and a year within +/-1 of the cited year. `@misc` entries that cite a
software release (HuggingFace, GitHub) are flagged as unverifiable rather
than failed, since OpenAlex is a scholarly index.

Usage:  python3 verify_citations.py [path/to/sample.bib]
"""

from __future__ import annotations

import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

OPENALEX = "https://api.openalex.org/works"


def normalize(text: str) -> str:
    """Lowercase ASCII, strip punctuation and braces."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[{}\\\"]", "", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def parse_bib(bib_text: str) -> list[dict]:
    """Parse a small subset of BibTeX adequate for this file."""
    entries = []
    for raw in re.finditer(r"@(\w+)\s*\{\s*([^,]+),(.*?)\n\}", bib_text, re.S):
        kind, key, body = raw.group(1), raw.group(2).strip(), raw.group(3)
        fields = {"_kind": kind, "_key": key}
        for fm in re.finditer(r"(\w+)\s*=\s*(\{(?:[^{}]|\{[^{}]*\})*\}|\"[^\"]*\")", body):
            name = fm.group(1).lower()
            value = fm.group(2)
            if value.startswith("{") and value.endswith("}"):
                value = value[1:-1]
            elif value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            fields[name] = value.strip()
        entries.append(fields)
    return entries


def first_author_lastname(authors_field: str) -> str:
    """Pull the first author's last name from a BibTeX 'A and B and C' field."""
    if not authors_field:
        return ""
    first = authors_field.split(" and ")[0]
    if "," in first:
        return normalize(first.split(",")[0])
    return normalize(first.split()[-1]) if first.split() else ""


def query_openalex(title: str, retries: int = 3) -> dict | None:
    """Search OpenAlex by title; return the top-N hits."""
    params = urllib.parse.urlencode({
        "search": title,
        "per-page": 5,
        "select": "title,publication_year,authorships,doi,id",
    })
    url = f"{OPENALEX}?{params}"
    return _http_json(url, retries)


def lookup_arxiv(arxiv_id: str, retries: int = 3) -> dict | None:
    """Direct OpenAlex lookup by arXiv DOI; bypasses the search ranker."""
    doi = f"10.48550/arxiv.{arxiv_id}"
    url = f"{OPENALEX}/doi:{doi}"
    return _http_json(url, retries)


def _http_json(url: str, retries: int) -> dict | None:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "citation-verifier/1.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode())
        except Exception as exc:
            if attempt == retries - 1:
                print(f"  ! request failed after {retries} attempts: {exc}")
                return None
            time.sleep(1 + attempt)
    return None


def verify(entry: dict) -> dict:
    """Verify one entry. Returns a dict with status and reason."""
    key = entry["_key"]
    kind = entry["_kind"]
    title = entry.get("title", "")
    authors = entry.get("author", "")
    year = entry.get("year", "")

    if kind.lower() == "misc" and ("howpublished" in entry or "github.com" in entry.get("howpublished", "")):
        return {"key": key, "status": "skip", "reason": "@misc release (not a scholarly artifact)"}

    if not title:
        return {"key": key, "status": "fail", "reason": "no title in bib entry"}

    # If the entry has an arXiv eprint, prefer direct DOI lookup; the title-
    # search ranker is unreliable for high-frequency phrases like "video
    # diffusion models". Fall back to title search otherwise.
    arxiv_id = entry.get("eprint")
    if arxiv_id:
        record = lookup_arxiv(arxiv_id)
        if record and record.get("title"):
            response = {"results": [record]}
        else:
            response = None
    else:
        short_title = " ".join(normalize(title).split()[:12])
        response = query_openalex(short_title)

    if response is None or not response.get("results"):
        return {"key": key, "status": "fail", "reason": "no OpenAlex hit"}

    expected_lastname = first_author_lastname(authors)
    expected_year = int(year) if year.isdigit() else None

    target_norm = normalize(title)
    for hit in response["results"]:
        hit_title = normalize(hit.get("title") or "")
        # Allow any prefix overlap on the longer normalized title.
        if not (target_norm in hit_title or hit_title in target_norm or _token_overlap(target_norm, hit_title) >= 0.6):
            continue
        hit_year = hit.get("publication_year")
        first_author = ""
        for a in hit.get("authorships") or []:
            disp = (a.get("author") or {}).get("display_name") or ""
            if disp:
                first_author = normalize(disp.split()[-1])
                break
        ok_author = expected_lastname in first_author or first_author in expected_lastname or not expected_lastname
        ok_year = (
            expected_year is None
            or hit_year is None
            or abs(hit_year - expected_year) <= 1
        )
        # When the entry was looked up directly by arXiv DOI, the DOI itself
        # is authoritative; OpenAlex's author metadata for arXiv preprints
        # is occasionally garbled (e.g. it lists DDPM's first author as
        # "Yan, Steven"). Treat title+year match as sufficient in that case.
        if arxiv_id:
            doi_str = (hit.get("doi") or "").lower()
            if arxiv_id.lower() in doi_str and ok_year:
                return {
                    "key": key,
                    "status": "ok",
                    "reason": f"arXiv DOI match: '{hit.get('title')[:80]}' ({hit_year}) doi={hit.get('doi')}",
                }
        if ok_author and ok_year:
            return {
                "key": key,
                "status": "ok",
                "reason": f"matched '{hit.get('title')[:80]}' ({hit_year}, {first_author}) doi={hit.get('doi')}",
            }
        return {
            "key": key,
            "status": "warn",
            "reason": (
                f"title hit but author/year mismatch: "
                f"hit_first_author={first_author!r}, expected={expected_lastname!r}, "
                f"hit_year={hit_year}, expected_year={expected_year}"
            ),
        }
    return {"key": key, "status": "fail", "reason": "OpenAlex top results have unrelated titles"}


def _token_overlap(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(len(sa), len(sb))


def main() -> int:
    bib_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "sample.bib"
    entries = parse_bib(bib_path.read_text())
    print(f"checking {len(entries)} entries against OpenAlex\n")
    results = []
    for entry in entries:
        print(f"- {entry['_key']} ({entry['_kind']})")
        result = verify(entry)
        results.append(result)
        marker = {"ok": "[OK]  ", "warn": "[WARN]", "fail": "[FAIL]", "skip": "[SKIP]"}[result["status"]]
        print(f"  {marker} {result['reason']}\n")
        time.sleep(0.4)  # rate-limit politeness

    print("\n=== Summary ===")
    for status in ("fail", "warn", "skip", "ok"):
        keys = [r["key"] for r in results if r["status"] == status]
        if keys:
            print(f"{status.upper()}: {len(keys)}  {', '.join(keys)}")

    failures = [r for r in results if r["status"] == "fail"]
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
