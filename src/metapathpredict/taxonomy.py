"""
Taxonomic lineages of genomes, from NCBI Taxonomy, cached in a JSON file.

A species-disjoint split still leaves close relatives on both sides of it (a test genome whose genus
or family is in the training set), which makes a model look better than it is on new organisms. The
lineages here let the split keep whole families together and let evaluation say how related each test
genome is to the training data.

Cache format: {"<taxid>": {"phylum": ..., "class": ..., "order": ..., "family": ..., "genus": ...}}.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

RANKS = ("phylum", "class", "order", "family", "genus")
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def parse_taxonomy_xml(xml: bytes) -> dict[str, dict[str, str | None]]:
    """Lineage (by rank) of every taxon in an efetch db=taxonomy response, also under merged ids."""
    lineages: dict[str, dict[str, str | None]] = {}
    for taxon in ET.fromstring(xml).findall("Taxon"):
        names = {node.findtext("Rank"): node.findtext("ScientificName") for node in taxon.findall("LineageEx/Taxon")}
        names[taxon.findtext("Rank")] = taxon.findtext("ScientificName")
        entry = {rank: names.get(rank) for rank in RANKS}
        lineages[taxon.findtext("TaxId")] = entry
        for alias in taxon.findall("AkaTaxIds/TaxId"):
            lineages[alias.text] = entry
    return lineages


def _efetch(taxids: list[str]) -> bytes:
    body = urllib.parse.urlencode({"db": "taxonomy", "id": ",".join(taxids), "retmode": "xml"}).encode()
    request = urllib.request.Request(EFETCH, data=body, headers={"User-Agent": "metapathpredict/0.1"})
    return urllib.request.urlopen(request, timeout=90).read()


def load_lineages(path: str | Path) -> dict[str, dict[str, str | None]]:
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


def fetch_lineages(taxids: Iterable[str | int], cache_path: str | Path, batch: int = 150, pause: float = 0.5) -> dict:
    """
    Lineages for `taxids`; only ids missing from the cache are requested from NCBI, and the cache
    file is updated. Returns the whole cache.
    """
    cache_path = Path(cache_path)
    cache = load_lineages(cache_path)
    missing = sorted({str(t) for t in taxids} - set(cache))
    if missing:
        logger.info(f"Fetching {len(missing)} lineages from NCBI Taxonomy")
    for start in range(0, len(missing), batch):
        cache.update(parse_taxonomy_xml(_efetch(missing[start:start + batch])))
        time.sleep(pause)
    if missing:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache))
        tmp.replace(cache_path)
    return cache


def lineage_of(lineages: dict, taxid: str | int) -> dict[str, str | None]:
    return lineages.get(str(taxid)) or {rank: None for rank in RANKS}


# Columns in split_assignments.tsv. Prefixed because that file already has a `class` column (one of
# the 8 taxonomic classes of the dataset) which the NCBI rank "class" must not overwrite.
LINEAGE_COLUMNS = {rank: f"lineage_{rank}" for rank in RANKS}


def lineage_columns(lineages: dict, taxid: str | int) -> dict[str, str]:
    """{"lineage_phylum": ..., ...} for one genome; unknown ranks are empty strings."""
    lineage = lineage_of(lineages, taxid)
    return {LINEAGE_COLUMNS[rank]: lineage[rank] or "" for rank in RANKS}


def parse_lineage_strings(xml: bytes) -> dict[str, str]:
    """Full lineage string ("cellular organisms; Eukaryota; Sar; ...") of every taxon in an efetch response."""
    strings: dict[str, str] = {}
    for taxon in ET.fromstring(xml).findall("Taxon"):
        text = taxon.findtext("Lineage") or ""
        strings[taxon.findtext("TaxId")] = text
        for alias in taxon.findall("AkaTaxIds/TaxId"):
            strings[alias.text] = text
    return strings


def fetch_full_lineages(taxids: Iterable[str | int], cache_path: str | Path, batch: int = 150, pause: float = 0.5) -> dict[str, str]:
    """Full lineage strings for `taxids`, cached like fetch_lineages (only missing ids are requested)."""
    cache_path = Path(cache_path)
    cache = load_lineages(cache_path)
    missing = sorted({str(t) for t in taxids} - set(cache))
    for start in range(0, len(missing), batch):
        cache.update(parse_lineage_strings(_efetch(missing[start:start + batch])))
        time.sleep(pause)
    if missing:
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache))
        tmp.replace(cache_path)
    return cache
