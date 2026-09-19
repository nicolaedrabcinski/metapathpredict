"""Lineage cache and novelty-driven genome selection."""

import importlib.util
import json
import random
from pathlib import Path

from metapathpredict import taxonomy

XML = b"""<TaxaSet>
<Taxon><TaxId>9606</TaxId><ScientificName>Homo sapiens</ScientificName><Rank>species</Rank>
  <AkaTaxIds><TaxId>1111</TaxId></AkaTaxIds>
  <LineageEx>
    <Taxon><ScientificName>Chordata</ScientificName><Rank>phylum</Rank></Taxon>
    <Taxon><ScientificName>Mammalia</ScientificName><Rank>class</Rank></Taxon>
    <Taxon><ScientificName>Hominidae</ScientificName><Rank>family</Rank></Taxon>
    <Taxon><ScientificName>Homo</ScientificName><Rank>genus</Rank></Taxon>
  </LineageEx></Taxon>
<Taxon><TaxId>7</TaxId><ScientificName>Some genus</ScientificName><Rank>genus</Rank><LineageEx></LineageEx></Taxon>
</TaxaSet>"""


def test_parse_reads_ranks_aliases_and_the_taxon_itself():
    lin = taxonomy.parse_taxonomy_xml(XML)
    assert lin["9606"] == {"phylum": "Chordata", "class": "Mammalia", "order": None, "family": "Hominidae", "genus": "Homo"}
    assert lin["1111"] == lin["9606"]  # merged id
    assert lin["7"]["genus"] == "Some genus" and lin["7"]["family"] is None  # rank of the taxon itself


def test_fetch_requests_only_missing_ids_and_updates_the_cache(tmp_path, monkeypatch):
    cache = tmp_path / "lineages.json"
    cache.write_text(json.dumps({"1": {r: "x" for r in taxonomy.RANKS}}))
    calls = []
    monkeypatch.setattr(taxonomy, "_efetch", lambda ids: calls.append(ids) or XML)
    result = taxonomy.fetch_lineages([1, 9606], cache, pause=0)
    assert calls == [["9606"]] and "9606" in result and "1" in result
    assert "9606" in json.loads(cache.read_text())
    taxonomy.fetch_lineages([1, 9606], cache, pause=0)
    assert len(calls) == 1  # everything cached now


def test_unknown_taxid_gives_an_empty_lineage():
    assert taxonomy.lineage_of({}, 5) == {rank: None for rank in taxonomy.RANKS}


def _downloader():
    spec = importlib.util.spec_from_file_location("dl", Path(__file__).resolve().parents[1] / "scripts/download_diverse_genomes.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _lin(family, order="O", genus=None):
    return {"phylum": "P", "class": "C", "order": order, "family": family, "genus": genus or family + "us"}


def test_selection_prefers_families_not_yet_in_the_manifest_and_caps_genera():
    dl = _downloader()
    lineages = {"1": _lin("A", genus="Ag"), "2": _lin("A", genus="Ah"),
                "3": _lin("A", genus="Ai"), "4": _lin("B"), "5": _lin("C"), "6": _lin("B", genus="Bx")}
    existing = [{"species_taxid": "1", "organism": "Ag one"}, {"species_taxid": "2", "organism": "Ah two"}]
    candidates = [{"species_taxid": str(i), "organism_name": f"sp{i}"} for i in (3, 4, 5, 6)]
    picked = dl.select_new_genomes(candidates, 2, existing, lineages, random.Random(0))
    assert {r["species_taxid"] for r in picked} <= {"4", "5", "6"}
    assert {lineages[r["species_taxid"]]["family"] for r in picked} == {"B", "C"}  # A is already covered twice


def test_selection_respects_the_genus_cap_and_the_requested_number():
    dl = _downloader()
    lineages = {str(i): _lin("F", genus="G") for i in range(1, 6)}
    candidates = [{"species_taxid": str(i), "organism_name": f"G sp{i}"} for i in range(1, 6)]
    picked = dl.select_new_genomes(candidates, 4, [], lineages, random.Random(0))
    assert len(picked) == dl.MAX_PER_GENUS  # only two per genus, even though four were wanted
