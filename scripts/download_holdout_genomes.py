#!/usr/bin/env python3
"""
Download NCBI RefSeq genomes NOT already present in data/input_large/,
for a genuinely held-out benchmark (as opposed to the train/val/test split,
which is held out at the fragment level but drawn from the same genomes the
model's augmentation/pretraining has already seen sequence context from).

Usage:
    python scripts/download_holdout_genomes.py --output data/holdout --per-group 8
"""

from __future__ import annotations

import argparse
import gzip
import logging
import re
import subprocess
from pathlib import Path
from urllib.request import urlretrieve

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REFSEQ_FTP = "https://ftp.ncbi.nlm.nih.gov/genomes/refseq"
ASSEMBLY_SUMMARY_URLS = {
    "bacteria": f"{REFSEQ_FTP}/bacteria/assembly_summary.txt",
    "fungi": f"{REFSEQ_FTP}/fungi/assembly_summary.txt",
    "viral": f"{REFSEQ_FTP}/viral/assembly_summary.txt",
}
# class label each group maps to, matching cli.py prepare_command's class_mapping
GROUP_TO_CLASS = {"bacteria": "bacteria", "fungi": "eukaryotic", "viral": "viral"}


def existing_accessions(input_dir: Path) -> set[str]:
    """Collect every accession already used in data/input_large/*.fasta."""
    seen = set()
    pattern = re.compile(r"^>(\S+)")
    for fasta in input_dir.glob("*.fasta"):
        with open(fasta) as f:
            for line in f:
                if line.startswith(">"):
                    m = pattern.match(line)
                    if m:
                        seen.add(m.group(1))
    logger.info(f"{len(seen)} accessions already used in {input_dir}")
    return seen


def parse_assembly_summary(summary_file: Path, exclude: set[str], n: int) -> list[dict]:
    genomes = []
    header = None
    with open(summary_file) as f:
        for line in f:
            stripped = line.lstrip("#").strip()
            if stripped.startswith("assembly_accession"):
                header = stripped.split("\t")
                continue
            if line.startswith("#") or header is None:
                continue
            fields = line.strip().split("\t")
            if len(fields) < len(header):
                fields.extend([""] * (len(header) - len(fields)))
            row = dict(zip(header, fields))
            if row.get("assembly_level", "") != "Complete Genome":
                continue
            ftp_path = row.get("ftp_path", "")
            if ftp_path in ("na", ""):
                continue
            accession = row.get("assembly_accession", "")
            if accession in exclude:
                continue
            genomes.append({
                "accession": accession,
                "organism": row.get("organism_name", ""),
                "ftp_path": ftp_path,
            })
            if len(genomes) >= n:
                break
    return genomes


def download_file(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(["wget", "-q", "-O", str(dest), url], timeout=120, capture_output=True)
        return result.returncode == 0 and dest.exists() and dest.stat().st_size > 0
    except Exception as e:
        logger.warning(f"Failed to download {url}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/holdout")
    parser.add_argument("--input-dir", default="data/input_large")
    parser.add_argument("--per-group", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = Path(args.input_dir)

    exclude = existing_accessions(input_dir)

    for group, class_name in GROUP_TO_CLASS.items():
        dl_dir = output_dir / "downloads" / group
        dl_dir.mkdir(parents=True, exist_ok=True)

        summary_file = dl_dir / "assembly_summary.txt"
        if not summary_file.exists():
            logger.info(f"Downloading assembly summary for {group}...")
            download_file(ASSEMBLY_SUMMARY_URLS[group], summary_file)

        genomes = parse_assembly_summary(summary_file, exclude, args.per_group)
        logger.info(f"Selected {len(genomes)} unseen {group} genomes")

        out_fasta = output_dir / f"{class_name}_holdout.fasta"
        with open(out_fasta, "w") as out:
            for i, genome in enumerate(genomes):
                ftp_path = genome["ftp_path"].replace("ftp://", "https://").rstrip("/")
                basename = ftp_path.split("/")[-1]
                genome_url = f"{ftp_path}/{basename}_genomic.fna.gz"
                local_gz = dl_dir / f"{genome['accession']}.fna.gz"

                if not local_gz.exists() or local_gz.stat().st_size == 0:
                    logger.info(f"  [{i+1}/{len(genomes)}] {genome['organism']} ({genome['accession']})")
                    if not download_file(genome_url, local_gz):
                        logger.warning(f"  Failed: {genome['accession']} ({genome_url})")
                        continue

                try:
                    with gzip.open(local_gz, "rt") as gz:
                        out.write(gz.read())
                except Exception as e:
                    logger.warning(f"  Failed to decompress {local_gz}: {e}")

        logger.info(f"Wrote {out_fasta}")


if __name__ == "__main__":
    main()
