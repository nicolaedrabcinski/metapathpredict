#!/usr/bin/env python3
"""
Download genomic data from NCBI RefSeq FTP for MetaPathPredict training.

Downloads bacterial, eukaryotic, and viral genomes directly from NCBI FTP,
then merges with existing data.

Usage:
    python scripts/download_ncbi_data.py --output data/input_large
    python scripts/download_ncbi_data.py --output data/input_large --merge-existing
"""

import argparse
import gzip
import logging
import os
import subprocess
import sys
from pathlib import Path
from urllib.request import urlretrieve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# NCBI FTP base URLs for RefSeq genomes
REFSEQ_FTP = "https://ftp.ncbi.nlm.nih.gov/genomes/refseq"

# Assembly summary files contain genome metadata + FTP paths
ASSEMBLY_SUMMARY_URLS = {
    "bacteria": f"{REFSEQ_FTP}/bacteria/assembly_summary.txt",
    "fungi": f"{REFSEQ_FTP}/fungi/assembly_summary.txt",
    "protozoa": f"{REFSEQ_FTP}/protozoa/assembly_summary.txt",
    "viral": f"{REFSEQ_FTP}/viral/assembly_summary.txt",
}


def download_file(url: str, dest: Path) -> bool:
    """Download a file using wget (supports FTP and HTTPS)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            ["wget", "-q", "-O", str(dest), url],
            timeout=120,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0 and dest.exists() and dest.stat().st_size > 0
    except (subprocess.TimeoutExpired, Exception) as e:
        logger.warning(f"Failed to download {url}: {e}")
        return False


def parse_assembly_summary(summary_file: Path, refseq_category: str = None,
                           assembly_level: str = "Complete Genome",
                           max_genomes: int = 500) -> list[dict]:
    """Parse NCBI assembly_summary.txt and return genome info."""
    genomes = []
    with open(summary_file) as f:
        header = None
        for line in f:
            # Header line starts with #assembly_accession (no space)
            stripped = line.lstrip("#").strip()
            if stripped.startswith("assembly_accession"):
                header = stripped.split("\t")
                continue
            if line.startswith("#"):
                continue
            if header is None:
                continue

            fields = line.strip().split("\t")
            if len(fields) < len(header):
                # Pad with empty strings
                fields.extend([""] * (len(header) - len(fields)))

            row = dict(zip(header, fields))

            # Filter by assembly level
            if assembly_level and row.get("assembly_level", "") != assembly_level:
                continue

            # Filter by refseq category
            if refseq_category:
                cat = row.get("refseq_category", "")
                if refseq_category not in cat:
                    continue

            ftp_path = row.get("ftp_path", "")
            if ftp_path == "na" or not ftp_path:
                continue

            genomes.append({
                "accession": row.get("assembly_accession", ""),
                "organism": row.get("organism_name", ""),
                "ftp_path": ftp_path,
            })

            if len(genomes) >= max_genomes:
                break

    return genomes


def download_genomes(group: str, output_dir: Path, max_genomes: int,
                     refseq_category: str = None,
                     assembly_level: str = "Complete Genome") -> Path:
    """Download genomes for a taxonomic group from NCBI FTP."""
    out_fasta = output_dir / f"{group}_refseq.fasta"
    if out_fasta.exists() and out_fasta.stat().st_size > 0:
        logger.info(f"{group} data already exists: {out_fasta}")
        return out_fasta

    dl_dir = output_dir / "downloads" / group
    dl_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download assembly summary
    summary_url = ASSEMBLY_SUMMARY_URLS.get(group)
    if not summary_url:
        logger.error(f"Unknown group: {group}")
        return out_fasta

    summary_file = dl_dir / "assembly_summary.txt"
    if not summary_file.exists() or summary_file.stat().st_size == 0:
        logger.info(f"Downloading assembly summary for {group}...")
        if not download_file(summary_url, summary_file):
            logger.error(f"Failed to download assembly summary for {group}")
            return out_fasta

    # Step 2: Parse and select genomes
    logger.info(f"Parsing assembly summary for {group}...")
    genomes = parse_assembly_summary(
        summary_file,
        refseq_category=refseq_category,
        assembly_level=assembly_level,
        max_genomes=max_genomes,
    )
    logger.info(f"Found {len(genomes)} {group} genomes matching criteria")

    if not genomes and refseq_category:
        # Fallback: try without category filter
        logger.info(f"Retrying without refseq_category filter...")
        genomes = parse_assembly_summary(
            summary_file,
            refseq_category=None,
            assembly_level=assembly_level,
            max_genomes=max_genomes,
        )
        logger.info(f"Found {len(genomes)} {group} genomes (no category filter)")

    if not genomes and assembly_level == "Complete Genome":
        # Fallback: try with Chromosome level too
        logger.info(f"Retrying with Chromosome level...")
        genomes = parse_assembly_summary(
            summary_file,
            refseq_category=None,
            assembly_level=None,  # any level
            max_genomes=max_genomes,
        )
        logger.info(f"Found {len(genomes)} {group} genomes (any level)")

    # Step 3: Download individual genomes
    logger.info(f"Downloading {len(genomes)} {group} genomes...")
    downloaded = 0
    with open(out_fasta, "w") as out:
        for i, genome in enumerate(genomes):
            ftp_path = genome["ftp_path"]
            # Convert FTP to HTTPS
            ftp_path = ftp_path.replace("ftp://", "https://")
            basename = ftp_path.split("/")[-1]
            genome_url = f"{ftp_path}/{basename}_genomic.fna.gz"

            local_gz = dl_dir / f"{genome['accession']}.fna.gz"

            if not local_gz.exists() or local_gz.stat().st_size == 0:
                logger.info(f"  [{i+1}/{len(genomes)}] Downloading {genome['organism']} ({genome['accession']})...")
                if not download_file(genome_url, local_gz):
                    logger.warning(f"  Failed: {genome['accession']}")
                    continue

            # Decompress and append
            try:
                with gzip.open(local_gz, "rt") as f:
                    for line in f:
                        out.write(line)
                downloaded += 1
            except Exception as e:
                logger.warning(f"  Error reading {local_gz}: {e}")
                local_gz.unlink(missing_ok=True)

            if (i + 1) % 50 == 0:
                logger.info(f"  Progress: {i+1}/{len(genomes)} processed, {downloaded} successful")

    size_mb = out_fasta.stat().st_size / 1e6
    logger.info(f"Downloaded {downloaded} {group} genomes → {out_fasta} ({size_mb:.0f} MB)")
    return out_fasta


def download_bacteria(output_dir: Path, max_genomes: int = 500) -> Path:
    """Download bacterial genomes (reference + representative)."""
    return download_genomes(
        "bacteria", output_dir, max_genomes,
        refseq_category="reference",
        assembly_level="Complete Genome",
    )


def download_eukaryotic(output_dir: Path, max_genomes: int = 100) -> Path:
    """Download eukaryotic genomes (fungi + protozoa)."""
    out_fasta = output_dir / "eukaryotic_refseq.fasta"
    if out_fasta.exists() and out_fasta.stat().st_size > 0:
        logger.info(f"Eukaryotic data already exists: {out_fasta}")
        return out_fasta

    # Download fungi and protozoa separately, then merge
    per_group = max_genomes // 2

    fungi_fasta = download_genomes(
        "fungi", output_dir, per_group,
        refseq_category=None,
        assembly_level="Complete Genome",
    )
    protozoa_fasta = download_genomes(
        "protozoa", output_dir, per_group,
        refseq_category=None,
        assembly_level=None,
    )

    # Merge into one eukaryotic file
    logger.info("Merging eukaryotic genomes...")
    with open(out_fasta, "w") as out:
        for src in [fungi_fasta, protozoa_fasta]:
            if src.exists() and src.stat().st_size > 0:
                with open(src) as f:
                    for line in f:
                        out.write(line)

    count = sum(1 for line in open(out_fasta) if line.startswith(">"))
    size_mb = out_fasta.stat().st_size / 1e6
    logger.info(f"Eukaryotic total: {count} sequences, {size_mb:.0f} MB")
    return out_fasta


def download_viruses(output_dir: Path, max_genomes: int = 50000) -> Path:
    """Download viral genomes."""
    return download_genomes(
        "viral", output_dir, max_genomes,
        refseq_category=None,
        assembly_level="Complete Genome",
    )


def main():
    parser = argparse.ArgumentParser(description="Download NCBI data for MetaPathPredict")
    parser.add_argument(
        "--output", "-o",
        default="data/input_large",
        help="Output directory for downloaded data",
    )
    parser.add_argument(
        "--max-bacteria-genomes",
        type=int,
        default=500,
        help="Max bacterial genomes to download",
    )
    parser.add_argument(
        "--max-eukaryotic-genomes",
        type=int,
        default=100,
        help="Max eukaryotic genomes to download",
    )
    parser.add_argument(
        "--max-viral-genomes",
        type=int,
        default=50000,
        help="Max viral genomes to download",
    )
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="Merge with existing data in data/input/",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download, only merge existing data",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=["bacteria", "eukaryotic", "viruses", "all"],
        default=["all"],
        help="Which groups to download",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_dir = Path("data/input")
    final_dir = Path("data/input_merged")
    final_dir.mkdir(parents=True, exist_ok=True)

    groups = args.groups
    if "all" in groups:
        groups = ["bacteria", "eukaryotic", "viruses"]

    if not args.skip_download:
        if "bacteria" in groups:
            download_bacteria(output_dir, args.max_bacteria_genomes)
        if "eukaryotic" in groups:
            download_eukaryotic(output_dir, args.max_eukaryotic_genomes)
        if "viruses" in groups:
            download_viruses(output_dir, args.max_viral_genomes)

    # Merge with existing data
    if args.merge_existing:
        for class_name, existing_name, new_pattern in [
            ("bacteria", "bacteria.fasta", "bacteria_*.fasta"),
            ("eukaryotic", "eucaryotic.fasta", "eukaryotic_*.fasta"),
            ("viruses", "viruses.fasta", "viruses_*.fasta"),
        ]:
            merged = final_dir / f"{class_name}.fasta"
            with open(merged, "w") as out:
                # Write existing
                existing = existing_dir / existing_name
                if existing.exists():
                    logger.info(f"Adding existing: {existing}")
                    with open(existing) as f:
                        for line in f:
                            out.write(line)

                # Write new downloads
                for new_file in sorted(output_dir.glob(new_pattern)):
                    logger.info(f"Adding downloaded: {new_file}")
                    with open(new_file) as f:
                        for line in f:
                            out.write(line)

            count = sum(1 for line in open(merged) if line.startswith(">"))
            size_mb = merged.stat().st_size / 1e6
            logger.info(f"{class_name}: {count} sequences, {size_mb:.0f} MB → {merged}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
