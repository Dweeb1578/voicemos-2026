"""Download all pretraining datasets.

Usage:
    python data/download.py --output data/datasets --datasets bvcc tmhint audiomos25t3
"""

import argparse
import os
import subprocess
import tarfile
import zipfile

import gdown
import requests
from tqdm import tqdm

TMHINT_REPO = "https://github.com/dhimasryan/TMHINT-QI-VoiceMOS2023.git"
TMHINT_AUDIO_ID = "1TMDiz6dnS76hxyeAcCQxeSqqEOH4UDN0"  # 2.2GB zip: TMHINTQI/train/ wavs
BVCC_ZENODO_RECORD = "6572573"

AUDIOMOS25T3_TRAIN_ID = "1IoxKU_dS8uDdMEFZc8IBLp0he8Vz5xOH"
AUDIOMOS25T3_DEV_LABELS_ID = "1i6gfL4eukxXe1bGjwxul5se_wyAnm_Eo"
AUDIOMOS25T3_EVALSET_ID = "1HJcmPKwoe2vckmznfuaD5pJKd5EyFMDY"
AUDIOMOS25T3_EVALSET_LABELS_ID = "1bLfPCv3YQvKPyYrmqVqYbF1UuuuEucRg"


def _download_file(url: str, dest: str) -> None:
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def _extract(path: str, out: str) -> None:
    if path.endswith(".tar.gz") or path.endswith(".tgz"):
        with tarfile.open(path, "r:gz") as tf:
            tf.extractall(out)
    elif path.endswith(".zip"):
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(out)


def download_bvcc(output_dir: str) -> None:
    """Download BVCC dataset from Zenodo record 6572573."""
    out = os.path.join(output_dir, "bvcc")
    os.makedirs(out, exist_ok=True)
    api_url = f"https://zenodo.org/api/records/{BVCC_ZENODO_RECORD}"
    resp = requests.get(api_url)
    resp.raise_for_status()
    for f in resp.json()["files"]:
        fname, url = f["key"], f["links"]["self"]
        dest = os.path.join(out, fname)
        if os.path.exists(dest):
            print(f"  Skipping {fname} (exists)")
        else:
            print(f"  Downloading {fname}...")
            _download_file(url, dest)
        _extract(dest, out)
    print(f"BVCC -> {out}")


def download_tmhint(output_dir: str) -> None:
    """Clone TMHINT-QI labels from GitHub and download audio from Google Drive."""
    out = os.path.join(output_dir, "tmhint")
    if not os.path.exists(out):
        subprocess.run(["git", "clone", TMHINT_REPO, out], check=True)
    else:
        print(f"TMHINT labels exist at {out}, skipping clone.")

    # Audio zip extracts to TMHINTQI/train/ inside out/
    audio_zip = os.path.join(out, "tmhint_audio.zip")
    wav_dir = os.path.join(out, "TMHINTQI", "train")
    if os.path.exists(wav_dir):
        print("  TMHINT audio already extracted, skipping.")
    else:
        print("  Downloading TMHINT audio (~2.2 GB)...")
        gdown.download(id=TMHINT_AUDIO_ID, output=audio_zip, quiet=False)
        _extract(audio_zip, out)

    print(f"TMHINT -> {out}")


def download_audiomos25t3(output_dir: str) -> None:
    """Download AudioMOS 2025 Track 3 from Google Drive."""
    out = os.path.join(output_dir, "audiomos25t3")
    os.makedirs(out, exist_ok=True)
    files = [
        (AUDIOMOS25T3_TRAIN_ID, "train.zip"),
        (AUDIOMOS25T3_DEV_LABELS_ID, "dev_labels.csv"),
        (AUDIOMOS25T3_EVALSET_ID, "evalset.zip"),
        (AUDIOMOS25T3_EVALSET_LABELS_ID, "evalset_labels.csv"),
    ]
    for file_id, fname in files:
        dest = os.path.join(out, fname)
        if os.path.exists(dest):
            print(f"  Skipping {fname} (exists)")
            continue
        print(f"  Downloading {fname}...")
        result = gdown.download(id=file_id, output=dest, quiet=False)
        if result is None:
            print(f"  WARNING: {fname} download failed (skipping)")
            continue
        if dest.endswith(".zip"):
            _extract(dest, out)
    print(f"AudioMOS 2025 T3 -> {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/datasets")
    parser.add_argument(
        "--datasets", nargs="+",
        choices=["bvcc", "tmhint", "audiomos25t3"],
        default=["bvcc", "tmhint", "audiomos25t3"],
    )
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    if "bvcc" in args.datasets:
        download_bvcc(args.output)
    if "tmhint" in args.datasets:
        download_tmhint(args.output)
    if "audiomos25t3" in args.datasets:
        download_audiomos25t3(args.output)
    print("All downloads complete.")


if __name__ == "__main__":
    main()
