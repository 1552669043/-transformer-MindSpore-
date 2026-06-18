import argparse
import os
import urllib.request
import zipfile
from pathlib import Path


URL = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"


def install_proxy(proxy):
    if not proxy:
        return
    handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
    opener = urllib.request.build_opener(handler)
    urllib.request.install_opener(opener)


def remote_size(url):
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            size = response.headers.get("Content-Length")
    except Exception:
        return 0
    return int(size) if size else 0


def download(url, dest):
    dest = Path(dest)
    total = remote_size(url)
    done = dest.stat().st_size if dest.exists() else 0
    if total and done == total:
        print(f"Skip existing complete file: {dest}")
        return
    if done and total and done > total:
        dest.unlink()
        done = 0

    headers = {}
    mode = "wb"
    if done > 0:
        headers["Range"] = f"bytes={done}-"
        mode = "ab"
        print(f"Resuming {dest.name}: {done}/{total or '?'} bytes")
    else:
        print(f"Downloading {url} -> {dest}")

    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code == 416:
            print(f"Remote reports file already complete: {dest}")
            return
        if done > 0:
            print(f"Resume is not supported, restarting: {dest}")
            dest.unlink(missing_ok=True)
            return download(url, dest)
        raise

    with response, open(dest, mode) as f:
        if not total:
            content_length = response.headers.get("Content-Length")
            total = int(content_length) + done if content_length else 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r{dest.name}: {done * 100.0 / total:5.1f}%", end="")
        print()
    if total and os.path.getsize(dest) != total:
        raise RuntimeError(f"Incomplete download: {dest} ({os.path.getsize(dest)} / {total} bytes)")


def safe_extract(zip_path, dest):
    dest = Path(dest).resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if dest not in target.parents and target != dest:
                raise RuntimeError(f"Unsafe zip member: {member.filename}")
        zf.extractall(dest)


def main():
    parser = argparse.ArgumentParser(description="Download ADE20K ADEChallengeData2016.")
    parser.add_argument("--root", default="data/ade20k")
    parser.add_argument("--proxy", default=None, help="Optional proxy, e.g. http://127.0.0.1:7890")
    parser.add_argument("--extract-only", action="store_true", help="Only extract an existing ADEChallengeData2016.zip.")
    args = parser.parse_args()

    install_proxy(args.proxy)
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "ADEChallengeData2016.zip"
    if not args.extract_only:
        download(URL, archive)
    elif not archive.exists():
        raise FileNotFoundError(f"Can not find {archive}")
    print(f"Extracting {archive}")
    safe_extract(archive, root)

    base = root / "ADEChallengeData2016"
    expected = [
        base / "images" / "training",
        base / "images" / "validation",
        base / "annotations" / "training",
        base / "annotations" / "validation",
    ]
    for path in expected:
        if not path.exists():
            raise FileNotFoundError(path)
    print(f"ADE20K is ready at {base}")


if __name__ == "__main__":
    main()
