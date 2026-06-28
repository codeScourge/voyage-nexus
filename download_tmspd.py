#!/usr/bin/env python3
"""
download_tmspd.py — threaded bulk downloader for the T-MSPD SciDB dataset.

Usage:
    python download_tmspd.py --dry-run        # size estimate, no files written
    python download_tmspd.py                   # download (threaded)
    python download_tmspd.py --workers 8       # override pool size
    python download_tmspd.py --check           # only run triplet integrity pass

Concurrency notes:
  * Each worker thread gets its OWN requests.Session (thread-local). Sharing a
    single session's connection pool across threads is the classic cause of
    flaky bulk downloads -- don't.
  * SciDB returns HTML throttle/error pages (HTTP 200) under heavy concurrent
    load. The HTML guard catches these, but a high worker count just trades
    speed for a long re-pull list. Start at 4 and raise only if failures stay
    near zero.
  * The CSV log and the progress counter are lock-guarded so output from
    multiple threads doesn't interleave or get lost.

Integrity notes (unchanged from the serial version):
  * URLs are session-bound; failed auth = HTML page with HTTP 200. We inspect
    Content-Type AND first bytes and refuse to write HTML as a binary file.
  * Raw sEMG files are all named data.bdf / evt.bdf with no subject id, so we
    reconstruct the full dataset path from the URL's `path=` param.
  * Raw EEG is a triplet (.cdt + .cdt.dpo + .cdt.ceo); check_eeg_triplets()
    flags any recording missing a sibling.
  * Writes go to .part then atomic rename; re-running skips finished files.
"""

import argparse
import csv
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

import requests

# ---- CONFIG -------------------------------------------------------------
URL_FILE    = "00fead6aa5dc4f2cb03d98fd29c5aad4.txt"
OUT_ROOT    = Path("./T-MSPD")
PATH_FILTER = ["2.raw data"]  
MIN_BYTES   = 256                            # evt.bdf can be legitimately tiny
REQUEST_TIMEOUT = 120
MAX_WORKERS = 4                              # start low; raise if failures ~0
POLITE_DELAY = 0.0                           # per-thread; usually leave at 0

SESSION_COOKIE = None   # "JSESSIONID=ABC; other=val" if auth is required
COOKIE_DOMAIN  = "download.scidb.cn"
# -------------------------------------------------------------------------

_thread_local = threading.local()
_log_lock = threading.Lock()
_count_lock = threading.Lock()
_done = 0


def get_session() -> requests.Session:
    """One session per worker thread."""
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": "Mozilla/5.0"})
        if SESSION_COOKIE:
            for pair in SESSION_COOKIE.split(";"): #type: ignore
                if "=" in pair:
                    k, v = pair.strip().split("=", 1)
                    s.cookies.set(k, v, domain=COOKIE_DOMAIN)
        _thread_local.session = s
    return s


def load_urls() -> list:
    lines = Path(URL_FILE).read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip().startswith("http")]


def parse_target(url: str) -> Path:
    q = parse_qs(urlparse(url).query)
    rel = unquote(q["path"][0]).lstrip("/")
    return OUT_ROOT / rel


def passes_filter(target: Path) -> bool:
    if PATH_FILTER is None:
        return True
    return any(f in str(target) for f in PATH_FILTER)


def looks_like_html(first_bytes: bytes) -> bool:
    head = first_bytes[:512].lstrip().lower()
    return head.startswith(b"<!doctype html") or head.startswith(b"<html")


# ---- DRY RUN (threaded HEAD) --------------------------------------------
def _head_size(url: str) -> int:
    target = parse_target(url)
    if not passes_filter(target):
        return -1   # sentinel: filtered out
    try:
        r = get_session().head(url, allow_redirects=True, timeout=REQUEST_TIMEOUT)
        return int(r.headers.get("Content-Length", 0))
    except Exception:
        return 0


def dry_run(urls: list, workers: int) -> None:
    total = counted = unknown = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_head_size, u): u for u in urls}
        for i, fut in enumerate(as_completed(futs), 1):
            size = fut.result()
            if size == -1:
                continue
            if size > 0:
                total += size
                counted += 1
            else:
                unknown += 1
            if i % 50 == 0:
                print(f"  ...probed {i} URLs, running total {total/1e9:.2f} GB",
                      flush=True)
    print("-" * 60)
    print(f"Files matching filter with known size: {counted}")
    print(f"Files with unknown size (HEAD gave no length): {unknown}")
    print(f"Estimated total: {total/1e9:.2f} GB "
          f"(unknown-size files not included)")


# ---- DOWNLOAD (threaded) ------------------------------------------------
def download_one(url: str) -> str:
    target = parse_target(url)
    if not passes_filter(target):
        return "skip-filter"
    if target.exists() and target.stat().st_size > MIN_BYTES:
        return "skip-exists"

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(target) + ".part")

    with get_session().get(url, stream=True, timeout=REQUEST_TIMEOUT) as r:
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "")
        first = next(r.iter_content(chunk_size=8192), b"")
        if "text/html" in ctype or looks_like_html(first):
            raise RuntimeError("got HTML, not a file (auth/cookie expired?)")
        with open(tmp, "wb") as f:
            f.write(first)
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)

    if tmp.stat().st_size < MIN_BYTES:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("file suspiciously small")
    tmp.replace(target)
    if POLITE_DELAY:
        time.sleep(POLITE_DELAY)
    return f"ok ({target.stat().st_size/1e6:.1f} MB)"


def run_download(urls: list, workers: int) -> None:
    global _done
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    n = len(urls)
    print(f"{n} URLs loaded; filter={PATH_FILTER}; workers={workers}")

    log_path = OUT_ROOT / "_download_log.csv"
    log = open(log_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(log)

    def task(url: str):
        global _done
        try:
            status = download_one(url)
        except Exception as e:
            status = f"FAIL: {e}"
        with _count_lock:
            _done += 1
            idx = _done
        with _log_lock:
            print(f"[{idx}/{n}] {status}", flush=True)
            writer.writerow([idx, status, url])
            log.flush()
        return status

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(task, urls))

    log.close()
    check_eeg_triplets(OUT_ROOT)


# ---- INTEGRITY ----------------------------------------------------------
def check_eeg_triplets(root: Path) -> None:
    missing = []
    cdts = list(root.rglob("*.cdt"))
    for cdt in cdts:
        for ext in (".cdt.dpo", ".cdt.ceo"):
            sib = Path(str(cdt)[:-4] + ext)
            if not sib.exists() or sib.stat().st_size < MIN_BYTES:
                missing.append(str(sib))
    print("-" * 60)
    if missing:
        print(f"INCOMPLETE EEG TRIPLETS ({len(missing)} files) "
              f"across {len(cdts)} .cdt recordings:")
        for m in missing:
            print("  ", m)
    else:
        print(f"All {len(cdts)} EEG triplets complete.")


# ---- ENTRY --------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = ap.parse_args()

    if args.check:
        check_eeg_triplets(OUT_ROOT)
        return

    urls = load_urls()
    if not urls:
        sys.exit(f"No URLs found in {URL_FILE}")

    if args.dry_run:
        dry_run(urls, args.workers)
    else:
        run_download(urls, args.workers)


if __name__ == "__main__":
    main()