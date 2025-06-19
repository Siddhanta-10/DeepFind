"""
This script starts from a single Confluence page ID, then walks the tree of
child pages entirely via the REST API.  All pages are saved locally as HTML, so
later stages of the pipeline can clean and index them.


Replace ``ROOT_PAGE_ID`` with the ID of your root wiki page and either export
``CONFLUENCE_TOKEN`` in your environment or hard‑code ``BEARER_TOKEN`` for
quick tests.

All downloaded files go to ``confluence_pages/`` in the current directory.
"""

import os
import logging
import time
import random
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


BASE_URL = "https://confluence.org.com"  # change to your Confluence base
BEARER_TOKEN = os.getenv("CONFLUENCE_TOKEN", "MY_BEARER_TOKEN")
OUTPUT_DIR = "downloaded_wikis/"
ROOT_PAGE_ID = "123456"  # <-- put your main page ID here


def create_session(token: str) -> requests.Session:
    """Return a requests.Session with auth + retry handling."""
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {token}",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/114.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
    })

    retry = Retry(
        total=10,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def safe_get_json(session: requests.Session, url: str, *, params=None, timeout=10):
    """GET a URL that returns JSON, with built‑in delay and retry."""
    for _ in range(10):
        try:
            time.sleep(0.5 + random.random() * 0.5)  # 0.5–1.0 s polite delay
            resp = session.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", "5"))
                logging.warning(f"Rate limited. Sleeping for {retry_after}s")
                time.sleep(retry_after)
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logging.warning(f"Temporary issue fetching {url}: {e}. Retrying...")
            time.sleep(5)
    raise RuntimeError(f"Failed to GET {url} after multiple retries")


def get_page_content(session: requests.Session, page_id: str):
    """Return (title, html) for a single Confluence page."""
    url = f"{BASE_URL}/rest/api/content/{page_id}"
    params = {"expand": "body.storage"}
    data = safe_get_json(session, url, params=params)
    title = data.get("title", f"page_{page_id}")
    html = data["body"]["storage"]["value"]
    return title, html


def get_child_pages(session: requests.Session, page_id: str, *, limit=50, start=0):
    """Return list of direct child pages for *page_id* and count of results."""
    url = f"{BASE_URL}/rest/api/content/{page_id}/child/page"
    params = {"limit": limit, "start": start}
    data = safe_get_json(session, url, params=params)
    results = data.get("results", [])
    return results, len(results)


def sanitize_filename(text: str, max_len=80) -> str:
    """Return filename‑safe version of *text* (alnum + -_.) limited to *max_len*."""
    return "".join(c if c.isalnum() or c in "-_ ." else "_" for c in text)[:max_len]


def save_html(page_id: str, title: str, html: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"{page_id}__{sanitize_filename(title)}.html"
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(html)
    logging.info("Saved %s", path)


def crawl_page(session: requests.Session, page_id: str, visited: set[str]):
    if page_id in visited:
        return
    visited.add(page_id)

    # Fetch & save this page
    try:
        title, html = get_page_content(session, page_id)
        save_html(page_id, title, html)
    except Exception as exc:
        logging.error("Failed to fetch/save page %s: %s", page_id, exc)
        return

    # Recurse into child pages (handle pagination)
    start = 0
    while True:
        try:
            children, count = get_child_pages(session, page_id, start=start)
        except Exception as exc:
            logging.warning("Failed to fetch children of page %s: %s", page_id, exc)
            break

        if count == 0:
            break

        for child in children:
            child_id = child["id"]
            crawl_page(session, child_id, visited)

        if count < 50:  # fewer than limit → last page of results
            break
        start += 50



def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    session = create_session(BEARER_TOKEN)
    visited: set[str] = set()
    try:
        crawl_page(session, ROOT_PAGE_ID, visited)
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
    logging.info("Done. Crawled %d pages.", len(visited))


if __name__ == "__main__":
    main()
