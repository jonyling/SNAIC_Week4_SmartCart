#!/usr/bin/env python3
"""
Collect NATURAL (in-store / in-market) Singapore grocery photos, as a companion
to collect_sg_groceries.py's clean studio product shots.

Source: Wikimedia Commons (CC-licensed contributor photos of real SG supermarkets
and wet markets) + Openverse. These are genuine shelf/aisle/stall photos, not
product packshots — the same spirit as GroceryStoreDataset's "natural in-store
phone photo" images.

Two output kinds:
  1. Scene photos (wide aisle/storefront/stall shots, not tied to one product)
     -> sg_dataset/natural_scenes/<retailer_or_market>/
  2. Per-class shelf photos (title/description names one of our 25 fine classes)
     -> sg_dataset/natural/<Coarse>/<Fine>/

Educational / course-demo use only. Wikimedia Commons images retain their
original CC license (recorded per-row in sg_natural_raw.csv); re-check license
text before any redistribution beyond course use.
"""

from __future__ import annotations

import csv
import re
import time
from datetime import date
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
NATURAL_SCENES = HERE / "sg_dataset" / "natural_scenes"
NATURAL_PRODUCTS = HERE / "sg_dataset" / "natural"
RAW_CSV = HERE / "sg_natural_raw.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-SG,en;q=0.9",
}
WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"
OPENVERSE_API = "https://api.openverse.org/v1/images/"

# Real Singapore retailers/markets -> scene search queries. These pull wide
# aisle / storefront / stall photos, not tied to a single product class.
SCENE_QUERIES: list[tuple[str, list[str]]] = [
    ("FairPrice", ["NTUC FairPrice supermarket Singapore aisle", "FairPrice Xtra Singapore interior"]),
    ("Sheng-Siong", ["Sheng Siong supermarket Singapore interior"]),
    ("Cold-Storage", ["Cold Storage supermarket Singapore interior"]),
    ("Giant", ["Giant hypermarket Singapore interior"]),
    ("Wet-Market", ["wet market Singapore vegetable stall", "Singapore wet market fruit stall", "Tekka market Singapore produce"]),
]

# Per fine-class: Wikimedia/Openverse queries biased toward an in-store shot of
# THIS product. Most will find nothing relevant -- that's expected and reported,
# not papered over.
PRODUCT_QUERIES: dict[str, list[str]] = {
    "Milo-Powder": ["Milo Singapore supermarket shelf", "Milo tin FairPrice Singapore"],
    "Yeos-Chrysanthemum-Tea": ["Yeo's Singapore supermarket shelf"],
    "Pokka-Green-Tea": ["Pokka Singapore supermarket shelf"],
    "FN-Seasons": ["F&N Seasons Singapore supermarket shelf"],
    "Ice-Mountain-Water": ["Ice Mountain water Singapore supermarket shelf"],
    "Marigold-HL-Milk": ["Marigold milk Singapore supermarket fridge"],
    "Meiji-Fresh-Milk": ["Meiji milk Singapore supermarket fridge"],
    "Gardenia-White-Bread": ["Gardenia bread Singapore supermarket shelf"],
    "Kaya-Spread": ["kaya spread Singapore supermarket shelf"],
    "Khong-Guan-Biscuits": ["Khong Guan biscuits Singapore supermarket shelf"],
    "Camel-Nuts": ["Camel nuts Singapore supermarket shelf"],
    "Maggi-Curry-Noodles": ["Maggi noodles Singapore supermarket shelf"],
    "Koka-Noodles": ["Koka noodles Singapore supermarket shelf"],
    "Prima-Taste-Laksa": ["Prima Taste Singapore supermarket shelf"],
    "Ayam-Brand-Sardines": ["Ayam Brand sardines Singapore supermarket shelf"],
    "SongHe-Rice": ["rice Singapore supermarket shelf"],
    "KCT-Chilli-Sauce": ["chilli sauce Singapore supermarket shelf"],
    "Lee-Kum-Kee-Oyster-Sauce": ["Lee Kum Kee Singapore supermarket shelf"],
    "Kailan": ["kailan Singapore wet market", "kai lan vegetable Singapore market"],
    "Chye-Sim": ["chye sim Singapore wet market", "choy sum Singapore market"],
    "Red-Chilli": ["red chilli Singapore wet market"],
    "Local-Lime": ["lime fruit Singapore market"],
    "Local-Mango": ["mango Singapore market"],
    "Bananas": ["banana Singapore market"],
    "Pasar-Fresh-Eggs": ["eggs Singapore supermarket shelf"],
}

FINE_TO_COARSE = {
    "Milo-Powder": "Beverages", "Yeos-Chrysanthemum-Tea": "Beverages", "Pokka-Green-Tea": "Beverages",
    "FN-Seasons": "Beverages", "Ice-Mountain-Water": "Beverages",
    "Marigold-HL-Milk": "Dairy", "Meiji-Fresh-Milk": "Dairy", "Pasar-Fresh-Eggs": "Dairy",
    "Gardenia-White-Bread": "Bakery", "Kaya-Spread": "Spreads",
    "Khong-Guan-Biscuits": "Snacks", "Camel-Nuts": "Snacks",
    "Maggi-Curry-Noodles": "Noodles", "Koka-Noodles": "Noodles",
    "Prima-Taste-Laksa": "Convenience", "Ayam-Brand-Sardines": "Canned", "SongHe-Rice": "Staples",
    "KCT-Chilli-Sauce": "Sauces", "Lee-Kum-Kee-Oyster-Sauce": "Sauces",
    "Kailan": "Vegetables", "Chye-Sim": "Vegetables", "Red-Chilli": "Vegetables",
    "Local-Lime": "Fruit", "Local-Mango": "Fruit", "Bananas": "Fruit",
}

# Reject obvious non-retail-context results (logos, maps, unrelated food closeups).
TITLE_STOPWORDS = (
    "logo", "map", "flag", "coat of arms", "signage only", "restaurant menu",
    "recipe", "cooked dish", "plate of", "bowl of",
)
RETAIL_CONTEXT_HINTS = (
    "supermarket", "market", "shelf", "aisle", "store", "shop", "fairprice",
    "sheng siong", "cold storage", "giant", "hypermarket", "grocery", "stall",
    "fridge", "mart",
)


def safe_slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")


def get_with_backoff(session: requests.Session, url: str, *, params: dict, retries: int = 5, base_sleep: float = 3.0) -> requests.Response | None:
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, headers=HEADERS, timeout=30)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(base_sleep * (attempt + 1))
                continue
            r.raise_for_status()
            return r
        except requests.RequestException:
            time.sleep(base_sleep * (attempt + 1))
    return None


def wikimedia_search(session: requests.Session, query: str, limit: int = 25) -> list[dict]:
    r = get_with_backoff(
        session,
        WIKIMEDIA_API,
        params={
            "action": "query",
            "generator": "search",
            "gsrsearch": f"filetype:bitmap {query}",
            "gsrnamespace": 6,
            "gsrlimit": min(limit, 50),
            "prop": "imageinfo",
            "iiprop": "url|thumburl|mime|size|extmetadata",
            "iiurlwidth": 1024,
            "format": "json",
        },
    )
    if r is None:
        return []
    pages = r.json().get("query", {}).get("pages", {})
    out = []
    for page in pages.values():
        title = str(page.get("title") or "")
        info = (page.get("imageinfo") or [{}])[0]
        if not str(info.get("mime", "")).startswith("image/"):
            continue
        url = info.get("thumburl") or info.get("url")
        if not url:
            continue
        meta = info.get("extmetadata") or {}
        license_name = (meta.get("LicenseShortName") or {}).get("value", "unknown")
        w, h = info.get("width") or 0, info.get("height") or 0
        out.append({"title": title, "url": url, "license": license_name, "w": w, "h": h, "source": "wikimedia"})
    return out


def openverse_search(session: requests.Session, query: str, limit: int = 20) -> list[dict]:
    r = get_with_backoff(session, OPENVERSE_API, params={"q": query, "page_size": limit})
    if r is None:
        return []
    out = []
    for item in r.json().get("results", []):
        url = item.get("url")
        if not url:
            continue
        out.append({
            "title": item.get("title") or query,
            "url": url,
            "license": item.get("license") or "unknown",
            "w": item.get("width") or 0,
            "h": item.get("height") or 0,
            "source": "openverse",
        })
    return out


def looks_retail_context(title: str) -> bool:
    low = title.lower()
    if any(sw in low for sw in TITLE_STOPWORDS):
        return False
    return any(h in low for h in RETAIL_CONTEXT_HINTS)


def is_valid_image(path: Path, min_side: int = 300, min_pixels: int = 150_000) -> bool:
    try:
        from PIL import Image

        with Image.open(path) as im:
            w, h = im.size
            if w < min_side or h < min_side or w * h < min_pixels:
                return False
            ratio = w / h
            return 0.3 <= ratio <= 3.2
    except Exception:
        return path.stat().st_size > 20_000


def download(session: requests.Session, url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 5000 and is_valid_image(dest):
        return True
    try:
        r = session.get(url, headers=HEADERS, timeout=45)
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "")
        if "image" not in ctype and not url.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            return False
        dest.write_bytes(r.content)
        return dest.stat().st_size > 5000 and is_valid_image(dest)
    except requests.RequestException:
        return False


FIELDNAMES = [
    "kind", "fine", "coarse", "retailer_or_market", "title", "license",
    "image_source", "image_url", "local_path", "scrape_date",
]


def main() -> int:
    session = requests.Session()
    seen_urls: set[str] = set()
    rows_written = 0

    csv_file = RAW_CSV.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
    writer.writeheader()

    def emit(row: dict) -> None:
        nonlocal rows_written
        writer.writerow(row)
        csv_file.flush()
        rows_written += 1

    def search_both(query: str, wiki_limit: int, ov_limit: int) -> list[dict]:
        try:
            wiki = wikimedia_search(session, query, limit=wiki_limit)
        except Exception as exc:  # noqa: BLE001 - keep collecting
            print(f"  [warn] wikimedia query failed: {query!r}: {exc}")
            wiki = []
        time.sleep(1.0)
        try:
            ov = openverse_search(session, query, limit=ov_limit)
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] openverse query failed: {query!r}: {exc}")
            ov = []
        return wiki + ov

    try:
        print("=== SCENE photos (wide aisle / storefront / stall) ===")
        for retailer, queries in SCENE_QUERIES:
            got = 0
            out_dir = NATURAL_SCENES / retailer
            for q in queries:
                for cand in search_both(q, 25, 15):
                    if cand["url"] in seen_urls:
                        continue
                    if not looks_retail_context(cand["title"]):
                        continue
                    if cand["w"] and cand["w"] < 500:
                        continue
                    seen_urls.add(cand["url"])
                    idx = got + 1
                    dest = out_dir / f"{retailer}_{idx:03d}.jpg"
                    if download(session, cand["url"], dest):
                        got += 1
                        emit({
                            "kind": "scene", "fine": "", "coarse": "", "retailer_or_market": retailer,
                            "title": cand["title"], "license": cand["license"], "image_source": cand["source"],
                            "image_url": cand["url"], "local_path": str(dest.relative_to(HERE)),
                            "scrape_date": date.today().isoformat(),
                        })
                time.sleep(1.0)
            print(f"  {retailer}: {got} scene photos")

        print("\n=== PER-PRODUCT natural shelf photos ===")
        for fine, queries in PRODUCT_QUERIES.items():
            coarse = FINE_TO_COARSE[fine]
            got = 0
            out_dir = NATURAL_PRODUCTS / coarse / fine
            for q in queries:
                for cand in search_both(q, 20, 15):
                    if cand["url"] in seen_urls:
                        continue
                    if not looks_retail_context(cand["title"] + " " + q):
                        if fine not in {"Kailan", "Chye-Sim", "Red-Chilli", "Local-Lime", "Local-Mango", "Bananas"}:
                            continue
                    seen_urls.add(cand["url"])
                    idx = got + 1
                    dest = out_dir / f"{fine}_natural_{idx:03d}.jpg"
                    if download(session, cand["url"], dest):
                        got += 1
                        emit({
                            "kind": "product", "fine": fine, "coarse": coarse, "retailer_or_market": "",
                            "title": cand["title"], "license": cand["license"], "image_source": cand["source"],
                            "image_url": cand["url"], "local_path": str(dest.relative_to(HERE)),
                            "scrape_date": date.today().isoformat(),
                        })
                time.sleep(1.0)
            print(f"  {fine}: {got} natural shelf photos" if got else f"  {fine}: 0 (none found)")
    finally:
        csv_file.close()

    with RAW_CSV.open() as f:
        all_rows = list(csv.DictReader(f))
    scenes = sum(1 for r in all_rows if r["kind"] == "scene")
    products = sum(1 for r in all_rows if r["kind"] == "product")
    print(f"\n==== SUMMARY ====\nscene photos: {scenes}\nper-product photos: {products}\ntotal: {len(all_rows)}")
    print(f"wrote {RAW_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
