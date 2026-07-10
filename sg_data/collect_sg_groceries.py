#!/usr/bin/env python3
"""
Collect a Singapore grocery image+price dataset for SmartCart.

Primary prices: FairPrice search pages.
Images: FairPrice product shots, then supplemental Bing Images and Wikimedia
Commons to reach --target-images per class (default 25).

Educational / course-demo use only.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import time
from datetime import date
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE / "sg_dataset"
IMAGES_ROOT = OUT_ROOT / "images"
ICONIC_ROOT = OUT_ROOT / "iconic"

SEARCH_URL = "https://www.fairprice.com.sg/search?query={query}"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-SG,en;q=0.9",
}

# (fine, coarse, query, must_all substrings, any_of substrings, exclude substrings)
# A product matches when:
#   - every must_all token appears in the name (if non-empty), AND
#   - at least one any_of token appears (if non-empty), AND
#   - none of the exclude tokens appear.
CLASSES: list[tuple[str, str, str, list[str], list[str], list[str]]] = [
    ("Milo-Powder", "Beverages", "milo powder", ["milo"], ["powder", "refill", "tin", "epicer", "gao kosong"], ["uht", "packet drink", "ready to drink", "activ-go", "nugget", "cereal", "bar"]),
    ("Yeos-Chrysanthemum-Tea", "Beverages", "yeo chrysanthemum", ["yeo"], ["chrysanthemum"], ["carton", "ctn", "biscuit"]),
    ("Pokka-Green-Tea", "Beverages", "pokka green tea", ["pokka"], ["green tea", "sencha", "houjicha", "jasmine"], ["carton", "ctn", "[bcrs]"]),
    ("FN-Seasons", "Beverages", "f&n seasons", ["seasons"], ["f&n", "fraser"], ["carton", "ctn", "[bcrs]", " sparkling"]),
    ("Ice-Mountain-Water", "Beverages", "f&n ice mountain", ["ice mountain"], ["drinking", "pure", "bottle"], ["sparkling", "lemon", "peach", "grapefruit", "classic"]),
    ("Marigold-HL-Milk", "Dairy", "marigold hl milk", ["marigold", "hl"], ["milk"], ["uht packet", "yoghurt", "yogurt", "soy", "cultured"]),
    ("Meiji-Fresh-Milk", "Dairy", "meiji pasteurized fresh milk", ["meiji"], ["fresh milk", "pasteurized"], ["yoghurt", "yogurt", "chocolate", "flavoured", "low fat"]),
    ("Gardenia-White-Bread", "Bakery", "gardenia white bread", ["gardenia"], ["white bread", "enriched white", "wholemeal"], ["bun", "wrap", "pancake", "toast", "sandwich"]),
    ("Kaya-Spread", "Spreads", "nyonya kaya", [], ["kaya", "nonya", "nyonya"], ["butter", "toast set", "biscuit", "bread"]),
    ("Khong-Guan-Biscuits", "Snacks", "khong guan biscuits", ["khong guan"], ["biscuit", "cracker", "marie", "sandwich"], ["carton"]),
    ("Camel-Nuts", "Snacks", "camel nuts", ["camel"], ["nut", "almond", "cashew", "peanut", "pistachio"], ["snack mix bar"]),
    ("Maggi-Curry-Noodles", "Noodles", "maggi curry noodles", ["maggi", "curry"], ["noodle"], ["ketchup", "sauce", "paste", "seasoning"]),
    ("Koka-Noodles", "Noodles", "koka noodles", ["koka"], ["noodle"], []),
    ("Prima-Taste-Laksa", "Convenience", "prima taste laksa", ["prima taste"], ["laksa"], ["fried rice"]),
    ("Ayam-Brand-Sardines", "Canned", "ayam brand sardines", ["ayam brand"], ["sardine"], []),
    ("SongHe-Rice", "Staples", "song he rice", [], ["songhe", "song he"], []),
    ("KCT-Chilli-Sauce", "Sauces", "maggi chilli sauce", [], ["chilli sauce", "chili sauce"], ["oil", "flake", "powder", "paste", "dry", "crispy", "noodle", "karagarashi"]),
    ("Lee-Kum-Kee-Oyster-Sauce", "Sauces", "lee kum kee oyster sauce", ["lee kum kee"], ["oyster"], []),
    ("Kailan", "Vegetables", "kailan", [], ["kailan", "kai lan"], ["sauce", "paste", "seasoning", "seed", "oil"]),
    ("Chye-Sim", "Vegetables", "chye sim", [], ["chye sim", "cai xin", "choy sum"], ["sauce", "paste", "seed", "oil"]),
    ("Red-Chilli", "Vegetables", "simply finest red chilli", [], ["red chilli", "red chili", "bird's eye", "bird eye"], ["flake", "powder", "sauce", "salt", "butter", "vermicelli", "chips", "capsicum", "oil", "paste", "crushed", "brand", "dried"]),
    # Fresh citrus / fruit only — reject baby food, juice, mangosteen lookalikes, etc.
    ("Local-Lime", "Fruit", "seedless lime", [], ["seedless lime", "fresh lime", "large lime", "calamansi"], ["leaf", "juice", "soda", "dish", "candy", "butter", "season", "sprite", "drink", "flavour", "flavor", "nuts", "caviar", "finger", "pearl", "concentrate"]),
    ("Local-Mango", "Fruit", "fresh mango", [], ["mango"], ["juice", "drink", "pudding", "yoghurt", "yogurt", "dried", "puree", "ice cream", "candy", "mangosteen", "frozen", "diced", "crisp", "spread", "sauce", "chutney", "sweet potato", "baby", "gerber", "organix"]),
    ("Bananas", "Fruit", "sumifru banana", [], ["banana"], ["leaf", "leaves", "shallot", "gerber", "organix", "cerelac", "puree", "puff", "cereal", "grain", "baby", "milk", "chip", "cake", "bread", "biscuit", "cookie", "yoghurt", "yogurt", "smoothie", "blossom", "shallots"]),
    ("Pasar-Fresh-Eggs", "Dairy", "pasar fresh eggs", ["pasar"], ["egg"], ["quail", "century", "salted", "balut"]),
]

# Extra FairPrice queries + web image search strings to reach target_images per class.
# Bing/Wikimedia are used only after FairPrice is exhausted (prices stay FairPrice-only).
CLASS_SUPPLEMENTS: dict[str, dict[str, list[str]]] = {
    "Milo-Powder": {
        "fp_extra": ["milo tin", "milo refill", "milo gao kosong"],
        "bing": ["Milo chocolate malt powder tin Nestle product packshot", "Milo powder 400g tin supermarket"],
    },
    "Yeos-Chrysanthemum-Tea": {
        "fp_extra": ["yeo's chrysanthemum packet", "yeo packet drink chrysanthemum"],
        "bing": ["Yeo's chrysanthemum tea packet drink Singapore product"],
    },
    "Pokka-Green-Tea": {
        "fp_extra": ["pokka jasmine green tea", "pokka sencha"],
        "bing": ["Pokka green tea bottle can Singapore product"],
    },
    "FN-Seasons": {
        "fp_extra": ["f&n seasons lychee", "f&n seasons ice lemon"],
        "bing": ["F&N Seasons packet drink Singapore product"],
    },
    "Ice-Mountain-Water": {
        "fp_extra": ["ice mountain pure drinking water", "f&n ice mountain bottle"],
        "openverse": ["plastic water bottle product", "drinking water bottle pack", "mineral water bottle"],
        "bing": [
            "Ice Mountain drinking water bottle Singapore F&N",
            "F&N Ice Mountain 600ml bottle water",
            "bottled drinking water product packshot",
        ],
    },
    "Marigold-HL-Milk": {
        "fp_extra": ["marigold hl milk 1l", "marigold hl milk 2l"],
        "bing": ["Marigold HL milk carton Singapore product"],
    },
    "Meiji-Fresh-Milk": {
        "fp_extra": ["meiji milk 1l", "meiji milk 2l"],
        "bing": ["Meiji fresh milk carton Singapore product"],
    },
    "Gardenia-White-Bread": {
        "fp_extra": ["gardenia bread loaf", "gardenia soft white"],
        "bing": ["Gardenia white bread loaf Singapore product packaging"],
    },
    "Kaya-Spread": {
        "fp_extra": ["fairprice kaya", "glory kaya", "wang kaya"],
        "bing": ["Singapore kaya spread jar coconut jam product"],
    },
    "Khong-Guan-Biscuits": {
        "fp_extra": ["khong guan cream crackers", "khong guan biscuits tin"],
        "bing": ["Khong Guan biscuits tin Singapore product"],
    },
    "Camel-Nuts": {
        "fp_extra": ["camel roasted peanuts", "camel cashew"],
        "bing": ["Camel nuts packet Singapore roasted peanuts product"],
    },
    "Maggi-Curry-Noodles": {
        "fp_extra": ["maggi curry instant noodles", "maggi 2 minute noodles curry"],
        "bing": ["Maggi curry instant noodles packet product"],
    },
    "Koka-Noodles": {
        "fp_extra": ["koka instant noodles", "koka laksa noodles"],
        "bing": ["Koka instant noodles cup Singapore product"],
    },
    "Prima-Taste-Laksa": {
        "fp_extra": ["prima taste laksa paste", "prima taste laksa kit"],
        "bing": ["Prima Taste laksa paste kit Singapore product"],
    },
    "Ayam-Brand-Sardines": {
        "fp_extra": ["ayam brand sardines tomato", "ayam brand sardines chili"],
        "bing": ["Ayam Brand sardines tin Singapore product"],
    },
    "SongHe-Rice": {
        "fp_extra": ["songhe thai hom mali", "songhe jasmine rice"],
        "bing": ["SongHe rice bag Singapore Thai hom mali product"],
    },
    "KCT-Chilli-Sauce": {
        "fp_extra": ["heinz chili sauce", "sin sin chili sauce", "maggi chili sauce garlic", "abc chili sauce"],
        "bing": ["Maggi chilli sauce bottle Singapore product", "Heinz chili sauce bottle product"],
    },
    "Lee-Kum-Kee-Oyster-Sauce": {
        "fp_extra": ["lee kum kee oyster sauce bottle"],
        "bing": ["Lee Kum Kee oyster sauce bottle product"],
    },
    "Kailan": {
        "fp_extra": ["kai lan vegetable", "chinese broccoli"],
        "wiki": ["gai lan vegetable", "chinese broccoli kai lan"],
        "bing": ["kailan kai lan vegetable fresh produce"],
        "openverse": ["gai lan vegetable", "chinese broccoli"],
    },
    "Chye-Sim": {
        "fp_extra": ["choy sum vegetable", "cai xin"],
        "wiki": ["choy sum vegetable", "choy sum greens"],
        "bing": ["chye sim choy sum vegetable fresh"],
        "openverse": ["choy sum vegetable"],
    },
    "Red-Chilli": {
        "fp_extra": ["red chilli fresh", "bird eye chilli"],
        "wiki": ["red chili pepper vegetable", "bird eye chili"],
        "openverse": ["red chili pepper fresh", "bird eye chili", "red hot chili peppers"],
        "bing": ["fresh red chili pepper isolated white background", "bird eye chilli pepper fresh"],
    },
    "Local-Lime": {
        "fp_extra": ["thygrace seedless lime", "agro fresh lime", "fresh lime fruit"],
        "wiki": ["lime fruit citrus white background", "key lime fruit"],
        "openverse": ["fresh lime fruit", "green lime citrus", "limes on white"],
        "bing": ["fresh seedless lime fruit produce packshot", "green lime fruit white background"],
    },
    "Local-Mango": {
        "fp_extra": ["fresh mango fruit", "india mango", "honey mango fruit", "mango fruit pasar"],
        "wiki": ["mango fruit ripe yellow", "mangifera indica fruit"],
        "openverse": ["ripe mango fruit", "fresh mango produce", "mango fruit white background"],
        "bing": ["fresh mango fruit produce supermarket", "ripe mango packshot white background"],
    },
    "Bananas": {
        "fp_extra": ["sumifru banana", "pasar banana", "philippines banana", "cavendish banana fruit"],
        "wiki": ["banana fruit bunch white background", "cavendish banana"],
        "openverse": ["banana fruit bunch", "yellow bananas produce", "bananas white background"],
        "bing": [
            "fresh banana bunch supermarket produce",
            "cavendish banana fruit packshot",
            "sumifru banana philippines produce",
        ],
    },
    "Pasar-Fresh-Eggs": {
        "fp_extra": ["pasar eggs 10", "pasar eggs 15", "pasar large eggs"],
        "bing": ["Pasar fresh eggs carton Singapore NTUC"],
        "openverse": ["egg carton fresh eggs", "chicken eggs carton"],
    },
}

BING_URL_BLOCK = (
    "youtube.com", "ytimg.com", "facebook.com", "instagram.com", "tiktok.com",
    "pinterest.com/pin", "skyline", "city-view", "travel-guide", "hotel",
    "logo.svg", "/icon/", "avatar", "emoji", "giphy.com", "meme",
    "fabric", "etsystatic", "fibre2fashion", "threadart", "walmartimages",
    "alicdn.com", "shopify.com/s/files",
)
WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"

# Produce + thin FP classes: NEVER pad with Bing (returns fabric/memes). Prefer fewer
# real FairPrice shots + local synthetic augment instead.
FAIRPRICE_ONLY_CLASSES = {
    "Bananas",
    "Local-Mango",
    "Local-Lime",
    "Red-Chilli",
    "Kailan",
    "Chye-Sim",
}

NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json"[^>]*>(.*?)</script>',
    re.S,
)

# Extra phrase blocklist applied to FairPrice product names (case-insensitive).
GLOBAL_NAME_BLOCKLIST = (
    "gerber",
    "organix",
    "cerelac",
    "puffs",
    "puree pouch",
    "baby cereal",
    "banana leaves",
    "banana leaf",
    "banana shallot",
    "shallots",
    "mangosteen",
    "lime caviar",
    "finger pearls",
    "great grains",
    "fruit crisps",
    "fruit rolls",
    "fruit spread",
    "fruit cordial",
    "st.dalfour",
    "oh so healthy",
    "oob organic",
    "berryfield",
    "frozen gold mango",
    "pickle",
    "pulp",
    "cordial",
    "twinings",
    "tea bags",
    "body lotio",
    "body lotion",
    "marula",
    "hey chips",
    "crisps",
    "dove replenishing",
    "bear fruit",
)


def safe_slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")


def iter_products(obj):
    if isinstance(obj, dict):
        if "name" in obj and "storeSpecificData" in obj:
            yield obj
        for v in obj.values():
            yield from iter_products(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from iter_products(v)


def extract_price(prod: dict) -> float | None:
    ssd = prod.get("storeSpecificData")
    if not isinstance(ssd, list) or not ssd:
        return None
    row = ssd[0] if isinstance(ssd[0], dict) else {}
    raw = row.get("mrp") or row.get("price")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def extract_brand(prod: dict) -> str:
    brand = prod.get("brand") or prod.get("brandName")
    if isinstance(brand, dict):
        return str(brand.get("name") or "")
    return str(brand or "")


def extract_size(prod: dict) -> str:
    meta = prod.get("metaData") or {}
    if isinstance(meta, dict):
        for key in ("Size", "size", "Weight", "weight", "Unit Of Measurement"):
            if meta.get(key):
                return str(meta[key])
    for key in ("displayUnit", "packSize", "unitOfMeasurement"):
        if prod.get(key):
            return str(prod[key])
    return ""


def extract_images(prod: dict) -> list[str]:
    imgs = prod.get("images") or []
    if not isinstance(imgs, list):
        return []
    out = []
    for u in imgs:
        if isinstance(u, str) and u.startswith("http"):
            out.append(u)
        elif isinstance(u, dict):
            for k in ("url", "src", "image", "xl"):
                if isinstance(u.get(k), str) and u[k].startswith("http"):
                    out.append(u[k])
                    break
    seen = set()
    deduped = []
    for u in out:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def name_matches(name: str, must_all: list[str], any_of: list[str], exclude: list[str]) -> bool:
    low = name.lower()
    if any(b in low for b in GLOBAL_NAME_BLOCKLIST):
        return False
    if any(ex.lower() in low for ex in exclude):
        return False
    if must_all and not all(tok.lower() in low for tok in must_all):
        return False
    if any_of and not any(tok.lower() in low for tok in any_of):
        return False
    return True


def looks_like_bulk(name: str, size: str) -> bool:
    blob = f"{name} {size}".lower()
    # Keep multipacks that we can normalize (N x ...) — only drop explicit carton warehouse SKUs
    # when DisplayUnit does not expose an N-pack pattern we can divide.
    if re.search(r"\b\d+\s*x\s*\d+", blob):
        return False
    bulk_markers = ("carton", " ctn", "ctn ", "pack of")
    return any(m in blob for m in bulk_markers)


def extract_display_unit(prod: dict) -> str:
    meta = prod.get("metaData") or {}
    if isinstance(meta, dict) and meta.get("DisplayUnit"):
        return str(meta["DisplayUnit"])
    return extract_size(prod)


def pack_count(name: str, size: str) -> int:
    blob = f"{name} {size}"
    m = re.search(r"(\d+)\s*x\s*\d+", blob, flags=re.I)
    if m:
        n = int(m.group(1))
        return n if n > 1 else 1
    return 1


def unit_price(prod: dict) -> float | None:
    price = extract_price(prod)
    if price is None:
        return None
    size = extract_display_unit(prod)
    name = str(prod.get("name") or "")
    n = pack_count(name, size)
    return price / n


def fetch_search(session: requests.Session, query: str, retries: int = 4) -> list[dict]:
    url = SEARCH_URL.format(query=requests.utils.quote(query))
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            r = session.get(url, headers=HEADERS, timeout=45)
            if r.status_code in (502, 503, 504, 429):
                time.sleep(1.5 * (attempt + 1))
                continue
            r.raise_for_status()
            m = NEXT_DATA_RE.search(r.text)
            if not m:
                raise RuntimeError(f"__NEXT_DATA__ not found for query={query!r}")
            data = json.loads(m.group(1))
            products = list(iter_products(data))
            seen = set()
            unique = []
            for p in products:
                key = (p.get("name"), (extract_images(p) or [None])[0])
                if key in seen:
                    continue
                seen.add(key)
                unique.append(p)
            return unique
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(1.5 * (attempt + 1))
    if last_exc:
        raise last_exc
    raise RuntimeError(f"Failed to fetch query={query!r}")


def is_valid_image(path: Path, min_side: int = 120, min_pixels: int = 20000) -> bool:
    try:
        from PIL import Image

        with Image.open(path) as im:
            w, h = im.size
            if w < min_side or h < min_side or w * h < min_pixels:
                return False
            ratio = w / h
            return 0.25 <= ratio <= 4.0
    except Exception:
        return path.stat().st_size > 8000


def download_image(session: requests.Session, url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000 and is_valid_image(dest):
        return True
    try:
        r = session.get(url, headers=HEADERS, timeout=45)
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "")
        if "image" not in ctype and not url.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            return False
        dest.write_bytes(r.content)
        return dest.stat().st_size > 1000 and is_valid_image(dest)
    except requests.RequestException:
        return False


def bing_image_urls(session: requests.Session, query: str, limit: int = 80) -> list[str]:
    urls: list[str] = []
    for first in (1, 35, 70, 105):
        r = session.get(
            "https://www.bing.com/images/search",
            params={"q": query, "form": "HDRSC2", "first": first, "count": 35},
            headers=HEADERS,
            timeout=30,
        )
        r.raise_for_status()
        found = re.findall(r'murl&quot;:&quot;(https?://[^&]+?)&quot;', r.text)
        if not found:
            found = re.findall(r'"murl":"(https?://[^"]+)"', r.text)
        for u in found:
            low = u.lower()
            if any(b in low for b in BING_URL_BLOCK):
                continue
            if u not in urls:
                urls.append(u)
        if len(urls) >= limit:
            break
        time.sleep(0.4)
    return urls[:limit]


def openverse_image_urls(session: requests.Session, query: str, limit: int = 50) -> list[str]:
    urls: list[str] = []
    page = 1
    while len(urls) < limit and page <= 5:
        r = session.get(
            "https://api.openverse.org/v1/images/",
            params={"q": query, "page_size": 20, "page": page},
            headers=HEADERS,
            timeout=30,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            break
        for item in results:
            url = item.get("url")
            if url and url not in urls:
                urls.append(url)
        page += 1
        time.sleep(0.35)
    return urls[:limit]


def wikimedia_image_urls(session: requests.Session, query: str, limit: int = 40) -> list[str]:
    r = session.get(
        WIKIMEDIA_API,
        params={
            "action": "query",
            "generator": "search",
            "gsrsearch": f"filetype:bitmap {query}",
            "gsrnamespace": 6,
            "gsrlimit": min(limit, 50),
            "prop": "imageinfo",
            "iiprop": "url|thumburl|mime|size",
            "iiurlwidth": 800,
            "format": "json",
        },
        headers=HEADERS,
        timeout=30,
    )
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", {})
    urls: list[str] = []
    for page in pages.values():
        info = (page.get("imageinfo") or [{}])[0]
        if not str(info.get("mime", "")).startswith("image/"):
            continue
        url = info.get("thumburl") or info.get("url")
        if url and url not in urls:
            urls.append(url)
    return urls[:limit]


def looks_like_fresh_produce(name: str, fine: str) -> bool:
    """Reject packaged foods that merely mention a fruit/veg keyword."""
    low = name.lower()
    if fine == "Bananas":
        # Accept Sumifru / Pasar / YayaPapaya fruit SKUs; reject chips etc.
        fruit_ok = any(
            k in low
            for k in (
                "sumifru",
                "pasar",
                "yayapapaya",
                "cavendish",
                "kamsookwang",
                "poongmi",
                "sweet mountain",
                "philippines banana",
                "rose banana",
            )
        )
        return fruit_ok and "chip" not in low and "crisp" not in low
    if fine == "Local-Mango":
        return any(
            k in low
            for k in ("fresh mango", "raw mango", "rumani", "benishan", "alphonso mango", "honey mango", "mango fruit")
        ) or (low.endswith("mango") and "fresh" in low)
    if fine == "Local-Lime":
        return "lime" in low and any(k in low for k in ("seedless", "fresh", "thygrace", "agro", "calamansi", "large lime"))
    if fine == "Red-Chilli":
        return any(k in low for k in ("red chilli", "red chili", "bird")) and "sauce" not in low
    return True


def match_fairprice_products(
    products: list[dict],
    must_all: list[str],
    any_of: list[str],
    exclude: list[str],
    *,
    strict: bool,
    fine: str = "",
) -> list[dict]:
    matched = []
    for p in products:
        name = str(p.get("name") or "")
        imgs = extract_images(p)
        size = extract_display_unit(p)
        if not name or not imgs:
            continue
        if fine and fine in {"Bananas", "Local-Mango", "Local-Lime", "Red-Chilli"}:
            if not looks_like_fresh_produce(name, fine):
                continue
        if strict:
            if extract_price(p) is None:
                continue
            if not name_matches(name, must_all, any_of, exclude):
                continue
            if looks_like_bulk(name, size):
                continue
        else:
            if any(b in name.lower() for b in GLOBAL_NAME_BLOCKLIST):
                continue
            if exclude and any(ex.lower() in name.lower() for ex in exclude):
                continue
            if must_all and not all(tok.lower() in name.lower() for tok in must_all):
                continue
            if any_of and not any(tok.lower() in name.lower() for tok in any_of):
                continue
        matched.append(p)
    return matched


def fp_product_score(p: dict, must_all: list[str], any_of: list[str]) -> tuple:
    name = str(p.get("name") or "").lower()
    size = extract_display_unit(p)
    hits = sum(1 for f in (must_all + any_of) if f.lower() in name)
    packs = pack_count(name, size)
    shelf = extract_price(p) or 999.0
    return (-hits, 0 if packs == 1 else 1, shelf, len(name))


def append_downloaded_image(
    session: requests.Session,
    *,
    fine: str,
    coarse: str,
    img_url: str,
    image_source: str,
    product_name: str,
    brand: str,
    size: str,
    shelf_price: float | None,
    per_unit: float | None,
    source_url: str,
    class_dir: Path,
    iconic_path: Path,
    img_idx: int,
    delay: float,
    seen_urls: set[str],
) -> tuple[int, Path | None, dict | None]:
    if img_url in seen_urls:
        return img_idx, None, None
    seen_urls.add(img_url)
    dest = class_dir / f"{fine}_{img_idx:03d}.jpg"
    ok = download_image(session, img_url, dest)
    time.sleep(delay * 0.35)
    if not ok:
        if dest.exists():
            dest.unlink(missing_ok=True)
        return img_idx, None, None
    if not iconic_path.exists() or iconic_path.stat().st_size < 1000:
        iconic_path.write_bytes(dest.read_bytes())
    row = {
        "fine": fine,
        "coarse": coarse,
        "product_name": product_name,
        "brand": brand,
        "size": size,
        "unit_price_sgd": f"{shelf_price:.2f}" if shelf_price is not None else "",
        "per_unit_sgd": f"{per_unit:.2f}" if per_unit is not None else "",
        "image_source": image_source,
        "image_url": img_url,
        "source_url": source_url,
        "local_path": str(dest.relative_to(HERE)),
        "scrape_date": date.today().isoformat(),
    }
    return img_idx + 1, dest, row


def local_augment_to_target(
    *,
    fine: str,
    coarse: str,
    seeds: list[Path],
    seed_rows: list[dict],
    class_dir: Path,
    target_images: int,
    img_idx: int,
) -> tuple[list[Path], list[dict], int]:
    """Pad a class with simple PIL transforms of clean FairPrice seeds (no web junk)."""
    if not seeds or len(seeds) >= target_images:
        return [], [], img_idx

    try:
        from PIL import Image, ImageEnhance, ImageOps
    except ImportError:
        print(f"  skip local augment (Pillow missing); have {len(seeds)} real images")
        return [], [], img_idx

    extra_paths: list[Path] = []
    extra_rows: list[dict] = []
    ops = [
        ("flip", lambda im: ImageOps.mirror(im)),
        ("rot8", lambda im: im.rotate(8, expand=True, fillcolor=(255, 255, 255))),
        ("rot-8", lambda im: im.rotate(-8, expand=True, fillcolor=(255, 255, 255))),
        ("bright", lambda im: ImageEnhance.Brightness(im).enhance(1.15)),
        ("dark", lambda im: ImageEnhance.Brightness(im).enhance(0.85)),
        ("contrast", lambda im: ImageEnhance.Contrast(im).enhance(1.2)),
        ("sat", lambda im: ImageEnhance.Color(im).enhance(1.15)),
        ("crop", lambda im: ImageOps.fit(im, (max(128, int(im.width * 0.85)), max(128, int(im.height * 0.85))))),
    ]

    i = 0
    while len(seeds) + len(extra_paths) < target_images:
        src = seeds[i % len(seeds)]
        seed_row = seed_rows[i % len(seed_rows)]
        name, op = ops[i % len(ops)]
        i += 1
        try:
            with Image.open(src) as im:
                im = im.convert("RGB")
                out = op(im)
                img_idx += 1
                dest = class_dir / f"{fine}_{img_idx:03d}_aug_{name}.jpg"
                out.save(dest, quality=92)
        except Exception:
            continue
        extra_paths.append(dest)
        extra_rows.append(
            {
                **{k: seed_row.get(k, "") for k in (
                    "fine", "coarse", "product_name", "brand", "size",
                    "unit_price_sgd", "per_unit_sgd", "image_url", "source_url", "scrape_date",
                )},
                "fine": fine,
                "coarse": coarse,
                "image_source": f"augment_{name}",
                "local_path": str(dest.relative_to(HERE)),
                "product_name": f"{seed_row.get('product_name', fine)} [aug:{name}]",
            }
        )
        if i > target_images * 8:
            break

    return extra_paths, extra_rows, img_idx


def collect_class(
    session: requests.Session,
    fine: str,
    coarse: str,
    query: str,
    must_all: list[str],
    any_of: list[str],
    exclude: list[str],
    max_products: int,
    max_images_per_product: int,
    target_images: int,
    delay: float,
    allow_web_fill: bool = True,
) -> tuple[list[dict], list[Path], float | None]:
    """Returns (raw_rows, image_paths, median_shelf_price from FairPrice)."""
    supplements = CLASS_SUPPLEMENTS.get(fine, {})
    fp_queries = [query] + supplements.get("fp_extra", [])
    # Do NOT default openverse queries to bing strings — that pulls junk.
    openverse_queries = supplements.get("openverse", [])
    bing_queries = supplements.get("bing", [])
    wiki_queries = supplements.get("wiki", [])
    fp_only = fine in FAIRPRICE_ONLY_CLASSES or not allow_web_fill

    class_dir = IMAGES_ROOT / coarse / fine
    class_dir.mkdir(parents=True, exist_ok=True)
    iconic_path = ICONIC_ROOT / f"{fine}.jpg"
    ICONIC_ROOT.mkdir(parents=True, exist_ok=True)

    for old in class_dir.glob("*.jpg"):
        old.unlink()
    if iconic_path.exists():
        iconic_path.unlink()

    raw_rows: list[dict] = []
    image_paths: list[Path] = []
    prices: list[float] = []
    seen_urls: set[str] = set()
    img_idx = 0
    seen_product_keys: set[tuple] = set()

    for fp_q in fp_queries:
        if len(image_paths) >= target_images:
            break
        products = fetch_search(session, fp_q)
        time.sleep(delay)
        strict = fp_q == query
        matched = match_fairprice_products(
            products, must_all, any_of, exclude, strict=strict, fine=fine
        )
        matched.sort(key=lambda p: fp_product_score(p, must_all, any_of))
        cap = max_products if strict else max_products * 2
        for p in matched[:cap]:
            if len(image_paths) >= target_images:
                break
            name = str(p.get("name") or "")
            key = (name, (extract_images(p) or [None])[0])
            if key in seen_product_keys:
                continue
            seen_product_keys.add(key)

            shelf = extract_price(p)
            per_unit = unit_price(p)
            # Count every accepted FairPrice SKU toward median price (not only primary query).
            if shelf is not None:
                prices.append(shelf)

            brand = extract_brand(p)
            size = extract_display_unit(p)
            slug = p.get("slug") or ""
            source = (
                f"https://www.fairprice.com.sg/product/{slug}"
                if slug
                else SEARCH_URL.format(query=fp_q)
            )
            for img_url in extract_images(p)[:max_images_per_product]:
                if len(image_paths) >= target_images:
                    break
                img_idx, dest, row = append_downloaded_image(
                    session,
                    fine=fine,
                    coarse=coarse,
                    img_url=img_url,
                    image_source="fairprice",
                    product_name=name,
                    brand=brand,
                    size=size,
                    shelf_price=shelf,
                    per_unit=per_unit,
                    source_url=source,
                    class_dir=class_dir,
                    iconic_path=iconic_path,
                    img_idx=img_idx + 1,
                    delay=delay,
                    seen_urls=seen_urls,
                )
                if dest and row:
                    image_paths.append(dest)
                    raw_rows.append(row)

    if not fp_only:
        for ov_q in openverse_queries:
            if len(image_paths) >= target_images:
                break
            try:
                urls = openverse_image_urls(session, ov_q, limit=60)
            except requests.RequestException:
                continue
            time.sleep(delay * 0.4)
            for img_url in urls:
                if len(image_paths) >= target_images:
                    break
                img_idx, dest, row = append_downloaded_image(
                    session,
                    fine=fine,
                    coarse=coarse,
                    img_url=img_url,
                    image_source="openverse",
                    product_name=f"{fine} (Openverse: {ov_q})",
                    brand="",
                    size="",
                    shelf_price=None,
                    per_unit=None,
                    source_url=img_url,
                    class_dir=class_dir,
                    iconic_path=iconic_path,
                    img_idx=img_idx + 1,
                    delay=delay,
                    seen_urls=seen_urls,
                )
                if dest and row:
                    image_paths.append(dest)
                    raw_rows.append(row)

        for wiki_q in wiki_queries:
            if len(image_paths) >= target_images:
                break
            for img_url in wikimedia_image_urls(session, wiki_q, limit=40):
                if len(image_paths) >= target_images:
                    break
                img_idx, dest, row = append_downloaded_image(
                    session,
                    fine=fine,
                    coarse=coarse,
                    img_url=img_url,
                    image_source="wikimedia",
                    product_name=f"{fine} (Wikimedia: {wiki_q})",
                    brand="",
                    size="",
                    shelf_price=None,
                    per_unit=None,
                    source_url=img_url,
                    class_dir=class_dir,
                    iconic_path=iconic_path,
                    img_idx=img_idx + 1,
                    delay=delay,
                    seen_urls=seen_urls,
                )
                if dest and row:
                    image_paths.append(dest)
                    raw_rows.append(row)
            time.sleep(delay * 0.5)

        for bing_q in bing_queries:
            if len(image_paths) >= target_images:
                break
            try:
                urls = bing_image_urls(session, bing_q, limit=100)
            except requests.RequestException:
                continue
            time.sleep(delay * 0.5)
            for img_url in urls:
                if len(image_paths) >= target_images:
                    break
                # Extra hard reject of fabric / textile hosts for all classes.
                low = img_url.lower()
                if any(b in low for b in ("fabric", "etsy", "textile", "cotton", "walmartimages")):
                    continue
                img_idx, dest, row = append_downloaded_image(
                    session,
                    fine=fine,
                    coarse=coarse,
                    img_url=img_url,
                    image_source="bing",
                    product_name=f"{fine} (Bing: {bing_q})",
                    brand="",
                    size="",
                    shelf_price=None,
                    per_unit=None,
                    source_url=img_url,
                    class_dir=class_dir,
                    iconic_path=iconic_path,
                    img_idx=img_idx + 1,
                    delay=delay,
                    seen_urls=seen_urls,
                )
                if dest and row:
                    image_paths.append(dest)
                    raw_rows.append(row)
    else:
        print(f"  FairPrice-only mode ({len(image_paths)} real shots); local augment if needed")

    if len(image_paths) < target_images and image_paths:
        extra_paths, extra_rows, img_idx = local_augment_to_target(
            fine=fine,
            coarse=coarse,
            seeds=list(image_paths),
            seed_rows=list(raw_rows),
            class_dir=class_dir,
            target_images=target_images,
            img_idx=img_idx,
        )
        image_paths.extend(extra_paths)
        raw_rows.extend(extra_rows)
        if extra_paths:
            print(f"  local augment +{len(extra_paths)} -> {len(image_paths)} images")

    median_price = statistics.median(prices) if prices else None
    return raw_rows, image_paths, median_price


def write_csvs(raw_rows: list[dict], price_rows: list[dict], label_rows: list[dict]) -> None:
    raw_path = HERE / "sg_products_raw.csv"
    labels_path = HERE / "labels.csv"
    catalog_path = HERE / "catalog_prices.csv"

    raw_fields = [
        "fine",
        "coarse",
        "product_name",
        "brand",
        "size",
        "unit_price_sgd",
        "per_unit_sgd",
        "image_source",
        "image_url",
        "source_url",
        "local_path",
        "scrape_date",
    ]
    with raw_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=raw_fields)
        w.writeheader()
        w.writerows(raw_rows)

    with labels_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "coarse", "fine"])
        w.writeheader()
        w.writerows(label_rows)

    with catalog_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["fine", "coarse", "unit_price", "currency", "unit"])
        w.writeheader()
        w.writerows(price_rows)

    print(f"wrote {raw_path}")
    print(f"wrote {labels_path}")
    print(f"wrote {catalog_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-products", type=int, default=30, help="Max FairPrice SKUs per query (strict)")
    parser.add_argument("--max-images-per-product", type=int, default=12, help="Max angles per FairPrice SKU")
    parser.add_argument("--target-images", type=int, default=40, help="Target images per class")
    parser.add_argument("--delay", type=float, default=0.45, help="Seconds between HTTP requests")
    parser.add_argument(
        "--fairprice-only",
        action="store_true",
        help="Disable Bing/Openverse/Wiki for ALL classes (safest)",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Optional subset of fine class names to collect",
    )
    args = parser.parse_args()

    selected = CLASSES
    if args.classes:
        wanted = set(args.classes)
        selected = [c for c in CLASSES if c[0] in wanted]
        missing = wanted - {c[0] for c in selected}
        if missing:
            raise SystemExit(f"Unknown class names: {sorted(missing)}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    all_raw: list[dict] = []
    label_rows: list[dict] = []
    price_rows: list[dict] = []
    failures: list[str] = []

    # When refreshing a subset, keep other classes' rows from the previous CSVs.
    if args.classes:
        labels_path = HERE / "labels.csv"
        raw_path = HERE / "sg_products_raw.csv"
        catalog_path = HERE / "catalog_prices.csv"
        wanted = set(args.classes)
        if labels_path.exists():
            label_rows = [r for r in csv.DictReader(labels_path.open()) if r["fine"] not in wanted]
        if raw_path.exists():
            all_raw = [r for r in csv.DictReader(raw_path.open()) if r["fine"] not in wanted]
        if catalog_path.exists():
            price_rows = [r for r in csv.DictReader(catalog_path.open()) if r["fine"] not in wanted]

    for fine, coarse, query, must_all, any_of, exclude in selected:
        print(f"\n=== {fine} ({coarse}) query={query!r} ===")
        try:
            raw, paths, median = collect_class(
                session,
                fine=fine,
                coarse=coarse,
                query=query,
                must_all=must_all,
                any_of=any_of,
                exclude=exclude,
                max_products=args.max_products,
                max_images_per_product=args.max_images_per_product,
                target_images=args.target_images,
                delay=args.delay,
                allow_web_fill=not args.fairprice_only,
            )
        except Exception as exc:  # noqa: BLE001 - keep collecting other classes
            print(f"FAILED {fine}: {exc}")
            failures.append(f"{fine}: {exc}")
            continue

        print(f"  images: {len(paths)} (target {args.target_images}), median SGD={median}")
        if len(paths) < args.target_images:
            print(f"  WARNING: only {len(paths)}/{args.target_images} images")
        if not paths:
            failures.append(f"{fine}: no images downloaded")
            continue

        all_raw.extend(raw)
        for p in paths:
            label_rows.append(
                {
                    "path": str(p.relative_to(HERE)),
                    "coarse": coarse,
                    "fine": fine,
                }
            )
        if median is not None:
            price_rows.append(
                {
                    "fine": fine,
                    "coarse": coarse,
                    "unit_price": f"{median:.2f}",
                    "currency": "SGD",
                    "unit": "each",
                }
            )
        elif paths:
            failures.append(f"{fine}: no FairPrice price")

    price_rows.sort(key=lambda r: r["fine"])
    label_rows.sort(key=lambda r: (r["fine"], r["path"]))
    write_csvs(all_raw, price_rows, label_rows)

    print("\n==== SUMMARY ====")
    print(f"classes ok: {len(price_rows)} / {len(selected)}")
    print(f"total images: {len(label_rows)}")
    print(f"raw provenance rows: {len(all_raw)}")
    if failures:
        print("failures:")
        for f in failures:
            print(" -", f)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
