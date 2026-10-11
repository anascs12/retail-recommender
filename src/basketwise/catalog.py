"""Import a store's public product catalogue from its website.

Order and customer data are private and are never scraped. The catalogue (names,
descriptions, categories, tags, prices) is public, and most store platforms publish it
in a structured feed:

- Shopify      /products.json                       (public, paginated)
- WooCommerce  /wp-json/wc/store/v1/products        (public Store API)
- Other sites  sitemap.xml → product pages → schema.org Product data (JSON-LD)

Etiquette: robots.txt is respected, requests are paced (≥1 s apart), the crawler
identifies itself, and the number of pages is capped. Intended for owners importing
their own store.
"""

from __future__ import annotations

import html
import json
import re
import time
import urllib.robotparser
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests

from . import remote

PAUSE_S = 1.0
MAX_PRODUCTS = 5000
MAX_SITEMAP_PAGES = 300
CATALOG_BYTES = 20 * 1024 * 1024
COLUMNS = ["product_id", "title", "description", "category", "brand", "tags", "price", "url", "image"]


class CatalogError(remote.LinkError):
    pass


@dataclass
class Catalog:
    products: pd.DataFrame
    platform: str
    base_url: str
    notes: list[str] = field(default_factory=list)


def clean_text(value) -> str:
    text = re.sub(r"<[^>]+>", " ", html.unescape(str(value or "")))
    return re.sub(r"\s+", " ", text).strip()


def normalise_base(url: str) -> str:
    url = url.strip()
    if not re.match(r"^https?://", url):
        url = "https://" + url
    parsed = urlparse(url)
    if not parsed.hostname:
        raise CatalogError("Please enter a website address such as https://yourshop.com")
    return f"{parsed.scheme}://{parsed.netloc}"


class PoliteClient:
    """Paced, robots.txt-aware client built on the SSRF-safe getter."""

    def __init__(self, base: str, session: requests.Session | None = None, pause: float = PAUSE_S,
                 sleep: Callable[[float], None] = time.sleep):
        self.base, self.session, self.pause, self.sleep = base, session or requests.Session(), pause, sleep
        self.last, self.refused = 0.0, 0
        self.robots = urllib.robotparser.RobotFileParser()
        try:
            r = remote.safe_get(urljoin(base, "/robots.txt"), self.session, max_bytes=512 * 1024)
            self.robots.parse(r.data.decode("utf-8", "ignore").splitlines() if r.status == 200 else [])
        except (remote.LinkError, requests.RequestException):
            self.robots.parse([])
        delay = self.robots.crawl_delay(remote.USER_AGENT) or self.robots.crawl_delay("*")
        if delay:
            self.pause = max(self.pause, min(float(delay), 10.0))

    def allowed(self, url: str) -> bool:
        return self.robots.can_fetch(remote.USER_AGENT, url)

    def get(self, path_or_url: str) -> remote.SafeResponse | None:
        url = urljoin(self.base, path_or_url)
        if not self.allowed(url):
            return None
        wait = self.pause - (time.monotonic() - self.last)
        if wait > 0:
            self.sleep(wait)
        self.last = time.monotonic()
        response = remote.safe_get(url, self.session, max_bytes=CATALOG_BYTES,
                                   headers={"Accept": "application/json, text/html;q=0.9, */*;q=0.5"})
        if response.status in (403, 429):
            self.refused += 1
        return response


def _json(response: remote.SafeResponse | None):
    if response is None or response.status != 200:
        return None
    try:
        return json.loads(response.data)
    except ValueError:
        return None


# ------------------------------------------------------------------ platforms
def shopify(client: PoliteClient, limit: int) -> list[dict] | None:
    rows, page = [], 1
    while len(rows) < limit:
        data = _json(client.get(f"/products.json?limit=250&page={page}"))
        if not isinstance(data, dict) or "products" not in data:
            return rows or None
        batch = data["products"]
        if not batch:
            break
        for p in batch:
            variant = (p.get("variants") or [{}])[0]
            rows.append({
                "product_id": str(variant.get("sku") or p.get("id")), "title": p.get("title"),
                "description": clean_text(p.get("body_html"))[:1000], "category": p.get("product_type"),
                "brand": p.get("vendor"), "tags": ", ".join(p.get("tags", []) if isinstance(p.get("tags"), list)
                                                          else str(p.get("tags", "")).split(",")),
                "price": pd.to_numeric(variant.get("price"), errors="coerce"),
                "url": urljoin(client.base, f"/products/{p.get('handle')}"),
                "image": ((p.get("images") or [{}])[0]).get("src"),
            })
        page += 1
    return rows


def woocommerce(client: PoliteClient, limit: int) -> list[dict] | None:
    rows, page = [], 1
    while len(rows) < limit:
        data = _json(client.get(f"/wp-json/wc/store/v1/products?per_page=100&page={page}"))
        if not isinstance(data, list):
            return rows or None
        if not data:
            break
        for p in data:
            prices = p.get("prices") or {}
            minor = int(prices.get("currency_minor_unit") or 0)
            price = pd.to_numeric(prices.get("price"), errors="coerce")
            rows.append({
                "product_id": str(p.get("sku") or p.get("id")), "title": clean_text(p.get("name")),
                "description": clean_text(p.get("short_description") or p.get("description"))[:1000],
                "category": ", ".join(c.get("name", "") for c in p.get("categories", [])),
                "brand": ", ".join(b.get("name", "") for b in p.get("brands", [])) if p.get("brands") else "",
                "tags": ", ".join(t.get("name", "") for t in p.get("tags", [])),
                "price": price / 10 ** minor if pd.notna(price) else None,
                "url": p.get("permalink"), "image": ((p.get("images") or [{}])[0]).get("src"),
            })
        page += 1
    return rows


def _product_from_jsonld(obj, page_url: str) -> dict | None:
    if isinstance(obj, list):
        for item in obj:
            found = _product_from_jsonld(item, page_url)
            if found:
                return found
        return None
    if not isinstance(obj, dict):
        return None
    if "@graph" in obj:
        return _product_from_jsonld(obj["@graph"], page_url)
    kind = obj.get("@type")
    kinds = set(kind) if isinstance(kind, list) else {kind}
    if kinds & {"Product", "ProductGroup"}:  # ProductGroup = a product with size/colour variants
        offers = obj.get("offers") or {}
        variants = obj.get("hasVariant") or []
        if not offers and variants and isinstance(variants[0], dict):
            offers = variants[0].get("offers") or {}
        offer = offers[0] if isinstance(offers, list) and offers else offers
        brand = obj.get("brand")
        image = obj.get("image")
        return {
            "product_id": str(obj.get("sku") or obj.get("productID") or page_url), "title": clean_text(obj.get("name")),
            "description": clean_text(obj.get("description"))[:1000], "category": clean_text(obj.get("category")),
            "brand": clean_text(brand.get("name") if isinstance(brand, dict) else brand), "tags": "",
            "price": pd.to_numeric((offer or {}).get("price") or (offer or {}).get("lowPrice"), errors="coerce"),
            "url": obj.get("url") or page_url,
            "image": image[0] if isinstance(image, list) and image else image if isinstance(image, str) else None,
        }
    return None


def _meta(html_text: str, *names: str) -> str | None:
    for name in names:
        match = re.search(rf'<meta[^>]+(?:property|name|itemprop)="{re.escape(name)}"[^>]+content="([^"]*)"', html_text,
                          flags=re.IGNORECASE) or re.search(
            rf'<meta[^>]+content="([^"]*)"[^>]+(?:property|name|itemprop)="{re.escape(name)}"', html_text, flags=re.IGNORECASE)
        if match and match.group(1).strip():
            return html.unescape(match.group(1).strip())
    return None


def category_from_url(url: str) -> str:
    """'/collections/womens-body-mists/products/x.html' -> 'womens body mists'."""
    parts = [p for p in urlparse(url).path.split("/") if p]
    for marker in ("collections", "category", "product-category", "c"):
        if marker in parts and parts.index(marker) + 1 < len(parts):
            return parts[parts.index(marker) + 1].replace("-", " ")
    return ""


def _product_from_meta(html_text: str, page_url: str) -> dict | None:
    """Fallback for sites without schema.org Product data: Open Graph and microdata meta tags."""
    title = _meta(html_text, "og:title", "twitter:title")
    if not title:
        return None
    price = _meta(html_text, "product:price:amount", "og:price:amount", "price")
    return {
        "product_id": _meta(html_text, "product:retailer_item_id", "sku") or page_url, "title": clean_text(title),
        "description": clean_text(_meta(html_text, "og:description", "description"))[:1000],
        "category": category_from_url(page_url), "brand": clean_text(_meta(html_text, "product:brand", "og:site_name")),
        "tags": "", "price": pd.to_numeric(str(price or "").replace(",", ""), errors="coerce"),
        "url": page_url, "image": _meta(html_text, "og:image"),
    }


def _sitemap_product_urls(client: PoliteClient, cap: int) -> list[str]:
    # Sitemaps declared in robots.txt first (the standard place), then the conventional location.
    queue, seen, urls = [*(client.robots.site_maps() or []), "/sitemap.xml", "/sitemap_index.xml"], set(), []
    while queue and len(seen) < 20 and len(urls) < cap:
        loc = queue.pop(0)
        if loc in seen:
            continue
        seen.add(loc)
        response = client.get(loc)
        if response is None or response.status != 200:
            continue
        try:
            root = ET.fromstring(response.data)
        except ET.ParseError:
            continue
        product_sitemap = "product" in loc.lower()   # e.g. sitemap_products_1.xml lists only products
        for node in root.iter():
            namespace, _, local = node.tag.rpartition("}")
            if local == "loc" and "image" not in namespace and "video" not in namespace and node.text:
                text = node.text.strip()
                if text.endswith(".xml"):
                    # follow product sitemaps first; other child sitemaps only if nothing better is known
                    (queue.insert(0, text) if "product" in text.lower() else queue.append(text))
                elif product_sitemap or re.search(r"/(product|products|p|item|shop)/", text):
                    urls.append(text)
    return urls[:cap]


def schema_org(client: PoliteClient, limit: int) -> list[dict] | None:
    rows = []
    for url in _sitemap_product_urls(client, min(limit, MAX_SITEMAP_PAGES)):
        response = client.get(url)
        if response is None or response.status != 200:
            continue
        page = response.data.decode("utf-8", "ignore")
        product = None
        for block in re.findall(r'<script[^>]+application/ld\+json[^>]*>(.*?)</script>', page, flags=re.DOTALL | re.IGNORECASE):
            try:
                product = _product_from_jsonld(json.loads(block.strip()), url)
            except ValueError:
                continue
            if product:
                break
        product = product or _product_from_meta(page, url)
        if product:
            product["category"] = product["category"] or category_from_url(url)
            rows.append(product)
    return rows or None


def strip_site_name(title: str, site: str) -> str:
    """'XOXO Body Mist Sapphire PK' / 'Mug | My Shop' -> product name only."""
    title = re.sub(r"\s*[|\-–—]\s*[^|\-–—]{2,40}$", "", title) if re.search(r"\s[|\-–—]\s", title) else title
    if site and title.lower().endswith(site.lower()) and len(title) > len(site) + 3:
        title = title[: -len(site)].rstrip(" -|–—")
    return title.strip()


def tidy(products: pd.DataFrame) -> pd.DataFrame:
    """Drop empty titles, non-products (gift cards, shipping) and duplicates; trim site names from titles."""
    from .cleaning import is_non_product_name

    p = products.copy()
    p["title"] = p["title"].fillna("").astype(str).str.strip()
    site = p["brand"].fillna("").astype(str).mode()
    site_name = site.iat[0] if len(site) and (p["brand"] == site.iat[0]).mean() > 0.8 else ""
    p["title"] = [strip_site_name(t, site_name) for t in p["title"]]
    p = p[(p["title"].str.len() > 0) & ~is_non_product_name(p["title"])]
    return p.drop_duplicates("product_id").reset_index(drop=True)


def import_catalog(website: str, limit: int = MAX_PRODUCTS, session: requests.Session | None = None,
                   sleep: Callable[[float], None] = time.sleep) -> Catalog:
    base = normalise_base(website)
    try:  # follow the homepage redirect to the store's real address (e.g. shop.pk -> pk.shop.pk)
        landing = remote.safe_get(base + "/", session, max_bytes=CATALOG_BYTES)
        base = normalise_base(landing.url)
    except (remote.LinkError, requests.RequestException):
        pass
    client = PoliteClient(base, session=session, sleep=sleep)
    notes = []
    for platform, reader in [("Shopify", shopify), ("WooCommerce", woocommerce), ("Website (product pages)", schema_org)]:
        rows = reader(client, limit)
        if rows:
            products = tidy(pd.DataFrame(rows, columns=COLUMNS)).head(limit)
            if len(products) >= limit:
                notes.append(f"Stopped at {limit:,} products.")
            return Catalog(products, platform, base, notes)
    if client.refused:
        raise CatalogError("This website refused automated access (it blocks bots), so its catalogue can't be "
                           "imported this way. If it's your store, export products as CSV instead.")
    if not client.allowed(urljoin(base, "/products.json")) and not client.allowed(urljoin(base, "/sitemap.xml")):
        raise CatalogError("This website's robots.txt does not allow automated access, so its catalogue can't be "
                           "imported. If it's your store, export products as CSV instead.")
    raise CatalogError("No product catalogue was found on this website. Supported: Shopify, WooCommerce, and sites "
                       "that publish schema.org product data. You can also upload a products CSV.")
