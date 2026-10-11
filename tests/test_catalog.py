import json
import socket

import pandas as pd
import pytest

from basketwise import catalog, similar

BASE = "https://shop.example"


class Resp:
    def __init__(self, status=200, body=b"", headers=None):
        self.status_code, self.body, self.headers = status, body, headers or {}
        self.is_redirect = status in (301, 302, 303, 307, 308)

    def iter_content(self, chunk_size):
        yield self.body


class FakeSite:
    """Serves a dict of path -> (status, body); anything else is a 404."""

    def __init__(self, pages):
        self.pages, self.requested = pages, []

    def get(self, url, **kwargs):
        path = url.replace(BASE, "") or "/"
        self.requested.append(path)
        status, body = self.pages.get(path, (404, b""))
        return Resp(status, body if isinstance(body, bytes) else body.encode())


@pytest.fixture(autouse=True)
def public_dns(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo",
                        lambda host, port, *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))])


def no_sleep(_):
    pass


SHOPIFY_PAGE = json.dumps({"products": [
    {"id": 1, "title": "Blue Ceramic Mug", "handle": "blue-mug", "body_html": "<p>Hand-made <b>mug</b></p>",
     "product_type": "Mugs", "vendor": "Clay Co", "tags": ["kitchen", "gift"],
     "variants": [{"sku": "MUG-B", "price": "1200.00"}], "images": [{"src": "https://cdn/x.jpg"}]},
    {"id": 2, "title": "Gift Card", "handle": "gift-card", "product_type": "", "vendor": "Clay Co", "tags": [],
     "variants": [{"sku": "GC", "price": "1000"}]},
]})


def test_shopify_catalogue_is_read_and_gift_cards_are_dropped():
    site = FakeSite({"/products.json?limit=250&page=1": (200, SHOPIFY_PAGE),
                     "/products.json?limit=250&page=2": (200, json.dumps({"products": []}))})
    result = catalog.import_catalog(BASE, session=site, sleep=no_sleep)
    assert result.platform == "Shopify"
    row = result.products.iloc[0]
    assert len(result.products) == 1
    assert (row["product_id"], row["category"], row["price"]) == ("MUG-B", "Mugs", 1200.0)
    assert row["description"] == "Hand-made mug" and row["url"] == f"{BASE}/products/blue-mug"


def test_woocommerce_store_api_prices_use_minor_units():
    products = [{"id": 7, "sku": "", "name": "Lawn Suit", "short_description": "<p>3-piece</p>",
                 "categories": [{"name": "Unstitched"}], "tags": [{"name": "summer"}],
                 "prices": {"price": "450000", "currency_minor_unit": 2}, "permalink": f"{BASE}/p/lawn"}]
    site = FakeSite({"/wp-json/wc/store/v1/products?per_page=100&page=1": (200, json.dumps(products)),
                     "/wp-json/wc/store/v1/products?per_page=100&page=2": (200, "[]")})
    result = catalog.import_catalog(BASE, session=site, sleep=no_sleep)
    row = result.products.iloc[0]
    assert result.platform == "WooCommerce" and row["price"] == 4500.0 and row["category"] == "Unstitched"


SITEMAP = f"""<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
  xmlns:image="http://www.google.com/schemas/sitemap-image/1.1">
  <url><loc>{BASE}/collections/body-mists/products/a.html</loc>
       <image:image><image:loc>{BASE}/img/a.jpg</image:loc></image:image></url>
  <url><loc>{BASE}/collections/body-mists/products/b.html</loc></url>
</urlset>"""
PAGE_JSONLD = """<html><script type="application/ld+json">{"@type": "ProductGroup", "name": "Rose Mist",
  "hasVariant": [{"offers": {"price": "1500"}}], "sku": "RM-1"}</script></html>"""
PAGE_META = """<html><head><meta property="og:title" content="Oud Mist | Shop Example">
  <meta property="og:site_name" content="Shop Example"><meta property="product:price:amount" content="2,100">
  </head></html>"""


def test_product_pages_fallback_reads_json_ld_and_open_graph_but_not_images():
    site = FakeSite({"/robots.txt": (200, f"User-agent: *\nAllow: /\nSitemap: {BASE}/sitemap_products_1.xml"),
                     "/sitemap_products_1.xml": (200, SITEMAP),
                     "/collections/body-mists/products/a.html": (200, PAGE_JSONLD),
                     "/collections/body-mists/products/b.html": (200, PAGE_META)})
    result = catalog.import_catalog(BASE, session=site, sleep=no_sleep)
    products = result.products.set_index("title")
    assert result.platform == "Website (product pages)"
    assert products.loc["Rose Mist", "price"] == 1500.0           # ProductGroup → first variant's offer
    assert products.loc["Oud Mist", "price"] == 2100.0            # Open Graph, site name stripped
    assert set(products["category"]) == {"body mists"}            # category taken from the URL
    assert "/img/a.jpg" not in site.requested                     # image:loc entries are not crawled


def test_robots_txt_is_respected():
    site = FakeSite({"/robots.txt": (200, "User-agent: *\nDisallow: /")})
    with pytest.raises(catalog.CatalogError, match="robots.txt"):
        catalog.import_catalog(BASE, session=site, sleep=no_sleep)
    assert all(p in {"/", "/robots.txt"} for p in site.requested)


def test_sites_that_block_bots_get_a_clear_message():
    site = FakeSite({"/products.json?limit=250&page=1": (403, ""), "/sitemap.xml": (403, "")})
    with pytest.raises(catalog.CatalogError, match="blocks bots"):
        catalog.import_catalog(BASE, session=site, sleep=no_sleep)


def test_requests_are_paced():
    pauses = []
    site = FakeSite({"/products.json?limit=250&page=1": (200, SHOPIFY_PAGE),
                     "/products.json?limit=250&page=2": (200, json.dumps({"products": []}))})
    catalog.import_catalog(BASE, session=site, sleep=pauses.append)
    assert pauses and all(p > 0 for p in pauses)


def test_similar_products_prefer_same_kind_and_price_range():
    products = pd.DataFrame({
        "product_id": ["m1", "m2", "m3", "s1"],
        "title": ["Blue Ceramic Mug", "Red Ceramic Mug", "Gold Ceramic Mug Luxury", "Cotton Lawn Suit"],
        "description": ["", "", "", ""], "category": ["Mugs", "Mugs", "Mugs", "Suits"],
        "brand": ["", "", "", ""], "tags": ["kitchen", "kitchen", "kitchen", "summer"],
        "price": [1000, 1100, 9000, 4500], "url": ["u1", "u2", "u3", "u4"], "image": [None] * 4,
    })
    pairs = similar.similar_products(products, per_product=2)
    blue = pairs[pairs["product_id"] == "m1"]
    assert blue.iloc[0]["similar_id"] == "m2"                     # same kind and similar price ranks first
    assert "s1" not in set(blue["similar_id"])
