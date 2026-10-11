"""Fetch an order export from a shared link (Google Sheets / Drive, Dropbox, OneDrive or a direct URL).

Safety: the app may run on a public server, so links are fetched defensively. Only http(s),
no private, loopback or link-local addresses (including after redirects), a size cap and timeouts.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from dataclasses import dataclass
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

import requests

MAX_BYTES = 100 * 1024 * 1024
TIMEOUT = (10, 60)          # connect, read (seconds)
MAX_REDIRECTS = 5
USER_AGENT = "BasketWise/0.1 (+order import)"


class LinkError(ValueError):
    """A problem the store owner can fix; the message is shown in the app."""


@dataclass
class Fetched:
    name: str
    data: bytes
    source: str      # e.g. "Google Sheets"


def to_download_url(url: str) -> tuple[str, str, str]:
    """Turn a share link into a direct download link. Returns (url, source, suggested file name)."""
    url = url.strip()
    parsed = urlparse(url)
    host = parsed.netloc.lower()

    if "docs.google.com" in host and "/spreadsheets/d/" in parsed.path:
        sheet_id = re.search(r"/spreadsheets/d/([A-Za-z0-9_-]+)", parsed.path).group(1)
        gid = parse_qs(parsed.query).get("gid", [None])[0]
        if gid is None:  # the tab id usually sits in the fragment: .../edit#gid=123
            fragment_gid = re.search(r"gid=(\d+)", parsed.fragment or "")
            gid = fragment_gid.group(1) if fragment_gid else None
        query = {"format": "xlsx"} if gid is None else {"format": "csv", "gid": gid}
        name = "google_sheet.xlsx" if gid is None else "google_sheet.csv"
        return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?{urlencode(query)}", "Google Sheets", name

    if "drive.google.com" in host:
        match = re.search(r"/file/d/([A-Za-z0-9_-]+)", parsed.path) or re.search(r"[?&]id=([A-Za-z0-9_-]+)", url)
        if not match:
            raise LinkError("This Google Drive link doesn't point to a single file. Open the file and copy its link.")
        return f"https://drive.google.com/uc?export=download&id={match.group(1)}", "Google Drive", "drive_file"

    if host.endswith("dropbox.com"):
        query = parse_qs(parsed.query)
        query["dl"] = ["1"]
        return urlunparse(parsed._replace(query=urlencode(query, doseq=True))), "Dropbox", parsed.path.rsplit("/", 1)[-1]

    if host in {"1drv.ms", "onedrive.live.com"} or host.endswith(".sharepoint.com"):
        sep = "&" if parsed.query else "?"
        return f"{url}{sep}download=1", "OneDrive", "onedrive_file"

    if parsed.scheme not in {"http", "https"}:
        raise LinkError("Please paste a link that starts with https://")
    return url, "Direct link", parsed.path.rsplit("/", 1)[-1] or "download"


def check_public_host(url: str) -> None:
    """Refuse links that resolve to private, loopback, link-local or otherwise internal addresses."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise LinkError("Please paste a link that starts with https://")
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise LinkError("That website could not be found. Check the link.") from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global:
            raise LinkError("Links to private or internal network addresses are not allowed.")


def _file_name(response, fallback: str) -> str:
    disposition = response.headers.get("content-disposition", "")
    match = re.search(r"filename\*?=(?:UTF-8'')?\"?([^\";]+)", disposition)
    name = match.group(1) if match else fallback
    content_type = response.headers.get("content-type", "")
    if not name.lower().endswith((".csv", ".xlsx", ".xls")):
        name += ".xlsx" if "spreadsheetml" in content_type or "excel" in content_type else ".csv"
    return name


@dataclass
class SafeResponse:
    status: int
    url: str                      # final URL after redirects
    headers: dict
    data: bytes


def safe_get(url: str, session: requests.Session | None = None, max_bytes: int | None = None,
             headers: dict | None = None) -> SafeResponse:
    """GET with every hop checked against internal addresses, manual redirects, timeouts and a size cap."""
    session = session or requests.Session()
    limit = max_bytes or MAX_BYTES
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        check_public_host(current)
        response = session.get(current, stream=True, timeout=TIMEOUT, allow_redirects=False,
                               headers={"User-Agent": USER_AGENT, **(headers or {})})
        if response.is_redirect or response.status_code in (301, 302, 303, 307, 308):
            current = urljoin(current, response.headers.get("location", ""))
            continue
        break
    else:
        raise LinkError("The link redirected too many times.")
    declared = int(response.headers.get("content-length") or 0)
    if declared > limit:
        raise LinkError(f"The file is larger than {limit // 1024 // 1024} MB.")
    chunks, total = [], 0
    if response.status_code < 400:
        for chunk in response.iter_content(chunk_size=1 << 16):
            total += len(chunk)
            if total > limit:
                raise LinkError(f"The file is larger than {limit // 1024 // 1024} MB.")
            chunks.append(chunk)
    return SafeResponse(response.status_code, current, dict(response.headers), b"".join(chunks))


def fetch(url: str, session: requests.Session | None = None) -> Fetched:
    download_url, source, fallback = to_download_url(url)
    response = safe_get(download_url, session)
    current = response.url

    if response.status in (401, 403) or (source == "Google Sheets" and "accounts.google.com" in current):
        raise LinkError(f"This {source} file is private. Set sharing to “Anyone with the link can view” and try again.")
    if response.status == 404:
        raise LinkError("Nothing was found at that link. Check that it is correct and still shared.")
    if response.status >= 400:
        raise LinkError(f"The link returned an error ({response.status}).")
    data = response.data

    head = data[:2000].lstrip().lower()
    if head.startswith((b"<!doctype html", b"<html")):
        if source == "Direct link":
            raise LinkError("That link opens a web page, not a CSV or Excel file. Paste the link to your exported "
                            "orders file itself. A store's website address can't be used, because orders are private.")
        raise LinkError(f"{source} returned a web page instead of the file. Make sure it's shared with “Anyone with "
                        "the link can view” (Google Drive may also refuse to share very large files this way).")
    return Fetched(name=_file_name(response, fallback), data=data, source=source)
