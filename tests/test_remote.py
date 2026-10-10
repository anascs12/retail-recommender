import socket

import pytest

from basketwise import remote


@pytest.mark.parametrize("link, expected_url, source", [
    ("https://docs.google.com/spreadsheets/d/AbC_123/edit#gid=456",
     "https://docs.google.com/spreadsheets/d/AbC_123/export?format=csv&gid=456", "Google Sheets"),
    ("https://docs.google.com/spreadsheets/d/AbC_123/edit?usp=sharing",
     "https://docs.google.com/spreadsheets/d/AbC_123/export?format=xlsx", "Google Sheets"),
    ("https://drive.google.com/file/d/FiLe-9/view?usp=sharing",
     "https://drive.google.com/uc?export=download&id=FiLe-9", "Google Drive"),
    ("https://www.dropbox.com/scl/fi/xyz/orders.csv?rlkey=k&dl=0",
     "https://www.dropbox.com/scl/fi/xyz/orders.csv?rlkey=k&dl=1", "Dropbox"),
    ("https://shop.example.com/exports/orders.xlsx", "https://shop.example.com/exports/orders.xlsx", "Direct link"),
])
def test_share_links_become_download_links(link, expected_url, source):
    url, detected, _ = remote.to_download_url(link)
    assert url == expected_url and detected == source


def test_non_web_links_are_rejected():
    with pytest.raises(remote.LinkError, match="https://"):
        remote.to_download_url("file:///etc/passwd")


def fake_dns(mapping):
    def getaddrinfo(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (mapping[host], port))]
    return getaddrinfo


@pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.5", "192.168.1.10", "169.254.169.254", "0.0.0.0"])
def test_internal_addresses_are_blocked(monkeypatch, address):
    monkeypatch.setattr(socket, "getaddrinfo", fake_dns({"evil.example": address}))
    with pytest.raises(remote.LinkError, match="private or internal"):
        remote.check_public_host("https://evil.example/orders.csv")


class FakeResponse:
    def __init__(self, status=200, body=b"", headers=None):
        self.status_code, self.body, self.headers = status, body, headers or {}
        self.is_redirect = status in (301, 302, 303, 307, 308)

    def iter_content(self, chunk_size):
        for i in range(0, len(self.body), chunk_size):
            yield self.body[i:i + chunk_size]


class FakeSession:
    def __init__(self, responses):
        self.responses, self.requested = responses, []

    def get(self, url, **kwargs):
        assert kwargs["allow_redirects"] is False   # redirects are followed manually and re-checked
        self.requested.append(url)
        return self.responses[url]


PUBLIC = {"files.example": "93.184.216.34", "internal.example": "10.1.2.3"}


def test_redirect_to_an_internal_address_is_blocked(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", fake_dns(PUBLIC))
    session = FakeSession({"https://files.example/orders.csv":
                           FakeResponse(302, headers={"location": "https://internal.example/secret"})})
    with pytest.raises(remote.LinkError, match="private or internal"):
        remote.fetch("https://files.example/orders.csv", session)


def test_successful_download_with_file_name(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", fake_dns(PUBLIC))
    body = b"Order ID,Item Name,Qty,Order Date\n1,Mug,2,2025-01-01\n"
    session = FakeSession({"https://files.example/orders.csv": FakeResponse(
        200, body, {"content-disposition": 'attachment; filename="march_orders.csv"', "content-length": str(len(body))})})
    fetched = remote.fetch("https://files.example/orders.csv", session)
    assert fetched.name == "march_orders.csv" and fetched.data == body


def test_private_files_get_a_helpful_message(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", fake_dns(PUBLIC))
    session = FakeSession({"https://files.example/orders.csv": FakeResponse(403)})
    with pytest.raises(remote.LinkError, match="private"):
        remote.fetch("https://files.example/orders.csv", session)


def test_html_pages_are_not_treated_as_data(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", fake_dns(PUBLIC))
    session = FakeSession({"https://files.example/orders.csv": FakeResponse(200, b"<!DOCTYPE html><html>Sign in")})
    with pytest.raises(remote.LinkError, match="web page"):
        remote.fetch("https://files.example/orders.csv", session)


def test_oversized_files_are_refused(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", fake_dns(PUBLIC))
    monkeypatch.setattr(remote, "MAX_BYTES", 10)
    session = FakeSession({"https://files.example/orders.csv": FakeResponse(200, b"x" * 50)})
    with pytest.raises(remote.LinkError, match="larger than"):
        remote.fetch("https://files.example/orders.csv", session)
