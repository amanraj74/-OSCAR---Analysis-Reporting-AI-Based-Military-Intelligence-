"""Tests for the Wikipedia client (no live HTTP required for unit tests)."""

from __future__ import annotations


def test_normalize_title_basic():
    from src.ingestion.wikipedia import WikipediaClient

    assert WikipediaClient.normalize_title("Wagner Group") == "Wagner_Group"
    assert WikipediaClient.normalize_title("F-16") == "F-16"
    assert WikipediaClient.normalize_title("Kyiv, Ukraine") == "Kyiv"
    assert WikipediaClient.normalize_title("Hamas (Palestinian group)") == "Hamas"
    assert WikipediaClient.normalize_title("") == ""
    assert WikipediaClient.normalize_title("  U.S.A.  ") == "USA"


def test_normalize_title_handles_special_chars():
    from src.ingestion.wikipedia import WikipediaClient

    # Parentheticals are stripped
    assert WikipediaClient.normalize_title("ATACMS (M142) missiles") == "ATACMS_missiles"
    # Slashes and quotes are removed
    assert "Iron_Dome" in WikipediaClient.normalize_title("Iron Dome / David's Sling")


def test_normalize_title_strips_punctuation():
    from src.ingestion.wikipedia import WikipediaClient

    assert WikipediaClient.normalize_title("B-21 'Raider'") == "B-21_Raider"
    assert WikipediaClient.normalize_title("F-16/F-15") == "F-16F-15"


def test_wikipedia_client_init():
    from src.ingestion.wikipedia import WikipediaClient

    client = WikipediaClient()
    assert "OSCAR" in client._session.headers["User-Agent"]


def test_get_summary_handles_network_failure():
    from src.ingestion.wikipedia import WikipediaClient

    client = WikipediaClient()
    summary = client.get_summary("")
    assert summary is None


def test_get_summary_handles_invalid_title():
    from src.ingestion.wikipedia import WikipediaClient

    client = WikipediaClient()
    summary = client.get_summary("!!!@@@###$$$")
    assert summary is None


def test_get_pageviews_handles_invalid_input():
    from src.ingestion.wikipedia import WikipediaClient

    client = WikipediaClient()
    assert client.get_pageviews("") == 0
    assert client.get_pageviews("!!!@@@###$$$") == 0


def test_get_summary_parses_response():
    """Test the parsing logic with a mock response."""
    from unittest.mock import MagicMock

    from src.ingestion.wikipedia import WikipediaClient

    client = WikipediaClient()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "title": "F-16 Fighting Falcon",
        "extract": "The F-16 Fighting Falcon is a single-engine multirole fighter aircraft.",
        "description": "Single-engine multirole fighter",
        "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/F-16"}},
        "thumbnail": {"source": "https://upload.wikimedia.org/..."},
    }
    client._session.get = MagicMock(return_value=mock_resp)

    summary = client.get_summary("F-16")
    assert summary is not None
    assert summary["title"] == "F-16 Fighting Falcon"
    assert "F-16" in summary["extract"]
    assert "wikipedia.org" in summary["url"]
    assert "F-16" in summary["url"]


def test_get_summary_handles_none_thumbnail():
    """Wikipedia returns null for articles without images."""
    from unittest.mock import MagicMock

    from src.ingestion.wikipedia import WikipediaClient

    client = WikipediaClient()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "title": "Test",
        "extract": "An extract.",
        "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Test"}},
        "thumbnail": None,
    }
    client._session.get = MagicMock(return_value=mock_resp)

    summary = client.get_summary("Test")
    assert summary is not None
    assert summary["thumbnail"] is None


def test_get_pageviews_sums_list_items():
    """Wikipedia returns items as a list of {accesses: int} dicts."""
    from unittest.mock import MagicMock

    from src.ingestion.wikipedia import WikipediaClient

    client = WikipediaClient()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "items": [
            {"project": "en.wikipedia", "article": "F-16", "accesses": 100},
            {"project": "en.wikipedia", "article": "F-16", "accesses": 150},
            {"project": "en.wikipedia", "article": "F-16", "accesses": 200},
        ]
    }
    client._session.get = MagicMock(return_value=mock_resp)

    total = client.get_pageviews("F-16", days=3)
    assert total == 450


def test_get_pageviews_handles_dict_items():
    """Fallback for when items is a dict (older endpoints do this)."""
    from unittest.mock import MagicMock

    from src.ingestion.wikipedia import WikipediaClient

    client = WikipediaClient()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "items": {
            "20250601": 100,
            "20250602": 150,
        }
    }
    client._session.get = MagicMock(return_value=mock_resp)

    total = client.get_pageviews("F-16", days=2)
    assert total == 250


def test_get_pageviews_handles_empty_items():
    from unittest.mock import MagicMock

    from src.ingestion.wikipedia import WikipediaClient

    client = WikipediaClient()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"items": []}
    client._session.get = MagicMock(return_value=mock_resp)

    assert client.get_pageviews("Test") == 0


def test_get_pageviews_handles_404():
    from unittest.mock import MagicMock

    from src.ingestion.wikipedia import WikipediaClient

    client = WikipediaClient()
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    client._session.get = MagicMock(return_value=mock_resp)

    assert client.get_pageviews("NonexistentArticle_xyz123") == 0


def test_wikipedia_summary_keys():
    """Verify the summary dict has all expected keys."""
    from unittest.mock import MagicMock

    from src.ingestion.wikipedia import WikipediaClient

    client = WikipediaClient()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "title": "Test",
        "extract": "An extract.",
        "description": "A desc.",
        "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Test"}},
        "thumbnail": None,
    }
    client._session.get = MagicMock(return_value=mock_resp)

    summary = client.get_summary("Test")
    assert set(summary.keys()) >= {
        "title",
        "extract",
        "url",
        "thumbnail",
        "description",
    }
