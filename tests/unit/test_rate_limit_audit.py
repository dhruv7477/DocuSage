"""
tests/unit/test_rate_limit_audit.py
─────────────────────────────────────
Unit tests for rate limiting helpers and audit log utilities.
No real HTTP server needed — tests the pure logic.
"""

from unittest.mock import MagicMock


from src.core.audit import _get_client_ip, _resolve_action
from src.core.rate_limit import chat_limit, default_limit


# ── _resolve_action ───────────────────────────────────────────────

class TestResolveAction:
    def test_skips_health(self):
        assert _resolve_action("GET", "/health") is None

    def test_skips_docs(self):
        assert _resolve_action("GET", "/docs") is None

    def test_skips_plain_get_list(self):
        # GET /api/v1/documents is not sensitive — skip
        assert _resolve_action("GET", "/api/v1/documents") is None

    def test_maps_register(self):
        assert _resolve_action("POST", "/api/v1/auth/register") == "auth.register"

    def test_maps_login(self):
        assert _resolve_action("POST", "/api/v1/auth/login") == "auth.login"

    def test_maps_get_me(self):
        # /me is sensitive GET
        assert _resolve_action("GET", "/api/v1/auth/me") == "auth.me"

    def test_maps_document_upload(self):
        assert _resolve_action("POST", "/api/v1/documents/upload") == "document.upload"

    def test_maps_document_delete(self):
        assert _resolve_action("DELETE", "/api/v1/documents/abc-123") == "document.delete"

    def test_maps_chat_query(self):
        assert _resolve_action("POST", "/api/v1/chat/abc/query") == "chat.query"

    def test_maps_chat_delete(self):
        assert _resolve_action("DELETE", "/api/v1/chat/abc") == "chat.delete"

    def test_maps_admin_read(self):
        assert _resolve_action("GET", "/api/v1/admin/users") == "admin.read"

    def test_maps_admin_update(self):
        assert _resolve_action("PATCH", "/api/v1/admin/users/abc") == "admin.update_user"

    def test_unmapped_post_gets_fallback(self):
        action = _resolve_action("POST", "/api/v1/unknown/endpoint")
        assert action == "api.post"

    def test_unmapped_get_returns_none(self):
        assert _resolve_action("GET", "/api/v1/unknown/endpoint") is None


# ── _get_client_ip ────────────────────────────────────────────────

class TestGetClientIp:
    def _make_request(self, forwarded_for=None, client_host=None):
        request = MagicMock()
        headers = {}
        if forwarded_for:
            headers["x-forwarded-for"] = forwarded_for
        request.headers = headers
        if client_host:
            request.client = MagicMock()
            request.client.host = client_host
        else:
            request.client = None
        return request

    def test_uses_x_forwarded_for(self):
        req = self._make_request(forwarded_for="1.2.3.4, 5.6.7.8")
        assert _get_client_ip(req) == "1.2.3.4"

    def test_uses_client_host_when_no_forwarded(self):
        req = self._make_request(client_host="9.10.11.12")
        assert _get_client_ip(req) == "9.10.11.12"

    def test_returns_none_when_no_client(self):
        req = self._make_request()
        assert _get_client_ip(req) is None

    def test_x_forwarded_for_strips_whitespace(self):
        req = self._make_request(forwarded_for="  203.0.113.1  , 10.0.0.1")
        assert _get_client_ip(req) == "203.0.113.1"


# ── Rate limit string helpers ─────────────────────────────────────

class TestRateLimitHelpers:
    def test_default_limit_format(self):
        limit = default_limit()
        assert "/minute" in limit
        assert limit.split("/")[0].isdigit()

    def test_chat_limit_format(self):
        limit = chat_limit()
        assert "/minute" in limit
        # Chat limit should be <= default limit
        chat_n = int(chat_limit().split("/")[0])
        default_n = int(default_limit().split("/")[0])
        assert chat_n <= default_n

    def test_chat_limit_is_more_restrictive(self):
        chat_n = int(chat_limit().split("/")[0])
        default_n = int(default_limit().split("/")[0])
        assert chat_n < default_n
