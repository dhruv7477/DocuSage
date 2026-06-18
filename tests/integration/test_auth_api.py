"""
tests/integration/test_auth_api.py
────────────────────────────────────
Integration tests for /api/v1/auth/* endpoints.

Uses FastAPI TestClient + mocked DB (no real Postgres needed).
Tests the full request → route → service → response cycle.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import create_app


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
def client(app):
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


class TestRegisterEndpoint:

    def test_register_returns_201(self, client):
        with patch("src.api.v1.auth.AuthService") as MockService:
            mock_svc = AsyncMock()
            mock_user = MagicMock()
            mock_user.id = uuid.uuid4()
            mock_user.email = "newuser@example.com"
            mock_user.full_name = "New User"
            mock_user.role = "member"
            mock_user.is_active = True
            mock_user.is_email_verified = False
            mock_svc.register.return_value = mock_user
            MockService.return_value = mock_svc

            resp = client.post(
                "/api/v1/auth/register",
                json={
                    "email": "newuser@example.com",
                    "password": "StrongPass1!",
                    "full_name": "New User",
                },
            )
        assert resp.status_code == 201

    def test_register_missing_email_returns_422(self, client):
        resp = client.post(
            "/api/v1/auth/register",
            json={"password": "StrongPass1!", "full_name": "No Email"},
        )
        assert resp.status_code == 422

    def test_register_short_password_returns_422(self, client):
        resp = client.post(
            "/api/v1/auth/register",
            json={"email": "x@x.com", "password": "short", "full_name": "X"},
        )
        assert resp.status_code == 422


class TestLoginEndpoint:

    def test_login_success_returns_tokens(self, client):
        with patch("src.api.v1.auth.AuthService") as MockService:
            mock_svc = AsyncMock()
            mock_svc.login.return_value = MagicMock(
                access_token="access.token.here",
                refresh_token="refresh.token.here",
                token_type="bearer",
            )
            MockService.return_value = mock_svc

            resp = client.post(
                "/api/v1/auth/login",
                json={"email": "alice@example.com", "password": "Password1!"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    def test_login_missing_fields_returns_422(self, client):
        resp = client.post("/api/v1/auth/login", json={"email": "x@x.com"})
        assert resp.status_code == 422

    def test_login_invalid_credentials_returns_401(self, client):
        from src.core.exceptions import AuthenticationError
        with patch("src.api.v1.auth.AuthService") as MockService:
            mock_svc = AsyncMock()
            mock_svc.login.side_effect = AuthenticationError("Invalid credentials.")
            MockService.return_value = mock_svc

            resp = client.post(
                "/api/v1/auth/login",
                json={"email": "x@x.com", "password": "wrongpass"},
            )
        assert resp.status_code == 401
        assert resp.json()["error_code"] == "AUTHENTICATION_FAILED"


class TestMeEndpoint:

    def test_me_without_token_returns_403(self, client):
        resp = client.get("/api/v1/auth/me")
        assert resp.status_code in (401, 403)

    def test_me_with_invalid_token_returns_401(self, client):
        resp = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer not.a.real.token"},
        )
        assert resp.status_code in (401, 403)

    def test_me_with_valid_token_returns_user(self, client):
        # Generate a real JWT, mock the DB lookup
        from src.configs.config import get_settings
        from src.core.security import create_access_token
        import uuid

        settings = get_settings()
        user_id = uuid.uuid4()
        token = create_access_token(
            subject=str(user_id),
            secret_key=settings.secret_key,
            algorithm=settings.algorithm,
            expires_minutes=settings.access_token_expire_minutes,
        )

        from datetime import datetime, timezone
        from src.models.user import User, UserRole
        now = datetime.now(timezone.utc)
        mock_user = User()
        mock_user.id = user_id
        mock_user.email = "alice@example.com"
        mock_user.full_name = "Alice"
        mock_user.role = UserRole.MEMBER
        mock_user.is_active = True
        mock_user.is_email_verified = True
        mock_user.created_at = now
        mock_user.updated_at = now

        # get_current_user resolves the user via its `db` dependency —
        # override get_db so the lookup hits a mock session, not Postgres.
        from src.db.session import get_db
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_db.execute = AsyncMock(return_value=mock_result)

        async def _fake_db():
            yield mock_db

        client.app.dependency_overrides[get_db] = _fake_db
        try:
            resp = client.get(
                "/api/v1/auth/me",
                headers={"Authorization": f"Bearer {token}"},
            )
        finally:
            client.app.dependency_overrides.pop(get_db, None)

        assert resp.status_code == 200
        assert resp.json()["email"] == "alice@example.com"


class TestHealthEndpoint:

    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
