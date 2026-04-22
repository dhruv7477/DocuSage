"""
tests/unit/test_security.py
─────────────────────────────
Unit tests for src/core/security.py — no DB, no FastAPI.
Pure function tests with maximum coverage.
"""

import pytest

from src.core.exceptions import TokenExpiredError, TokenInvalidError
from src.core.security import (
    ACCESS_TOKEN_TYPE,
    REFRESH_TOKEN_TYPE,
    create_access_token,
    create_refresh_token,
    decode_token,
    extract_user_id,
    hash_password,
    verify_password,
)

SECRET = "test-secret-key-at-least-32-characters-long"
ALGO = "HS256"
USER_ID = "123e4567-e89b-12d3-a456-426614174000"


# ── Password Tests ────────────────────────────────────────────────

class TestPasswordHashing:
    def test_hash_is_not_plaintext(self):
        hashed = hash_password("MyPassword1")
        assert hashed != "MyPassword1"

    def test_verify_correct_password(self):
        hashed = hash_password("MyPassword1")
        assert verify_password("MyPassword1", hashed) is True

    def test_verify_wrong_password(self):
        hashed = hash_password("MyPassword1")
        assert verify_password("WrongPassword", hashed) is False

    def test_different_hashes_same_input(self):
        """bcrypt uses random salt — two hashes of same input differ."""
        h1 = hash_password("MyPassword1")
        h2 = hash_password("MyPassword1")
        assert h1 != h2


# ── Token Tests ───────────────────────────────────────────────────

class TestAccessToken:
    def test_create_and_decode(self):
        token = create_access_token(USER_ID, SECRET, ALGO, expires_minutes=30)
        payload = decode_token(token, SECRET, ALGO, ACCESS_TOKEN_TYPE)
        assert payload["sub"] == USER_ID
        assert payload["type"] == ACCESS_TOKEN_TYPE

    def test_extra_claims_included(self):
        token = create_access_token(
            USER_ID, SECRET, ALGO, 30, extra_claims={"role": "admin"}
        )
        payload = decode_token(token, SECRET, ALGO)
        assert payload["role"] == "admin"

    def test_wrong_secret_raises(self):
        token = create_access_token(USER_ID, SECRET, ALGO, 30)
        with pytest.raises(TokenInvalidError):
            decode_token(token, "wrong-secret-key-padding-padding-pad", ALGO)

    def test_wrong_type_raises(self):
        """Access token should not pass as refresh token."""
        token = create_access_token(USER_ID, SECRET, ALGO, 30)
        with pytest.raises(TokenInvalidError):
            decode_token(token, SECRET, ALGO, expected_type=REFRESH_TOKEN_TYPE)

    def test_expired_token_raises(self):
        token = create_access_token(USER_ID, SECRET, ALGO, expires_minutes=-1)
        with pytest.raises(TokenExpiredError):
            decode_token(token, SECRET, ALGO)


class TestRefreshToken:
    def test_create_and_decode(self):
        token = create_refresh_token(USER_ID, SECRET, ALGO, expires_days=7)
        payload = decode_token(token, SECRET, ALGO, REFRESH_TOKEN_TYPE)
        assert payload["sub"] == USER_ID
        assert payload["type"] == REFRESH_TOKEN_TYPE


class TestExtractUserId:
    def test_extract_valid(self):
        token = create_access_token(USER_ID, SECRET, ALGO, 30)
        payload = decode_token(token, SECRET, ALGO)
        assert extract_user_id(payload) == USER_ID

    def test_missing_sub_raises(self):
        with pytest.raises(TokenInvalidError):
            extract_user_id({})
