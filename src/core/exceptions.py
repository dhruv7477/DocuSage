"""
src/core/exceptions.py
──────────────────────
Custom exception hierarchy for PrivateAI.

Design:
- All domain exceptions inherit from PrivateAIException
- Each exception carries an HTTP status code and error code string
- FastAPI exception handlers in main.py convert these to JSON responses
- This keeps HTTP concerns OUT of the service layer
"""

from http import HTTPStatus


class PrivateAIException(Exception):
    """Base exception. All custom exceptions inherit from this."""

    status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR.value
    error_code: str = "INTERNAL_ERROR"

    def __init__(self, message: str, detail: str | None = None) -> None:
        self.message = message
        self.detail = detail
        super().__init__(message)

    def to_dict(self) -> dict:
        return {
            "error_code": self.error_code,
            "message": self.message,
            "detail": self.detail,
        }


# ── Auth Exceptions ──────────────────────────────────────────────

class AuthenticationError(PrivateAIException):
    """Invalid credentials or expired token."""
    status_code = HTTPStatus.UNAUTHORIZED.value
    error_code = "AUTHENTICATION_FAILED"


class TokenExpiredError(PrivateAIException):
    """JWT token has expired."""
    status_code = HTTPStatus.UNAUTHORIZED.value
    error_code = "TOKEN_EXPIRED"


class TokenInvalidError(PrivateAIException):
    """JWT token is malformed or tampered."""
    status_code = HTTPStatus.UNAUTHORIZED.value
    error_code = "TOKEN_INVALID"


class InsufficientPermissionsError(PrivateAIException):
    """User lacks required role/permission for this action."""
    status_code = HTTPStatus.FORBIDDEN.value
    error_code = "INSUFFICIENT_PERMISSIONS"


# ── User Exceptions ──────────────────────────────────────────────

class UserNotFoundError(PrivateAIException):
    status_code = HTTPStatus.NOT_FOUND.value
    error_code = "USER_NOT_FOUND"


class UserAlreadyExistsError(PrivateAIException):
    status_code = HTTPStatus.CONFLICT.value
    error_code = "USER_ALREADY_EXISTS"


class UserInactiveError(PrivateAIException):
    status_code = HTTPStatus.FORBIDDEN.value
    error_code = "USER_INACTIVE"


# ── Workspace Exceptions ─────────────────────────────────────────

class WorkspaceNotFoundError(PrivateAIException):
    status_code = HTTPStatus.NOT_FOUND.value
    error_code = "WORKSPACE_NOT_FOUND"


class WorkspaceAccessDeniedError(PrivateAIException):
    """User is not a member of this workspace."""
    status_code = HTTPStatus.FORBIDDEN.value
    error_code = "WORKSPACE_ACCESS_DENIED"


class WorkspaceLimitReachedError(PrivateAIException):
    status_code = HTTPStatus.PAYMENT_REQUIRED.value
    error_code = "WORKSPACE_LIMIT_REACHED"


# ── Document Exceptions ──────────────────────────────────────────

class DocumentNotFoundError(PrivateAIException):
    status_code = HTTPStatus.NOT_FOUND.value
    error_code = "DOCUMENT_NOT_FOUND"


class DocumentAccessDeniedError(PrivateAIException):
    """User does not own this document."""
    status_code = HTTPStatus.FORBIDDEN.value
    error_code = "DOCUMENT_ACCESS_DENIED"


class UnsupportedFileTypeError(PrivateAIException):
    status_code = HTTPStatus.UNPROCESSABLE_ENTITY.value
    error_code = "UNSUPPORTED_FILE_TYPE"


class FileTooLargeError(PrivateAIException):
    status_code = HTTPStatus.REQUEST_ENTITY_TOO_LARGE.value
    error_code = "FILE_TOO_LARGE"


class DocumentProcessingError(PrivateAIException):
    status_code = HTTPStatus.INTERNAL_SERVER_ERROR.value
    error_code = "DOCUMENT_PROCESSING_FAILED"


# ── Chat Exceptions ──────────────────────────────────────────────

class ChatNotFoundError(PrivateAIException):
    status_code = HTTPStatus.NOT_FOUND.value
    error_code = "CHAT_NOT_FOUND"


class ChatAccessDeniedError(PrivateAIException):
    status_code = HTTPStatus.FORBIDDEN.value
    error_code = "CHAT_ACCESS_DENIED"


# ── Infrastructure Exceptions ────────────────────────────────────

class VectorStoreError(PrivateAIException):
    status_code = HTTPStatus.INTERNAL_SERVER_ERROR.value
    error_code = "VECTOR_STORE_ERROR"


class StorageError(PrivateAIException):
    status_code = HTTPStatus.INTERNAL_SERVER_ERROR.value
    error_code = "STORAGE_ERROR"


class LLMProviderError(PrivateAIException):
    status_code = HTTPStatus.BAD_GATEWAY.value
    error_code = "LLM_PROVIDER_ERROR"


class EmbeddingError(PrivateAIException):
    status_code = HTTPStatus.INTERNAL_SERVER_ERROR.value
    error_code = "EMBEDDING_ERROR"


class EmbeddingDimensionMismatch(VectorStoreError):
    """
    Raised when an upsert attempts to write vectors of size N into a
    Qdrant collection that was created with size M ≠ N. This happens
    most commonly when `EMBEDDING_MODEL` is changed without reindexing.
    The detail message includes the remediation script command.
    """
    error_code = "EMBEDDING_DIMENSION_MISMATCH"


class RateLimitExceededError(PrivateAIException):
    status_code = HTTPStatus.TOO_MANY_REQUESTS.value
    error_code = "RATE_LIMIT_EXCEEDED"