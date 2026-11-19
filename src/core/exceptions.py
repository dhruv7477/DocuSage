"""
src/core/exceptions.py
──────────────────────
Custom exception hierarchy for DocuSage.

Design:
- All domain exceptions inherit from DocuSageError
- Each exception carries an HTTP status code and error code string
- FastAPI exception handlers in main.py convert these to JSON responses
- This keeps HTTP concerns OUT of the service layer
"""

from http import HTTPStatus


class DocuSageError(Exception):
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


class AuthenticationError(DocuSageError):
    """Invalid credentials or expired token."""

    status_code = HTTPStatus.UNAUTHORIZED.value
    error_code = "AUTHENTICATION_FAILED"


class TokenExpiredError(DocuSageError):
    """JWT token has expired."""

    status_code = HTTPStatus.UNAUTHORIZED.value
    error_code = "TOKEN_EXPIRED"


class TokenInvalidError(DocuSageError):
    """JWT token is malformed or tampered."""

    status_code = HTTPStatus.UNAUTHORIZED.value
    error_code = "TOKEN_INVALID"


class InsufficientPermissionsError(DocuSageError):
    """User lacks required role/permission for this action."""

    status_code = HTTPStatus.FORBIDDEN.value
    error_code = "INSUFFICIENT_PERMISSIONS"


# ── User Exceptions ──────────────────────────────────────────────


class UserNotFoundError(DocuSageError):
    status_code = HTTPStatus.NOT_FOUND.value
    error_code = "USER_NOT_FOUND"


class UserAlreadyExistsError(DocuSageError):
    status_code = HTTPStatus.CONFLICT.value
    error_code = "USER_ALREADY_EXISTS"


class UserInactiveError(DocuSageError):
    status_code = HTTPStatus.FORBIDDEN.value
    error_code = "USER_INACTIVE"


# ── Workspace Exceptions ─────────────────────────────────────────


class WorkspaceNotFoundError(DocuSageError):
    status_code = HTTPStatus.NOT_FOUND.value
    error_code = "WORKSPACE_NOT_FOUND"


class WorkspaceAccessDeniedError(DocuSageError):
    """User is not a member of this workspace."""

    status_code = HTTPStatus.FORBIDDEN.value
    error_code = "WORKSPACE_ACCESS_DENIED"


class WorkspaceLimitReachedError(DocuSageError):
    status_code = HTTPStatus.PAYMENT_REQUIRED.value
    error_code = "WORKSPACE_LIMIT_REACHED"


class QuotaExceededError(DocuSageError):
    """A per-user quota (documents, storage, questions per day) is used up."""

    status_code = HTTPStatus.TOO_MANY_REQUESTS.value
    error_code = "QUOTA_EXCEEDED"


# ── Document Exceptions ──────────────────────────────────────────


class DocumentNotFoundError(DocuSageError):
    status_code = HTTPStatus.NOT_FOUND.value
    error_code = "DOCUMENT_NOT_FOUND"


class DocumentAccessDeniedError(DocuSageError):
    """User does not own this document."""

    status_code = HTTPStatus.FORBIDDEN.value
    error_code = "DOCUMENT_ACCESS_DENIED"


class UnsupportedFileTypeError(DocuSageError):
    status_code = HTTPStatus.UNPROCESSABLE_ENTITY.value
    error_code = "UNSUPPORTED_FILE_TYPE"


class FileTooLargeError(DocuSageError):
    status_code = HTTPStatus.REQUEST_ENTITY_TOO_LARGE.value
    error_code = "FILE_TOO_LARGE"


class DocumentProcessingError(DocuSageError):
    status_code = HTTPStatus.INTERNAL_SERVER_ERROR.value
    error_code = "DOCUMENT_PROCESSING_FAILED"


# ── Chat Exceptions ──────────────────────────────────────────────


class ChatNotFoundError(DocuSageError):
    status_code = HTTPStatus.NOT_FOUND.value
    error_code = "CHAT_NOT_FOUND"


class ChatAccessDeniedError(DocuSageError):
    status_code = HTTPStatus.FORBIDDEN.value
    error_code = "CHAT_ACCESS_DENIED"


# ── Infrastructure Exceptions ────────────────────────────────────


class VectorStoreError(DocuSageError):
    status_code = HTTPStatus.INTERNAL_SERVER_ERROR.value
    error_code = "VECTOR_STORE_ERROR"


class StorageError(DocuSageError):
    status_code = HTTPStatus.INTERNAL_SERVER_ERROR.value
    error_code = "STORAGE_ERROR"


class LLMProviderError(DocuSageError):
    status_code = HTTPStatus.BAD_GATEWAY.value
    error_code = "LLM_PROVIDER_ERROR"


class LLMRateLimitedError(LLMProviderError):
    """The hosted LLM refused the call with a rate limit (HTTP 429)."""

    status_code = HTTPStatus.SERVICE_UNAVAILABLE.value
    error_code = "LLM_RATE_LIMITED"


class EmbeddingError(DocuSageError):
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


class RateLimitExceededError(DocuSageError):
    status_code = HTTPStatus.TOO_MANY_REQUESTS.value
    error_code = "RATE_LIMIT_EXCEEDED"
