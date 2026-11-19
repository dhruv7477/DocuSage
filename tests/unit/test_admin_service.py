"""
tests/unit/test_admin_service.py
──────────────────────────────────
Unit tests for AdminService — all DB calls mocked.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.exceptions import InsufficientPermissionsError, UserNotFoundError
from src.models.audit_log import AuditLog
from src.models.user import User, UserRole
from src.services.admin_service import AdminService


def _make_user(
    user_id=None,
    email="alice@example.com",
    role=UserRole.MEMBER,
    is_active=True,
):
    user = User()
    user.id = user_id or uuid.uuid4()
    user.email = email
    user.full_name = "Alice"
    user.role = role
    user.is_active = is_active
    user.is_email_verified = True
    user.created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return user


def _make_db():
    db = AsyncMock()
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    db.execute = AsyncMock()
    db.add = MagicMock()
    db.refresh = AsyncMock()
    return db


class TestListUsers:
    @pytest.mark.asyncio
    async def test_returns_users(self):
        db = _make_db()
        user = _make_user()

        # Mock count query
        count_result = MagicMock()
        count_result.scalar_one.return_value = 1

        # Mock users query
        users_result = MagicMock()
        users_result.scalars.return_value.all.return_value = [user]

        # Mock workspace/doc count subqueries
        sub_result = MagicMock()
        sub_result.scalar_one_or_none.return_value = 2

        db.execute.side_effect = [count_result, users_result, sub_result, sub_result]

        service = AdminService(db)
        users, total = await service.list_users()

        assert total == 1
        assert len(users) == 1
        assert users[0].email == "alice@example.com"

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_users(self):
        db = _make_db()

        count_result = MagicMock()
        count_result.scalar_one.return_value = 0

        users_result = MagicMock()
        users_result.scalars.return_value.all.return_value = []

        db.execute.side_effect = [count_result, users_result]

        service = AdminService(db)
        users, total = await service.list_users()

        assert total == 0
        assert users == []


class TestGetUser:
    @pytest.mark.asyncio
    async def test_returns_user_by_id(self):
        db = _make_db()
        user = _make_user()

        fetch_result = MagicMock()
        fetch_result.scalar_one_or_none.return_value = user

        sub_result = MagicMock()
        sub_result.scalar_one_or_none.return_value = 0

        db.execute.side_effect = [fetch_result, sub_result, sub_result]

        service = AdminService(db)
        view = await service.get_user(user.id)

        assert view.email == user.email
        assert view.id == str(user.id)

    @pytest.mark.asyncio
    async def test_raises_not_found_for_missing_user(self):
        db = _make_db()

        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        db.execute.return_value = result

        service = AdminService(db)
        with pytest.raises(UserNotFoundError):
            await service.get_user(uuid.uuid4())


class TestUpdateUser:
    @pytest.mark.asyncio
    async def test_disable_user(self):
        db = _make_db()
        user = _make_user()
        admin_id = uuid.uuid4()

        fetch_result = MagicMock()
        fetch_result.scalar_one_or_none.return_value = user
        sub_result = MagicMock()
        sub_result.scalar_one_or_none.return_value = 0

        db.execute.side_effect = [
            fetch_result,  # _fetch_user in update_user
            MagicMock(),  # the UPDATE stmt
            fetch_result,  # _fetch_user in get_user (re-fetch)
            sub_result,  # ws count
            sub_result,  # doc count
        ]

        service = AdminService(db)
        await service.update_user(user.id, is_active=False, acting_admin_id=admin_id)

        assert db.commit.called

    @pytest.mark.asyncio
    async def test_cannot_deactivate_self(self):
        db = _make_db()
        admin_id = uuid.uuid4()

        service = AdminService(db)
        with pytest.raises(InsufficientPermissionsError):
            await service.update_user(
                admin_id, is_active=False, acting_admin_id=admin_id
            )

    @pytest.mark.asyncio
    async def test_cannot_promote_to_super_admin(self):
        db = _make_db()
        service = AdminService(db)
        with pytest.raises(InsufficientPermissionsError):
            await service.update_user(
                uuid.uuid4(),
                role="super_admin",
                acting_admin_id=uuid.uuid4(),
            )


class TestGetStats:
    @pytest.mark.asyncio
    async def test_returns_stats_dataclass(self):
        db = _make_db()

        def _scalar_result(n):
            r = MagicMock()
            r.scalar_one_or_none.return_value = n
            return r

        # 6 scalar queries + 1 group-by query
        group_result = MagicMock()
        group_result.all.return_value = []

        db.execute.side_effect = [
            _scalar_result(10),  # total_users
            _scalar_result(8),  # active_users
            _scalar_result(5),  # total_workspaces
            _scalar_result(20),  # total_documents
            _scalar_result(15),  # total_chats
            _scalar_result(100),  # total_messages
            group_result,  # documents_by_status
        ]

        service = AdminService(db)
        stats = await service.get_stats()

        assert stats.total_users == 10
        assert stats.active_users == 8
        assert stats.total_documents == 20


class TestListAuditLogs:
    @pytest.mark.asyncio
    async def test_returns_paginated_logs(self):
        db = _make_db()

        entry = AuditLog()
        entry.id = 1
        entry.user_id = str(uuid.uuid4())
        entry.user_email = "alice@example.com"
        entry.action = "document.upload"
        entry.resource_type = "document"
        entry.resource_id = str(uuid.uuid4())
        entry.method = "POST"
        entry.path = "/api/v1/documents/upload"
        entry.status_code = 202
        entry.ip_address = "127.0.0.1"
        entry.duration_ms = 45
        entry.extra = None
        entry.created_at = datetime(2025, 6, 1, tzinfo=timezone.utc)

        count_result = MagicMock()
        count_result.scalar_one.return_value = 1

        logs_result = MagicMock()
        logs_result.scalars.return_value.all.return_value = [entry]

        db.execute.side_effect = [count_result, logs_result]

        service = AdminService(db)
        logs, total = await service.list_audit_logs()

        assert total == 1
        assert logs[0].action == "document.upload"
        assert logs[0].status_code == 202

    @pytest.mark.asyncio
    async def test_returns_empty_list(self):
        db = _make_db()

        count_result = MagicMock()
        count_result.scalar_one.return_value = 0

        logs_result = MagicMock()
        logs_result.scalars.return_value.all.return_value = []

        db.execute.side_effect = [count_result, logs_result]

        service = AdminService(db)
        logs, total = await service.list_audit_logs()

        assert total == 0
        assert logs == []
