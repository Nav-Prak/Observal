# SPDX-FileCopyrightText: 2026 Hari Srinivasan <harisrini21@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-only

import json
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True, scope="module")
def init_key_manager_for_module(tmp_path_factory):
    from services.crypto import init_key_manager

    key_dir = tmp_path_factory.mktemp("keys")
    init_key_manager(key_dir=str(key_dir), key_password=None)


class FakeRedis:
    def __init__(self):
        self.store: dict[str, str] = {}
        self.sets: dict[str, set[str]] = {}

    async def setex(self, key, _ttl, value):
        self.store[key] = value

    async def get(self, key):
        return self.store.get(key)

    async def getdel(self, key):
        value = self.store.get(key)
        self.store.pop(key, None)
        return value

    async def delete(self, *keys):
        for key in keys:
            self.store.pop(key, None)
            self.sets.pop(key, None)

    async def incr(self, key):
        value = int(self.store.get(key, "0")) + 1
        self.store[key] = str(value)
        return value

    async def expire(self, _key, _ttl):
        return True

    async def sadd(self, key, value):
        self.sets.setdefault(key, set()).add(value)

    async def smembers(self, key):
        return set(self.sets.get(key, set()))


def _make_user(**overrides):
    from models.user import UserRole

    user = MagicMock()
    user.id = overrides.get("id", uuid.uuid4())
    user.email = overrides.get("email", "user@example.com")
    user.username = overrides.get("username", "user")
    user.name = overrides.get("name", "Test User")
    user.role = overrides.get("role", UserRole.user)
    user.avatar_url = None
    user.created_at = overrides.get("created_at", datetime.now(UTC))
    user.org_id = overrides.get("org_id", uuid.uuid4())
    user.auth_provider = "local"
    user.verify_password = MagicMock(return_value=overrides.get("password_ok", True))
    user.set_password = MagicMock()
    return user


def _mock_db_for_user(user):
    result = MagicMock()
    result.scalar_one_or_none.return_value = user
    db = AsyncMock()
    db.execute = AsyncMock(return_value=result)
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    return db


def _app_with_db(db):
    from api.deps import get_db
    from api.ratelimit import limiter
    from main import app

    limiter.enabled = False

    async def _get_db():
        yield db

    app.dependency_overrides[get_db] = _get_db
    return app


def _cleanup_app():
    from main import app

    app.dependency_overrides.clear()


def _emitted_event_types(emit_mock):
    return [call.args[0].event_type for call in emit_mock.await_args_list]


@pytest.mark.asyncio
async def test_login_sets_http_only_refresh_cookie():
    from httpx import ASGITransport, AsyncClient

    from api.routes.auth import REFRESH_COOKIE_NAME

    user = _make_user(password_ok=True)
    app = _app_with_db(_mock_db_for_user(user))
    redis = FakeRedis()

    try:
        with patch("api.routes.auth.get_redis", return_value=redis):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.post("/api/v1/auth/login", json={"email": user.email, "password": "correct"})

        assert resp.status_code == 200
        assert resp.cookies.get(REFRESH_COOKIE_NAME)
        cookie = resp.headers["set-cookie"]
        assert REFRESH_COOKIE_NAME in cookie
        assert "HttpOnly" in cookie
        assert "SameSite=lax" in cookie
    finally:
        _cleanup_app()


@pytest.mark.asyncio
async def test_failed_logins_lock_account_by_user_id():
    from httpx import ASGITransport, AsyncClient

    from api.routes.auth import AUTH_FAIL_LIMIT
    from services.security_events import EventType

    user = _make_user(password_ok=False)
    app = _app_with_db(_mock_db_for_user(user))
    redis = FakeRedis()

    try:
        with (
            patch("api.routes.auth.get_redis", return_value=redis),
            patch("api.routes.auth.emit_security_event", new=AsyncMock()) as emit,
        ):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                for _ in range(AUTH_FAIL_LIMIT):
                    resp = await client.post("/api/v1/auth/login", json={"email": user.email, "password": "wrong"})
                    assert resp.status_code == 401

                locked = await client.post("/api/v1/auth/login", json={"email": user.email, "password": "wrong"})

        assert locked.status_code == 429
        assert "temporarily locked" in locked.json()["detail"]
        assert await redis.get(f"auth:lock:{user.id}") == "1"
        assert EventType.ACCOUNT_LOCKED in _emitted_event_types(emit)
    finally:
        _cleanup_app()


@pytest.mark.asyncio
async def test_successful_login_clears_failed_login_counter():
    from httpx import ASGITransport, AsyncClient

    user = _make_user(password_ok=False)
    app = _app_with_db(_mock_db_for_user(user))
    redis = FakeRedis()

    try:
        with (
            patch("api.routes.auth.get_redis", return_value=redis),
            patch("api.routes.auth.emit_security_event", new=AsyncMock()),
        ):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                failed = await client.post("/api/v1/auth/login", json={"email": user.email, "password": "wrong"})
                user.verify_password.return_value = True
                succeeded = await client.post(
                    "/api/v1/auth/login",
                    json={"email": user.email, "password": "correct"},
                )

        assert failed.status_code == 401
        assert succeeded.status_code == 200
        assert await redis.get(f"auth:fail:{user.id}") is None
        assert await redis.get(f"auth:lock:{user.id}") is None
    finally:
        _cleanup_app()


@pytest.mark.asyncio
async def test_token_endpoint_sets_http_only_refresh_cookie():
    from httpx import ASGITransport, AsyncClient

    from api.routes.auth import REFRESH_COOKIE_NAME

    user = _make_user(password_ok=True)
    app = _app_with_db(_mock_db_for_user(user))
    redis = FakeRedis()

    try:
        with patch("api.routes.auth.get_redis", return_value=redis):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.post("/api/v1/auth/token", json={"email": user.email, "password": "correct"})

        assert resp.status_code == 200
        assert resp.cookies.get(REFRESH_COOKIE_NAME)
        assert "HttpOnly" in resp.headers["set-cookie"]
    finally:
        _cleanup_app()


@pytest.mark.asyncio
async def test_exchange_code_sets_http_only_refresh_cookie():
    from httpx import ASGITransport, AsyncClient

    from api.routes.auth import REFRESH_COOKIE_NAME
    from services.jwt_service import create_access_token, create_refresh_token

    user = _make_user()
    access_token, _ = create_access_token(user.id, user.role)
    refresh_token, _ = create_refresh_token(user.id, user.role)
    redis = FakeRedis()
    redis.store["oauth_code:abc123"] = json.dumps(
        {
            "user_id": str(user.id),
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_in": 900,
        }
    )
    app = _app_with_db(_mock_db_for_user(user))

    try:
        with patch("api.routes.auth.get_redis", return_value=redis):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.post("/api/v1/auth/exchange", json={"code": "abc123"})

        assert resp.status_code == 200
        assert resp.cookies.get(REFRESH_COOKIE_NAME)
        assert "oauth_code:abc123" not in redis.store
    finally:
        _cleanup_app()


@pytest.mark.asyncio
async def test_refresh_can_use_http_only_cookie_and_rotates_cookie():
    from httpx import ASGITransport, AsyncClient

    from api.routes.auth import REFRESH_COOKIE_NAME
    from services.jwt_service import create_refresh_token
    from services.security_events import EventType

    user = _make_user()
    token, jti = create_refresh_token(user.id, user.role)
    redis = FakeRedis()
    await redis.setex(f"refresh_jti:{jti}", 86400, str(user.id))

    app = _app_with_db(_mock_db_for_user(user))

    try:
        with (
            patch("api.routes.auth.get_redis", return_value=redis),
            patch("api.routes.auth.emit_security_event", new=AsyncMock()) as emit,
        ):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                client.cookies.set(REFRESH_COOKIE_NAME, token)
                resp = await client.post("/api/v1/auth/token/refresh")

        assert resp.status_code == 200
        assert resp.json()["access_token"]
        assert resp.cookies.get(REFRESH_COOKIE_NAME)
        assert await redis.get(f"refresh_jti:{jti}") is None
        assert EventType.TOKEN_REFRESH in _emitted_event_types(emit)
    finally:
        _cleanup_app()


@pytest.mark.asyncio
async def test_refresh_rejects_user_level_revocation():
    from httpx import ASGITransport, AsyncClient

    from api.routes.auth import REFRESH_COOKIE_NAME
    from services.jwt_service import create_refresh_token

    user = _make_user()
    token, jti = create_refresh_token(user.id, user.role)
    redis = FakeRedis()
    await redis.setex(f"refresh_jti:{jti}", 86400, str(user.id))
    await redis.setex(f"revoked_user:{user.id}", 86400, "1")

    app = _app_with_db(_mock_db_for_user(user))

    try:
        with patch("api.routes.auth.get_redis", return_value=redis):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                client.cookies.set(REFRESH_COOKIE_NAME, token)
                resp = await client.post("/api/v1/auth/token/refresh")

        assert resp.status_code == 401
        assert "revoked or expired" in resp.json()["detail"]
    finally:
        _cleanup_app()


@pytest.mark.asyncio
async def test_change_password_revokes_sessions_and_clears_cookie():
    from httpx import ASGITransport, AsyncClient

    from api.deps import get_current_user
    from api.routes.auth import REFRESH_COOKIE_NAME
    from services.security_events import EventType

    user = _make_user(password_ok=True)
    db = AsyncMock()
    redis = FakeRedis()
    app = _app_with_db(db)

    async def _current_user():
        return user

    app.dependency_overrides[get_current_user] = _current_user

    try:
        with (
            patch("api.routes.auth.get_redis", return_value=redis),
            patch("api.routes.auth.emit_security_event", new=AsyncMock()) as emit,
        ):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                client.cookies.set(REFRESH_COOKIE_NAME, "old-refresh")
                resp = await client.put(
                    "/api/v1/auth/profile/password",
                    json={"current_password": "OldStr0ng!Pass#2026", "new_password": "NewStr0ng!Pass#2026"},
                    headers={"Authorization": "Bearer access"},
                )

        assert resp.status_code == 200
        assert await redis.get(f"revoked_user:{user.id}") == "1"
        assert f"{REFRESH_COOKIE_NAME}=" in resp.headers["set-cookie"]
        assert "Max-Age=0" in resp.headers["set-cookie"]
        assert EventType.PASSWORD_CHANGED in _emitted_event_types(emit)
    finally:
        _cleanup_app()


@pytest.mark.asyncio
async def test_logout_accepts_cookie_refresh_without_body():
    from httpx import ASGITransport, AsyncClient

    from api.deps import get_current_user
    from api.routes.auth import REFRESH_COOKIE_NAME
    from services.jwt_service import create_refresh_token

    user = _make_user()
    refresh_token, refresh_jti = create_refresh_token(user.id, user.role)
    redis = FakeRedis()
    await redis.setex(f"refresh_jti:{refresh_jti}", 86400, str(user.id))
    app = _app_with_db(AsyncMock())

    async def _current_user():
        return user

    app.dependency_overrides[get_current_user] = _current_user

    try:
        with patch("api.routes.auth.get_redis", return_value=redis):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                client.cookies.set(REFRESH_COOKIE_NAME, refresh_token)
                resp = await client.post("/api/v1/auth/logout")

        assert resp.status_code == 200
        assert await redis.get(f"revoked_user:{user.id}") == "1"
        assert await redis.get(f"refresh_jti:{refresh_jti}") is None
        assert "Max-Age=0" in resp.headers["set-cookie"]
    finally:
        _cleanup_app()


@pytest.mark.asyncio
async def test_revoke_accepts_cookie_refresh_without_body():
    from httpx import ASGITransport, AsyncClient

    from api.routes.auth import REFRESH_COOKIE_NAME
    from services.jwt_service import create_refresh_token

    user = _make_user()
    refresh_token, refresh_jti = create_refresh_token(user.id, user.role)
    redis = FakeRedis()
    await redis.setex(f"refresh_jti:{refresh_jti}", 86400, str(user.id))
    app = _app_with_db(AsyncMock())

    try:
        with patch("api.routes.auth.get_redis", return_value=redis):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                client.cookies.set(REFRESH_COOKIE_NAME, refresh_token)
                resp = await client.post("/api/v1/auth/token/revoke")

        assert resp.status_code == 200
        assert await redis.get(f"refresh_jti:{refresh_jti}") is None
        assert "Max-Age=0" in resp.headers["set-cookie"]
    finally:
        _cleanup_app()


@pytest.mark.asyncio
async def test_admin_password_reset_revokes_target_sessions():
    from api.routes.admin.users import reset_user_password
    from models.user import UserRole
    from schemas.admin import AdminResetPasswordRequest
    from services.security_events import EventType

    admin = _make_user(email="admin@example.com", role=UserRole.admin)
    target = _make_user(email="target@example.com", role=UserRole.user)
    db = _mock_db_for_user(target)
    redis = FakeRedis()

    with (
        patch("services.redis.get_redis", return_value=redis),
        patch("api.routes.admin.users.emit_security_event", new=AsyncMock()) as emit,
    ):
        resp = await reset_user_password(
            target.id,
            AdminResetPasswordRequest(new_password="NewStr0ng!Pass#2026"),
            db,
            admin,
        )

    assert resp["message"] == f"Password reset for {target.email}"
    assert await redis.get(f"revoked_user:{target.id}") == "1"
    assert await redis.get(f"must_change_password:{target.id}") == "1"
    assert EventType.ADMIN_PASSWORD_RESET in _emitted_event_types(emit)
