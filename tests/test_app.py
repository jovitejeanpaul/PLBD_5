"""
test_app.py
============
Tests des endpoints REST et du flux d'authentification de l'app FastAPI.

    pytest tests/test_app.py -v
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="module")
def db_init(tmp_path_factory):
    """Initialise une DB temporaire pour les tests."""
    import app.database as db_mod
    tmp = tmp_path_factory.mktemp("db")
    db_mod.DATABASE = tmp / "test.db"
    db_mod.initialize_database()
    return db_mod.DATABASE


@pytest.fixture
def client(db_init):
    """Client HTTP synchrone pour tester l'app."""
    from httpx import AsyncClient, ASGITransport
    from app.main import app
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


# ══════════════════════════════════════════════════════════════════════
# AUTHENTIFICATION
# ══════════════════════════════════════════════════════════════════════

class TestAuth:

    @pytest.mark.anyio
    async def test_root_redirects_without_auth(self, client):
        r = await client.get("/", follow_redirects=False)
        assert r.status_code == 302
        assert r.headers["location"] == "/login"

    @pytest.mark.anyio
    async def test_login_page_returns_html(self, client):
        r = await client.get("/login")
        assert r.status_code == 200
        assert "AQUA-SENS" in r.text

    @pytest.mark.anyio
    async def test_login_bad_password(self, client):
        r = await client.post("/login", json={"username": "admin", "password": "wrong"})
        assert r.status_code == 401

    @pytest.mark.anyio
    async def test_login_unknown_user(self, client):
        r = await client.post("/login", json={"username": "nobody", "password": "x"})
        assert r.status_code == 401

    @pytest.mark.anyio
    async def test_login_success_sets_cookie(self, client):
        r = await client.post("/login", json={"username": "admin", "password": "Admin@2026"})
        assert r.status_code == 200
        assert "access_token" in r.headers.get("set-cookie", "")

    @pytest.mark.anyio
    async def test_authenticated_access(self, client):
        await client.post("/login", json={"username": "admin", "password": "Admin@2026"})
        r = await client.get("/", follow_redirects=False)
        assert r.status_code == 200

    @pytest.mark.anyio
    async def test_logout_clears_cookie(self, client):
        await client.post("/login", json={"username": "admin", "password": "Admin@2026"})
        r = await client.post("/logout")
        assert r.status_code == 200
        r2 = await client.get("/", follow_redirects=False)
        assert r2.status_code == 302

    @pytest.mark.anyio
    async def test_me_returns_role(self, client):
        await client.post("/login", json={"username": "admin", "password": "Admin@2026"})
        r = await client.get("/me")
        assert r.status_code == 200
        assert r.json()["role"] == "administrator"


# ══════════════════════════════════════════════════════════════════════
# INSCRIPTION
# ══════════════════════════════════════════════════════════════════════

class TestRegistration:

    @pytest.mark.anyio
    async def test_register_creates_operator(self, client):
        r = await client.post("/register", json={"username": "newuser", "password": "Test@123"})
        assert r.status_code == 200
        assert r.json()["user"]["role"] == "operator"

    @pytest.mark.anyio
    async def test_register_duplicate_rejected(self, client):
        await client.post("/register", json={"username": "dup", "password": "Test@123"})
        r = await client.post("/register", json={"username": "dup", "password": "Test@123"})
        assert r.status_code == 400

    @pytest.mark.anyio
    async def test_registered_user_can_login(self, client):
        await client.post("/register", json={"username": "logintest", "password": "Test@123"})
        r = await client.post("/login", json={"username": "logintest", "password": "Test@123"})
        assert r.status_code == 200


# ══════════════════════════════════════════════════════════════════════
# ROLES
# ══════════════════════════════════════════════════════════════════════

class TestRoles:

    @pytest.mark.anyio
    async def test_operator_cannot_activate_filter(self, client):
        await client.post("/register", json={"username": "op_filter", "password": "Test@123"})
        await client.post("/login", json={"username": "op_filter", "password": "Test@123"})
        r = await client.post("/api/filter/activate")
        assert r.status_code == 403

    @pytest.mark.anyio
    async def test_operator_cannot_list_users(self, client):
        await client.post("/register", json={"username": "op_users", "password": "Test@123"})
        await client.post("/login", json={"username": "op_users", "password": "Test@123"})
        r = await client.get("/api/users")
        assert r.status_code == 403

    @pytest.mark.anyio
    async def test_operator_can_view_status(self, client):
        await client.post("/register", json={"username": "op_status", "password": "Test@123"})
        await client.post("/login", json={"username": "op_status", "password": "Test@123"})
        r = await client.get("/api/status")
        assert r.status_code == 200

    @pytest.mark.anyio
    async def test_admin_can_create_user(self, client):
        await client.post("/login", json={"username": "admin", "password": "Admin@2026"})
        r = await client.post("/api/users", json={"username": "created", "password": "x", "role": "operator"})
        assert r.status_code == 200

    @pytest.mark.anyio
    async def test_admin_can_delete_user(self, client):
        await client.post("/login", json={"username": "admin", "password": "Admin@2026"})
        await client.post("/api/users", json={"username": "todelete", "password": "x", "role": "operator"})
        r = await client.delete("/api/users/todelete")
        assert r.status_code == 200

    @pytest.mark.anyio
    async def test_cannot_delete_admin(self, client):
        await client.post("/login", json={"username": "admin", "password": "Admin@2026"})
        r = await client.delete("/api/users/admin")
        assert r.status_code == 400


# ══════════════════════════════════════════════════════════════════════
# API PROTEGEES
# ══════════════════════════════════════════════════════════════════════

class TestProtectedAPIs:

    @pytest.mark.anyio
    async def test_shap_without_auth(self, client):
        await client.post("/logout")
        r = await client.get("/api/shap")
        assert r.status_code == 401

    @pytest.mark.anyio
    async def test_status_without_auth(self, client):
        await client.post("/logout")
        r = await client.get("/api/status")
        assert r.status_code == 401

    @pytest.mark.anyio
    async def test_models_endpoint(self, client):
        await client.post("/login", json={"username": "admin", "password": "Admin@2026"})
        r = await client.get("/api/models")
        assert r.status_code == 200
        d = r.json()
        assert "active_model" in d
        assert "models" in d

    @pytest.mark.anyio
    async def test_filter_status_endpoint(self, client):
        await client.post("/login", json={"username": "admin", "password": "Admin@2026"})
        r = await client.get("/api/filter/status")
        assert r.status_code == 200
        assert "running" in r.json()

    @pytest.mark.anyio
    async def test_history_endpoints(self, client):
        await client.post("/login", json={"username": "admin", "password": "Admin@2026"})
        for endpoint in ["/api/history/diagnostic", "/api/history/filtration", "/api/history/forecast"]:
            r = await client.get(endpoint)
            assert r.status_code == 200
            assert isinstance(r.json(), list)

    @pytest.mark.anyio
    async def test_config_get(self, client):
        await client.post("/login", json={"username": "admin", "password": "Admin@2026"})
        r = await client.get("/api/config")
        assert r.status_code == 200
        assert "pump_duration_s" in r.json()

    @pytest.mark.anyio
    async def test_invalid_model_select(self, client):
        await client.post("/login", json={"username": "admin", "password": "Admin@2026"})
        r = await client.post("/api/models/select/nonexistent")
        assert r.status_code == 404
