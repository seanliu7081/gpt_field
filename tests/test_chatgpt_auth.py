"""Authentication boundary tests; all network and credentials are synthetic."""

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import stat
import time
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest

jwt = pytest.importorskip("jwt")
from cryptography.hazmat.primitives.asymmetric import rsa

from gpt_field.chatgpt_auth import (AUTHORIZE_URL, DIRECT_SCOPE, DISCOVERY_URL, ISSUER,
                                  JWKS_URL, RESOURCE, SCOPES, TOKEN_URL, AuthError,
                                  AuthorizationAttempt, OpenAIAuthService,
                                  SubscriptionCredentials, TokenStore)


CLIENT = "oaiapp_test_registration"
SUBJECT = "synthetic-account-identity"


@pytest.fixture(scope="module")
def signing_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def id_token(key, **overrides):
    now = int(time.time())
    claims = {"iss": ISSUER, "sub": SUBJECT, "aud": CLIENT, "iat": now - 1,
              "exp": now + 3600, "nonce": "expected-nonce", "email": "synthetic@example.test"}
    claims.update(overrides)
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": "test-key"})


class FakeServer:
    def __init__(self, key):
        public = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key()))
        self.keys = {"keys": [{**public, "kid": "test-key", "use": "sig", "alg": "RS256"}]}
        self.discovery = {"issuer": ISSUER, "authorization_endpoint": AUTHORIZE_URL,
                          "token_endpoint": TOKEN_URL, "jwks_uri": JWKS_URL,
                          "id_token_signing_alg_values_supported": ["RS256"]}
        self.response = {"id_token": id_token(key), "access_token": "synthetic-access-secret",
                         "refresh_token": "synthetic-refresh-secret", "scope": SCOPES,
                         "expires_in": 3600, "token_type": "Bearer"}
        self.calls = []

    def __call__(self, url, form=None):
        self.calls.append((url, form))
        if url == DISCOVERY_URL:
            return dict(self.discovery)
        if url == JWKS_URL:
            return dict(self.keys)
        assert url == TOKEN_URL
        return dict(self.response)


@pytest.fixture
def server(signing_key):
    return FakeServer(signing_key)


def make_attempt(**kwargs):
    return AuthorizationAttempt("urn:uuid:907b7be6-b7a6-479c-8c2c-aeb6ef74b4bf",
                                "http://127.0.0.1:1455/auth/callback", nonce="expected-nonce", **kwargs)


def callback(attempt, **overrides):
    values = {"state": attempt.state, "code": "synthetic-authorization-code", "client_id": CLIENT}
    values.update(overrides)
    return urlencode(values)


def saved_record():
    now = time.time()
    return {"schema_version": 1, "issuer": ISSUER, "subject": SUBJECT,
            "email": "synthetic@example.test", "client_id": CLIENT,
            "ext_agent_host_id": "urn:uuid:907b7be6-b7a6-479c-8c2c-aeb6ef74b4bf",
            "auth_nonce": "expected-nonce", "id_token": "synthetic-retained-id-secret",
            "access_token": "synthetic-access-secret", "refresh_token": "synthetic-refresh-secret",
            "token_type": "Bearer", "scopes": SCOPES.split(), "expires_at": now + 3600,
            "saved_at": now}


def test_registration_uses_pkce_and_issued_client_id_in_exchange(server):
    attempt = make_attempt()
    url = urlsplit(attempt.authorization_url())
    params = parse_qs(url.query)
    assert url.scheme == "https" and url.netloc == "auth.openai.com"
    assert params["client_id"] == ["dynamic_agent_client"]
    assert params["code_challenge_method"] == ["S256"]
    assert params["code_challenge"] != [attempt.verifier]
    assert "id_token_hint" not in params
    result = attempt.accept_callback(callback(attempt), OpenAIAuthService(server))
    assert result["client_id"] == CLIENT and result["subject"] == SUBJECT
    form = next(form for url, form in server.calls if url == TOKEN_URL)
    assert form["client_id"] == CLIENT
    assert form["code_verifier"] == attempt.verifier
    assert form["redirect_uri"] == "http://127.0.0.1:1455/auth/callback"
    assert form["resource"] == RESOURCE
    assert "client_secret" not in form
    with pytest.raises(AuthError, match="already used"):
        attempt.accept_callback(callback(attempt), OpenAIAuthService(server))


@pytest.mark.parametrize("query", ["state=wrong", "state=%E6%B5%8B%E8%AF%95", "state=a&state=b"])
def test_invalid_states_reject_before_exchange(server, query):
    attempt = make_attempt()
    with pytest.raises(AuthError):
        attempt.accept_callback(query, OpenAIAuthService(server))
    assert server.calls == [] and not attempt.consumed


def test_oauth_error_validates_state_and_never_exchanges(server):
    attempt = make_attempt()
    with pytest.raises(AuthError, match="denied"):
        attempt.accept_callback(callback(attempt, error="access_denied"), OpenAIAuthService(server))
    assert attempt.consumed and server.calls == []


@pytest.mark.parametrize("issued", ["dynamic_agent_client", "", "../escape", "oaiapp_x/escape"])
def test_registration_requires_issued_safe_id(server, issued):
    attempt = make_attempt()
    with pytest.raises(AuthError, match="client ID"):
        attempt.accept_callback(callback(attempt, client_id=issued), OpenAIAuthService(server))
    assert server.calls == []


def test_returning_account_reuses_client_and_accepts_missing_callback_client(server):
    attempt = make_attempt(selected_record=saved_record())
    params = parse_qs(urlsplit(attempt.authorization_url()).query)
    assert params["client_id"] == [CLIENT]
    assert params["id_token_hint"] == ["synthetic-retained-id-secret"]
    assert "agent_name_hint" not in params
    result = attempt.accept_callback(urlencode({"state": attempt.state, "code": "code"}), OpenAIAuthService(server))
    assert result["client_id"] == CLIENT


def test_returning_client_substitution_rejected_before_exchange(server):
    attempt = make_attempt(selected_record=saved_record())
    with pytest.raises(AuthError, match="differs"):
        attempt.accept_callback(callback(attempt, client_id="oaiapp_other"), OpenAIAuthService(server))
    assert server.calls == []


def test_expired_authorization_attempt_rejected(server):
    attempt = make_attempt(created_at=time.monotonic() - 601)
    with pytest.raises(AuthError, match="expired"):
        attempt.accept_callback(callback(attempt), OpenAIAuthService(server))
    assert server.calls == []


@pytest.mark.parametrize("uri", ["http://localhost:1455/auth/callback", "https://127.0.0.1:1455/auth/callback",
                                 "http://127.0.0.1:1455/callback", "http://0.0.0.0:1455/auth/callback",
                                 "http://127.0.0.1:1455/auth/callback?extra=1"])
def test_redirect_requires_exact_loopback_callback(uri):
    with pytest.raises(AuthError, match="loopback"):
        AuthorizationAttempt("host", uri)


@pytest.mark.parametrize("changed", [{"iss": "https://wrong.example"}, {"aud": "oaiapp_someone_else"},
                                     {"exp": 1}, {"nonce": "wrong"}, {"sub": ""}, {"iat": time.time() + 300},
                                     {"azp": "oaiapp_someone_else"}, {"aud": [CLIENT, "oaiapp_other"]}])
def test_invalid_signed_identity_claims_rejected(signing_key, server, changed):
    service = OpenAIAuthService(server)
    with pytest.raises(AuthError):
        service.validate_identity(id_token(signing_key, **changed), CLIENT, nonce="expected-nonce")


def test_invalid_signature_and_algorithm_rejected(signing_key, server):
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    with pytest.raises(AuthError):
        OpenAIAuthService(server).validate_identity(id_token(other), CLIENT, nonce="expected-nonce")
    bad = jwt.encode({"sub": SUBJECT}, "x" * 32, algorithm="HS256", headers={"kid": "test-key"})
    with pytest.raises(AuthError):
        OpenAIAuthService(server).validate_identity(bad, CLIENT, nonce="expected-nonce")


def test_returning_identity_mismatch_does_not_replace_registration(signing_key, server):
    server.response["id_token"] = id_token(signing_key, sub="a-different-account")
    attempt = make_attempt(selected_record=saved_record())
    with pytest.raises(AuthError, match="does not match"):
        attempt.accept_callback(callback(attempt), OpenAIAuthService(server))


def test_identity_without_direct_grant_cannot_authorize_inference(server):
    server.response["scope"] = "openid profile email"
    attempt = make_attempt()
    with pytest.raises(AuthError, match="plan use was not granted"):
        attempt.accept_callback(callback(attempt), OpenAIAuthService(server))


def test_discovery_does_not_allow_redirecting_token_destination(server):
    server.discovery["token_endpoint"] = "https://untrusted.example/token"
    with pytest.raises(AuthError, match="unexpected endpoints"):
        OpenAIAuthService(server).exchange({"grant_type": "refresh_token"})
    assert all(url != TOKEN_URL for url, _ in server.calls)


def test_profile_store_atomic_private_and_status_redacted(tmp_path):
    store = TokenStore(tmp_path / "private")
    host = store.host_id()
    assert TokenStore(store.path).host_id() == host
    store.save(saved_record(), make_active=True)
    for path in store.path.rglob("*"):
        assert stat.S_IMODE(path.stat().st_mode) == (0o700 if path.is_dir() else 0o600)
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o700
    serialized = json.dumps(store.status())
    assert store.status()["connected"]
    for secret in ["synthetic-access-secret", "synthetic-refresh-secret", "synthetic-retained-id-secret", SUBJECT, "synthetic@example.test"]:
        assert secret not in serialized
    assert not list(store.path.rglob(".pending-*"))


def test_profiles_remain_separate_and_failed_identity_cannot_overwrite(tmp_path):
    store = TokenStore(tmp_path)
    first = saved_record()
    store.save(first, make_active=True)
    second = {**first, "client_id": "oaiapp_second", "subject": "another-account"}
    store.save(second)
    assert store.load()["client_id"] == CLIENT
    store.select("oaiapp_second")
    assert store.load()["subject"] == "another-account"
    with pytest.raises(AuthError, match="another account"):
        store.save({**first, "subject": "substituted"}, make_active=True)
    assert store.load()["client_id"] == "oaiapp_second"
    assert store.load(CLIENT)["subject"] == SUBJECT


def test_symlinked_store_and_credential_are_rejected(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    with pytest.raises(AuthError, match="symlink"):
        TokenStore(alias)
    store = TokenStore(actual)
    store.save(saved_record(), make_active=True)
    profile = actual / "profiles" / (CLIENT + ".json")
    outside = tmp_path / "external.json"
    profile.rename(outside)
    profile.symlink_to(outside)
    with pytest.raises(AuthError):
        store.load()


def test_world_readable_credentials_rejected(tmp_path):
    store = TokenStore(tmp_path)
    store.save(saved_record(), make_active=True)
    (tmp_path / "profiles" / (CLIENT + ".json")).chmod(0o644)
    with pytest.raises(AuthError, match="permissions"):
        store.load()


def test_fresh_token_never_calls_network(tmp_path, server):
    provider = SubscriptionCredentials(tmp_path, service=OpenAIAuthService(server))
    provider.store.save(saved_record(), make_active=True)
    assert provider.access_token() == "synthetic-access-secret"
    assert server.calls == []


def test_refresh_rotates_atomically_retains_omitted_scopes_and_serializes(tmp_path, server):
    provider = SubscriptionCredentials(tmp_path, service=OpenAIAuthService(server))
    provider.store.save({**saved_record(), "expires_at": time.time() - 1}, make_active=True)
    server.response.pop("scope")
    server.response.pop("id_token")
    server.response.update(access_token="rotated-access", refresh_token="rotated-refresh")
    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = list(pool.map(lambda _: provider.access_token(), range(2)))
    assert outputs == ["rotated-access", "rotated-access"]
    forms = [form for url, form in server.calls if url == TOKEN_URL]
    assert len(forms) == 1
    assert forms[0] == {"grant_type": "refresh_token", "client_id": CLIENT,
                        "refresh_token": "synthetic-refresh-secret", "resource": RESOURCE}
    stored = provider.store.load()
    assert stored["refresh_token"] == "rotated-refresh"
    assert set(stored["scopes"]) == set(SCOPES.split())
    assert stored["id_token"] == "synthetic-retained-id-secret"


@pytest.mark.parametrize("scope", [SCOPES + " unexpected.admin", "openid profile email"])
def test_refresh_never_escalates_or_loses_required_grant(tmp_path, server, scope):
    provider = SubscriptionCredentials(tmp_path, service=OpenAIAuthService(server))
    original = {**saved_record(), "expires_at": time.time() - 1}
    provider.store.save(original, make_active=True)
    server.response["scope"] = scope
    with pytest.raises(AuthError):
        provider.access_token()
    assert provider.store.load() == original


def test_refresh_revalidates_new_id_token_identity(tmp_path, signing_key, server):
    provider = SubscriptionCredentials(tmp_path, service=OpenAIAuthService(server))
    original = {**saved_record(), "expires_at": time.time() - 1}
    provider.store.save(original, make_active=True)
    server.response["id_token"] = id_token(signing_key, sub="different-account")
    with pytest.raises(AuthError, match="does not match"):
        provider.access_token()
    assert provider.store.load() == original


def test_failed_atomic_write_keeps_previous_token_set(tmp_path, monkeypatch):
    store = TokenStore(tmp_path)
    original = saved_record()
    store.save(original, make_active=True)
    def fail_replace(*args):
        raise OSError("synthetic write failure")
    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(AuthError, match="Cannot save"):
        store.save({**original, "access_token": "new-secret"})
    assert store.load() == original
    assert not list(tmp_path.rglob(".pending-*"))


def test_cli_status_has_no_credentials_or_network(tmp_path, capsys):
    path = Path(__file__).resolve().parents[1] / "scripts/chatgpt_plan_login.py"
    spec = importlib.util.spec_from_file_location("chatgpt_plan_login_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--store-dir", str(tmp_path), "status"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output == {"connected": False, "selected_profile": None, "profiles": []}
