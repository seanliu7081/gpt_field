"""Official Sign in with ChatGPT public-client authentication.

Credentials live outside experiment artifacts. This module deliberately neither
reads Codex's own login files nor falls back to an API key. Its network requests
are restricted to the public OpenAI OAuth/OIDC endpoints, never inference.
"""

from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import dataclass, field
import fcntl
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import tempfile
import time
from typing import Any, Callable, Iterator
import urllib.error
import urllib.parse
import urllib.request
import uuid


ISSUER = "https://auth.openai.com"
DISCOVERY_URL = ISSUER + "/.well-known/openid-configuration"
AUTHORIZE_URL = ISSUER + "/api/accounts/authorize"
TOKEN_URL = ISSUER + "/api/accounts/oauth/token"
JWKS_URL = ISSUER + "/.well-known/jwks.json"
RESOURCE = "https://api.openai.com/v1"
DIRECT_SCOPE = "chatgpt.tokens.use.direct"
SCOPES = "openid profile email offline_access resource.invoke " + DIRECT_SCOPE
APP_NAME = "GPT Field Robot Planning"
DEFAULT_STORE = Path(__file__).resolve().parents[2] / ".private" / "chatgpt_plan"
_CLIENT_ID = re.compile(r"oaiapp_[A-Za-z0-9_-]{1,240}\Z")
_MAX_JSON = 1024 * 1024


class AuthError(RuntimeError):
    """A deliberately sanitized error, safe to show without credential leakage."""


def _client_id(value: Any) -> str:
    if not isinstance(value, str) or not _CLIENT_ID.fullmatch(value):
        raise AuthError("The issued ChatGPT client ID is missing or invalid.")
    return value


def _secret(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 65536:
        raise AuthError(f"The {label} is missing or invalid.")
    return value


def _expiry(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AuthError("The token lifetime is invalid.")
    if not math.isfinite(value) or value <= 0:
        raise AuthError("The token lifetime is invalid.")
    return float(value)


def _scope_set(value: Any) -> set[str]:
    if not isinstance(value, str):
        raise AuthError("The token response did not provide granted scopes.")
    result = set(value.split())
    if DIRECT_SCOPE not in result:
        raise AuthError("ChatGPT plan use was not granted. Enable plan access and sign in again.")
    return result


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _request_json(url: str, form: dict[str, str] | None = None) -> dict[str, Any]:
    if url not in {DISCOVERY_URL, TOKEN_URL, JWKS_URL}:
        raise AuthError("Refusing an unexpected authentication endpoint.")
    headers = {"Accept": "application/json", "User-Agent": "gpt-field-siwc/1"}
    payload = None
    if form is not None:
        payload = urllib.parse.urlencode(form).encode("ascii")
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    request = urllib.request.Request(url, data=payload, headers=headers)
    try:
        with urllib.request.build_opener(_NoRedirect()).open(request, timeout=30) as response:
            data = response.read(_MAX_JSON + 1)
        if len(data) > _MAX_JSON:
            raise AuthError("The authentication response was unexpectedly large.")
        result = json.loads(data)
        if not isinstance(result, dict):
            raise AuthError("The authentication server returned invalid JSON.")
        return result
    except urllib.error.HTTPError as exc:
        # Never report response bodies, URLs, request data, or exception reprs.
        raise AuthError(f"Authentication request rejected (HTTP {exc.code}); sign in again.") from None
    except (OSError, ValueError, urllib.error.URLError):
        raise AuthError("Cannot reach the authentication service or parse its response.") from None


class OpenAIAuthService:
    def __init__(self, request_json: Callable = _request_json):
        self.request_json = request_json
        self._discovery_checked = False
        self._keys: dict[str, Any] | None = None
        self._keys_at = 0.0

    def discovery(self) -> None:
        if self._discovery_checked:
            return
        result = self.request_json(DISCOVERY_URL)
        expected = {"issuer": ISSUER, "authorization_endpoint": AUTHORIZE_URL,
                    "token_endpoint": TOKEN_URL, "jwks_uri": JWKS_URL}
        if any(result.get(key) != value for key, value in expected.items()):
            raise AuthError("The authentication discovery document has unexpected endpoints.")
        if "RS256" not in result.get("id_token_signing_alg_values_supported", []):
            raise AuthError("The authentication server no longer advertises the supported signature algorithm.")
        self._discovery_checked = True

    def exchange(self, form: dict[str, str]) -> dict[str, Any]:
        self.discovery()
        return self.request_json(TOKEN_URL, form)

    def validate_identity(self, token: str, client_id: str, *, nonce: str | None,
                          subject: str | None = None) -> dict[str, Any]:
        try:
            import jwt
        except ImportError:
            raise AuthError("Install the subscription extra (PyJWT[crypto]) before signing in.") from None
        self.discovery()
        try:
            header = jwt.get_unverified_header(token)
            kid = header.get("kid")
            if header.get("alg") != "RS256" or not isinstance(kid, str) or not kid or len(kid) > 256:
                raise AuthError("The ID token uses an unsupported signing key or algorithm.")
            if self._keys is None or time.time() - self._keys_at > 600:
                self._keys = self.request_json(JWKS_URL)
                self._keys_at = time.time()
            matches = self._matching_keys(kid)
            if not matches:
                self._keys = self.request_json(JWKS_URL)
                self._keys_at = time.time()
                matches = self._matching_keys(kid)
            if len(matches) != 1:
                raise AuthError("The ID token signing key could not be verified.")
            key = jwt.PyJWK.from_dict(matches[0], algorithm="RS256").key
            claims = jwt.decode(token, key, algorithms=["RS256"], audience=_client_id(client_id),
                                issuer=ISSUER, leeway=5,
                                options={"require": ["iss", "aud", "sub", "exp", "iat"]})
            token_subject = claims.get("sub")
            if not isinstance(token_subject, str) or not token_subject or len(token_subject) > 512:
                raise AuthError("The ID token has no valid account identity.")
            if nonce is not None and (not isinstance(claims.get("nonce"), str)
                                      or not hmac.compare_digest(claims["nonce"], nonce)):
                raise AuthError("The ID token nonce did not match this sign-in attempt.")
            if subject is not None and not hmac.compare_digest(token_subject, subject):
                raise AuthError("The signed-in account does not match the selected registration.")
            audience = claims["aud"]
            if isinstance(audience, list) and len(audience) > 1 and claims.get("azp") != client_id:
                raise AuthError("The ID token authorized party does not match this app.")
            if "azp" in claims and claims["azp"] != client_id:
                raise AuthError("The ID token authorized party does not match this app.")
            return claims
        except AuthError:
            raise
        except Exception:
            raise AuthError("The ID token signature or identity claims could not be verified.") from None

    def _matching_keys(self, kid: str) -> list[dict[str, Any]]:
        keys = self._keys.get("keys", []) if isinstance(self._keys, dict) else []
        if not isinstance(keys, list):
            return []
        return [key for key in keys if isinstance(key, dict) and key.get("kid") == kid
                and key.get("kty") == "RSA" and key.get("use", "sig") == "sig"
                and key.get("alg", "RS256") == "RS256"]


class TokenStore:
    """Protected profiles keyed by issued client ID, with serialized refreshes."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else DEFAULT_STORE
        self.path = self.path.absolute()
        self._directory(self.path)
        self._directory(self.path / "profiles")

    @staticmethod
    def _directory(path: Path) -> None:
        try:
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
            info = path.lstat()
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
                raise AuthError("Credential storage must be an owner-controlled directory, not a symlink.")
            path.chmod(0o700)
        except AuthError:
            raise
        except OSError:
            raise AuthError("Cannot create protected ChatGPT credential storage.") from None

    @contextmanager
    def locked(self) -> Iterator[None]:
        fd = None
        try:
            fd = os.open(self.path / ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise AuthError("The credential lock has unsafe ownership or permissions.")
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        except OSError:
            raise AuthError("Cannot lock protected ChatGPT credential storage.") from None
        finally:
            if fd is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)

    def _read(self, path: Path) -> dict[str, Any] | None:
        fd = None
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise AuthError("A credential file has unsafe ownership or permissions.")
            with os.fdopen(fd, "r", encoding="utf-8") as handle:
                fd = None
                data = handle.read(_MAX_JSON + 1)
            value = json.loads(data)
            if len(data) > _MAX_JSON or not isinstance(value, dict):
                raise AuthError("A protected credential record is invalid.")
            return value
        except FileNotFoundError:
            return None
        except AuthError:
            raise
        except (OSError, ValueError):
            raise AuthError("Cannot read a protected credential record.") from None
        finally:
            if fd is not None:
                os.close(fd)

    def _write(self, path: Path, value: dict[str, Any]) -> None:
        temporary = None
        try:
            if path.is_symlink():
                raise AuthError("Refusing a symlink in credential storage.")
            fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                os.fchmod(handle.fileno(), 0o600)
                json.dump(value, handle, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            temporary = None
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except AuthError:
            raise
        except (OSError, ValueError, TypeError):
            raise AuthError("Cannot save the protected credential record.") from None
        finally:
            if temporary is not None:
                os.unlink(temporary)

    def host_id(self) -> str:
        with self.locked():
            saved = self._read(self.path / "host.json")
            if saved is None:
                saved = {"ext_agent_host_id": "urn:uuid:" + str(uuid.uuid4())}
                self._write(self.path / "host.json", saved)
            value = saved.get("ext_agent_host_id")
            if not isinstance(value, str) or not value.startswith("urn:uuid:"):
                raise AuthError("The saved agent host identifier is invalid.")
            try:
                uuid.UUID(value[9:])
            except ValueError:
                raise AuthError("The saved agent host identifier is invalid.") from None
            return value

    def _active(self) -> str | None:
        active = self._read(self.path / "active.json")
        return _client_id(active.get("client_id")) if active else None

    def _load(self, client_id: str | None = None) -> dict[str, Any]:
        selected = _client_id(client_id) if client_id else self._active()
        if selected is None:
            raise AuthError("No ChatGPT account is connected. Run chatgpt_plan_login.py login first.")
        record = self._read(self.path / "profiles" / (selected + ".json"))
        if record is None or record.get("client_id") != selected:
            raise AuthError("The selected ChatGPT registration has no saved credentials.")
        self._validate_record(record)
        return record

    @staticmethod
    def _validate_record(record: dict[str, Any]) -> None:
        _client_id(record.get("client_id"))
        if record.get("issuer") != ISSUER or not isinstance(record.get("subject"), str) or not record["subject"]:
            raise AuthError("The saved account identity is invalid.")
        if record.get("token_type", "").lower() != "bearer":
            raise AuthError("The saved token type is invalid.")
        for name in ["access_token", "id_token"]:
            _secret(record.get(name), name.replace("_", " "))
        scopes = record.get("scopes")
        if not isinstance(scopes, list) or any(not isinstance(s, str) for s in scopes):
            raise AuthError("The saved granted scopes are invalid.")
        _scope_set(" ".join(scopes))
        _expiry(record.get("expires_at"))

    def load(self, client_id: str | None = None) -> dict[str, Any]:
        with self.locked():
            return self._load(client_id)

    def save(self, record: dict[str, Any], *, make_active: bool = False) -> None:
        self._validate_record(record)
        with self.locked():
            client_id = record["client_id"]
            old = self._read(self.path / "profiles" / (client_id + ".json"))
            if old and (old.get("subject") != record["subject"] or old.get("issuer") != record["issuer"]):
                raise AuthError("Refusing to replace a registration with another account's identity.")
            self._write(self.path / "profiles" / (client_id + ".json"), record)
            if make_active:
                self._write(self.path / "active.json", {"client_id": client_id})

    def select(self, client_id: str) -> None:
        with self.locked():
            record = self._load(client_id)
            self._write(self.path / "active.json", {"client_id": record["client_id"]})

    def status(self, client_id: str | None = None) -> dict[str, Any]:
        with self.locked():
            active = self._active()
            selected = _client_id(client_id) if client_id else active
            profiles = []
            for file in sorted((self.path / "profiles").glob("*.json")):
                record = self._load(file.stem)
                profiles.append({"client_id": record["client_id"], "active": record["client_id"] == active,
                                 "plan_use_granted": DIRECT_SCOPE in record["scopes"],
                                 "expires_at": record["expires_at"],
                                 "expired": record["expires_at"] <= time.time(),
                                 "refresh_available": bool(record.get("refresh_token"))})
            return {"connected": any(p["client_id"] == selected for p in profiles),
                    "selected_profile": selected, "profiles": profiles}


@dataclass(repr=False)
class AuthorizationAttempt:
    host_id: str
    redirect_uri: str
    selected_record: dict[str, Any] | None = field(default=None, repr=False)
    registered_client_id: str | None = None
    state: str = field(default_factory=lambda: secrets.token_urlsafe(32), repr=False)
    nonce: str = field(default_factory=lambda: secrets.token_urlsafe(32), repr=False)
    verifier: str = field(default_factory=lambda: secrets.token_urlsafe(64), repr=False)
    created_at: float = field(default_factory=time.monotonic)
    consumed: bool = False

    def __post_init__(self):
        parsed = urllib.parse.urlsplit(self.redirect_uri)
        if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or not parsed.port
                or parsed.path != "/auth/callback" or parsed.query or parsed.fragment
                or parsed.username or parsed.password):
            raise AuthError("OAuth requires an exact 127.0.0.1 loopback /auth/callback URI.")
        if self.selected_record:
            self.registered_client_id = _client_id(self.selected_record.get("client_id"))
        elif self.registered_client_id is not None:
            _client_id(self.registered_client_id)

    def authorization_url(self) -> str:
        challenge = base64.urlsafe_b64encode(hashlib.sha256(self.verifier.encode("ascii")).digest()).rstrip(b"=").decode("ascii")
        params = {"client_id": self.registered_client_id or "dynamic_agent_client",
                  "ext_agent_host_id": self.host_id, "response_type": "code",
                  "redirect_uri": self.redirect_uri, "scope": SCOPES, "resource": RESOURCE,
                  "state": self.state, "nonce": self.nonce, "code_challenge_method": "S256",
                  "code_challenge": challenge}
        if not self.registered_client_id:
            params["agent_name_hint"] = APP_NAME
        if self.selected_record:
            params["id_token_hint"] = self.selected_record["id_token"]
            email = self.selected_record.get("email")
            if isinstance(email, str) and email:
                params["login_hint"] = email
        return AUTHORIZE_URL + "?" + urllib.parse.urlencode(params)

    def accept_callback(self, query: str, service: OpenAIAuthService) -> dict[str, Any]:
        if self.consumed or time.monotonic() - self.created_at > 600:
            raise AuthError("This sign-in attempt has expired or was already used.")
        if len(query) > 16384:
            raise AuthError("The sign-in callback is invalid.")
        try:
            values = urllib.parse.parse_qs(query, keep_blank_values=True, max_num_fields=16)
        except ValueError:
            raise AuthError("The sign-in callback is invalid.") from None
        if any(len(value) != 1 for value in values.values()):
            raise AuthError("The sign-in callback contains duplicate parameters.")
        fields = {key: value[0] for key, value in values.items()}
        if not hmac.compare_digest(fields.get("state", "").encode("utf-8"), self.state.encode("utf-8")):
            raise AuthError("The sign-in state did not match. Continue the original browser sign-in.")
        self.consumed = True
        if "error" in fields:
            raise AuthError("ChatGPT authorization was denied or could not be completed.")
        issued = fields.get("client_id")
        if self.registered_client_id is None:
            self.registered_client_id = _client_id(issued)
        elif issued is not None and issued != self.registered_client_id:
            raise AuthError("The callback client ID differs from the selected registration.")
        client_id = _client_id(self.registered_client_id)
        code = _secret(fields.get("code"), "authorization code")
        response = service.exchange({"grant_type": "authorization_code", "client_id": client_id,
                                     "code": code, "code_verifier": self.verifier,
                                     "redirect_uri": self.redirect_uri, "resource": RESOURCE})
        scopes = _scope_set(response.get("scope"))
        token = _secret(response.get("id_token"), "ID token")
        claims = service.validate_identity(token, client_id, nonce=self.nonce,
                                            subject=self.selected_record["subject"] if self.selected_record else None)
        if response.get("token_type", "").lower() != "bearer":
            raise AuthError("The authentication server returned an unsupported token type.")
        now = time.time()
        record = {"schema_version": 1, "issuer": ISSUER, "subject": claims["sub"],
                  "email": claims.get("email"), "client_id": client_id,
                  "ext_agent_host_id": self.host_id, "auth_nonce": self.nonce,
                  "id_token": token, "access_token": _secret(response.get("access_token"), "access token"),
                  "refresh_token": _secret(response.get("refresh_token"), "refresh token"),
                  "token_type": "Bearer", "scopes": sorted(scopes),
                  "expires_at": now + _expiry(response.get("expires_in")), "saved_at": now}
        return record


class SubscriptionCredentials:
    """Supply a direct-use subscription token; never substitute API credentials."""

    def __init__(self, store_dir: str | Path | None = None, profile: str | None = None,
                 *, service: OpenAIAuthService | None = None):
        self.store = TokenStore(store_dir)
        self.profile = _client_id(profile) if profile else None
        self.service = service or OpenAIAuthService()

    def status(self) -> dict[str, Any]:
        return self.store.status(self.profile)

    def access_token(self) -> str:
        # The lock covers the complete network refresh and atomic replacement;
        # another process subsequently reads the newly rotated refresh token.
        with self.store.locked():
            record = self.store._load(self.profile)
            if record["expires_at"] > time.time() + 60:
                return record["access_token"]
            refresh = _secret(record.get("refresh_token"), "refresh token; sign in again if absent")
            response = self.service.exchange({"grant_type": "refresh_token", "client_id": record["client_id"],
                                              "refresh_token": refresh, "resource": RESOURCE})
            old_scopes = set(record["scopes"])
            granted = _scope_set(response.get("scope", " ".join(old_scopes)))
            if not granted.issubset(old_scopes):
                raise AuthError("The refresh response unexpectedly expanded the granted permissions.")
            if response.get("token_type", "").lower() != "bearer":
                raise AuthError("The refresh response has an unsupported token type.")
            updated = dict(record)
            if "id_token" in response:
                token = _secret(response["id_token"], "ID token")
                claims = self.service.validate_identity(token, record["client_id"], nonce=None,
                                                       subject=record["subject"])
                if "nonce" in claims and claims["nonce"] != record.get("auth_nonce"):
                    raise AuthError("The refreshed ID token nonce differs from the original session.")
                updated["id_token"] = token
                updated["email"] = claims.get("email", record.get("email"))
            now = time.time()
            updated.update({"access_token": _secret(response.get("access_token"), "access token"),
                            "refresh_token": _secret(response.get("refresh_token", refresh), "refresh token"),
                            "scopes": sorted(granted), "expires_at": now + _expiry(response.get("expires_in")),
                            "saved_at": now})
            self.store._validate_record(updated)
            self.store._write(self.store.path / "profiles" / (record["client_id"] + ".json"), updated)
            return updated["access_token"]


CredentialProvider = SubscriptionCredentials
