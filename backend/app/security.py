"""Password hashing, JWT issuance, and application-level field encryption.

Password hashing uses stdlib ``hashlib.scrypt`` rather than passlib+bcrypt.
scrypt is memory-hard, is in the standard library on every supported platform,
and sidesteps the passlib 1.7.4 / bcrypt 4.x incompatibility that breaks
installs on fresh environments.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt

from app.config import settings
from app.errors import AuthError
from app.logging_conf import get_logger

log = get_logger(__name__)

# scrypt work factors. n=2**14 keeps verification near ~50ms on a laptop.
_SCRYPT_N = 2 ** 14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32
_SALT_BYTES = 16


# --------------------------------------------------------------------------
# passwords
# --------------------------------------------------------------------------


def hash_password(password: str) -> str:
    """Return ``scrypt$n$r$p$salt$digest`` with a fresh random salt."""
    if not password:
        raise ValueError("password must not be empty")
    salt = secrets.token_bytes(_SALT_BYTES)
    digest = hashlib.scrypt(
        password.encode("utf-8"), salt=salt,
        n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=_SCRYPT_DKLEN,
    )
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${base64.b64encode(salt).decode()}${base64.b64encode(digest).decode()}"


def verify_password(password: str, encoded: str) -> bool:
    """Constant-time verification against a stored hash."""
    try:
        scheme, n_s, r_s, p_s, salt_b64, digest_b64 = encoded.split("$")
        if scheme != "scrypt":
            return False
        expected = base64.b64decode(digest_b64)
        actual = hashlib.scrypt(
            password.encode("utf-8"), salt=base64.b64decode(salt_b64),
            n=int(n_s), r=int(r_s), p=int(p_s), dklen=len(expected),
        )
    except Exception:
        return False
    return hmac.compare_digest(expected, actual)


# --------------------------------------------------------------------------
# tokens
# --------------------------------------------------------------------------


def create_access_token(subject: str, *, extra: dict | None = None) -> str:
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": subject,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=settings.access_token_expire_minutes)).timestamp()),
        "iss": "cognitive-twin",
    }
    if extra:
        payload.update(extra)
    return jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict:
    try:
        return jwt.decode(
            token, settings.secret_key,
            algorithms=[settings.jwt_algorithm], issuer="cognitive-twin",
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("Session expired. Please sign in again.") from exc
    except jwt.PyJWTError as exc:
        raise AuthError("Invalid authentication token.") from exc


# --------------------------------------------------------------------------
# field-level encryption at rest
# --------------------------------------------------------------------------


class FieldCipher:
    """Symmetric encryption for free-text personal data.

    Disk encryption protects a stolen volume; this protects a leaked database
    dump. When no key is configured the cipher is a no-op and says so loudly at
    startup, so a production deployment cannot quietly run unencrypted.
    """

    def __init__(self, key: str = "") -> None:
        self._fernet = None
        if key:
            try:
                from cryptography.fernet import Fernet

                self._fernet = Fernet(key.encode() if isinstance(key, str) else key)
            except Exception as exc:  # pragma: no cover - config error path
                raise ValueError(f"FIELD_ENCRYPTION_KEY is not a valid Fernet key: {exc}") from exc

    @property
    def enabled(self) -> bool:
        return self._fernet is not None

    def encrypt(self, plaintext: str | None) -> str | None:
        if plaintext is None or self._fernet is None:
            return plaintext
        return "enc:v1:" + self._fernet.encrypt(plaintext.encode("utf-8")).decode("ascii")

    def decrypt(self, value: str | None) -> str | None:
        if value is None or self._fernet is None or not value.startswith("enc:v1:"):
            return value
        try:
            return self._fernet.decrypt(value[7:].encode("ascii")).decode("utf-8")
        except Exception:  # pragma: no cover - corrupted/rotated key
            log.error("failed to decrypt a stored field; returning placeholder")
            return "[unreadable: encryption key mismatch]"


cipher = FieldCipher(settings.field_encryption_key)


def generate_fernet_key() -> str:
    from cryptography.fernet import Fernet

    return Fernet.generate_key().decode()


if __name__ == "__main__":  # pragma: no cover - developer helper
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "keygen":
        print(generate_fernet_key())
    else:
        print("usage: python -m app.security keygen")
        print("secret_key suggestion:", os.urandom(32).hex())
