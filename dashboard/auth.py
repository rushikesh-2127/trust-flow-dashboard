"""Local user database and bcrypt password handling for the dashboard."""

import datetime
import json
from pathlib import Path

import bcrypt

USERS_FILE = Path(__file__).resolve().parent / "users.json"
MAX_PASSWORD_BYTES = 72  # bcrypt ignores/rejects anything longer


def hash_password(password: str) -> str:
    """Hash a plaintext password with bcrypt (per-user random salt).

    :param password: Raw password string to hash.
    :returns: bcrypt hash string.
    :raises ValueError: If the password exceeds bcrypt's 72-byte limit.
    """
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        raise ValueError(f"Password must be at most {MAX_PASSWORD_BYTES} bytes.")
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, stored_hash: str) -> bool:
    """Check *password* against a stored bcrypt hash.

    :param password: Raw password string supplied at sign-in.
    :param stored_hash: Hash previously produced by :func:`hash_password`.
    :returns: True if the password matches. Legacy SHA-256 hashes never match
        (those accounts must be re-created).
    """
    try:
        return bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8"))
    except ValueError:
        return False


def load_users() -> dict:
    """Load user credentials from the JSON persistence file.

    Initializes the user database with default demo credentials if the file
    does not exist.

    :returns: Dictionary mapping user email to profile information and password hash.
    """
    if not USERS_FILE.exists():
        default_users = {
            "demo@trustfl.org": {
                "password_hash": hash_password("TrustFL@2026"),
                "name": "Demo User",
                "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
        }
        save_users(default_users)
        return default_users
    try:
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # Keep the unreadable file for recovery instead of overwriting it
        # the next time someone signs up.
        USERS_FILE.replace(USERS_FILE.with_suffix(".corrupt"))
        return load_users()


def save_users(users: dict):
    """Persist user account records to the JSON user database.

    :param users: Dictionary mapping user email to profile details and credentials.
    """
    tmp = USERS_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(users, f, indent=2)
    tmp.replace(USERS_FILE)  # atomic: a crash mid-write can't truncate the user DB
