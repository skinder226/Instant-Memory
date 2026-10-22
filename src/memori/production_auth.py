"""Fail-closed OIDC authorization and public URL validation for the Streamlit deployment.

Authentication is performed by Streamlit's OIDC integration. These helpers add
application authorization: only verified identities explicitly allowed by
email or email domain receive a tenant. They do not validate arbitrary JWTs.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from urllib.parse import urlparse


def _truthy_claim(value: object) -> bool:
    return value is True or (isinstance(value, str) and value.strip().lower() == "true")


def authorize_oidc_claims(
    claims: Mapping[str, object],
    *,
    allowed_emails: str,
    allowed_domains: str,
) -> str:
    """Return a stable tenant ID only for a verified, explicitly allowed OIDC user."""
    subject = str(claims.get("sub") or "").strip()
    issuer = str(claims.get("iss") or "").strip().rstrip("/")
    email = str(claims.get("email") or "").strip().lower()
    email_verified = _truthy_claim(claims.get("email_verified"))

    if not subject or not issuer:
        raise ValueError("OIDC identity must include non-empty issuer and subject claims.")
    if not email or not email_verified or "@" not in email:
        raise PermissionError("A verified email claim is required for production access.")

    emails = {item.strip().lower() for item in allowed_emails.split(",") if item.strip()}
    domains = {
        item.strip().lower().lstrip("@").rstrip(".")
        for item in allowed_domains.split(",")
        if item.strip()
    }
    if not emails and not domains:
        raise PermissionError(
            "Production access is disabled until MEMORI_OIDC_ALLOWED_EMAILS "
            "or MEMORI_OIDC_ALLOWED_EMAIL_DOMAINS is configured."
        )

    domain = email.rsplit("@", 1)[1].rstrip(".")
    if email not in emails and domain not in domains:
        raise PermissionError("This identity is not authorized to access Instant-Memory.")

    digest = hashlib.sha256(f"{issuer}\0{subject}".encode("utf-8")).hexdigest()
    return "oidc_" + digest


def validate_public_https_url(value: str) -> str:
    """Require an absolute HTTPS URL with a host and no embedded credentials."""
    parsed = urlparse((value or "").strip())
    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("MEMORI_PUBLIC_URL must be an absolute HTTPS URL without embedded credentials.")
    return parsed.geturl()
