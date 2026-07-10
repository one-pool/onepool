"""Per-session TLS identity.

The host generates an ephemeral key + self-signed certificate in memory when
the pool starts and discards both when it stops — nothing touches disk. Clients
can't chain-verify a self-signed cert, so instead the cert's SHA-256
fingerprint travels in the mDNS advertisement and is cryptographically bound
into the join handshake (see ``SessionCode.auth_mac``).
"""

from __future__ import annotations

import datetime
import hashlib
import ssl
import tempfile
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID


def make_session_identity() -> tuple[bytes, bytes, str]:
    """Return (cert_pem, key_pem, sha256_fingerprint) for a fresh session."""
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "onepool-session")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=7))
        .sign(key, hashes.SHA256())
    )
    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    key_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    return cert_pem, key_pem, fingerprint(cert.public_bytes(serialization.Encoding.DER))


def fingerprint(cert_der: bytes) -> str:
    return hashlib.sha256(cert_der).hexdigest()


def host_ssl_context(cert_pem: bytes, key_pem: bytes) -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    # ssl requires files for cert chains; use a private temp dir, removed at once.
    with tempfile.TemporaryDirectory(prefix="onepool-tls-") as tmp:
        cert_path = Path(tmp) / "cert.pem"
        key_path = Path(tmp) / "key.pem"
        cert_path.write_bytes(cert_pem)
        key_path.write_bytes(key_pem)
        ctx.load_cert_chain(cert_path, key_path)
    return ctx


def client_ssl_context() -> ssl.SSLContext:
    # No CA chain to verify against — authenticity comes from the fingerprint
    # check + MAC binding in the handshake, not from PKI.
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def peer_fingerprint(writer) -> str:
    """SHA-256 fingerprint of the certificate the peer actually presented."""
    ssl_obj = writer.get_extra_info("ssl_object")
    cert_der = ssl_obj.getpeercert(binary_form=True)
    if cert_der is None:
        raise ConnectionError("peer presented no TLS certificate")
    return fingerprint(cert_der)
