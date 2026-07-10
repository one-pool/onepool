"""Session codes: the human-friendly key that forms a pool.

``onepool up`` prints a code like ``amber-fox-73``. People in the room type it
into ``onepool join``. The code doubles as the pool's shared secret, so it is
never broadcast: discovery advertises only ``code_id``, a salted hash prefix.

Threat model is "stranger on the same WiFi", not offline attackers: the code
space (~1.3M) is small but every failed authentication costs a forced delay at
the host, making online guessing impractical for a session that lives an hour.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass

_ADJECTIVES = (
    "amber", "bold", "brisk", "calm", "civic", "clever", "coral", "crisp",
    "dusty", "eager", "early", "fable", "fancy", "fleet", "glad", "grand",
    "happy", "hardy", "hazel", "ivory", "jolly", "keen", "lively", "lucky",
    "lunar", "mellow", "merry", "misty", "noble", "north", "olive", "opal",
    "pearl", "plucky", "prime", "quiet", "rapid", "regal", "royal", "rustic",
    "sage", "sunny", "swift", "tidal", "topaz", "vivid", "wild", "witty",
)
_NOUNS = (
    "badger", "bison", "cedar", "comet", "crane", "delta", "dune", "eagle",
    "ember", "falcon", "fern", "fox", "gale", "grove", "harbor", "hawk",
    "heron", "iris", "jade", "koala", "lark", "lotus", "lynx", "maple",
    "marsh", "meadow", "otter", "owl", "panda", "pine", "prairie", "quill",
    "raven", "reef", "ridge", "river", "robin", "sparrow", "spruce", "stone",
    "summit", "swan", "thistle", "tiger", "trail", "walnut", "willow", "wren",
)

_ID_SALT = b"onepool-code-id-v1"


@dataclass(frozen=True)
class SessionCode:
    """A pool session code, e.g. ``amber-fox-73``."""

    code: str

    @classmethod
    def generate(cls) -> SessionCode:
        adj = secrets.choice(_ADJECTIVES)
        noun = secrets.choice(_NOUNS)
        num = secrets.randbelow(90) + 10  # two digits, no leading zero
        return cls(f"{adj}-{noun}-{num}")

    @classmethod
    def parse(cls, text: str) -> SessionCode:
        code = text.strip().lower()
        if len(code.split("-")) != 3:
            raise ValueError(f"session codes look like 'amber-fox-73', got: {text!r}")
        return cls(code)

    @property
    def code_id(self) -> str:
        """Public identifier safe to broadcast over mDNS (does not reveal the code)."""
        return hashlib.sha256(_ID_SALT + self.code.encode()).hexdigest()[:12]

    @property
    def secret(self) -> bytes:
        return hashlib.sha256(b"onepool-secret-v1" + self.code.encode()).digest()

    def auth_mac(self, host_nonce: bytes, client_nonce: bytes, cert_fingerprint: str) -> bytes:
        """MAC proving knowledge of the code, bound to this TLS identity.

        Binding the host's certificate fingerprint into the transcript means a
        machine-in-the-middle presenting its own certificate produces a MAC the
        real host rejects.
        """
        transcript = host_nonce + client_nonce + cert_fingerprint.encode()
        return hmac.new(self.secret, transcript, hashlib.sha256).digest()


def new_nonce() -> bytes:
    return secrets.token_bytes(16)
