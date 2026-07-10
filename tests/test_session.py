"""Tests for session codes and their crypto helpers."""

import pytest

from onepool.session import SessionCode


def test_generate_shape():
    code = SessionCode.generate()
    parts = code.code.split("-")
    assert len(parts) == 3
    assert parts[2].isdigit()


def test_parse_normalizes():
    assert SessionCode.parse("  Amber-Fox-73 ").code == "amber-fox-73"


def test_parse_rejects_garbage():
    with pytest.raises(ValueError):
        SessionCode.parse("not a code")


def test_code_id_is_stable_and_hides_code():
    a = SessionCode.parse("amber-fox-73")
    b = SessionCode.parse("amber-fox-73")
    assert a.code_id == b.code_id
    assert len(a.code_id) == 12
    assert "amber" not in a.code_id


def test_different_codes_different_ids():
    assert SessionCode.parse("amber-fox-73").code_id != SessionCode.parse("amber-fox-74").code_id


def test_auth_mac_binds_all_inputs():
    code = SessionCode.parse("amber-fox-73")
    base = code.auth_mac(b"h" * 16, b"c" * 16, "fp")
    assert base != code.auth_mac(b"x" * 16, b"c" * 16, "fp")  # host nonce
    assert base != code.auth_mac(b"h" * 16, b"x" * 16, "fp")  # client nonce
    assert base != code.auth_mac(b"h" * 16, b"c" * 16, "other")  # cert fingerprint
    assert base != SessionCode.parse("bold-owl-42").auth_mac(b"h" * 16, b"c" * 16, "fp")
