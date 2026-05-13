"""UDP teleop bind parsing."""

from __future__ import annotations

import pytest

from legacy_vr_code.teleop_udp_bind import parse_udp_bind


def test_parse_udp_bind_default_host() -> None:
    h, p = parse_udp_bind(":5005")
    assert h == "0.0.0.0"
    assert p == 5005


def test_parse_udp_bind_explicit() -> None:
    h, p = parse_udp_bind("192.168.1.2:7777")
    assert h == "192.168.1.2"
    assert p == 7777


def test_parse_udp_bind_invalid_port() -> None:
    with pytest.raises(ValueError):
        parse_udp_bind("0.0.0.0:99999")


def test_parse_udp_bind_no_colon() -> None:
    with pytest.raises(ValueError):
        parse_udp_bind("5005")
