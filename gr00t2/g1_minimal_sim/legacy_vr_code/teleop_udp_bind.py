"""Parse ``HOST:PORT`` for ``--udp-teleop`` (no MuJoCo dependency)."""


def parse_udp_bind(spec: str) -> tuple[str, int]:
    """Parse ``HOST:PORT`` (e.g. ``0.0.0.0:5005``). Empty host → ``0.0.0.0``."""
    if ":" not in spec:
        raise ValueError(f"udp bind must be HOST:PORT, got {spec!r}")
    host, _, port_s = spec.rpartition(":")
    host = host.strip() or "0.0.0.0"
    port = int(port_s.strip())
    if not (0 < port < 65536):
        raise ValueError(f"invalid UDP port: {port}")
    return host, port
