"""Frozen split-machine VR / UDP teleop and pose-stream reference code.

New work lives under ``vr_teleop/`` (scaffold). This package stays importable so
``--udp-teleop`` and pytest keep working until the refactor replaces it.

Run pose lab (from ``g1_minimal_sim``)::

    python -m legacy_vr_code.vr_teleop_test_streaming receive --bind 0.0.0.0:5006
    python -m legacy_vr_code.openvr_teleop_client --host <sim-ip> --port 5005
"""
