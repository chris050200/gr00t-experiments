#!/usr/bin/env python3
"""Open a passive MuJoCo viewer for the table PnP apple scene (no Gear WBC / policy).

Equivalent one-liner (from this directory)::

    python -m mujoco.viewer g1_gear_wbc_table_pnp_apple.xml

This script steps ``mj_step`` in a loop so the free apple can settle under gravity.
Requires a display (or ``MUJOCO_GL`` / EGL setup compatible with ``mujoco.viewer``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import mujoco.viewer


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--hands",
        action="store_true",
        help="Load g1_gear_wbc_hands_table_pnp_apple.xml instead of the no-hands MJCF.",
    )
    args = p.parse_args()
    here = Path(__file__).resolve().parent
    name = (
        "g1_gear_wbc_hands_table_pnp_apple.xml"
        if args.hands
        else "g1_gear_wbc_table_pnp_apple.xml"
    )
    xml = here / name
    if not xml.is_file():
        raise SystemExit(f"MJCF not found: {xml}")

    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
