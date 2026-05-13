# Inference observation — one-pass checklist (Tier-B apple PnP)

**Purpose:** Single entry point for a **disciplined observation / lighting iteration** on `run_gr00t_inference.py --scene table_pnp_apple` vs Tier-A (`rollout_policy.py` + `LMPnPAppleToPlateDC_G1_gear_wbc`). Use when you suspect **ego RGB / render domain** and want measurable gates before another qualitative video.

**Stop rule:** If `scripts/hybrid_policy_get_action.py` already shows **`hybrid_ego` ≈ rollout ref** on `action.*` but **live** inference still collapses after a few seconds, **exit this pass** — prioritize **oracle action-chunk replay**, **`--inference-trace-jsonl`**, and **multi-step** dumps (see **`ACTIVE_gr00t_inference_wiring.md` §10**). Do not loop MJCF forever.

---

## Preconditions

- Same **policy server** + **checkpoint** for Tier-A and Tier-B runs.
- WholeBodyControl venv `python` for anything importing `gr00t`.
- Tier-B scene: **`--scene table_pnp_apple`** (not `stylish_diner` for PnP parity).

---

## 1. Capture paired step-0 dumps

**A — rollout (from `Isaac-GR00T/`):**

```bash
cd /path/to/Isaac-GR00T
/path/to/.venv/bin/python gr00t/eval/rollout_policy.py \
  --policy_client_host 127.0.0.1 --policy_client_port 2000 \
  --model_path "" \
  --env_name gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc \
  --n_episodes 1 --max_episode_steps 200 --n_action_steps 30 --n_envs 1 \
  --policy-ab-dump /tmp/gr00t_ab_rollout_obs_pass
```

**B — minimal (from `g1_minimal_sim/`):**

```bash
cd /path/to/g1_minimal_sim
/path/to/.venv/bin/python scripts/run_gr00t_inference.py \
  --scene table_pnp_apple \
  --policy-host 127.0.0.1 --policy-port 2000 \
  --max-steps 50 --n-action-steps 30 \
  --policy-ab-dump /tmp/gr00t_ab_minimal_obs_pass
```

---

## 2. Diff dumps (`state.*`, `video.ego_view`, `action.*`)

```bash
cd /path/to/g1_minimal_sim
python scripts/compare_policy_ab_dumps.py \
  -a /tmp/gr00t_ab_rollout_obs_pass \
  -b /tmp/gr00t_ab_minimal_obs_pass \
  --exit-on never
```

**Watch:**

- **`[state.*]`** — must stay **OK** at default `state_atol` (Bug #8 regression gate).
- **`video.ego_view`** — per-channel **mean RGB delta** vs Tier-A; target **shrink toward 0** (exact match unlikely across GL paths).
- **`action.navigate_command`** and arm rows — first-chunk alignment vs Tier-A.

For strict state gate only:

```bash
python scripts/compare_policy_ab_dumps.py -a ... -b ... --exit-on strict
```

---

## 3. Hybrid `get_action` (vision causality at step 0)

```bash
cd /path/to/g1_minimal_sim
python scripts/hybrid_policy_get_action.py \
  --rollout /tmp/gr00t_ab_rollout_obs_pass \
  --minimal /tmp/gr00t_ab_minimal_obs_pass \
  --policy-host 127.0.0.1 --policy-port 2000 \
  --modes minimal hybrid_ego rollout
```

**Read:** `hybrid_ego` vs **rollout ref** — if **`navigate_command` / arms** jump from WARN → OK vs **`minimal`**, **ego pixels still drive** the gap.

---

## 4. MJCF edit scope (Phase 4b style)

**Edit only** (keep Isaac vendored `gear_lab.xml` unchanged):

- `g1_minimal_sim/scenes/table_pnp_apple/g1_gear_wbc_table_pnp_apple.xml`
- `g1_minimal_sim/scenes/table_pnp_apple/g1_gear_wbc_hands_table_pnp_apple.xml`
- `g1_minimal_sim/scenes/lab_dc_layout/lab_dc_world.xml`
- `g1_minimal_sim/scenes/lab_dc_layout/gen_lab_dc_world_xml.py` (so regen does not revert lighting)

Typical levers: `<visual><headlight .../>`, worldbody `<light .../>` diffuse/specular.

---

## 5. Re-measure

Repeat **§1–§3** after each MJCF tweak until:

- mean RGB delta plateaus or hits ~0, **and**
- qualitative short clip improves, **or**
- hybrid vs minimal margin no longer shrinks (**stop rule**).

---

## 6. Qualitative gate

`run_gr00t_inference.py --scene table_pnp_apple` + `--save-video`; check **`ACTIVE_gr00t_inference_wiring.md` §1** four checkboxes.

---

## Related docs

- Full wiring audit + §10 experiments: **`ACTIVE_gr00t_inference_wiring.md`**
- Operator dump harness: **`techContext.md`** § GR00T policy I/O numerical A/B dump
- Key audit for rollout-only obs keys: **`scripts/audit_policy_obs_keys.py`**
