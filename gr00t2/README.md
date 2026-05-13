# gr00t2

Workspace for **G1 minimal sim** experiments next to **Isaac-GR00T**.

## Layout

- **`g1_minimal_sim/`** — MuJoCo + Gear WBC sandbox (tracked by this repo). Includes the canonical project memory bank under `g1_minimal_sim/memory-bank/` (start at `INDEX.md`).
- **`Isaac-GR00T/`** — **not committed** (separate git clone; avoids nested-repo issues). After cloning this repo, add Isaac-GR00T:

  ```bash
  cd /path/to/gr00t2
  git clone <your-Isaac-GR00T-remote> Isaac-GR00T
  ```

  Paths in `g1_minimal_sim` expect `Isaac-GR00T` as a **sibling folder of `g1_minimal_sim`** under `gr00t2/`.

## Tags

- **`pre-vr-teleop-rebuild`** — snapshot before the VR teleop / VR lab rebuild.

## Push to GitHub (private)

Local git is initialized on **`main`** with that tag. If `gh repo create` fails with **“Resource not accessible by personal access token”**, create the repo in the GitHub web UI (**New repository** → **Private**), then:

```bash
cd /path/to/gr00t2
git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
git push -u origin main
git push origin pre-vr-teleop-rebuild
```

For **`gh`**, use a token with the **`repo`** scope (classic) or repository creation permission (fine-grained).
