# aletheia

SAR forensics: detect manually tampered (copy-move, splicing, inpainting) and
AI-generated Sentinel-1 images, with an image-level score and a pixel-level
anomaly map. `README.md` at the root gives the motivation and the approach.
Code is organised as one feature per folder of `features/` (module + notebook
+ README); read a feature's README before changing it.

## Environment

- **All the code runs on Google Colab** (Linux VM, GPU): patch extraction,
  tamper generation, training, inference. The user's PC (Windows, CPU only)
  is for editing, git, and the upstream preprocessing. Colab installs with
  pip on its own VM, nothing reaches the PC; torch/torchvision come
  preinstalled with CUDA: never reinstall or pin them in a Colab cell.
- Upstream of this repo, the user preprocesses the Sentinel-1 GRD scenes on
  their PC with rosarium (SNAP `gpt`, not available on Colab) and puts the
  resulting GeoTIFFs on Google Drive. SNAP is never a dependency of this repo.
- Dependencies come from PyPI and must be cross-platform. Anything not
  installable with a plain `pip install` on Colab is a decision to bring to
  the user, not to take: ask whether it is worth it before writing code that
  needs it. External binaries, if any, are located at runtime
  (`shutil.which`), never hard-coded.
- Data and checkpoints live on Google Drive (`MyDrive/aletheia/`), never in
  git. Notebooks mount Drive, copy the inputs to the VM's local disk (one zip
  rather than many small files: Drive is slow per file), and write outputs
  and checkpoints back to Drive so a disconnected session can resume.

## Working rules

- Tasks and priorities live in the GitHub Project
  https://github.com/users/bonlieuguillaume/projects/5 (issues of the repo),
  driven with `gh` (needs the `project` scope). Milestones carry the versions
  (v0.1 copy-move mockup, v0.2 SAR physics cues, v0.3 splicing, v1
  extensions); the Priority field orders the backlog inside a milestone; work
  is split with sub-issues. A known limit goes in the feature's README as
  well, since a clone does not carry the issues.
- The user runs the notebooks on Colab and the tests on real data. Do not
  install packages on their PC, or run commands against their SAR products,
  unless they explicitly ask. Ad-hoc checks belong in the scratchpad
  directory; tests that write through the features redirect their data dirs to
  the scratchpad, never the repo's `data/`.
- Developed on Windows, **but every feature must run unchanged on Linux
  (Colab) and macOS**. Rules: `pathlib` and `/`-agnostic paths, no hard-coded
  drive letters outside the users' parameter cells, no Windows-only modules
  or shell calls (`cmd`, PowerShell, `%VAR%`), no reliance on case-insensitive
  file names, and every OS-specific branch (`sys.platform`) gets its Unix
  side. Shell scripts stay POSIX-ish bash: no GNU-only flags (`readlink -f`,
  `sed -i` without suffix). Line endings are pinned by `.gitattributes`: keep
  new file kinds in there. This binds what goes **into the repo**, not the
  conversation: commands the user is meant to type and one-off checks can be
  Windows-only.
- Windows shells: the "Miniforge Prompt" is `cmd.exe`; VS Code's integrated
  terminal is PowerShell. Quoting differs between the two, which matters when
  documenting CLI examples. Console output must stay ASCII (`->` not `→`):
  the cmd console is cp1252.
- **Never overwrite the user's own values.** Parameter cells hold what they
  chose: input paths, Drive paths, hyper-parameters. `NotebookEdit` rewrites a
  whole cell, so editing one line of such a cell silently restores every other
  value from whenever the cell was last read. Re-read the cell immediately
  before writing it, carry the current values across verbatim, and if a value
  cannot be confirmed, ask instead of guessing.
- `data/` mirrors the Drive layout locally; only the `.gitkeep` files are
  versioned. Each feature documents which folder it reads and which it
  writes; keep the producer and the consumer in sync when changing one.
- Experiments must be reproducible: fixed seeds, train/val/test split **by
  scene** (never patches of one scene on both sides), configs versioned in
  the repo, and each run's outputs record the commit and the config they came
  from.
- `docs/report.md` is the living design report: every choice, why it was
  made, and what is still open. Update it in the same step as the decision
  (and the issue it touches), rewriting the relevant section rather than
  appending; move a point out of *Open questions* once decided.
- From time to time, post a short progress update to the Claude Desktop
  session "Travailler sur aletheia en local" (via `ListAgents` /
  `SendMessage`), even though it can read this folder.

## Data

- Input: Sentinel-1 IW GRD amplitude, **VV only**, forest scenes for now,
  one GeoTIFF per scene plus the XML of its SNAP graph, produced by the
  user's own preprocessing (rosarium / SNAP) and put on Drive. No speckle
  filter and no extra multilook, but IW GRD is already multilooked by ESA
  (about 5x1, ENL ~4.4): keep it in mind for any speckle statistic.
- The repo, on Colab, tiles the scenes into 512x512 single-band `.tif`
  patches without georeferencing, with a manifest (patch -> scene, position,
  split), then generates the forgeries with their ground-truth masks.
- Validation and test sets are always fixed and precomputed (seeded); the
  training set is fixed too until on-the-fly generation is introduced.

## Conventions

- Code, comments and docstrings in English (numpy-style docstrings);
  conversation with the user in French.
- **Simple, readable code over clever code**: explicit loops and
  intermediate variables rather than dense one-liners, plenty of comments
  explaining the why (SAR specifics, design choices, link to the issue). The
  user reads and owns every line.
- Git: the first working version goes in a single commit; once it runs, the
  user sorts out which issues are done, and from then on **one branch per
  issue**, merged by the user. Do not move project cards (Status) unless
  asked.
- Tests run in **GitHub Actions** (Linux runners, small synthetic data), never
  on the user's PC; real runs happen on Colab, launched by the user.
  Rationale: `docs/github_actions.md`.
- Features import each other through the repository root on `sys.path`
  (`from features.<name>.<name> import ...`); notebooks import the module next
  to them (`from <name> import ...`) and are run from their own folder (on
  Colab: `%cd` into it after cloning).
- A notebook either drives its module or mirrors it. A mirror holds the same
  functions as the module: any change to one is applied to the other in the
  same edit. Say in the feature's README which kind it is.
- A notebook whose opening markdown cell describes what each cell does keeps
  that description in sync: adding, removing or reordering a cell means
  updating the table in the same edit. A stale walkthrough is worse than none.
- A module usable from the command line exposes `main(argv=None, prog=None)`
  and guards it with `if __name__ == "__main__":`.

## Maintaining this file

Keep it current as the repo evolves: add a line under Features for a new one,
and record a decision or constraint here once it is durable and not derivable
from the code. Be selective: this file is loaded into context every session,
so prefer rewriting an existing line over appending a new one, and drop what
no longer helps.

## Features

None yet: the code restarts from scratch, following the issues of the
project (v0.1 first).
