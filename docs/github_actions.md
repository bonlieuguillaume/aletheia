# Why GitHub Actions for the tests

## The constraint

All the code of aletheia runs on Google Colab. The user's PC is used for
editing, git and the upstream preprocessing only: nothing is installed on it
with pip, and it has no GPU. Yet the code must be checked before a long and
costly Colab run, and every change can break something.

## The solution

[GitHub Actions](https://docs.github.com/actions) runs the test suite on
GitHub's own Linux machines at every push and every pull request
(`.github/workflows/tests.yml`):

1. a fresh Ubuntu machine is started by GitHub;
2. it installs Python 3.12 (as on Colab), a CPU-only PyTorch and
   `requirements.txt`;
3. it runs `pytest` on the `tests/` folder;
4. the result appears on GitHub: a green tick or a red cross next to the
   commit, with the full log.

Nothing is installed on or run from the user's PC.

## What the tests check

The tests use **small synthetic data** (gamma-distributed speckle with
Sentinel-1-like statistics, an empty border like terrain correction leaves,
256x256 patches), never real SAR products:

- tiling: no-data rule, split by scene, patch sizes;
- copy-move: the paste is an exact copy of the source, nothing else changes,
  mask values, border margin, same seed = same forgery, manifest columns;
- metrics: MCC / F1 on the worked example of `architecture_v0.1.md`, source
  pixels ignored, score methods, the whole val -> test procedure;
- U-Net: output size = input size, reflection padding, loss ignoring the
  source, a 2-epoch training on CPU, resuming, single-patch prediction.

They check that the code does what it should, not how good the model is:
real trainings and real metrics happen on Colab.

## Why this choice

- **Nothing on the PC**: the tests run in GitHub's cloud, consistent with
  "everything runs in the cloud".
- **Free**: GitHub Actions is free for public repositories.
- **Automatic**: no step to remember; a broken change is visible before
  anyone launches a Colab run on it.
- **Same platform as Colab**: Linux, Python 3.12, which also catches
  Windows-only habits (paths, file names).
- **Standard practice**: continuous integration is how most open-source and
  industrial projects keep their code working.

## Alternatives considered

- **Testing on the PC**: requires installing PyTorch and the other packages
  locally, which is ruled out.
- **Testing only on Colab**: manual, slow (each session reinstalls
  everything), and it spends Colab time on checks that a free CPU machine
  does as well.
