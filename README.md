# NN_Stochastic_Volitility_Calibration_Project

Neural-network calibration of stochastic volatility models, after
**Horvath, Muguruza & Tomas (2020), *Deep Learning Volatility*** (SSRN 3322085) and the upstream
[NN-StochVol-Calibrations](https://github.com/amuguruza/NN-StochVol-Calibrations).

The main notebook is a **from-scratch reproduction of the rough Bergomi pipeline** on the JAX
backend: a Monte-Carlo pricer generates (parameters → implied-vol surface) data, a small ELU
network learns the pricing map, and Levenberg–Marquardt calibrates over the frozen network in
milliseconds. It also covers a digital-barrier surface extension and a vectorised GPU implied-vol
inverter. Unlike a pre-trained-weights reproduction, this one **trains and generates its own data**.

## Setup

```bash
git clone --recurse-submodules <this-repo-url>
cd NN_Stochastic_Volitility_Calibration_Project
git lfs pull                       # fetch the cached dataset (~276 MB, git-lfs)
pip install -r requirements.txt
jupyter notebook code/notebook.ipynb
```

The notebook runs on the **JAX** backend (it sets `KERAS_BACKEND=jax`); for GPU install
`jax[cuda12]` instead of plain `jax`.

## Structure

| Path | Description |
|------|-------------|
| `code/notebook.ipynb` | Main notebook — from-scratch rough Bergomi pricing, training, and calibration (committed **with outputs**). |
| `code/notebook.py` | Jupytext (`py:percent`) source paired with `code/notebook.ipynb` — edit either and run `jupytext --sync code/notebook.py` (from `code/`) to keep them in sync. |
| `data/` | Cached `.npy` datasets (`[params \| surface]` arrays plus standard errors), tracked with git-lfs. |
| `report/` | LaTeX write-up (`main.tex`) and figures. |
| `papers/` | Reference papers (Deep Learning Volatility; Functional CLT for Rough Volatility). |
| `original/` | Submodule — upstream source repo being reproduced (Heston, Rough Bergomi, 1-Factor). |
| `requirements.txt` | Python dependencies. |

## Running

`code/notebook.ipynb` defaults to `GENERATE = False`, loading the cached dataset from `data/`
(run `git lfs pull` first). Set `GENERATE = True` to re-price from scratch — GPU strongly
recommended (~2 h; ~5 h on CPU).

## Collaboration

- Work on feature branches (`feat/your-feature`) and open PRs into `main`
- Never commit directly to `main`
- Notebook outputs are committed (kept in git) so teammates don't have to re-run the notebook
- To add collaborators: go to **Settings → Collaborators and teams** on the GitHub repo page and invite by GitHub username or email
