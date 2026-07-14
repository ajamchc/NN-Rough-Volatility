# Plots to export — 2026-06-09 (calibration §5)

The §4 figures are done. What remains is exporting the **three calibration figures** that
currently exist only as inline cell outputs in `code/notebook.ipynb`. The optimizer-timing
figure (`calib_optimizers.png`) is already saved.

Convention: save into `report/` with `plt.savefig("<path>", dpi=200, bbox_inches="tight")`
**before** `plt.show()` (use `f"{REPORT_DIR}/<name>"` as the other cells do).

---

## 1. `report/calib_cdf.png` — empirical CDFs  (cell 52)
- **What:** left = CDF of parameter relative error (ξ₀, ν, ρ, H) with 95% quantile line;
  right = CDF of implied-vol surface RMSE with 99% quantile line. (Paper Figs 9–10 analog.)
- **Action:** add `fig.savefig(f"{REPORT_DIR}/calib_cdf.png", dpi=200, bbox_inches="tight")`
  before `plt.show()`.

## 2. `report/calib_scatter.png` — error vs true value  (cell 54)
- **What:** 2×2 panels, one per parameter; true value (x) vs relative error % (y), with the
  average/median annotation box. Shows H (and ρ) least identifiable.
- **Action:** add `fig.savefig(f"{REPORT_DIR}/calib_scatter.png", dpi=200, bbox_inches="tight")`
  before `plt.show()`.

## 3. `report/calib_smiles.png` — best/worst calibration fit  (cell 56)
- **What:** market (target) vs calibrated smiles across maturities, for the best- and
  worst-fit test surfaces.
- **Action:** the cell calls `plot_smile_comparison(...)`, which already accepts a `save=`
  argument — pass `save=f"{REPORT_DIR}/calib_smiles.png"`.

## (already done) `report/calib_optimizers.png` — optimizer timing  (cell 60)
- Avg calibration time per surface across optimisers (LM fastest). No action.

---

## Numbers already read from this run (for §5 prose — no need to re-report unless they change)
- NumPy mirror vs Keras: max diff `8.9e-16` (validates analytic Jacobian).
- Speed: **0.57 ms per surface** (LM, 1000 test surfaces).
- Median parameter relative error: ξ₀ 0.59%, ν 0.65%, ρ 0.91%, H 1.76%.
- Surface RMSE: median **0.095%**, 99% quantile **0.438%** (paper benchmark: 99% < 1%).
- Best/worst smile fit: 0.06% / 1.89% mean error.

## Checklist
- [ ] `calib_cdf.png` exported (cell 52)
- [ ] `calib_scatter.png` exported (cell 54)
- [ ] `calib_smiles.png` exported (cell 56, via `save=`)
- [x] `calib_optimizers.png` (cell 60)
