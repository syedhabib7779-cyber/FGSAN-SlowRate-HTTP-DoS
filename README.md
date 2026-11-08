# FGSAN — Feature-Grouped Self-Attention with Adversarial Training for Slow-Rate HTTP DoS Detection

Code accompanying the article

> S. H. U. Shah and S. Akram, "Feature-Grouped Self-Attention with Adversarial Training for Slow-Rate HTTP
> Denial-of-Service Detection," submitted, 2026.

FGSAN detects Slowloris and SlowHTTPTest flows from 27 CICFlowMeter features. The features are encoded
separately in volume (8), timing (14) and behavior (5) groups, and multi-head self-attention is applied
across the groups. The model is trained with feature-space PGD adversarial training under
functionality-preserving constraints. The evaluation covers clean accuracy, white-box and transferred
evasion, feasibility of the adversarial examples, native-prevalence alarm load, calibration and
generalization.

| Result (CICIDS2017 Wednesday, seed 42) | Value |
|---|---|
| Clean F1 / AUC | 0.961 / 0.996 |
| White-box PGD evasion (ε = 0.05) | 0.19% (3 / 1,586 clean-correct attack flows) |
| Transfer of FGSAN examples to RF / LightGBM / XGBoost | 88.0% / 94.8% / 99.8% |
| Feasibility-projected adversary / 20 directly controllable features | 0.06% / 36.1% |
| PR-AUC at the native 3% attack prevalence | 0.981 |

All attacks are in feature space; no packet-level traffic was replayed (see Section IV-G of the article).

## Repository layout

```
notebooks/
  FGSAN_main.ipynb                 main pipeline, Cells 1–21 (outputs from the reported run)
  archive/FGSAN_working_notebook_full.ipynb   full working notebook, including intermediate cells
experiments/                       reviewer-requested analyses, one script per Colab cell (see paper map)
utils/
  CHECKPOINT_1_save.py             saves models, scaler, splits and adversarial arrays to Drive
  CHECKPOINT_2_restore.py          restores them after a Colab disconnect (skips all training)
  RESTORE_main_model_direct.py     loads main_model.pt and checks F1 = 0.9612, 20 FP, 108 FN
  M20_publication_figure_style.py  figure style used for the published figures
archive/fgsan_reproducibility_patch.py   notes on the determinism fixes (already applied in the notebook)
data/README.md                     how to obtain CICIDS2017 and CSE-CIC-IDS2018
```

## Reproducing the results

The experiments were run on Google Colab (Python 3.13, PyTorch 2.14.1 + CUDA 13.0, NVIDIA T4,
scikit-learn 1.6.1, XGBoost 3.4.1, LightGBM 4.7.0).

1. Put `Wednesday-workingHours.pcap_ISCX.csv` in `MyDrive/cicids2017/` (see `data/README.md`).
2. Open `notebooks/FGSAN_main.ipynb` in Colab with a GPU runtime and run all cells. This trains the
   baselines and FGSAN and reproduces Tables IV–VI and Figs. 3, 4 and 6. Run `utils/CHECKPOINT_1_save.py`
   once training has finished.
3. For the scripts in `experiments/`, start a fresh session and run the definition cells of the notebook
   (Cells 1, 2, 3, 4, 6, 8, 9, 10 and 11). Then run `utils/CHECKPOINT_2_restore.py` and
   `utils/RESTORE_main_model_direct.py`, and paste or `%run` the experiment script. Each script lists in
   its header the variables it needs, and it writes a JSON result next to the checkpoint so a disconnect
   can resume.

To skip training, download `FGSAN_release.zip` from the
[Releases](../../releases) page and copy its `checkpoints/` and `splits/` folders into
`MyDrive/fgsan_ckpt/`. The release also contains `environment.json` (feature order, group assignment,
configuration and seed), exact package versions, MD5 checksums and the JSON outputs behind every table.

## Paper map (IEEE numbering)

| Script | Article | Supplement |
|---|---|---|
| `notebooks/FGSAN_main.ipynb` | §III, Algorithm 1, Tables II, IV–VI, Figs. 3, 4, 6 | S2–S5, S7, S15, S17 |
| `M1_cells_v2.py`, `M1_cell27_final.py` — feasibility constraints and projected adversary | §IV-D, Table VIII | S8 |
| `M2_cell28_attack_battery.py`, `M2_cell29_closeout.py` — gradient-masking checks, APGD, EOT/BPDA | §IV-C | S9 |
| `M3_cell30_fair_comparison.py` — common SPSA attack on clean/adversarially trained models | Table VII, Fig. 5 | — |
| `M4_cell31_leakage.py`, `M4_cell32_recalibration.py` — deduplication, temporal and cross-tool splits | §IV-E, Table IX | — |
| `M5_cell33_native_prevalence.py` — native-prevalence evaluation | §IV-E, Table X | — |
| `M6_cell34_per_tool.py`, `M6_cell35_external_ids2018.py` — per-tool results, CSE-CIC-IDS2018 | §IV-E | S11 |
| `M7_cell36_uncertainty.py`, `M8_cell37_transfer_accounting.py` | Table VI | S12 |
| `M9_cell38_group_sensitivity.py` — random and alternative groupings | — | S13 |
| `M10_cell_focal_vs_ce.py` — focal loss vs. cross-entropy | §III-B | S14 |
| `M11_cell_stress_realism.py` — benign-slow stress test | §IV-E | S15 |
| `M12_cell_deployment_cost.py` — inference cost and time-to-detection | §IV-E | S16 |
| `M14_cell_svm_parity.py` — SVM learning curve and full-data fit | §III-D | S18 |
| `M16_cell_attack_sensitivity.py` — ε, steps and restarts with CIs | §IV-C | S10 |
| `R2_A_threshold_on_validation.py` — matched-FPR operating points | — | S19 |
| `R2_B_transfer_bootstrap_CI.py` — transfer rates with bootstrap CIs | Table VI | S7 |
| `R2_C_prevalence_metrics.py` — precision and alerts across prevalences | §IV-E | S20 |
| `R5_fixable_limitations.py` — near-duplicates, calibration, cost, paired bootstrap, grouped split | §IV-E | S21–S24 |
| `R6_figures_and_release.py` — published figures and the release bundle | Figs. 3–6 | Figs. S1–S6 |
| `M21_cell_reproducibility_package.py` — split indices, masks and checksums | — | — |

## Data

CICIDS2017 and CSE-CIC-IDS2018 are public datasets of the Canadian Institute for Cybersecurity and are
not redistributed here (`data/README.md`).

## Citation

See `CITATION.cff`. The citation will be updated with the journal reference and DOI after publication.

## License

Code: MIT (see `LICENSE`). The datasets remain under their providers' terms.
