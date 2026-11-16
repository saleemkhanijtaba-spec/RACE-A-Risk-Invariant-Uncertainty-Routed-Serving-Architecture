# RACE: risk-invariant, uncertainty-routed serving for Classical Arabic GEC

Code and results for the paper "RACE: A Risk-Invariant, Uncertainty-Routed Serving
Architecture for Faithful Grammatical Error Correction of Classical Arabic on CPU Clusters".

## Layout

| Path | Contents |
|---|---|
| `code/` | Full pipeline (below) |
| `code/cs/` | Optional GPU, LLM and multi-node experiments (see `code/cs/README_CS.md`) |
| `code/third_party/` | Vendored plotting helpers, with their own licence notice |
| `results/` | CSV/JSON outputs of every experiment |

Source texts are not redistributed. They are the OpenITI corpus (https://github.com/OpenITI,
release 2023.1.8, https://doi.org/10.5281/zenodo.10007820) and the Tanzil Qur'an text
(simple-clean, version 1.1, https://tanzil.net/download/). Place them under `data/` as
described in `code/prep_data.py`. `code/gen_tables.py` writes LaTeX tables and `numbers.tex`
into the repository root.

## Pipeline

All steps run on CPU (tested on 2 cores, 7 GB RAM). `pip install -r code/requirements.txt`.

1. `python3 code/prep_data.py` – extracts sentences from the OpenITI works and the
   Tanzil Qur'an text, mines silver rhetorical / isti'ara labels from the two tafsir works,
   builds splits. `code/canonical_istiara.py` holds the 30 canonical exemplar tokens.
2. `python3 code/gec_data.py` – seeded synthetic error injection (six families) and
   data-derived edit tag set (|T| = 913).
3. `python3 code/pretrain.py` – MLM pretraining of the shared encoder (d=256, 4 layers).
4. `bash code/run_queue.sh code/queue_main.txt` then `code/queue_rest.txt` – trains
   RACE, baselines (BiLSTM, scratch transformer, pretrained tagger, stacked hybrid,
   seq2seq) and the seven ablations; each line is `model seed`. Predictions go to
   `results/preds/`.
5. `python3 code/analysis.py` – calibration, conformal risk control, rhetoric-gate sweeps,
   metrics, Welch/Holm statistics, ablation table data.
6. `python3 code/serve_bench.py` – measured latency, batching, worker scaling.
   `python3 code/routing_extra.py` – routing sweep, theta chosen on the development split, mixture risk,
   end-to-end conformal threshold. `python3 code/analysis_extra.py` – 1,000 calibration resplits,
   Qur'an/training-prose overlap, non-inferiority test, AUC intervals, duplicate check.
7. `python3 code/simulate.py` – discrete-event cluster model; `python3 code/load_test.py` – measured
   open-loop load test on the local node, compared with the model (run on an idle machine).
   `python3 code/multidomain.py build eval` – four extra held-out OpenITI works (`data/openiti_extra/`), recalibration.
   `python3 code/analysis_extra.py rcbase` – conformal risk control vs naive threshold vs Learn-then-Test.
   `python3 code/bootstrap_sig.py` – paired sentence-level bootstrap of F0.5 differences (Table 5).
   `python3 code/mem_profile.py` – resident memory of light and heavy serving workers.
   `python3 code/design_space.py` – risk-invariant design-space exploration of routing level and cores (Table 8).
   `python3 code/cost_all.py` – equal-footing latency/throughput of every model (idle machine).
   `code/cs/` – hyperparameter search (`make_queues.py`, `hp_select.py`), leak-free data (`leakfree_data.py`),
   evaluation of any tag (`eval_tags.py`), and GPU / multi-node scripts (`hf_tagger.py`, `hf_seq2seq.py`,
   `llm_correct.py`, `gen_eval.py`, `cluster_bench.py`; see `code/cs/README_CS.md`).
8. `python3 code/gen_tables.py && python3 code/make_figs.py && python3 code/figs_arch.py`
   – regenerates tables, `numbers.tex` and figures.
9. `pdflatex main && bibtex main && pdflatex main && pdflatex main`.

## Notes on scope

* Errors are synthetic; absolute F0.5 values are not comparable with QALB/ZAEBUC.
* No LLM was run; published LLM and QALB values are quoted as reported.
* Cluster results beyond two cores come from the simulator, not a physical cluster.
* Seeds: RACE and re-implemented baselines 5; seq2seq 3; ablations 3 per variant; tuned RACE and pretrained tagger 5; leak-free retraining 3.

## Data sources

OpenITI corpus, release 2023.1.8 (https://doi.org/10.5281/zenodo.10007820) and Tanzil Qur'an text
(https://tanzil.net), used under their respective licences. Checkpoints are
not included in the archive because of size; they are reproduced by steps 3–4.
