# Live Fault Detector: Evaluation and Limits

## Current model

The demo uses `model_runs/xgb_35run_20261013T082155Z/`. The trainer used 35
eligible measured runs and kept whole runs together: 28 train runs and seven
held-out runs (5,080 test rows).

| Fault class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Congestion | 0.961 | 0.958 | 0.959 |
| High latency | 0.959 | 1.000 | 0.979 |
| Link down | 0.949 | 0.963 | 0.956 |
| Normal | 0.995 | 0.994 | 0.995 |
| Packet loss | 0.971 | 0.978 | 0.975 |

The raw classifier on the grouped split scored 0.991 accuracy, 0.979 balanced
accuracy, 0.973 macro F1, and a 0.6% false-alarm rate on normal rows. This
split measures new runs
from the collected lab profiles; the independent profile result below is a
more difficult check.

## Independent severity profile

The clean final capture is `dataset_runs/run_001_20261013T082217Z/`. It used
500 ms latency and 18% packet loss on `r3-r4`, 1 Mbit/s congestion, and
link-down on `r2-r4`. The link-down location was selected so traffic moved
onto the `r3-r4` path before latency and packet loss were injected. Signal
checks confirmed that RTT reached a 500.2 ms median during latency, measured
ping loss reached 60% (median 10%) during the 18% loss injection, and the
throughput stream recorded a 10.5 median during congestion.

The final evaluation is
`model_runs/xgb_holdout_profile_run_001_20261013T082217Z_20261013T083012Z/`.
It scored 591 rows:

| Fault class | Precision | Recall | F1 | Rows |
|---|---:|---:|---:|---:|
| Congestion | 0.478 | 1.000 | 0.646 | 32 |
| High latency | 1.000 | 0.562 | 0.720 | 16 |
| Link down | 1.000 | 1.000 | 1.000 | 16 |
| Normal | 1.000 | 0.947 | 0.973 | 511 |
| Packet loss | 0.533 | 0.500 | 0.516 | 16 |

Overall accuracy was 0.929, balanced accuracy 0.802, macro F1 0.771, and
normal false-alarm rate 5.3%. The 92.9% accuracy is useful for a demo, but it
does not mean every fault is identified reliably: packet-loss recall is 50%,
high-latency recall is 56%, and congestion precision is 48%. Each fault class
has only 16–32 labeled samples in this one profile run.

For a same-capture baseline, the previous model
`xgb_30run_20261013T043628Z` scored 0.497 accuracy, 0.496 balanced accuracy,
0.381 macro F1, and 51.9% normal false alarms on the same 591 rows with the
same link-state override. The clean model improved accuracy by 43.2
percentage points and macro F1 by 39.0 points, while reducing normal false
alarms by 46.6 points. Its saved baseline comparison is
`model_runs/xgb_holdout_profile_run_001_20261013T082217Z_20261013T083532Z/`.

## Repeated held-out interpolation profile

A second, separately collected profile is
`dataset_runs/run_001_20261013T085801Z/`. It used 450 ms latency, 15% packet
loss, and 1.5 Mbit/s congestion—values between the measured training
profiles—and kept link-down on `r2-r4` so later faults remained on the active
path. The run is explicitly marked `training_eligible,false` with the reason
`locked_interpolation_holdout_do_not_train`.

The model scored 531 offline rows with 0.968 accuracy, 0.905 balanced
accuracy, 0.899 macro F1, and a 1.6% normal false-alarm rate. Per-class
precision/recall/F1 were congestion 0.821/0.889/0.853 (36 rows), high latency
1.000/0.778/0.875 (18), link-down 1.000/1.000/1.000 (16), normal
0.991/0.984/0.988 (445), and packet loss 0.700/0.875/0.778 (16).

The streaming detector produced 530 predictions aligned to offline labels
(one of 531 offline rows had no live prediction). On those matched rows it
scored 0.962 accuracy, 0.892 balanced accuracy, and 0.889 macro F1; it
generated 90 alert rows. Class F1 was 0.82 for congestion, 0.88 for high
latency, 0.97 for link-down, 0.98 for normal, and 0.80 for packet loss.
This confirms that the live path ran during a full collection, though the
large alert count includes multiple interface-level samples within each
fault window and is not a count of unique incidents.

Sensor checks found a 451.9 ms median RTT during the latency event, 10%
median ping loss during the 15% injection (with a 0–40% observed range), six
DOWN readings among seven `r2/eth2` link-state samples during link-down, and
10.5 median throughput feature value during congestion. Ping loss and link
state sampling are sparse, and a single capture is not enough to claim stable
performance. The per-class support is still small.

On the same capture, the previous `xgb_30run_20261013T043628Z` model scored
0.571 accuracy, 0.513 balanced accuracy, 0.400 macro F1, and a 43.6% normal
false-alarm rate. The current model improved accuracy by 39.7 percentage
points and macro F1 by 49.9 points, and reduced the normal false-alarm rate
by 42.0 points. The previous-model comparison is saved in
`model_runs/xgb_holdout_profile_run_001_20261013T085801Z_20261013T090652Z/`;
the current-model evaluation and predictions are in
`model_runs/xgb_holdout_profile_run_001_20261013T085801Z_20261013T090539Z/`.

A second independent repetition of the same profile is
`dataset_runs/run_001_20261013T090936Z/`, also locked out of training. On 530
rows it scored 0.975 accuracy, 0.952 balanced accuracy, 0.941 macro F1, and
2.0% normal false alarms. Per-class F1 was 0.865 for congestion, 0.941 for
high latency, 0.941 for link-down, 0.988 for normal, and 0.970 for packet
loss. Its time-aligned live stream matched 530 rows and scored 0.968 accuracy,
0.928 balanced accuracy, and 0.922 macro F1. The saved artifact is
`model_runs/xgb_holdout_profile_run_001_20261013T090936Z_20261013T091623Z/`.

Pooling the two held-out runs (1,061 offline rows; 1,060 live matches) gives
0.972 offline accuracy, 0.927 balanced accuracy, and 0.919 macro F1. Pooled
offline class F1 was 0.859 for congestion, 0.909 for high latency, 0.970 for
link-down, 0.988 for normal, and 0.870 for packet loss. The pooled live
stream scored 0.965 accuracy and 0.905 macro F1. Run-level variation remains:
the first run had 0.778 packet-loss F1, while the second had 0.970. Two
repetitions are stronger evidence than one, but still a small sample.

The second run’s signal checks showed 450.1 ms median RTT in the latency
window, 10% median measured ping loss during 15% injected loss (0–40% range),
six DOWN readings in seven `r2/eth2` samples during link-down, and a 10.5
median throughput feature during congestion. The intermittent ping-loss
estimate reinforces the need for more samples and trials.

## Held-out link-down location

`dataset_runs/run_001_20261013T063041Z/` moved link-down to `r1-r3`, which
was not used for link-down in the clean model's training set. On the valid
link-down event, the clean model plus the interface-state override had 1.000
precision and 0.944 recall (17 of 18 samples detected; 17 link-down
predictions total). This is one event, not repeated evidence of reliable
location transfer.

The later latency and packet-loss events in that run are not valid evaluation
samples: the initial link-down changed the route, and the later `r3-r4`
injections no longer affected measured traffic. The event labels remained in
the raw file for audit, but this run is marked `training_eligible,false` and
was excluded from training and all-class accuracy claims.

The class-specific evaluation artifact is
`model_runs/xgb_holdout_profile_run_001_20261013T063041Z_20261013T083350Z/`.
It evaluates link-down against all other rows: accuracy 0.998, balanced
accuracy 0.972, macro F1 0.985, and zero false link-down alarms among
non-target rows.

## Data validity and training exclusions

The fault schedule can change the active route. When the first link-down
event disables `r1-r3` or `r3-r4`, traffic may fail over to `r2-r4`; subsequent
latency or packet-loss injections on `r3-r4` can then be bypassed. Those runs
must not be treated as correctly labeled examples for all fault classes.

Four captures are retained but marked ineligible for training because their
fault order caused that bypass:

- `run_001_20261013T063041Z`
- `run_001_20261013T071739Z`
- `run_001_20261013T073501Z`
- `run_001_20261013T075915Z`

The trainer reads `training_eligible,false` in `run_metadata.csv` and skips
those runs. The verified 400 ms profile in
`dataset_runs/run_001_20261013T081308Z/` kept `r3-r4` on the active path and
measured the expected RTT, loss, and throughput changes.

## Earlier baselines

The earlier real-data model
`model_runs/xgb_30run_20261013T043628Z/` scored 0.807 accuracy and 0.678
macro F1 on its original run-held-out split. The older artifact
`xgb_30run_20261013T043016Z` scored 0.991 accuracy on that split while using
router/interface identity features; its score dropped substantially on
unseen profiles. The synthetic-to-real experiment
`model_runs/xgb_synthetic_to_real_20261013T045709Z/` scored 0.203 macro F1
and failed to detect high latency and packet loss. These remain useful
warnings against relying on a single accuracy number.

## Live detector behavior

`collect_dataset.sh` starts `live_detector.py` when `LIVE_DETECTOR=1`. It
scores new telemetry rows and appends timestamped predictions, confidence,
and alert status to each run's `live_predictions.csv`; console output is
saved in `live_detector.log`. Both live scoring and offline evaluation apply
the same direct override: a sampled interface state of DOWN is classified as
`link_down`. This prevents the classifier from relying only on the interface
locations present in its training data.

The current detector is a live lab prototype, not production AIOps. The
450 ms/15%/1.5 Mbit/s interpolation profile has two captures; the separate
500 ms/18%/1 Mbit/s severity profile has one. This is still too small to
establish performance stability across a broad range. Further work should
collect more repeated, verified-on-path profiles, especially for packet loss
and congestion, and preserve a new profile as an untouched evaluation set.
