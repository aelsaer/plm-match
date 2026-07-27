# MegaLoc5 + farthest8_diverse_desc results

Collected on 2026-07-25 from the local run folders and the saved VisualLocalization benchmark export.

Scope: native feature SfM where available, MegaLoc top-5 retrieval, fixed `farthest8_diverse_desc` point-memory selection, `obs8`.

## VisualLocalization benchmark datasets

Scores are benchmark triplets. The macro column is the unweighted average over the reported splits for that dataset.

| Dataset | Feature / map | Split scores | Macro |
|---|---|---|---:|
| Aachen Day-Night v1.1 | ALIKED+LG native fullmap SfM | day: 88.5 / 95.8 / 99.3; night: 77.0 / 90.6 / 99.5 | 82.8 / 93.2 / 99.4 |
| Aachen Day-Night v1.1 | SP+SG native fullmap SfM | missing for exact MegaLoc5 + farthest8 run | — |
| RobotCar Seasons v2 | ALIKED+LG native SfM | day: 65.0 / 95.0 / 100.0; night: 48.0 / 87.6 / 98.8 | 56.5 / 91.3 / 99.4 |
| RobotCar Seasons v2 | SP+SG native SfM | day: 64.7 / 94.6 / 100.0; night: 21.7 / 51.0 / 86.9 | 43.2 / 72.8 / 93.5 |
| CMU Seasons Extended | ALIKED+LG native SfM | urban: 94.1 / 97.3 / 99.3; suburban: 95.6 / 98.0 / 99.8; park: 86.2 / 90.8 / 97.4 | 92.0 / 95.4 / 98.8 |
| CMU Seasons Extended | SP+SG native SfM | missing for exact MegaLoc5 + farthest8 run | — |

## Cambridge Landmarks

Scores are median translation / rotation errors, averaged over the five scenes.

| Feature / map | KingsCollege | OldHospital | ShopFacade | StMarysChurch | GreatCourt | Macro avg. |
|---|---:|---:|---:|---:|---:|---:|
| ALIKED+LG native SfM | 10.54 cm / 0.191° | 12.55 cm / 0.261° | 4.36 cm / 0.188° | 6.63 cm / 0.217° | 14.66 cm / 0.090° | 9.75 cm / 0.190° |
| SP+SG native SfM | 10.15 cm / 0.179° | 12.43 cm / 0.277° | 4.08 cm / 0.192° | 6.54 cm / 0.213° | 14.52 cm / 0.087° | 9.55 cm / 0.189° |

## Missing exact runs

The exact MegaLoc5 + `farthest8_diverse_desc` submissions/results were not found for:

- Aachen SP+SG native fullmap SfM.
- CMU Extended SP+SG native SfM.

Nearby runs exist for those cases, but they use a different selector or retrieval method and should not be mixed into this table.
