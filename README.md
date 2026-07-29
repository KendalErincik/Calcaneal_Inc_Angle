# Automated Measurement of the Calcaneal Inclination Angle (CIA)

Interpretable deep-learning pipeline for automated measurement of the calcaneal
inclination angle (CIA) on lateral weight-bearing foot radiographs, using **partial
semantic segmentation** and deterministic landmark geometry.

The application used for annotation, model training and measurement is our own software,
**RadiologyMate**.

## Method (as reported in the manuscript)

Three anatomical boundaries are segmented, each by its **own independently trained
single-class U-Net** (a compact 4-level U-Net trained from scratch). Because every
structure has a dedicated network, there is **no inter-class loss weighting
hyperparameter** at all.

| Structure | Role | Validation Dice |
|---|---|---|
| Calcaneus inferior (CAL) | P2 + tangent search for P3 | 0.942 |
| 5th metatarsal head (M5) | P1 | 0.856 |
| Calcaneonavicular / inferior calcaneocuboid (CC) | P4 | 0.826 |

Each network is trained with an unweighted Dice + Binary Cross-Entropy loss and
**early stopping** on the validation Dice; per-class binarisation thresholds
(0.50 / 0.30 / 0.40) are selected on the recorded validation splits.

From the three masks the pipeline derives four landmarks deterministically:
P1 (inferior-most point of M5), P2 (inferior-most point of the calcaneus), P4
(posterior-inferior corner of CC) and P3 (lower tangent from P4 along the inferior
calcaneal contour). The CIA is the angle between the ground line P1->P2 and the
tangent line P4->P3. This deterministic step is identical regardless of how the masks
are produced and is provided here in [`cia_geometry.py`](cia_geometry.py).

## Held-out internal test performance

On a **held-out internal test set** of 105 radiographs from 76 patients, entirely
independent of the development cohort (779 radiographs / 724 patients; no patient in
both), the model was compared against three readers who measured every case
independently and blinded:

- Mean absolute error vs the 3-reader mean: **0.727 deg** (95% CI 0.533-0.922;
  cluster-bootstrap CI 0.569-0.926)
- Bias -0.110 deg; 95% limits of agreement -2.535 to +2.314 deg
- 94.3% of cases within 2 deg; ICC(A,1) 0.984 across model and readers, 0.994 among readers

Inter-annotator Dice between the two radiologists (50 development images re-annotated
from a blank canvas): 0.848 (CAL), 0.912 (M5), 0.819 (CC); passing the second
annotator's masks through the identical geometry changed the angle by 0.39 +/- 0.29 deg
(ICC 0.996).

## Repository contents

- `cia_geometry.py` - deterministic landmark extraction + CIA computation (numpy, opencv).
- `LICENSE` - MIT.
- `CITATION.cff` - how to cite this work.

## Reproducing the geometry

```python
import cv2, numpy as np
from cia_geometry import compute_from_masks

masks = {
    "calcaneus_inferior": cv2.imread("cal.png", 0),
    "metatarsal_5":       cv2.imread("m5.png", 0),
    "calcaneonavicular":  cv2.imread("cc.png", 0),
}
cia, P1, P2, P3, P4 = compute_from_masks(masks)
print(f"CIA = {cia:.1f} deg")
```

## Data availability

Patient radiographs cannot be shared publicly (retrospective clinical images);
requests are considered on a reasonable basis subject to institutional and ethical
approval.

## Citation

If you use this code, please cite the paper and this repository (see `CITATION.cff`).

## License

MIT - see `LICENSE`.
