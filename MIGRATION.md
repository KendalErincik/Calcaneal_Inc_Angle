# Repo migration to the revised (per-class) version + Zenodo release

The public repo `KendalErincik/Calcaneal_Inc_Angle` currently holds the PRE-revision
system (1 commit, 0 tags; README describes "U-Net + EfficientNet-B0, 3-channel output,
weighted BCE"). That contradicts the revised manuscript. Update it before creating any
release, so the Zenodo archive/DOI reflects the correct code.

## 1. Files in THIS folder (push these)
- `README.md`        - revised method (three from-scratch single-class U-Nets, no smp/EfficientNet).
- `cia_geometry.py`  - deterministic landmark + CIA geometry (numpy, opencv). Model-independent core.
- `CITATION.cff`     - fill the TODOs (co-authors; Zenodo DOI after step 4; journal once accepted).
- `LICENSE`          - MIT.
- `requirements.txt` - numpy, opencv-python (only what cia_geometry needs).

## 2. DELETE from the repo (old, contradicts the paper)
- `calcaneal_cia.py`          - old EfficientNet-B0 / 3-channel inference code.
- `calcaneal_cia_colab.ipynb` - old notebook; it imports `segmentation_models_pytorch`
  (smp / EfficientNet) and so contradicts the "from scratch, no pretrained encoder" method.
- old `README.md` and old `requirements.txt` - replaced by the versions here.

(If you want a runnable notebook, ask and a clean deterministic-geometry demo without
smp/torch can be written to match the README.)

## 3. Push (in your local clone of the repo)
```
cd <local Calcaneal_Inc_Angle repo>
git rm calcaneal_cia.py calcaneal_cia_colab.ipynb
copy /Y "...\Revizyon 2026-07\Calcaneal_Inc_Angle_repo_v2_READY\*" .
git add -A
git commit -m "Revise repo to per-class U-Net method; deterministic geometry; drop smp/EfficientNet code and outdated notebook"
git push
```

## 4. Release + Zenodo
1. GitHub -> Releases -> Create a new release -> new tag `v2.0.0` -> Publish.
   (Zenodo is already connected, so it archives THIS release automatically.)
2. Open the Zenodo record -> copy the **concept DOI** ("Cite all versions"),
   e.g. `10.5281/zenodo.XXXXXXXX`.
3. Put that DOI in `CITATION.cff` (uncomment the `doi:` line) and commit.
4. Send me the concept DOI -> it goes into the manuscript Data/Code Availability
   statement and the Response-to-Editor software-availability paragraph.

Note: connecting Zenodo does NOT archive past releases; only releases published after
connecting are archived. So you must publish a new release now to get a DOI. You do not
need to delete the old (untagged) state - the new release + concept DOI supersede it.
