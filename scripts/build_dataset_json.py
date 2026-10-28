"""Collect the beams of all flare_dose metadata files into one dataset JSON for the data loader.

Input:  META_DIR/metadata_<patient_id>.json, one file per patient: the beam target and a list of rays
Output: OUT_FILE, a list with one record per beam (ray):
    {"patient_id": "FLARE22_Tr_0001_0000",
     "ct_path":     ".../FLARE22_Tr_0001_0000.nii.gz",
     "dose_path":   ".../dose_FLARE22_Tr_0001_0000_angle_0.nii.gz",
     "beam_source": [x, y, z],      # mm, as in the metadata
     "beam_target": [x, y, z],      # mm
     "angle":       0,              # deg
     "energy":      157.25}         # MeV
"""

import json
from pathlib import Path

META_DIR = Path("flare_dose/output")          # metadata_*.json files
CT_DIR = Path("flare_CT")             # CT volumes, FLARE22 naming: <patient_id>.nii.gz
DOSE_DIR = META_DIR                           # dose volumes, named by output_filename in the metadata
OUT_FILE = Path("jsons/flare.json")
CHECK_FILES = True                            # True: leave out beams whose CT or dose file is missing

records = []
for meta_file in sorted(META_DIR.glob("metadata_*.json")):
    meta = json.loads(meta_file.read_text())
    patient_id = meta["patient_id"]
    for ray in meta["rays"]:
        records.append({
            "patient_id": patient_id,
            "ct_path": str((CT_DIR / f"{patient_id}.nii.gz").absolute()),
            "dose_path": str((DOSE_DIR / ray["output_filename"]).absolute()),
            "beam_source": ray["beam_source_mm"],
            "beam_target": meta["beam_target_mm"],
            "angle": ray["angle"],
            "energy": ray["beam_energy_MeV"],
        })

if CHECK_FILES:
    missing = {p for r in records for p in (r["ct_path"], r["dose_path"]) if not Path(p).is_file()}
    kept = [r for r in records if r["ct_path"] not in missing and r["dose_path"] not in missing]
    if missing:
        print(f"Left out {len(records) - len(kept)} beams: {len(missing)} files missing, e.g. {sorted(missing)[0]}")
    records = kept

OUT_FILE.write_text(json.dumps(records, indent=2))
print(f"{len(records)} beams of {len({r['patient_id'] for r in records})} patients -> {OUT_FILE}")
