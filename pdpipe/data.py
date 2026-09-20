"""Dataset loading, subject parsing, and cross-cohort feature harmonization."""
from __future__ import annotations

import io
import os
import re
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd

OXFORD_URL = "https://archive.ics.uci.edu/static/public/174/parkinsons.zip"
ISTANBUL_URL = (
    "https://archive.ics.uci.edu/static/public/470/"
    "parkinson+s+disease+classification.zip"
)

SUBJECT_RE = re.compile(r"^phon_(R\d+)_(S\d+)_(\d+)$")

# Oxford 22 acoustic features, in dataset column order.
OXFORD_FEATURES = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA",
    "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE",
]

# Oxford -> Istanbul mapping for the 16 acoustically equivalent measures.
# Each entry documents the shared acoustic definition being matched.
HARMONISED_MAP = {
    "MDVP:Jitter(%)":   ("locPctJitter",   "local jitter, relative"),
    "MDVP:Jitter(Abs)": ("locAbsJitter",   "local jitter, absolute (s)"),
    "MDVP:RAP":         ("rapJitter",      "relative average perturbation"),
    "MDVP:PPQ":         ("ppq5Jitter",     "5-point period perturbation quotient"),
    "Jitter:DDP":       ("ddpJitter",      "difference of differences of periods"),
    "MDVP:Shimmer":     ("locShimmer",     "local shimmer, relative"),
    "MDVP:Shimmer(dB)": ("locDbShimmer",   "local shimmer (dB)"),
    "Shimmer:APQ3":     ("apq3Shimmer",    "3-point amplitude perturbation quotient"),
    "Shimmer:APQ5":     ("apq5Shimmer",    "5-point amplitude perturbation quotient"),
    "MDVP:APQ":         ("apq11Shimmer",   "11-point amplitude perturbation quotient"),
    "Shimmer:DDA":      ("ddaShimmer",     "difference of differences of amplitudes"),
    "NHR":              ("meanNoiseToHarmHarmonicity", "noise-to-harmonics ratio"),
    "HNR":              ("meanHarmToNoiseHarmonicity", "harmonics-to-noise ratio (dB)"),
    "RPDE":             ("RPDE",           "recurrence period density entropy"),
    "DFA":              ("DFA",            "detrended fluctuation analysis"),
    "PPE":              ("PPE",            "pitch period entropy"),
}

# Oxford features with no Istanbul counterpart.
OXFORD_ONLY = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
               "spread1", "spread2", "D2"]


@dataclass
class Cohort:
    """A cohort of voice recordings with subject-level grouping."""
    name: str
    X: pd.DataFrame          # recording-level features
    y: np.ndarray            # recording-level labels (1 = PD)
    groups: np.ndarray       # subject identifier per recording
    features: list[str]

    def __len__(self) -> int:
        return len(self.X)

    @property
    def n_subjects(self) -> int:
        return len(np.unique(self.groups))

    def subject_labels(self) -> pd.Series:
        """One label per subject; asserts label homogeneity within subject."""
        s = pd.DataFrame({"g": self.groups, "y": self.y}).groupby("g")["y"]
        if (s.nunique() > 1).any():
            raise ValueError("Subject with heterogeneous labels detected.")
        return s.first()

    def summary(self) -> dict:
        sl = self.subject_labels()
        return {
            "cohort": self.name,
            "recordings": len(self.X),
            "subjects": self.n_subjects,
            "pd_subjects": int((sl == 1).sum()),
            "hc_subjects": int((sl == 0).sum()),
            "pd_recordings": int((self.y == 1).sum()),
            "hc_recordings": int((self.y == 0).sum()),
            "features": len(self.features),
        }


def _cache_path(root: str, name: str) -> str:
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, name)


def _download(url: str, dest: str) -> str:
    if os.path.exists(dest):
        return dest
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        f.write(r.read())
    return dest


def parse_subject_table(names: pd.Series) -> pd.DataFrame:
    """Deterministic filename -> (session, subject, recording index) parse.

    Every Oxford filename matches ``phon_<session>_<subject>_<index>``. The
    subject field is used verbatim as the grouping unit; no merging, renaming
    or heuristic reassignment is applied.
    """
    rows = []
    for n in names:
        m = SUBJECT_RE.match(n)
        if m is None:
            raise ValueError(f"Filename does not match expected pattern: {n}")
        rows.append({"filename": n, "session": m.group(1),
                     "subject": m.group(2), "recording_index": int(m.group(3))})
    return pd.DataFrame(rows)


def load_oxford(root: str = "data") -> Cohort:
    """UCI Parkinson's (Little et al., Oxford) — 195 recordings, 22 features."""
    zp = _download(OXFORD_URL, _cache_path(root, "oxford.zip"))
    with zipfile.ZipFile(zp) as z:
        member = [m for m in z.namelist() if m.endswith("parkinsons.data")][0]
        df = pd.read_csv(io.BytesIO(z.read(member)))
    meta = parse_subject_table(df["name"])
    return Cohort(
        name="Oxford",
        X=df[OXFORD_FEATURES].astype(float).reset_index(drop=True),
        y=df["status"].to_numpy(dtype=int),
        groups=meta["subject"].to_numpy(),
        features=list(OXFORD_FEATURES),
    )


def load_oxford_meta(root: str = "data") -> pd.DataFrame:
    """Filename -> subject mapping table with labels and recording counts."""
    zp = _download(OXFORD_URL, _cache_path(root, "oxford.zip"))
    with zipfile.ZipFile(zp) as z:
        member = [m for m in z.namelist() if m.endswith("parkinsons.data")][0]
        df = pd.read_csv(io.BytesIO(z.read(member)))
    meta = parse_subject_table(df["name"])
    meta["status"] = df["status"].to_numpy()
    return meta


def load_istanbul(root: str = "data") -> Cohort:
    """Sakar et al. Istanbul cohort — 756 recordings, 252 subjects."""
    zp = _download(ISTANBUL_URL, _cache_path(root, "istanbul.zip"))
    csv_path = _cache_path(root, "pd_speech_features.csv")
    if not os.path.exists(csv_path):
        with zipfile.ZipFile(zp) as z:
            rar = [m for m in z.namelist() if m.endswith(".rar")][0]
            rar_path = _cache_path(root, "pd_speech_features.rar")
            with open(rar_path, "wb") as f:
                f.write(z.read(rar))
        _extract_rar(rar_path, root)
    df = pd.read_csv(csv_path, header=1)
    ist_cols = [HARMONISED_MAP[k][0] for k in HARMONISED_MAP]
    X = df[ist_cols].astype(float).reset_index(drop=True)
    # Rename to the Oxford convention so a model fitted on Oxford can be
    # applied directly; the mapping is documented in HARMONISED_MAP.
    X.columns = list(HARMONISED_MAP.keys())
    return Cohort(
        name="Istanbul",
        X=X,
        y=df["class"].to_numpy(dtype=int),
        groups=df["id"].astype(str).to_numpy(),
        features=list(HARMONISED_MAP.keys()),  # renamed to Oxford convention
    )


def _extract_rar(rar_path: str, out_dir: str) -> None:
    """Extract the Istanbul .rar (Colab: apt-get install unrar-free)."""
    import shutil
    import subprocess
    exe = shutil.which("unrar") or shutil.which("unrar-free") or shutil.which("unar")
    if exe is None:
        raise RuntimeError(
            "No rar extractor found. On Colab run: "
            "!apt-get -qq install -y unrar-free"
        )
    subprocess.run([exe, "-x", rar_path], cwd=out_dir, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def harmonised_oxford(ox: Cohort) -> Cohort:
    """Oxford cohort restricted to the 16 features shared with Istanbul."""
    cols = list(HARMONISED_MAP.keys())
    return Cohort(name="Oxford-16", X=ox.X[cols].copy(), y=ox.y,
                  groups=ox.groups, features=cols)
