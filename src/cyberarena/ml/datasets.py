"""Dataset acquisition and preprocessing for the three cyberarena classifiers.

CLI:  python -m cyberarena.ml.datasets --all          (or --name malware|phishing|network|malware_tuandromd)

``malware`` is UCI NATICUSdroid (id 722, Android permissions). The older, much smaller TUANDROMD set
(id 855) stays loadable as ``malware_tuandromd`` but is not built or trained by ``--all``.

Output per dataset: ``data/processed/<name>.npz`` with X_train, y_train, X_val, y_val, X_test, y_test
(already standard-scaled, float32), ``feature_names`` and the scaler params (``scaler_mean``,
``scaler_scale``) plus ``source`` (dataset URL).

Labels are always 1 = malicious, 0 = benign.

Leakage guards:
* the scaler (and one-hot vocabulary, constant-column filter) is fit on the training split only;
* the UCI phishing, NATICUSdroid and TUANDROMD datasets contain thousands of exact duplicate rows
  (NATICUSdroid: 29.3k rows but only ~7.5k distinct, 14.7k malware rows collapse to ~2.6k; TUANDROMD:
  4.4k rows but only ~660 distinct vectors), which would leak between train and test and inflate
  metrics. Exact duplicates are dropped, and splits are *group-aware*: identical feature vectors
  (e.g. the same vector with conflicting labels) always land in the same split;
* no ID / metadata columns are kept (NSL-KDD's ``difficulty`` column is dropped).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from cyberarena.config import ROOT

NAMES = ("malware", "phishing", "network")  # built / trained by default
EXTRA_NAMES = ("malware_tuandromd",)  # loadable on request only
ALL_NAMES = NAMES + EXTRA_NAMES
SEED = 42
VAL_FRAC = 0.15
TEST_FRAC = 0.15

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


@dataclass(frozen=True)
class RemoteFile:
    filename: str
    url: str
    sha256: str
    min_bytes: int


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    source: str  # human-facing dataset page, recorded in metrics
    files: tuple[RemoteFile, ...]


_UCI = "https://archive.ics.uci.edu"
_NSL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master"

SPECS: dict[str, DatasetSpec] = {
    "phishing": DatasetSpec(
        "phishing",
        f"{_UCI}/dataset/327/phishing+websites",
        (
            RemoteFile(
                "phishing.csv",
                f"{_UCI}/static/public/327/data.csv",
                "35dd9aa6870643c292d68343d926869095292a7406d198305cae7435069b59c2",
                500_000,
            ),
        ),
    ),
    "malware": DatasetSpec(
        "malware",
        f"{_UCI}/dataset/722/naticusdroid+android+permissions+dataset",
        (
            RemoteFile(
                "naticusdroid.csv",
                f"{_UCI}/static/public/722/data.csv",
                "5764687009af15605f46941902d6a1c4c1a07618dfefb94a0275edb4869780ba",
                4_000_000,
            ),
        ),
    ),
    "malware_tuandromd": DatasetSpec(
        "malware_tuandromd",
        f"{_UCI}/dataset/855/tuandromd+tezpur+university+android+malware+dataset",
        (
            RemoteFile(
                "TUANDROMD.zip",
                f"{_UCI}/static/public/855/tuandromd+(tezpur+university+android+malware+dataset).zip",
                "441d2005b97f3816e8b64e692c5e5a0b629685091638aad9e67fd81f40a5cf5f",
                20_000,
            ),
        ),
    ),
    "network": DatasetSpec(
        "network",
        "https://www.unb.ca/cic/datasets/nsl.html (mirror: github.com/defcom17/NSL_KDD)",
        (
            RemoteFile(
                "KDDTrain+.txt",
                f"{_NSL}/KDDTrain%2B.txt",
                "1b86d2f957b33082081bba410fe129b475efebcc13c9014c3f447c8271aadf95",
                15_000_000,
            ),
            RemoteFile(
                "KDDTest+.txt",
                f"{_NSL}/KDDTest%2B.txt",
                "fa46b0935342616aa83b7c2578db355b6a7aaabbc492248172c7a1e8b7ab8f84",
                3_000_000,
            ),
        ),
    ),
}

NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty",
]  # fmt: skip
NSL_KDD_CATEGORICAL = ("protocol_type", "service", "flag")


# --------------------------------------------------------------------------- download


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify(path: Path, rf: RemoteFile) -> None:
    size = path.stat().st_size
    if size < rf.min_bytes:
        raise ValueError(f"{path} is {size} bytes, expected >= {rf.min_bytes}; delete it and re-run")
    digest = _sha256(path)
    if digest != rf.sha256:
        raise ValueError(f"{path} sha256 {digest} != expected {rf.sha256}; delete it and re-run")


def download(name: str, raw_dir: Path = RAW_DIR, force: bool = False) -> dict[str, Path]:
    """Fetch the raw files for ``name`` into ``raw_dir/<name>/`` (cached), verify size + sha256."""
    import requests  # local import: only needed when actually downloading

    spec = SPECS[name]
    out_dir = Path(raw_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for rf in spec.files:
        path = out_dir / rf.filename
        if force or not path.exists():
            print(f"[{name}] downloading {rf.url}")
            resp = requests.get(rf.url, timeout=300)
            resp.raise_for_status()
            tmp = path.with_suffix(path.suffix + ".part")
            tmp.write_bytes(resp.content)
            tmp.replace(path)
        else:
            print(f"[{name}] using cached {path}")
        _verify(path, rf)
        paths[rf.filename] = path
    return paths


# --------------------------------------------------------------------------- raw -> frames


@dataclass
class Frames:
    """Cleaned features/labels. ``X_test``/``y_test`` set only when the source ships its own test split."""

    X: pd.DataFrame
    y: np.ndarray
    X_test: pd.DataFrame | None = None
    y_test: np.ndarray | None = None
    categorical: tuple[str, ...] = ()


def clean_phishing(df: pd.DataFrame) -> Frames:
    """UCI 327: 30 ternary features in {-1, 0, 1}; target ``result`` is -1 = phishing, 1 = legitimate."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    target = next(c for c in df.columns if c.lower() == "result")
    df = df.dropna().drop_duplicates()
    y = (df[target].astype(int) == -1).astype(np.int64).to_numpy()
    X = df.drop(columns=[target]).astype(np.float32)
    return Frames(X, y)


def clean_malware(df: pd.DataFrame) -> Frames:
    """NATICUSdroid: 86 binary permission features; target ``Result`` is 1 = malware, 0 = benign."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    target = next(c for c in df.columns if c.lower() == "result")
    df = df.dropna().drop_duplicates()  # 29.3k rows -> ~7.5k; most malware rows are exact copies
    labels = set(df[target].astype(int).unique())
    if not labels <= {0, 1}:
        raise ValueError(f"unexpected NATICUSdroid labels: {sorted(labels)}")
    y = df[target].astype(np.int64).to_numpy()
    X = df.drop(columns=[target]).astype(np.float32)
    return Frames(X, y)


def clean_tuandromd(df: pd.DataFrame) -> Frames:
    """TUANDROMD: 241 binary permission/API features; ``Label`` is malware / goodware."""
    df = df.copy()
    # The UCI CSV ships with a case-insensitive find/replace of "no" -> "goodware" applied to its header
    # too (e.g. DOWNLOAD_WITHOUT_goodwareTIFICATION); undo it in column names only.
    df.columns = [
        c.strip().replace("getLastKgoodwarewn", "getLastKnown").replace("goodware", "NO") for c in df.columns
    ]
    df = df.dropna().drop_duplicates()  # 3.8k of 4.4k rows are exact copies of ~130 malware vectors
    label = df["Label"].astype(str).str.strip().str.lower()
    if not set(label.unique()) <= {"malware", "goodware"}:
        raise ValueError(f"unexpected TUANDROMD labels: {sorted(label.unique())}")
    y = (label == "malware").astype(np.int64).to_numpy()
    X = df.drop(columns=["Label"]).astype(np.float32)
    return Frames(X, y)


def _nsl_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    df = df.copy()
    if df.shape[1] == len(NSL_KDD_COLUMNS) - 1:  # some mirrors omit difficulty
        df.columns = NSL_KDD_COLUMNS[:-1]
    else:
        df.columns = NSL_KDD_COLUMNS
        df = df.drop(columns=["difficulty"])  # metadata about the record, not a traffic feature
    df = df.dropna()
    y = (df["label"].astype(str).str.strip() != "normal").astype(np.int64).to_numpy()
    X = df.drop(columns=["label"])
    for c in X.columns:
        if c not in NSL_KDD_CATEGORICAL:
            X[c] = X[c].astype(np.float32)
    return X, y


def clean_network(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Frames:
    """NSL-KDD: binary normal(0) vs any attack(1). Uses the official KDDTest+ as the test split."""
    X, y = _nsl_frame(train_df)
    X_te, y_te = _nsl_frame(test_df)
    return Frames(X, y, X_te, y_te, categorical=NSL_KDD_CATEGORICAL)


def load_frames(name: str, raw_dir: Path = RAW_DIR, force_download: bool = False) -> Frames:
    paths = download(name, raw_dir, force=force_download)
    if name == "phishing":
        return clean_phishing(pd.read_csv(paths["phishing.csv"]))
    if name == "malware":
        return clean_malware(pd.read_csv(paths["naticusdroid.csv"]))
    if name == "malware_tuandromd":
        with zipfile.ZipFile(paths["TUANDROMD.zip"]) as z:
            csv = next(n for n in z.namelist() if n.lower().endswith(".csv"))
            return clean_tuandromd(pd.read_csv(io.BytesIO(z.read(csv))))
    if name == "network":
        return clean_network(
            pd.read_csv(paths["KDDTrain+.txt"], header=None),
            pd.read_csv(paths["KDDTest+.txt"], header=None),
        )
    raise KeyError(name)


# --------------------------------------------------------------------------- split / encode / scale


def group_split(
    X: pd.DataFrame, y: np.ndarray, test_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified split where identical feature rows share a split. Returns boolean (keep, held_out) masks."""
    keys = pd.util.hash_pandas_object(X.reset_index(drop=True), index=False).to_numpy()
    _, inv = np.unique(keys, return_inverse=True)
    n_groups = int(inv.max()) + 1
    g_label = np.zeros(n_groups, dtype=np.int64)
    np.maximum.at(g_label, inv, np.asarray(y, dtype=np.int64))
    stratify = g_label if np.bincount(g_label, minlength=2).min() >= 2 else None
    _, g_out = train_test_split(
        np.arange(n_groups), test_size=test_frac, random_state=seed, stratify=stratify
    )
    held = np.isin(inv, g_out)
    return ~held, held


def one_hot(
    train: pd.DataFrame, others: list[pd.DataFrame], cols: tuple[str, ...]
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """One-hot ``cols`` with the vocabulary of ``train`` only; unseen categories become all-zero."""
    if not cols:
        return train, others
    enc_train = pd.get_dummies(train, columns=list(cols), prefix_sep="=", dtype=np.float32)
    enc_others = [
        pd.get_dummies(o, columns=list(cols), prefix_sep="=", dtype=np.float32).reindex(
            columns=enc_train.columns, fill_value=0.0
        )
        for o in others
    ]
    return enc_train, enc_others


def split_and_scale(frames: Frames, seed: int = SEED, val_frac: float = VAL_FRAC,
                    test_frac: float = TEST_FRAC) -> dict[str, np.ndarray]:  # fmt: skip
    """Split (group-aware, stratified), one-hot + drop constant cols + standard-scale, all fit on train."""
    X, y = frames.X.reset_index(drop=True), np.asarray(frames.y, dtype=np.int64)
    if frames.X_test is None:
        keep, held = group_split(X, y, test_frac, seed)
        X_test, y_test = X[held], y[held]
        X, y = X[keep].reset_index(drop=True), y[keep]
        rel_val = val_frac / (1.0 - test_frac)
    else:
        X_test, y_test = frames.X_test.reset_index(drop=True), np.asarray(frames.y_test, dtype=np.int64)
        rel_val = val_frac
    tr, va = group_split(X, y, rel_val, seed)
    X_train, X_val, y_train, y_val = X[tr], X[va], y[tr], y[va]

    X_train, (X_val, X_test) = one_hot(X_train, [X_val, X_test], frames.categorical)

    nonconst = X_train.columns[X_train.nunique(dropna=False) > 1]
    X_train, X_val, X_test = X_train[nonconst], X_val[nonconst], X_test[nonconst]

    scaler = StandardScaler().fit(X_train.to_numpy(np.float64))

    def tf(df: pd.DataFrame) -> np.ndarray:
        return scaler.transform(df.to_numpy(np.float64)).astype(np.float32)

    return {
        "X_train": tf(X_train), "y_train": y_train.astype(np.int64),
        "X_val": tf(X_val), "y_val": y_val.astype(np.int64),
        "X_test": tf(X_test), "y_test": np.asarray(y_test, dtype=np.int64),
        "feature_names": np.array([str(c) for c in nonconst]),
        "scaler_mean": scaler.mean_.astype(np.float64),
        "scaler_scale": scaler.scale_.astype(np.float64),
    }  # fmt: skip


# --------------------------------------------------------------------------- persist


def save_processed(name: str, arrays: dict[str, np.ndarray], source: str,
                   processed_dir: Path = PROCESSED_DIR) -> Path:  # fmt: skip
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    path = processed_dir / f"{name}.npz"
    np.savez_compressed(path, **arrays, source=np.array(source))
    return path


def load_processed(name: str, processed_dir: Path = PROCESSED_DIR) -> dict[str, np.ndarray]:
    path = Path(processed_dir) / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `python -m cyberarena.ml.datasets --all` first")
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def build(name: str, raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR,
          force_download: bool = False, seed: int = SEED) -> Path:  # fmt: skip
    frames = load_frames(name, raw_dir, force_download)
    arrays = split_and_scale(frames, seed=seed)
    path = save_processed(name, arrays, SPECS[name].source, processed_dir)
    pos = lambda a: f"{len(a)} rows, {a.mean():.1%} malicious"
    print(
        f"[{name}] {len(arrays['feature_names'])} features | train {pos(arrays['y_train'])} | "
        f"val {pos(arrays['y_val'])} | test {pos(arrays['y_test'])} -> {path}"
    )
    return path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="build all three datasets")
    g.add_argument("--name", choices=ALL_NAMES)
    ap.add_argument("--force-download", action="store_true")
    args = ap.parse_args(argv)
    for name in NAMES if args.all else (args.name,):
        build(name, force_download=args.force_download)


if __name__ == "__main__":
    main()
