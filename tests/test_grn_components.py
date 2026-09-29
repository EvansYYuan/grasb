import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_grn_drift import fit_target
from grn_drift import GRNDrift


def test_fit_target_can_shrink_toward_nonzero_prior():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(80, 2))
    y = 2.0 * x[:, 0] - 0.5 * x[:, 1] + 0.1 * rng.normal(size=80)
    frame = pd.DataFrame({"TF1": x[:, 0], "TF2": x[:, 1], "TARGET": y})
    _, beta, _ = fit_target(
        frame,
        "TARGET",
        ["TF1", "TF2"],
        np.zeros(80),
        alpha=1.0,
        beta_prior=np.array([2.0, -0.5]),
    )
    np.testing.assert_allclose(beta, [2.0, -0.5], atol=0.08)


def test_grn_artifact_preserves_fractional_interval_boundaries(tmp_path):
    artifact = {
        "genes": np.array(["TF", "TARGET"]),
        "component": "regulon",
        "intervals": [(3.3, 4.3), (4.3, 6.0)],
        "artifacts": {
            ("lineage", (3.3, 4.3)): {
                "edges": {1: (np.array([0]), np.array([2.0]))},
                "c": {1: 0.0},
                "gamma": 1.0,
                "n_cells": 50,
            },
            ("lineage", (4.3, 6.0)): {
                "edges": {1: (np.array([0]), np.array([3.0]))},
                "c": {1: 0.0},
                "gamma": 1.0,
                "n_cells": 50,
            },
        },
    }
    path = tmp_path / "grn.pkl"
    with path.open("wb") as handle:
        pickle.dump(artifact, handle)

    grn = GRNDrift.from_pkl(path)
    assert grn.interval_lo == [3.3, 4.3]
    assert grn.interval_hi == [4.3, 6.0]
    assert grn.interval_id_for_tp(4.0) == 0
    assert grn.interval_id_for_tp(5.3) == 1

