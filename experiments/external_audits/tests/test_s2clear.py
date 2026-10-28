import pickle

import numpy as np
import pytest

from experiments.external_audits.denethor import s2clear


def test_clp_scale_detection():
    assert s2clear.detect_clp_scale(np.array([0, 3, 250], dtype=np.uint8)) == 255
    assert s2clear.detect_clp_scale(np.array([0, 3, 99], dtype=np.uint8)) == 100


def test_clear_rules_thresholds():
    clp_mean = np.array([[39.9, 101.0, 103.0]])  # (T=1, n_fields=3)
    scl_clear = np.array([[0.9, 0.4, 0.9]])
    isdata = np.array([[1.0, 1.0, 0.5]])
    rules = s2clear.clear_rules(clp_mean, scl_clear, isdata, scale=255)
    assert rules["literal40"].tolist() == [[True, False, False]]
    assert rules["s2cp40"].tolist() == [[True, True, False]]  # threshold 0.40 * 255 = 102
    assert rules["scl"].tolist() == [[True, False, False]]  # needs >= 0.5 clear and isdata >= 0.99


class _StubBBox:
    def __init__(self):
        self.min_x, self.min_y, self.max_x, self.max_y = 432000.0, 5808000.0, 456000.0, 5832000.0
        self.crs = "EPSG:32633"


def test_load_bbox_grid_consistency(tmp_path, monkeypatch):
    p = tmp_path / "bbox.pkl"
    with open(p, "wb") as fh:
        pickle.dump(_StubBBox(), fh)
    minx, miny, maxx, maxy, crs = s2clear.load_bbox(p, tile="18E-242N", zone="33N")
    assert (minx, miny, maxx, maxy) == (432000.0, 5808000.0, 456000.0, 5832000.0)
    assert crs == "EPSG:32633"


def test_load_bbox_rejects_grid_mismatch(tmp_path):
    p = tmp_path / "bbox.pkl"
    with open(p, "wb") as fh:
        pickle.dump(_StubBBox(), fh)
    with pytest.raises(ValueError):
        s2clear.load_bbox(p, tile="19E-242N", zone="33N")  # 19 * 24000 = 456000 != 432000


def test_grid_bbox_from_tile_name():
    assert s2clear.grid_bbox("18E-242N", "33N") == (432000.0, 5808000.0, 456000.0, 5832000.0, "EPSG:32633")
    assert s2clear.grid_bbox("20E-259N", "34S")[4] == "EPSG:32734"
