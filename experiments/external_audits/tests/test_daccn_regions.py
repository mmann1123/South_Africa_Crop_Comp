import pytest

from experiments.external_audits.daccn import regions


class _FC:
    def __init__(self, n):
        self._n = n

    def filter(self, *a, **k):
        return self

    def size(self):
        return self

    def getInfo(self):
        return self._n

    def first(self):
        return "feature"


def test_region_lookup_single_feature(monkeypatch):
    import ee

    monkeypatch.setattr(ee, "FeatureCollection", lambda *a, **k: _FC(2))
    monkeypatch.setattr(ee, "Filter", type("F", (), {"eq": staticmethod(lambda *a, **k: None)}))
    with pytest.raises(ValueError):
        regions.lookup("Jilin", "China")
    monkeypatch.setattr(ee, "FeatureCollection", lambda *a, **k: _FC(1))
    assert regions.lookup("Jilin", "China") == "feature"


def test_regions_list_has_six():
    assert len(regions.REGIONS) == 6
    assert [r[0] for r in regions.REGIONS] == ["Minnesota", "Nebraska", "Wisconsin", "Iowa", "Heilongjiang", "Jilin"]
    assert regions.REGIONS[4][1] == "China"


def test_frame_label_by_country():
    assert regions.frame_label("United States of America") == "cdl_conf90"
    assert regions.frame_label("China") == "worldcover40"


def test_transfer_pairs():
    assert ("Minnesota", "Wisconsin") in regions.SPATIAL_PAIRS
    assert ("Nebraska", "Iowa") in regions.SPATIAL_PAIRS
    assert ("Minnesota", "Heilongjiang") in regions.SPATIAL_PAIRS
    assert ("Nebraska", "Jilin") in regions.SPATIAL_PAIRS
