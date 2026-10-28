from experiments.external_audits.denethor import catalogue


def test_s2_keys_paths():
    k = catalogue.s2_keys("s2-utm-33N-18E-242N-2018")
    assert k["CLP"] == "fusion-competition/sentinel-2/s2-utm-33N-18E-242N-2018/data/CLP.npy"
    assert k["SCL"] == "fusion-competition/sentinel-2/s2-utm-33N-18E-242N-2018/mask/SCL.npy"
    assert k["IS_DATA"] == "fusion-competition/sentinel-2/s2-utm-33N-18E-242N-2018/mask/IS_DATA.npy"
    assert k["timestamp"] == "fusion-competition/sentinel-2/s2-utm-33N-18E-242N-2018/timestamp.pkl"
    assert k["bbox"] == "fusion-competition/sentinel-2/s2-utm-33N-18E-242N-2018/bbox.pkl"
    assert k["meta_info"] == "fusion-competition/sentinel-2/s2-utm-33N-18E-242N-2018/meta_info.pkl"


def test_listing_size_known():
    assert catalogue.listing_size("fusion-competition/sentinel-2/s2-utm-33N-18E-242N-2018/data/CLP.npy") == 829440128


def test_pfqa_key():
    assert catalogue.pfqa_key("17E-243N", "2019-01-01") == "fusion-competition/planet/UTM-24000/33N/17E-243N/PF-QA/2019-01-01.tif"


def test_url():
    assert catalogue.url("fusion-competition/x") == "https://data.source.coop/esa/fusion-competition/x"


def test_tile_of_and_year_of():
    assert catalogue.tile_of("s2-utm-33N-18E-242N-2018") == "18E-242N"
    assert catalogue.year_of("s2-utm-33N-18E-242N-2018") == 2018
    assert catalogue.tile_of("s2-utm-34S-20E-259N-2017") == "20E-259N"


def test_constants():
    assert len(catalogue.S2_TILE_YEARS) == 4 and len(catalogue.SA_TILE_YEARS) == 3
    assert catalogue.LABELS["17E-243N"].endswith("br-17E-243N-crop-labels-test-2019.geojson")
