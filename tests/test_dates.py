from aa.util.dates import yyyymm_to_period


def test_period():
    assert str(yyyymm_to_period(196201)) == "1962-01"
