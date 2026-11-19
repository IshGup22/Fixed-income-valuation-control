from src.model import bond_price, yield_from_price, generate_inputs
import numpy as np

def test_par_bond():
    assert abs(bond_price(100,.05,.05,5)-100)<1e-9

def test_price_yield_inverse():
    p=bond_price(100,.045,.065,7)
    assert abs(yield_from_price(p,.045,7)-.065)<1e-9

def test_price_decreases_when_yields_rise():
    assert bond_price(100,.04,.06,10)<bond_price(100,.04,.04,10)

def test_attribution_reconciliation():
    df=generate_inputs()
    assert np.allclose(df.market_pl_usd, df.rate_pl_usd+df.spread_pl_usd,atol=.02)
    assert len(df)==160
    assert df.instrument.nunique()==160

def test_deterministic():
    assert generate_inputs().equals(generate_inputs())
