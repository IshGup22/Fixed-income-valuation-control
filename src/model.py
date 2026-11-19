"""Transparent fixed-coupon bond valuation and independent-price-control simulation.

All generated market inputs are synthetic; NOT bank-approved IPV or VA methodology.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

SEED=20261119
ASOF=pd.Timestamp('2026-11-19')
TENORS=np.array([1,2,3,5,7,10],dtype=float)
BASE_GOV=np.array([0.027,0.029,0.030,0.032,0.034,0.036])

def bond_price(face:float, coupon:float, ytm:float, years:float, frequency:int=2, elapsed_fraction:float=0.0)->float:
    """Clean price per 100 par, with fractional coupon-period discounting.

    elapsed_fraction is the proportion of the current coupon period elapsed;
    full-period cash-flow count is deliberately synthetic, not a date schedule.
    """
    periods=int(round(years*frequency))
    if periods<=0: raise ValueError('remaining life must be positive')
    cf=np.full(periods,100*coupon/frequency,dtype=float)
    cf[-1]+=100
    time=np.arange(1,periods+1)
    dirty=float(np.sum(cf/(1+ytm/frequency)**(time-elapsed_fraction)))
    accrued=100*coupon/frequency*elapsed_fraction
    return dirty-accrued

def yield_from_price(price:float,coupon:float,years:float)->float:
    lo,hi=-0.05,0.5
    for _ in range(80):
        mid=(lo+hi)/2
        if bond_price(100,coupon,mid,years)>price: lo=mid
        else: hi=mid
    return (lo+hi)/2

def generate_inputs(n=160, seed=SEED):
    rng=np.random.default_rng(seed)
    sector=rng.choice(['Government','Investment Grade','High Yield'],size=n,p=[0.34,0.49,0.17])
    tenor=rng.choice(TENORS,size=n,p=[.13,.19,.23,.21,.15,.09])
    coupon=rng.uniform(.022,.073,n)
    elapsed=rng.uniform(.05,.95,n)
    spread=np.where(sector=='Government',rng.uniform(.0002,.0018,n),np.where(sector=='Investment Grade',rng.uniform(.006,.020,n),rng.uniform(.025,.055,n)))
    gov=np.interp(tenor,TENORS,BASE_GOV)
    y0=gov+spread
    rate_move=rng.normal(.0006,.0003,n) # average +6bp benchmark shock
    spread_move=rng.normal(.0001,.00045,n)
    y1=y0+rate_move+spread_move
    face=rng.choice([100_000,250_000,500_000,1_000_000],n,p=[.20,.35,.30,.15])
    # quotes are independently generated around economically plausible true-model reference
    reference=np.array([bond_price(100,c,yy,t,elapsed_fraction=ee) for c,yy,t,ee in zip(coupon,y1,tenor,elapsed)])
    # simulated booking feeds contain occasional stale marks or mismarked yields
    booking_error=rng.normal(0,.075,n)
    anomaly_indices=rng.choice(n,size=16,replace=False)
    booking_error[anomaly_indices]+=rng.choice([-1,1],16)*rng.uniform(.65,1.7,16)
    desk_marks=reference+booking_error
    quotes=np.vstack([reference+rng.normal(0,.045,n) for _ in range(3)]).T
    # vendor failures: localized outliers, median resilient
    for idx in rng.choice(n,size=12,replace=False):
        quotes[idx,int(rng.integers(0,3))]+=rng.choice([-1,1])*rng.uniform(.8,1.9)
    rows=[]
    for i in range(n):
        b0=bond_price(100,coupon[i],y0[i],tenor[i],elapsed_fraction=elapsed[i]); b_rate=bond_price(100,coupon[i],y0[i]+rate_move[i],tenor[i],elapsed_fraction=elapsed[i]); b1=reference[i]
        base_value=b0*face[i]/100
        ending_value=b1*face[i]/100
        market_pl=(b1-b0)*face[i]/100
        rate_pl=(b_rate-b0)*face[i]/100
        spread_pl=(b1-b_rate)*face[i]/100
        dv01=(bond_price(100,coupon[i],y1[i]-.0001,tenor[i],elapsed_fraction=elapsed[i])-bond_price(100,coupon[i],y1[i]+.0001,tenor[i],elapsed_fraction=elapsed[i]))/2*face[i]/100
        independent=float(np.median(quotes[i])); difference=float(desk_marks[i]-independent)
        tolerance=.30 if sector[i]=='High Yield' else .20
        breach=abs(difference)>tolerance
        # conservative haircut *illustration* only; not an accounting VA or bank regulatory measure
        adjustment_proxy=max(0.,abs(difference)-tolerance)*face[i]/100
        rows.append(dict(instrument=f'BND-{i+1:04d}',sector=sector[i],tenor_years=int(tenor[i]),coupon_pct=round(100*coupon[i],3),accrual_fraction=round(elapsed[i],5),accrued_interest_per_100=round(100*coupon[i]/2*elapsed[i],6),face_usd=int(face[i]),start_ytm_pct=round(100*y0[i],4),end_ytm_pct=round(100*y1[i],4),rate_move_bp=round(rate_move[i]*10000,3),spread_move_bp=round(spread_move[i]*10000,3),start_price_per_100=round(b0,6),end_model_price_per_100=round(b1,6),desk_price_per_100=round(desk_marks[i],6),vendor_a=round(quotes[i,0],6),vendor_b=round(quotes[i,1],6),vendor_c=round(quotes[i,2],6),independent_price_per_100=round(independent,6),price_difference_per_100=round(difference,6),tolerance_per_100=tolerance,ipv_exception=bool(breach),valuation_difference_usd=round(difference*face[i]/100,2),illustrative_excess_diff_usd=round(adjustment_proxy,2),start_mv_usd=round(base_value,2),end_mv_usd=round(ending_value,2),market_pl_usd=round(market_pl,2),rate_pl_usd=round(rate_pl,2),spread_pl_usd=round(spread_pl,2),dv01_usd_per_bp=round(dv01,2)))
    return pd.DataFrame(rows)

def produce(outdir:Path, datadir:Path):
    outdir.mkdir(parents=True,exist_ok=True);datadir.mkdir(parents=True,exist_ok=True)
    df=generate_inputs()
    df.to_csv(datadir/'synthetic_bond_positions.csv',index=False)
    stats={'bonds':len(df),'portfolio_face_usd':int(df.face_usd.sum()),'portfolio_mv_usd':round(df.end_mv_usd.sum(),2),'ipv_exceptions':int(df.ipv_exception.sum()),'exception_rate_pct':round(100*df.ipv_exception.mean(),2),'total_market_pl_usd':round(df.market_pl_usd.sum(),2),'rates_pl_usd':round(df.rate_pl_usd.sum(),2),'spreads_pl_usd':round(df.spread_pl_usd.sum(),2),'dv01_usd_per_bp':round(df.dv01_usd_per_bp.sum(),2),'illustrative_excess_diff_usd':round(df.illustrative_excess_diff_usd.sum(),2),'max_price_difference_per_100':round(df.price_difference_per_100.abs().max(),3),'pricing_variance_reconciles':bool(np.allclose(df.market_pl_usd,df.rate_pl_usd+df.spread_pl_usd,atol=.02))}
    import json
    (outdir/'metrics.json').write_text(json.dumps(stats,indent=2))
    df.loc[df.ipv_exception].sort_values('illustrative_excess_diff_usd',ascending=False).to_csv(outdir/'ipv_exception_queue.csv',index=False)
    summary=df.groupby('sector',as_index=False).agg(positions=('instrument','count'),market_value_usd=('end_mv_usd','sum'),market_pl_usd=('market_pl_usd','sum'),rate_pl_usd=('rate_pl_usd','sum'),spread_pl_usd=('spread_pl_usd','sum'),exceptions=('ipv_exception','sum'),illustrative_excess_diff_usd=('illustrative_excess_diff_usd','sum'))
    summary.to_csv(outdir/'sector_summary.csv',index=False)
    return df,stats,summary

if __name__=='__main__':
    d=Path(__file__).resolve().parents[1]
    df,stats,summary=produce(d/'outputs',d/'data')
    print(stats)
