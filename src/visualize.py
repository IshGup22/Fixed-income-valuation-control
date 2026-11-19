from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
df=pd.read_csv(ROOT/'data/synthetic_bond_positions.csv')
fig,axes=plt.subplots(1,2,figsize=(12,4.8))
pl=[df.rate_pl_usd.sum(),df.spread_pl_usd.sum(),df.market_pl_usd.sum()]
axes[0].bar(['Rates','Credit spreads','Total'],[x/1000 for x in pl],color=['#3C75A6','#98B8D4','#17395C'])
axes[0].axhline(0,color='#777',linewidth=.8)
axes[0].set_title('Illustrative portfolio mark-to-market P&L')
axes[0].set_ylabel('P&L (USD thousands)')
axes[1].scatter(df.tenor_years,df.price_difference_per_100.abs(),c=df.ipv_exception.map({True:'#C43E43',False:'#347CA5'}),alpha=.65,s=22)
axes[1].set_title('Independent-price difference by bond tenor')
axes[1].set_xlabel('Maturity (years)')
axes[1].set_ylabel('Absolute difference (per $100 par)')
fig.suptitle('Fixed income valuation controls | synthetic market data',fontweight='bold',fontsize=13)
fig.tight_layout()
fig.savefig(ROOT/'outputs/analysis_dashboard.png',dpi=170,bbox_inches='tight')
plt.close(fig)
