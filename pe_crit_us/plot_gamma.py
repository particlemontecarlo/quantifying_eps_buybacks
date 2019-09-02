import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# plotting options
plt.rc('text', usetex=True)
font = { 'family' : 'serif',
        'size'   : 20}
plt.rc('font', **font)


sp_data = 'sp_buybacks_unclean.csv'
raw_data = pd.read_csv(sp_data)



# ok for the date column want to remove 00:00 and convert all to date object
raw_data_date = raw_data['QUARTER END'].str.strip()
mask_00 = raw_data_date.str.endswith(' 00:00')
raw_data_date[mask_00] = raw_data_date[mask_00].str[:-6]
raw_data_date = pd.to_datetime(raw_data_date)
raw_data_date.name = 'Date'

raw_data.index = raw_data_date


# clean buyback column
buyback_column = raw_data['BUYBACKS\n$ BILLIONS']
dol_mask = buyback_column.str.startswith('$')
buyback_column[dol_mask] = buyback_column[dol_mask].str[1:]

us_mask = buyback_column.str.startswith('US$')
buyback_column[us_mask] = buyback_column[us_mask].str[3:]
buyback_column.name = 'buyback'
buyback_column = buyback_column.astype(float)


# clean market capitalisation column
market_cap_col = raw_data['MARKET VALUE $ BILLIONS']
dol_mask = market_cap_col.str.startswith('$')
market_cap_col[dol_mask] = market_cap_col[dol_mask].str[1:]
market_cap_col = market_cap_col.str.replace(',','')
market_cap_col = market_cap_col.astype(float)


# gamma col
gamma_col = buyback_column/market_cap_col




fig2,ax2 = plt.subplots()


gamma_col.plot(ax=ax2,label='$\\gamma$')
#m_act.plot(ax=ax2)
ax2.set_xlabel('Date')
ax2.set_ylabel('$\\gamma=\\frac{S}{PN}$')
ax2.set_title('S\&P 500')


gamma_mean = gamma_col.mean()
ax2.axhline(gamma_mean,label='Mean',linestyle=':',color='darkgrey')
fig2.tight_layout()
plt.legend(loc='lower right')
plt.savefig('gamma_sp500.pdf',format='pdf')



