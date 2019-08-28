import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# plotting options
plt.rc('text', usetex=True)
font = { 'family' : 'serif',
        'size'   : 20}
plt.rc('font', **font)

tr_df = pd.read_csv("us_tax_csv.csv")
ir_df = pd.read_csv("us_ir_csv.csv")
sp_pe_df = pd.read_csv("sp_pe_csv.csv")


# process the data
tr_df["date"] = tr_df["date"].astype('datetime64[ns]')
tr_df = tr_df.set_index("date")

ir_df["date"] = ir_df["date"].astype('datetime64[ns]')
ir_df = ir_df.set_index("date")
ir_df = ir_df["10ybond"]
ir_df = ir_df[ir_df!=0]


sp_pe_df["date"] = sp_pe_df["date"].astype('datetime64[ns]')
sp_pe_df = sp_pe_df.dropna(axis=1)
sp_pe_df = sp_pe_df.set_index("date")

data_df = pd.concat([tr_df[['efftaxrate','corporatetax']],ir_df,sp_pe_df],axis=1)
data_df = data_df.fillna(method='ffill')


fig, ax1 = plt.subplots()

tick_color2 = 'C0'
data_df['10ybond'].plot(ax=ax1,color=tick_color2,label='10Y treasury rate (\%)')
ax1.tick_params('y', colors=tick_color2)
ax1.set_ylabel('US 10Y treasury rate (\%)')
ax1.set_xlabel('Date')
#plt.legend(loc='lower left')
#lgd = plt.legend(bbox_to_anchor=(0.2,-0.2), loc="lower left")


ax2 = ax1.twinx()
tick_color1 = 'C1'
(data_df['efftaxrate']*100).plot(ax=ax2,label='Effective tax rate (\%)',color=tick_color1)
(data_df['corporatetax']*100).plot(ax=ax2,label='Tax rate (\%)',color=tick_color1,linestyle='--')
ax2.set_ylabel('US corporate tax rate (\%)')
ax2.tick_params('y',colors=tick_color1)
ax2.set_xlabel('Date')
# Make the y-axis label, ticks and tick labels match the line color.

#plt.legend(loc='upper right')
#lgs = plt.legend(bbox_to_anchor=(0.8,-0.2), loc="lower right")
fig.tight_layout()
#plt.savefig('us_ir_tr.pdf',format='pdf',bbox_extra_artists=(lgd,),bbox_inches='tight')
plt.savefig('us_ir_tr.pdf',format='pdf',bbox_inches='tight')

# plot the critical pe ratios
fig1,ax1 = plt.subplots()

pe_crit_df_eff = 1./((1.-data_df['efftaxrate'])*data_df['10ybond']/100)
pe_crit_df_eff.plot(ax=ax1,label='$P_E$ (effective tax)')

pe_crit_df_actual = 1./((1.-data_df['corporatetax'])*data_df['10ybond']/100)
pe_crit_df_actual.plot(ax=ax1,label='$P_E$ (standard tax)')
ax1.set_xlabel('Date')
ax1.set_ylabel('P/E ratio')
plt.legend()

fig.tight_layout()
plt.savefig('us_crit_pe.pdf',format='pdf')


# plot the pe ratio for S&P 500
fig2,ax2 = plt.subplots()
m_eff = data_df['pe']/pe_crit_df_eff
#m_act = data_df['pe']/pe_crit_df_actual
m_eff.plot(ax=ax2,label='m')
#m_act.plot(ax=ax2)
ax2.set_xlabel('Date')
ax2.set_ylabel('$m=\\frac{P}{P_0}$')
ax2.set_title('S\&P 500')

m_mean = m_eff.mean()
m_median = m_eff.median()
ax2.axhline(m_mean,label='m (mean)',linestyle=':',color='darkgrey')
ax2.axhline(m_median,label='m (median)',linestyle='--',color='darkgrey')
med_txt = 'median = %.2f' % m_median
mean_txt = 'mean = %.2f' % m_mean
plt.text(data_df.index[data_df.shape[0]-1800],m_mean+0.1,mean_txt)
plt.text(data_df.index[100],m_median-0.2,med_txt)
fig.tight_layout()
plt.savefig('m_sp500.pdf',format='pdf')



plt.show()


