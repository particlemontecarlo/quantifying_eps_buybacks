import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# plotting options
plt.rc('text', usetex=True)
font = { 'family' : 'serif',
                'size'   : 20}
plt.rc('font', **font)


# sort out buyback data
bb_fname = 'aapl_buybacks.csv'
raw_buyback = pd.read_csv(bb_fname)

# process data (convert dates to datetime and ensure units aren't in e.g. '20.0B' format
buyback_date = pd.to_datetime(raw_buyback['date'])
d = dict(M='E6', B='E9', T='E12')
buyback_ts = raw_buyback['buybacks'].replace(d, regex=True).astype(float)
buyback_ts.index = buyback_date
buyback_ts.sort_index(inplace=True)

# sort out apple data
wrds_fname = 'aapl_wrds.csv'
raw_wrds = pd.read_csv(wrds_fname)

wrds_date = pd.to_datetime(raw_wrds['datadate'])
raw_wrds = raw_wrds.set_index(wrds_date)
raw_wrds.index.name = 'date'

# merge data together
buyback_df = pd.concat([raw_wrds,buyback_ts],axis=1)

# plot some things
df_ts = buyback_df['buybacks']
ax = df_ts.plot(kind='bar', x=df_ts.index, stacked=True,label='Buybacks $\$$',color='C0')

# Make most of the ticklabels empty so the labels don't get too crowded
ticklabels = ['']*len(df_ts.index)
ticklabels[::4] = [item.strftime('%Y') for item in df_ts.index[::4]]
ax.xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))
plt.gcf().autofmt_xdate()
ax.set_xlabel('Date')
ax.set_ylabel('Quaterly buyback ($\$$)')
ax.set_title('Apple')
plt.tight_layout()
plt.savefig('apple_buybacks.pdf',format='pdf')


# get the other important data
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



# get the pe for effective and actual tax rate 
pe_crit_df_eff = 1./((1.-data_df['efftaxrate'])*data_df['10ybond']/100)
pe_crit_df_actual = 1./((1.-data_df['corporatetax'])*data_df['10ybond']/100)
pe_ts = pe_crit_df_eff.resample('D')
pe_ts = pe_ts.fillna(method='ffill')
pe_ts = pe_ts.loc[buyback_df.index].copy()
pe_ts = pe_ts.rename('critpe')
data_df = pd.concat([buyback_df,pe_ts],axis=1)

gamma_col = (data_df['buybacks']/(data_df['mkvaltq']*1e6)).copy()
gamma_col = gamma_col.rename('gamma')
data_df = pd.concat([data_df,gamma_col],axis=1)








# plot the pe ratio for S&P 500
fig2,ax2 = plt.subplots()
m_eff = data_df['dilutedpe']/data_df['critpe']
#m_act = data_df['pe']/pe_crit_df_actual
m_eff.plot(ax=ax2,label='m')
#m_act.plot(ax=ax2)
ax2.set_xlabel('Date')
ax2.set_ylabel('$m=\\frac{P}{P^*}$')
ax2.set_title('Apple')

m_lq = m_eff.quantile(0.2)
m_median = m_eff.quantile(0.5)
m_uq = m_eff.quantile(0.8)
ax2.axhline(m_median,label='m (50%-ile)',linestyle='--',color='darkgrey')
ax2.axhline(m_lq,label='m (20%-ile)',linestyle=':',color='darkgrey')
ax2.axhline(m_uq,label='m (80%-ile)',linestyle=':',color='darkgrey')
#med_txt = 'median = %.2f' % m_median
#mean_txt = 'mean = %.2f' % m_mean
#plt.text(data_df.index[data_df.shape[0]-1800],m_mean+0.1,mean_txt)
#plt.text(data_df.index[100],m_median-0.2,med_txt)
fig2.tight_layout()
plt.savefig('m_apple.pdf',format='pdf')





# plot the pe ratio for AAPL
fig3,ax3 = plt.subplots()
gamma_ts = data_df['gamma']
gamma_ts.plot(ax=ax3,label='m')
ax3.set_xlabel('Date')
ax3.set_ylabel('$\\gamma=\\frac{S}{PN}$')
ax3.set_title('Apple')

gamma_median = gamma_ts.quantile(0.5)
gamma_lq = gamma_ts.quantile(0.2)
gamma_uq = gamma_ts.quantile(0.8)
ax3.axhline(gamma_median,label='m (median)',linestyle='--',color='darkgrey')
ax3.axhline(gamma_lq,label='m (20%-ile)',linestyle=':',color='darkgrey')
ax3.axhline(gamma_uq,label='m (80%-ile)',linestyle=':',color='darkgrey')
#med_txt = 'median = %.2f' % gamma_median
#mean_txt = 'mean = %.2f' % gamma_mean
#plt.text(data_df.index[data_df.shape[0]-1800],gamma_mean+0.1,mean_txt)
#plt.text(data_df.index[100],gamma_median-0.2,med_txt)
fig3.tight_layout()
plt.savefig('gamma_apple.pdf',format='pdf')










f4,ax4 = plt.subplots()
df_ts = data_df['eps']
ax4 = df_ts.plot(kind='bar', x=df_ts.index, stacked=True,label='EPS',ax=ax4,color='C0')#,marker='x')
#ax4 = df_ts.plot(kind='line',  label='EPS',ax=ax4,color='C0')#,marker='x')

# Make most of the ticklabels empty so the labels don't get too crowded
ticklabels = ['']*len(df_ts.index)
ticklabels[::4] = [item.strftime('%Y') for item in df_ts.index[::4]]
ax4.xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))
plt.gcf().autofmt_xdate()


# sequence of buybacks
n = 5.25
buyback_times = np.arange(0,n+ 0.25,0.25)


P = buyback_times.shape[0]
t_arr = np.linspace(0,n+0.5,1000)
gamma = 0.0093
i = 0.03
x = np.array([0.08])
x_start_vals = np.array([df_ts[0]])
x_m_vals = np.array([0.32])


def eps_enh_with_times(gamma,i,m,x,buyback_times,t_arr):
    eps_through_time = np.zeros(len(t_arr))
    for j in range(len(t_arr)):
        t = t_arr[j]
        log_natural_growth = t*np.log(1+x)
        curr_buyback_times = buyback_times[buyback_times<t]
        delta_buyback_times = t-curr_buyback_times
        n_buybacks = np.sum(buyback_times<t)
        P = curr_buyback_times.shape[0]


        logp1_i = np.log(1-m*gamma* (((1+i)/(1+x))**delta_buyback_times))
        logp1 = sum(logp1_i)
        logp2 = n_buybacks*np.log(1./(1.-gamma)) 
        log_prod = logp1 + logp2
        eps_through_time[j] = np.exp(log_natural_growth + log_prod)
        #import ipdb;ipdb.set_trace()

    return eps_through_time


def eps_compound(buyback_times,x,m):
    eps_through_time = eps_enh_with_times(gamma,i,m,x,buyback_times,t_arr)
    return eps_through_time

def plot_eps_compound(m,ax):
    lwidth=3
    growth_ratios = []
    for idx  in range(len(x)):
        x_i = x[idx]
        start_val = x_start_vals[idx]
        m = x_m_vals[idx]


        EPS_n0 = eps_compound(buyback_times,x_i,m)

        xplt = 4*t_arr
        #xplt = t_arr
        ax.plot(xplt,start_val*(EPS_n0),label='Buyback EPS',linewidth=lwidth,color='C1')

        naive_growth = (1+x_i)**t_arr
        ax.plot(xplt,start_val*(naive_growth),label='$x=%.3f$'%x_i,linewidth=lwidth,color='C3')

        growth_ratios.append((EPS_n0[-1]/EPS_n0[0],naive_growth[-1]/naive_growth[0]))

    ax.set_title('Apple\n$\\gamma=%.3f$, $i=%.3f$, $m=%.3f$' % (gamma,i,m))
    ax.set_ylabel('Quarterly EPS')
    ax.set_xlabel('Date')
    ax.legend(loc='upper left',fontsize=15)
    ax.set_ylim([4,12])

    return growth_ratios

# should we assemble a big dataframe????
# seems like if we want to use a dataframe we'll just need to
# be a little careful with diffing - on different dates






m = 0.25
growth_ratios= plot_eps_compound(m,ax4)
plt.tight_layout()
plt.savefig('apple_natural_eps.pdf',type='pdf')


# plot the earnings using nonlinear least squares







#plt.show()






