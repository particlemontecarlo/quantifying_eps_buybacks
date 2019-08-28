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




#######################################################################
# perform NLLS
def E_pred(c,n,E0,xi,gamma,m,iint):
    """
    function that predicts earnings given initial conditions and growth
    requires knowledge of time and number of buybacks
    """
    natural_growth  = (1+xi)**(c*n)
    prod_arr = (1/(1-gamma))*(1-m*gamma*(((1+iint)/(1+xi))**(c*np.arange(0,n))))
    factor2 = np.prod(prod_arr)
    factor3 = E0
    return natural_growth*factor2*factor3
    
# quick test, plotting E_pred for trial values
n_vals = np.arange(0,10)
E_vals1 = [E_pred(0.25,i,1,0.08,0.0093,0.25,0.03) for i in n_vals]
f,ax = plt.subplots()
plt.plot(n_vals,E_vals1,label='bb')
plt.legend(loc='best')

def cost_function(E0,xi):
    """
    for each time we have earnings reported then we return the value
    of the predicted earnings given initial conditions and growth
    """

    gamma = 0.01
    m = 0.25
    iint = 0.03
    E_vals_data = data_df['eps']
    n_buybacks_total = E_vals_data.size
    n_vals = np.arange(0,n_buybacks_total)
    E_vals = np.array([E_pred(0.25,i,E0,xi,gamma,m,iint) for i in n_vals])

    return sum((E_vals-E_vals_data)**2)
    
nx = 100
ny = 100

E0_vals = np.linspace(1,10,nx)
xi_vals = np.linspace(0.05,0.55,ny)
cost_res = np.empty([nx,ny])
for i in range(len(E0_vals)):
    for j in range(len(xi_vals)):
        E0 = E0_vals[i]
        xi = xi_vals[j]
        cost_res[i,j] = cost_function(E0,xi)

E0_indx = np.where(np.min(cost_res)==cost_res)[0][0]
E0_opt = E0_vals[E0_indx]
xi_indx = np.where(np.min(cost_res)==cost_res)[1][0]
xi_opt = xi_vals[xi_indx]


#######################################################################
# plot results
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
buyback_times = np.arange(0,n+0.25,0.25)


P = buyback_times.shape[0]
t_arr = np.linspace(0,n+0.5,1000)
gamma = 0.01
i = 0.03
x = xi_opt#np.array([0.08])
x_start_val = E0_opt#np.array([df_ts[0]])


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
    start_val = x_start_val

    EPS_n0 = eps_compound(buyback_times,x,m)

    xplt = 4*t_arr
    #xplt = t_arr
    ax.plot(xplt,start_val*(EPS_n0),label='Buyback EPS',linewidth=lwidth,color='C1')

    naive_growth = (1+x)**t_arr
    ax.plot(xplt,start_val*(naive_growth),label='$x=%.3f$'%x,linewidth=lwidth,color='C3')

    growth_ratios.append((EPS_n0[-1]/EPS_n0[0],naive_growth[-1]/naive_growth[0]))

    ax.set_title('Applen$\\gamma=%.2f$, $i=%.2f$, $m=%.2f$' % (gamma,i,m))
    ax.set_ylabel('Quarterly EPS')
    ax.set_xlabel('Date')
    ax.legend(loc='upper left',fontsize=15)
    ax.set_ylim([4,12])

    return growth_ratios

m = 0.25
growth_ratios= plot_eps_compound(m,ax4)
plt.tight_layout()
plt.savefig('apple_natural_eps.pdf',type='pdf')


# plot the earnings using nonlinear least squares







#plt.show()






