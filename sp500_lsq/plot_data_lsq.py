import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# plotting options
plt.rc('text', usetex=True)
font = { 'family' : 'serif',
                'size'   : 20}
plt.rc('font', **font)


sp_eps = 'sp_eps.csv'
raw_data = pd.read_csv(sp_eps)

date_col = raw_data['QUARTER END']
date_col = pd.to_datetime(date_col)
date_col.name = 'date'


oper_eps = raw_data['OPERATING EPS']
oper_eps = oper_eps.str.replace('US\$','')
oper_eps = oper_eps.astype(float)

rep_eps = raw_data['REPORTED EPS']
rep_eps = rep_eps.str.replace('US\$','')
rep_eps = rep_eps.astype(float)

data_df = pd.concat([rep_eps,oper_eps,date_col],axis=1)
data_df = data_df.set_index('date')
data_df = data_df.sort_index()


gamma = 0.0075
m = 0.7
iint = 0.03


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


    E_vals_data = data_df['OPERATING EPS']
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
df_ts = data_df['OPERATING EPS']
ax4 = df_ts.plot(kind='bar', x=df_ts.index, stacked=True,label='Realised EPS',ax=ax4,color='C0')#,marker='x')
#ax4 = df_ts.plot(kind='line',  label='EPS',ax=ax4,color='C0')#,marker='x')

# Make most of the ticklabels empty so the labels don't get too crowded
ticklabels = ['']*len(df_ts.index)
ticklabels[::4] = [item.strftime('%Y') for item in df_ts.index[::4]]
ax4.xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))
plt.gcf().autofmt_xdate()
plt.xticks(rotation='vertical')

# sequence of buybacks
n = 16.25
buyback_times = np.arange(0,n+0.25,0.25)


P = buyback_times.shape[0]
t_arr = np.linspace(0,n+0.5,1000)
i = iint
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
    ax.plot(xplt,start_val*(EPS_n0),label='$E\'_n$, buyback EPS',linewidth=lwidth,color='C1')

    naive_growth = (1+x)**t_arr
    ax.plot(xplt,start_val*(naive_growth),label='$E_n$ with $\\xi=%.3f$'%x,linewidth=lwidth,color='C3')

    growth_ratios.append((EPS_n0[-1]/EPS_n0[0],naive_growth[-1]/naive_growth[0]))

    ax.set_title('SP500 $\\gamma=%.3f$, $i=%.2f$, $m=%.2f$' % (gamma,i,m))
    ax.set_ylabel('Quarterly EPS')
    ax.set_xlabel('Date')
    ax.legend(loc='upper left',fontsize=15)
    ax.set_ylim([4,45])

    return growth_ratios

growth_ratios= plot_eps_compound(m,ax4)
plt.tight_layout()
plt.savefig('sp500_natural_eps.pdf',type='pdf')


# plot the earnings using nonlinear least squares







#plt.show()






