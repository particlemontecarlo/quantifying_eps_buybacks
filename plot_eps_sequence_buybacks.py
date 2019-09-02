import numpy as np
import matplotlib.pyplot as plt

#plt.rc('text', usetex=True)


# plotting options
plt.rc('text', usetex=True)
font = { 'family' : 'serif',
        'size'   : 20}
plt.rc('font', **font)






n = np.linspace(0,5)


# sequence of buybacks
n = 10.
buyback_times = np.arange(0,n+ 0.25,0.25)


P = buyback_times.shape[0]
t_arr = np.linspace(0,n,1000)
gamma = 0.01
i = 0.03
x = np.array([0.05,0.1,0.15])


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

    return eps_through_time


def eps_compound(buyback_times,x):
    eps_through_time = eps_enh_with_times(gamma,i,m,x,buyback_times,t_arr)
    return eps_through_time

def eps_enh_approx(gamma,i,m,x,buyback_times,t_arr):
    eps_through_time = np.zeros(len(t_arr))
    for j in range(len(t_arr)):
        t = t_arr[j]
        n_buybacks = np.sum(buyback_times<t)
        log_natural_growth = t*np.log(1+x)
        log_prod = n_buybacks*np.log((1./(1.-gamma)))
        eps_through_time[j] = np.exp(log_natural_growth + log_prod)

    return eps_through_time


def eps_compound_approx(buyback_times,x):
    eps_through_time = eps_enh_approx(gamma,i,m,x,buyback_times,t_arr)
    return eps_through_time

def plot_eps_compound(m):
    f,ax=plt.subplots(figsize=(5.5,3.5))
    f,ax=plt.subplots()
    lwidth=1
    for x_i in x:
        EPS_n0 = eps_compound(buyback_times,x_i)
        ax.plot(t_arr,100*(EPS_n0),label='$\\xi=%.3f$'%x_i,linewidth=lwidth)

        naive_growth = (1+x_i)**t_arr
        ax.plot(t_arr,100*(naive_growth),linestyle=':',linewidth=lwidth)

    ax.set_title('EPS after buybacks over %i years \n$\\gamma=%.3f$, $i=%.3f$, $m=%.3f$' % (n,gamma,i,m))
    ax.set_ylabel('$\\frac{E_n\'}{E_0}$ and $\\frac{E_n}{E_0}$ (\%)')
    ax.legend()
    plt.tight_layout()



def plot_relative_increase(m):
    f,ax=plt.subplots()
    lwidth=1
    for x_i in x:
        EPS_n0 = eps_compound(buyback_times,x_i)
        p1 = (EPS_n0)*100
        p2 = 100*(1+x_i)**t_arr

        ax.plot(t_arr,p1-p2,label='$\\xi=%.3f$'%x_i,linewidth=lwidth)

    ax.set_title('Difference in pc growth (%i years) \n$\\gamma=%.3f$, $i=%.3f$, $m=%.3f$' % (n,gamma,i,m))
    ax.set_ylabel('$\\frac{E_n\'}{E_0}-\\frac{E_n}{E_0}$ (\%)')
    ax.legend()

    plt.tight_layout()




m = 0.5
plot_eps_compound(m)
plt.savefig('figs/eps_compound_m%.3f.pdf' % m,format='pdf')


m = 0.5
plot_relative_increase(m)
plt.savefig('figs/eps_relative_compound_m%.3f.pdf' % m,format='pdf')


plt.show(block=False)




















