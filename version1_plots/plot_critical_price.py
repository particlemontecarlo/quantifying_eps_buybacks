import numpy as np
import matplotlib.pyplot as plt

# plotting options
plt.rc('text', usetex=True)
font = { 'family' : 'serif',
        'size'   : 20}
plt.rc('font', **font)



# nonasymptotic earnings enhancement


tr_arr = [0.1,0.2,0.3,0.4,0.5]

i_arr = np.linspace(0.01,0.1)
y_arrs = [1/((1-i)*i_arr) for i in tr_arr]


# plot critical prices
f,ax = plt.subplots()


for i in range(len(y_arrs)):
    x = i_arr*100
    y = y_arrs[i]
    ax.plot(x,y,label='$t=' + str(tr_arr[i]*100) +  '\%$ exact')
    ax.set_ylabel('Critical PE ratio')
    ax.set_xlabel('Interest rate (\%)')
    ax.grid()

plt.legend()
plt.tight_layout()
plt.savefig('figs/eps_critical_price_ir.pdf',format='pdf')







