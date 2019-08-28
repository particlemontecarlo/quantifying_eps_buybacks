import numpy as np
import matplotlib.pyplot as plt


# plotting options
plt.rc('text', usetex=True)
font = { 'family' : 'serif',
        'size'   : 20}
plt.rc('font', **font)



# nonasymptotic earnings enhancement
SNP0_1 = 0.02

n = np.linspace(0,5,10)
m = 0.1
i = 0.03
x = 0.1

#eps_through_time = ( ((1/m-1)-((1+i)/(1+x))**n)/(1-(1/m)*SNP0_1) ) * SNP0_1
#eps_through_time = ( ((1/m-1)-((1+i)/(1+x))**n)/(1-(1/m)*SNP0_1) ) * SNP0_1
eps_through_time = (  (1/(1-SNP0_1))  *  ( (1+x)**n  - m*SNP0_1*(1+i)**n  )  ) -1
#eps_through_time_asymp = ( ((1/m-1)-((1+i)/(1+x))**n)/(1) ) * SNP0_1

eps_naive =  (n*np.log(1+x))
eps_with_enh = np.log(1+eps_through_time) + eps_naive
#eps_with_enh_asymp = np.log(1+eps_through_time_asymp) + eps_naive


main = '$\\frac{S}{P N}=' + str(SNP0_1) +  '$ $m=' + str(m) + '$ $\\imath=' + str(i) + '$ $\\xi=' + str(x) + '$'

label_str = 'EPS natural'
label_str_enh = 'EPS with buybacks'
label_str_enh_asymp = 'EPS growth with buyback (approx.)'

x_plt = n


f,ax = plt.subplots()
ax.plot(x_plt,100*(np.exp(eps_naive)-1),label=label_str)
ax.plot(x_plt,100*(np.exp(eps_with_enh)-1),label=label_str_enh)
ax.set_title(main)
ax.set_ylabel('$\\frac{E\'_T- E_0}{E_0}$ (\%)')

ax.legend()
ax.grid()

plt.tight_layout()
plt.savefig('figs/eps_enhancement_through_time.pdf',format='pdf')


plt.show()



