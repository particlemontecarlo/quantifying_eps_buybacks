import numpy as np
import matplotlib.pyplot as plt



# plotting options
plt.rc('text', usetex=True)
font = { 'family' : 'serif',
        'size'   : 20}
plt.rc('font', **font)

# nonasymptotic earnings enhancement
SNP0_1 = 0.01
SNP0_2 = 0.05

x = np.linspace(0.1,1.25)
ratio1 = SNP0_1
ratio2 = SNP0_2

y1 = (1/x - 1 )/(1-ratio1/x)*ratio1
y2 = (1/x - 1 )/(1-ratio2/x)*ratio2


# asymptotic earnings enhancement

y1_asymptotic = (1/x - 1 )/(1)*ratio1
y2_asymptotic = (1/x - 1 )/(1)*ratio2

f,ax = plt.subplots()

ax.scatter(x,100*y1_asymptotic,label='$\\frac{S}{P N}=' + str(SNP0_1*100) +  '\%$ approximation',
        color='lightblue')
ax.scatter(x,100*y2_asymptotic,label='$\\frac{S}{P N}=' + str(SNP0_2*100) +  '\%$ approximation',
        color='grey')
ax.plot(x,100*y1,label='$\\frac{S}{P N}=' + str(SNP0_1*100) +  '\%$ exact')
ax.plot(x,100*y2,label='$\\frac{S}{P N}=' + str(SNP0_2*100) +  '\%$ exact')
ax.set_title('Earnings enhancement, $\Delta E$')
ax.set_ylabel('$\Delta E$ (\%)')
ax.set_xlabel('Price paid $m$ as a fraction of $\\frac{P}{P_0}$')
ax.grid()

plt.legend()

plt.tight_layout()
plt.savefig('figs/eps_enhancement_asymptotic.pdf',format='pdf')







