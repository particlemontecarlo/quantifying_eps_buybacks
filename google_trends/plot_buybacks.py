import pandas as pd
import matplotlib.pyplot as plt


# plotting options
plt.rc('text', usetex=True)
font = { 'family' : 'serif',
                'size'   : 20}
plt.rc('font', **font)


# we plot the value of buybacks for the sp500
df = pd.read_csv('sp500_simplified.csv')
date_col = pd.to_datetime(df['QUARTER END'])
df['DATE'] = date_col
df.set_index('DATE',inplace=True)


# we also plot the google trends analytics for the term 'share buyback' over a similar period
df_trends = pd.read_csv('sharebuyback_trends.csv')
date_col = pd.to_datetime(df_trends['Month'])
df_trends['DATE'] = date_col
df_trends.set_index('DATE',inplace=True)
df_trends = df_trends['sharebb_interest']



# plot the two series
f,ax=plt.subplots()
df['BUYBACKS'].plot()
plt.ylabel('USD (BN)')
plt.xlabel('')
plt.tight_layout()
plt.savefig('buybackssp500.pdf',format='pdf')

f,ax=plt.subplots()
df_trends.plot()
plt.ylabel('Google trend interest (\%)')
plt.xlabel('')
plt.tight_layout()
plt.savefig('googletrendsinterest.pdf',format='pdf')


#df_combined= pd.concat([df['BUYBACKS'],df_trends['share buyback: (']],axis=1)


