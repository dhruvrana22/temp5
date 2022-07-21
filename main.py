import wave
import struct
import sys
import csv
import numpy 
from scipy.io import wavfile
from scipy.signal import resample
import numpy as np
import pandas as pd
from vmdpy import VMD
import sys, os, os.path
from sklearn.decomposition import PCA

input_filename = 'input.wav'
if input_filename[-3:] != 'wav':
    print('WARNING!! Input File format should be *.wav')
    sys.exit()
samrate,data = wavfile.read(str('./' + input_filename))
np.savetxt('out.csv',data,delimiter=",")
df=pd.DataFrame(data)
f2=df.iloc[:,0]
f=df.iloc[:,1]
alpha = 2000       # moderate bandwidth constraint
tau = 0           # noise-tolerance (no strict fidelity enforcement)
K = 5              # 3 modes
DC = 0             # no DC part imposed
init = 1           # initialize omegas uniformly
tol = 1e-7
l=len(f)
k=l//40000
final=pd.DataFrame()
final1=pd.DataFrame()
for x in range(k+1):
	if x==k:
		temp=f2[k*40000:l,]
		temp2=f[k*40000:l,]
		u, u_hat, omega = VMD(temp, alpha, tau, K, DC, init, tol)
		u1, u_hat1, omega1 = VMD(temp2, alpha, tau, K, DC, init, tol)
		u=u.transpose()
		u1=u1.transpose()
		final=pd.concat([final,pd.DataFrame(u)],axis=0)
		final1=pd.concat([final1,pd.DataFrame(u1)],axis=0)
		print(final.shape)
	else:
		temp=f2.iloc[(x*40000):(40000*(x+1)),]
		temp2=f.iloc[(x*40000):(40000*(x+1)),]
		u, u_hat, omega = VMD(temp, alpha, tau, K, DC, init, tol)
		u1, u_hat1, omega1 = VMD(temp2, alpha, tau, K, DC, init, tol)
		u=u.transpose()
		u1=u1.transpose()
		final=pd.concat([final,pd.DataFrame(u)],axis=0)
		final1=pd.concat([final1,pd.DataFrame(u1)],axis=0)
		print(final.shape)
df=final
df2=final1
pca = PCA(n_components=5)
Xt = pca.fit_transform(df)
Xt2 = pca.fit_transform(df2)
fcol=Xt[:,0]
scol=Xt[:,1]
msum=fcol+scol
fcol=Xt2[:,0]
scol=Xt2[:,1]
nsum=fcol+scol
df3=pd.DataFrame()
df3=pd.concat([pd.DataFrame(msum),pd.DataFrame(nsum)],axis=1)
df3.to_csv('final.csv', header=False, index=False)
print("done")
