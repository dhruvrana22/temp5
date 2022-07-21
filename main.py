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

N=50000
input_filename = 'input.wav'
samrate,data = wavfile.read(str('./'+input_filename))
df=pd.DataFrame(data)
sig1=df.iloc[:,0]
sig2=df.iloc[:,1]
alpha=2000       # moderate bandwidth constraint
tau=0           # noise-tolerance (no strict fidelity enforcement)
K=5              # 5 modes
DC=0             # no DC part imposed
init=1           # initialize omegas uniformly
tol=1e-7
l=len(f)
k=l//N
vmd1=pd.DataFrame()
vmd2=pd.DataFrame()
for x in range(k+1):
	if x==k:
		temp1=sig1[k*N:l,]
		temp2=sig2[k*N:l,]
		u,u_hat,omega=VMD(temp1,alpha,tau,K,DC,init,tol)
		u1,u_hat1,omega1=VMD(temp2,alpha,tau,K,DC,init,tol)
		u=u.transpose()
		u1=u1.transpose()
		vmd1=pd.concat([vmd1,pd.DataFrame(u)],axis=0)
		vmd2=pd.concat([vmd2,pd.DataFrame(u1)],axis=0)
		print(vmd1.shape)
	else:
		temp1=sig1[(x*N):(N*(x+1)),]
		temp2=sig2[(x*N):(N*(x+1)),]
		u,u_hat,omega=VMD(temp1,alpha,tau,K,DC,init,tol)
		u1,u_hat1,omega1=VMD(temp2,alpha,tau,K,DC,init,tol)
		u=u.transpose()
		u1=u1.transpose()
		vmd1=pd.concat([vmd1,pd.DataFrame(u)],axis=0)
		vmd2=pd.concat([vmd2,pd.DataFrame(u1)],axis=0)
		print(vmd1.shape)
pca = PCA(n_components=5)
Xt = pca.fit_transform(vmd1)
Xt2 = pca.fit_transform(vmd2)
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
