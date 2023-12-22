#%%
from matrix_fact import *
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

from scipy.io import loadmat

# fetch dataset
ionosphere = fetch_ucirepo(id=52)

# data (as pandas dataframes)
X = ionosphere.data.features
y = ionosphere.data.targets

X_array = X.to_numpy()
X_array.shape
#%%

n = 1
basis = 15

media = 0
for i in range (n):
  mdl = SNMF(X_array, num_bases=basis)
  mdl.factorize(niter=10)
  media_linha = np.mean(mdl.H, axis=1)
  for j in range (basis):
   mdl.H = np.where(mdl.H[j] < media_linha[j]*0.001, 0, mdl.H)
  num_zeros = np.count_nonzero(~mdl.H.astype(bool))
  media = media + num_zeros/(basis*(len(mdl.H[0])))
  
media = media/n

sparsity = 1 - media

qual_fact = 100 - 100*mdl.frobenius_norm()/(np.linalg.norm(X_array))

print("sparsity: %s" % sparsity)
print("norma de frobenius: %s" % mdl.ferr)
print("percentagem dos dados que é explicada pela fatoração: %s\n" % qual_fact)

#%%

media = 0
n = 100
basis = 3
for i in range (n):
  mdl = SNMF(X_array, num_bases=basis)
  mdl.factorize(niter=100)
  media_linha = np.mean(mdl.H, axis=1)
  for j in range (basis):
   mdl.H = np.where(mdl.H[j] < media_linha[j]*0.001, 0, mdl.H)
  num_zeros = np.count_nonzero(~mdl.H.astype(bool))
  media = media + num_zeros/(basis*(len(mdl.H[0])))
  
media = media/n

sparsity = 1 - media

qual_fact = 100 - 100*mdl.frobenius_norm()/(np.linalg.norm(X_array))

print("Ionosphere:")
print("sparsity: %s" % sparsity)
print("norma de frobenius: %s" % mdl.ferr[99])
print("percentagem dos dados que é explicada pela fatoração: %s%%\n" % qual_fact)


#%%

data = loadmat(r'C:\Users\terti\jupyter_lab\projetos\matrix_fact-main\matrix_fact\Urban.mat')
dados_urbanos =  data['X']/1000

media = 0
n = 10
basis = 15
for i in range (n):
  dados_mdl = SNMF(dados_urbanos, num_bases=basis)
  dados_mdl.factorize(niter=10)
  media_linha = np.mean(dados_mdl.H, axis=1)
  for j in range (basis):
   dados_mdl.H = np.where(dados_mdl.H[j] < media_linha[j]*0.001, 0, dados_mdl.H)
  num_zeros = np.count_nonzero(~dados_mdl.H.astype(bool))
  media = media + num_zeros/(basis*(len(dados_mdl.H[0])))
  
media = media/n

sparsity = 1 - media

qual_fact = 100 - 100*dados_mdl.frobenius_norm()/(np.linalg.norm(dados_urbanos))

print("Dados Urbanos")
print("sparsity: %s" % sparsity)
print("norma de frobenius: %s" % dados_mdl.ferr)
print("percentagem dos dados que é explicada pela fatoração: %s%% \n" % qual_fact)

#%%
print(dados_mdl.W.shape)
"""import matrix_fact
import numpy as np
data = np.array([
    [1.0, 0.0, 2.0],
    [0.0, 1.0, 1.0]
])
nmf = matrix_fact.CHNMF(data, num_bases=2)
nmf.factorize(niter=100)
print(nmf.W, '\n')
print(nmf.H, '\n')
print(nmf.W @ nmf.H, '\n')
print(nmf.ferr)

"""
