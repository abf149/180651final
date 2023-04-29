from numpy import mean
from numpy import std
from numpy import absolute
from numpy.linalg import svd
import numpy as np
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import ElasticNet
import matplotlib.pyplot as plt

# data: n x p
# k: number of PCs
# max_iter=100: max SPCA iterations
# alpha: sum of lambda (L2-norm) and lambda_1 (L1-norm)
# l1_ratio: lambda_1/(lambda_1 + lambda)
def SPCA(data,k,max_iter=100,alpha=0.5,l1_ratio=1.0,verbose=False):
  def logprint(*args):
    if verbose:
      print(args)
    else:
      pass

  dV_list=[]

  logprint("SPCA")
  logprint("- Initialization")
  n=data.shape[0]
  p=data.shape[1]
  logprint("-- n:",n,"p:",p,"k:",k)
  B=np.random.rand(p,k)
  V=np.random.rand(p,k)
  V=V @ np.diag(1.0/np.sqrt(np.sum(np.multiply(V,V),0)))
  Vprev=V
  A=V[:,0:k]
  logprint("-- V0:",V.shape[0],"x",V.shape[1])
  logprint("-- A0:",A.shape[0],"x",A.shape[1])
  logprint("-- B0:",B.shape[0],"x",B.shape[1])

  logprint("- Solve")
  for idx in range(max_iter):
    logprint("-- idx=",idx)
    Y=data@A # Y = X * [a_1, a_2, ..., a_k]
    logprint("-- Y:",Y.shape[0],"x",Y.shape[1])
    logprint("-- Loop over principal components")
    for jdx in range(k):
      model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)     
      logprint("--- jdx=",jdx)
      logprint("--- ElasticNet model:",model)
      y=Y[:,jdx]
      B[:,jdx]=model.fit(data,y).coef_
    logprint("-- B:",B.shape[0],"x",B.shape[1])
    XTXB=np.transpose(data)@data@B
    logprint("-- XTXB:",XTXB.shape[0],XTXB.shape[1])
    u,s,vh=svd(XTXB, full_matrices=True)
    logprint("-- SVD(XTXB):")
    logprint("--- u:",u.shape[0],u.shape[1])
    logprint("--- s:",s.shape[0])
    logprint(s)
    logprint("--- vh:",vh.shape[0],vh.shape[1])
    #V=vh
    #print("-- V:",V.shape[0],V.shape[1])
    A=u[:,0:k]@np.transpose(vh)
    logprint("-- A:",A.shape[0],"x",A.shape[1])    
    logprint(B)
    Vprev=V
    V=B @ np.diag(1.0/np.sqrt(np.sum(np.multiply(B,B),0)))
    dV_list.append(np.sum((V-Vprev)*(V-Vprev)))

  plt.plot(dV_list)
  return V, data@V
