from numpy import mean
from numpy import std
from numpy import absolute
from numpy.linalg import svd
import numpy as np
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import ElasticNet
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Zero-mean, unit-variance,
# for each column of X
def standardize(X):
  n=X.shape[0]
  p=X.shape[1]
  #pint(np.sum(X,0))
  means=np.sum(X,0)/float(n)
    
  for pdx in range(p):
    A=X[:,pdx]-means[pdx]
    A=A/np.sqrt(np.sum(np.multiply(A,A)))
    X[:,pdx]=A    
    
  return X

# data: n x p
# k: number of PCs
# max_iter=100: max SPCA iterations
# alpha: sum of lambda (L2-norm) and lambda_1 (L1-norm)
# l1_ratio: lambda_1/(lambda_1 + lambda)
# verbose: print helpful debug info
def SPCA(data,k,max_iter=100,alpha=0.5,l1_ratio=1.0,verbose=False):
  def logprint(*args):
    if verbose:
      print(args)
    else:
      pass

  #
  # 1. Standardize (center and unit-column-variance) the data
  # 

  X=standardize(data)

  # for tracking convergence
  dV_list=[]

  # initialization
  logprint("SPCA")
  logprint("- Initialization")
  n=X.shape[0] # observations
  p=X.shape[1] # variables
  logprint("-- n:",n,"p:",p,"k:",k)

  #
  # 2. Initialize V with p PCA loadings, A with k PCA loadings
  #
  #V=np.random.rand(p,k) # TODO: should come from PCA
  #V=V @ np.diag(1.0/np.sqrt(np.sum(np.multiply(V,V),0)))
  model0=PCA(n_components=p, svd_solver='full')
  model0=model0.fit(X)
  V=model0.components_
  A=V[:,0:k]
  V=A #?
  logprint("-- V0:",V.shape[0],"x",V.shape[1])
  logprint("-- A0:",A.shape[0],"x",A.shape[1])
  logprint(A)

  #
  # 3. Iterative solver - until convergence is reached (TODO: metric?),
  #
  Vprev=V # for tracking convergence
  B=np.random.rand(p,k) # elastic net intermediate results
  logprint("-- B0:",B.shape[0],"x",B.shape[1])
  logprint("- Solve")
  for idx in range(max_iter):
    logprint("-- idx=",idx)
    #
    # 4. Compute elasticnet regression for each of k PCs:
    #    B[:,j] = argmin_b (A[:,j] - b)^T X^T X (A[:,j] - b) + \lambda \Vert b \Vert^2 + \lambda_{1,j} \Vert b \Vert_1
    #           = argmin_b (XA[:,j] - Xb)^T (XA[:,j] - Xb) + lambda \Vert b \Vert&2 + \lambda_{1,j} \Vert b \Vert_1
    #
    #    where $j \in [1,k]$, 
    #          $\lambda_{1,j} = \alpha l1\_ratio$, 
    #          $\lambda = \alpha (1 - l1\_ratio)$
    #
    #    and $A$ is the current best-guess at the loadings
    #
    #    Note that the expressions for B[:,j] above require (XA[:,j] - Xb) or equivalently X(A[:,j] - b),
    #    however Python's ElasticNet model.fit(X,y) routine wants to minimize (y - X), therefore for
    #    compatibility with Python's conventions we compute
    #
    #    y = XA[:,j]
    #
    #    and then later call
    #
    #    model.fit(X,y)
    #
    #    and extract the fitted coefficients to get B[:,j}
    #    
    #
    Y=X@A # Y = X * [a_1, a_2, ..., a_k]
    logprint("-- Y:",Y.shape[0],"x",Y.shape[1])
    logprint("-- Loop over principal components")
    for jdx in range(k):
      # elasticnet with desired L2, L1 lambdas
      model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)     
      logprint("--- jdx=",jdx)
      logprint("--- ElasticNet model:",model)
      # for compatibility with Python elasticnet library
      y=Y[:,jdx]
      B[:,jdx]=model.fit(X,y).coef_ # Get fitted coefficients
    logprint("-- B:",B.shape[0],"x",B.shape[1])
    #
    # 5. Update: A = u[:,0:k] v_h^T, where u, s, v_h = SVD(X^T XB)
    #
    XTXB=np.transpose(X)@X@B
    logprint("-- XTXB:",XTXB.shape[0],XTXB.shape[1])
    u,s,vh=svd(XTXB, full_matrices=True)
    A=u[:,0:k]@np.transpose(vh)
    logprint("-- SVD(XTXB):")
    logprint("--- u:",u.shape[0],u.shape[1])
    logprint("--- s:",s.shape[0])
    logprint(s)
    logprint("--- vh:",vh.shape[0],vh.shape[1])
    #logprint("-- A:",A.shape[0],"x",A.shape[1])    
    #logprint(B)
    #
    # Update: V[:,j] = B[:,j]/\Vert B[:,j] \Vert
    #         computed cleverly in one line using diagonal matrix
    #
    Vprev=V # for tracking convergence
    V=B @ np.diag(1.0/np.sqrt(np.sum(np.multiply(B,B),0)))
    dV_list.append(np.sum((V-Vprev)*(V-Vprev)))
    logprint("-- V:",V.shape[0],"x",V.shape[1])     
    logprint(V)

  plt.plot(dV_list)
  return X, V, X@V
