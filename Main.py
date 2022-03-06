# -*- coding: utf-8 -*-
"""
Elías D. Nino-Ruiz
Code for paper submitted
This is a temporary script file.
"""

#%% LIBS
import numpy as np
import matplotlib.pyplot as plt
import scipy as sci
import abc
from scipy.integrate import odeint
from sklearn.linear_model import Ridge
import seaborn as sns

#%% Analysis Equations

class Analysis(metaclass=abc.ABCMeta):
  def __init__(self):
    pass;
  def performassimilation():
    pass;
  def getanalysisstate():
    pass;
  def performcovarianceinflation():
    pass;
    
    
#%%% Analysis EnKF B-Loc
    
class AnalysisEnKFBLoc(Analysis):

  def __init__(self,model,r=1):
    self.model = model;
    self.model.createdecorrelatonmatrix(r);

  def performassimilation(self, background, observation):
    Xb = background.getensemble();
    Pb = background.getcovariancematrix();
    y = observation.getobservation();
    H = observation.getobservationoperator();
    R = observation.getdataerrorcovariance();
    n, ensemble_size = Xb.shape; 
    Ys = np.random.multivariate_normal(y, R, size=ensemble_size).T;
    L = self.model.getdecorrelationmatrix();
    Pb = L*np.cov(Xb);
    D = Ys-H@Xb;
    IN = R + H@(Pb@H.T);
    Z = np.linalg.solve(IN,D);
    self.Xa = Xb + Pb@(H.T@Z);
    return self.Xa;

  def getanalysisstate(self):
    return np.mean(self.Xa,axis=1);

  def getensemble(self):
    return self.Xa;
  
  def geterrorcovariance(self):
    return np.cov(self.Xa);
  
  def getanalysisstate(self):
    return np.mean(self.Xa,axis=1);

  def inflateensemble(self,inflation_factor):
    n,ensemble_size = self.Xa.shape;
    xa = self.getanalysisstate();
    DXa = self.Xa-np.outer(xa,np.ones(ensemble_size));
    self.Xa = np.outer(xa,np.ones(ensemble_size))+inflation_factor*DXa;

#%%% Analysis EnKF B-Loc Analysis EnKF Modified Cholesky decomposition
    
class AnalysisEnKFModifiedCholesky(Analysis):

  def __init__(self, model, r=1, regularization_factor=0.01):
    self.model = model;
    self.r = r;
    self.regularization_factor = regularization_factor;

  def getprecisionmatrix(self,DX,r,regularization_factor=0.01):
    n,ensemble_size = DX.shape;
    lr = Ridge(fit_intercept=False,alpha=regularization_factor);
    L = np.eye(n);
    D = np.zeros((n,n));
    D[0,0] = 1/np.var(DX[0,:]); #We are estimating D^{-1}
    for i in range(1,n):
      ind_prede = self.model.getpre(i,r);
      y = DX[i,:];
      X = DX[ind_prede,:].T;
      lr_fit = lr.fit(X,y);
      err_i = y-lr_fit.predict(X);
      D[i,i] = 1/np.var(err_i);
      L[i,ind_prede] = -lr_fit.coef_;
    
    return L.T@(D@L);

  def performassimilation(self, background, observation):
    Xb = background.getensemble();
    y = observation.getobservation();
    H = observation.getobservationoperator();
    R = observation.getdataerrorcovariance();
    n, ensemble_size = Xb.shape; 
    Ys = np.random.multivariate_normal(y, R, size=ensemble_size).T;
    xb = np.mean(Xb,axis=1);
    DX = Xb-np.outer(xb,np.ones(ensemble_size))
    Binv = self.getprecisionmatrix(DX,self.r,self.regularization_factor);
    D = Ys-H@Xb;
    Rinv = np.diag(np.reciprocal(np.diag(R)));
    IN = Binv + H.T@(Rinv@H);
    Z = np.linalg.solve(IN,H.T@(Rinv@D));
    self.Xa = Xb + Z;
    return self.Xa;

  def getanalysisstate(self):
    return np.mean(self.Xa,axis=1);
  
  def get_obs_errors(self, observation):
    y = observation.getobservation();
    H = observation.getobservationoperator();
    R = observation.getdataerrorcovariance();
    xa = self.getanalysisstate();
    return  np.linalg.norm(y-H@xa); 

  def getensemble(self):
    return self.Xa;
  
  def geterrorcovariance(self):
    return np.cov(self.Xa);


#%% Observation Class
class Observation:

  def __init__(self,m,n,std_obs=0.01,obs_operator_fixed=False,H=None):
    self.m = m;
    self.H = H;
    self.n = n;
    self.R = (std_obs**2)*np.eye(self.m,self.m);
    self.obs_operator_fixed = obs_operator_fixed;
    if self.obs_operator_fixed: self.setobservationoperator_fixed(self.H)

  def setobservationoperator(self,n):
    I = np.eye(n,n);
    H = np.random.choice(np.arange(0,n),self.m,replace=False);
    H = I[H,:];
    self.H = H;
    
  def setobservationoperator_fixed(self,H):
    n = self.n;
    I = np.eye(n,n);
    H = self.H;
    H = I[H,:];
    self.H = H;

  def generateobservation(self,x):
    if not self.obs_operator_fixed:
      self.setobservationoperator(x.size);
    self.y = self.H@x + np.random.multivariate_normal(np.zeros(self.m),self.R);
  
  def getobservation(self):
    return self.y;

  def getobservationoperator(self):
    return self.H;

  def getdataerrorcovariance(self):
    return self.R;
  
  def getprecisionerrorcovariance(self):
    return np.diag(np.reciprocal(np.diag(R)));
  
  def getanalysisstate(self):
    return np.mean(self.Xa,axis=1);

  def get_error(self, x):
      return np.linalg.norm(self.y-self.H @ x)

#%% Meta Class - Model
class Model(metaclass=abc.ABCMeta):
  def __init__(self):
    pass;
  def getinitialcondition():
    pass;
  def propagate():
    pass;
  def getnumberofvariables():
    pass;
  def createlocalizationmatrix():
    pass;
  def getlocalizationmatrix():
    pass;
  def getprecisionmatrix():
    pass;

#%%% Lorenz 96 Model
class Lorenz96(Model):

  n = 40;
  F = 8;

  def __init__(self,n = 40,F = 8):
    self.n = n;
    self.F = F;

  def lorenz96(self, x, t):
    n = self.n;
    F = self.F;
    return [(x[np.mod(i+1,n)]-x[i-2])*x[i-1]-x[i]+F for i in range(0,n)];

  def getnumberofvariables(self):
    return self.n;
  
  def getinitialcondition(self, seed = 10, T = np.arange(0,10,0.1)):
    n = self.n;
    np.random.seed(seed=10);
    x0 = np.random.randn(n);
    return self.propagate(x0,T);

  def propagate(self, x0, T, just_final_state=True):
    x1 = odeint(self.lorenz96,x0,T);
    if just_final_state:
      return x1[-1,:];
    else:
      return x1;
    
  def createdecorrelatonmatrix(self,r):
    n = self.n;
    L = np.zeros((n,n));
    for i in range(0,n):
      for j in range(i,n):
        dij = np.min([np.abs(i-j),np.abs((n-1)-j+i)]);
        L[i,j] = (dij**2)/(2*r**2);
        L[j,i] = L[i,j];
    self.L = np.exp(-L);
    
  def getdecorrelationmatrix(self):
    return self.L;
  
  def getngb(self,i,r):
    return np.arange(i-r,i+r+1)%(self.n);
  
  def getpre(self,i,r):
    ngb = self.getngb(i,r);
    return ngb[ngb<i];

#%% Forecast Class
class Background:
  def __init__(self, model, ensemble_size=200):
    self.model = model;
    self.ensemble_size = ensemble_size;

  def getinitialensemble(self, initial_perturbation = 0.05, time = np.arange(0,10,0.01)):
    n = self.model.getnumberofvariables();
    Xb = initial_perturbation*np.random.randn(n, self.ensemble_size);
    M = len(time);
    for e in range(0,self.ensemble_size):
      Xb[:,e] = self.model.propagate(Xb[:,e], time);
    self.Xb = Xb;
    return Xb;
  
  def forecaststep(self, Xb, time = np.arange(0,1,0.01)):
    ensemble_size = self.ensemble_size;
    for e in range(0,ensemble_size):
      Xb[:,e] = self.model.propagate(Xb[:,e], time);
    self.Xb = Xb;
    return Xb;

  def getensemblesize(self):
    return self.ensemble_size;
  
  def getensemble(self):
    return self.Xb;
  
  def getcovariancematrix(self):
    return np.cov(self.Xb);
  
  def getbackgroundstate(self):
    return np.mean(self.Xb,axis=1);

  def getmemberdeviations(self,scale=1):
      return scale*(self.Xb-np.outer(self.getbackgroundstate(),np.ones(self.getensemblesize())));

#%% Main program

N = 20;
M = 50;
n = 40;
r = 10;
sample_size = 1000;
Htra = np.array([2*i for i in range(0,int(n/2))], dtype=np.int32);
Hval = np.array([2*i+1 for i in range(0,int(n/2))], dtype=np.int32);
m_tra = Htra.size;
m_val = Hval.size;

print(Htra)
print(Hval)

model = Lorenz96();
background = Background(model,ensemble_size=N);
analysis = AnalysisEnKFModifiedCholesky(model, r=r, regularization_factor=500);
observation_tra = Observation(m_tra, n, std_obs=0.01, obs_operator_fixed=True, H=Htra);
observation_val = Observation(m_val, n, std_obs=0.01, obs_operator_fixed=True, H=Hval);

T = np.linspace(0,1,num=2);

Xb0 = background.getinitialensemble();
xt0 = model.getinitialcondition(); 

background_0 = background;
model_0 = model;


Xbk = Xb0;
xtk = xt0;

Xbk = background.forecaststep(Xbk, time = T);
xtk = model.propagate(xtk,T)

errora_k = [];
errorb_k = [];
for k in range(0, sample_size):
    observation_tra.generateobservation(xtk);
    observation_val.generateobservation(xtk);
    Xak_tra = analysis.performassimilation(background,observation_tra);
    
    xmb_k = background.getbackgroundstate();
    trab_error = observation_tra.get_error(xmb_k);
    valb_error = observation_val.get_error(xmb_k);
    errorb_k.append([trab_error, valb_error]);
    
    xma_k = analysis.getanalysisstate();
    traa_error = observation_tra.get_error(xma_k);
    vala_error = observation_val.get_error(xma_k);
    errora_k.append([traa_error, vala_error]);
    
errora_k = np.log10(np.array(errora_k, dtype=np.float32));
errorb_k = np.log10(np.array(errorb_k, dtype=np.float32));


sns.regplot(x=errorb_k[:,0], y=errorb_k[:,1], color='blue')
sns.regplot(x=errora_k[:,0], y=errora_k[:,1], color='red')






