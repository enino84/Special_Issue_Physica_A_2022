# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sci
import abc
from sklearn.linear_model import Ridge


class Analysis(metaclass=abc.ABCMeta):
  def __init__(self):
    pass;
  def performassimilation():
    pass;
  def getanalysisstate():
    pass;
  def performcovarianceinflation():
    pass;
    
    
#Analysis EnKF B-Loc
    
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

#Analysis EnKF Modified Cholesky decomposition
    
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



class Observation:

  def __init__(self,m,std_obs=0.01,obs_operator_fixed=False,H=None):
    self.m = m;
    self.H = H;
    self.R = (std_obs**2)*np.eye(self.m,self.m);
    self.obs_operator_fixed = obs_operator_fixed;

  def setobservationoperator(self,n):
    I = np.eye(n,n);
    H = np.random.choice(np.arange(0,n),self.m,replace=False);
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

  def inflateensemble(self,inflation_factor):
    n,ensemble_size = self.Xa.shape;
    xa = self.getanalysisstate();
    DXa = self.Xa-np.outer(xa,np.ones(ensemble_size));
    self.Xa = np.outer(xa,np.ones(ensemble_size))+inflation_factor*DXa;
    
#EnKF implementation Cholesky (ensemble space)