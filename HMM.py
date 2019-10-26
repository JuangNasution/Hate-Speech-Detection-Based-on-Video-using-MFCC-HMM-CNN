# -*- coding: utf-8 -*-
"""
Created on Tue May 21 09:15:33 2019

@author: XYZ
"""
import numpy as np
import math
np.seterr(divide='ignore',invalid='ignore')

def forward(y,A,B,pi):
  loglike=0
  T=len(y)
  N=len(A)
  alpha=np.zeros((T,N))
  scale=np.zeros((T))
  scale[0]=0
  for i in range(N):
    alpha[0,i]=pi[i]*B[y[0],i]
    scale[0]=scale[0]+alpha[0,i]
  #scale alpha[0,i]
  scale[0]=1/scale[0]
  for i in range(N):
    alpha[0,i]=scale[0]*alpha[0,i]
  #compute aplha[t,i]
  for t in range(1,T):
    scale[t]=0
    for i in range(N):
      alpha[t,i]=0
      for j in range(N):
        alpha[t,i]+=alpha[t-1,j]*A[j, i]
      alpha[t,i]=alpha[t,i]*B[y[t],i]
      scale[t]=scale[t]+alpha[t,i]
    #scale alpha[t,i]
    scale[t]=1/scale[t]
    for i in range(N):
      alpha[t,i]*=scale[t]
  loglike=sum(np.log10(scale))
  return -loglike
def forwback (ytrain, N):
    np.seterr(all='ignore')
    L=len(ytrain)
    M=max(ytrain[0])
    M+=1
    A=np.random.rand(N,N)
    B=np.random.rand(M,N)
    pi=np.random.rand(N)
    A=A/np.tile(sum(A),(N,1))
    B=B/np.tile(sum(B),(M,1))
    pi=pi/sum(pi)
    maxiters=1000
    k=0
    loglike=np.zeros((maxiters))
    loglikedif=math.inf
    loglikeold=-math.inf
    tol=0.001
    while(k<maxiters and loglikedif>=tol):
      k+=1
      Ac=np.zeros((N,N))
      Bc=np.zeros((M,N))
      pic=np.zeros((N))
      loglikec=np.zeros((L))
      for l in range(L):
        y=ytrain[l]
        T=len(y)
        alpha=np.zeros((T,N))
        beta=np.zeros((T,N))
        scale=np.zeros((T))
        scale[0]=0
        for i in range(N):
          alpha[0,i]=pi[i]*B[y[0],i]
          scale[0]=scale[0]+alpha[0,i]
        #scale alpha[0,i]
        scale[0]=1/scale[0]
        for i in range(N):
          alpha[0,i]=scale[0]*alpha[0,i]
        #compute aplha[t,i]
        for t in range(1,T):
          scale[t]=0
          for i in range(N):
            alpha[t,i]=0
            for j in range(N):
              alpha[t,i]+=alpha[t-1,j]*A[j,i]
            alpha[t,i]=alpha[t,i]*B[y[t],i]
            scale[t]=scale[t]+alpha[t,i]
          #scale alpha[t,i]
          scale[t]=1/scale[t]
          for i in range(N):
            alpha[t,i]=scale[t]*alpha[t,i]
        ####backward
        for i in range(N):
          beta[T-1,i]=scale[T-1]
        for t in range(T-2,-1,-1):
          for i in range(N):
            beta[t,i]=0
            for j in range(N):
              beta[t,i]+=A[i,j]*B[y[t+1],j]*beta[t+1,j]
            beta[t,i]*=scale[t]
        loglikec[l]=sum(np.log10(scale))
        gamma=np.zeros((T,N))
        gamma1=np.zeros((T,N,N))
        for t in range(T-1):
          for i in range(N):
            gamma[t,i]=0
            for j in range(N):
              gamma1[t,i,j]=(alpha[t,i]*A[i,j]*B[y[t+1],j]*beta[t+1,j])
              gamma[t,i]=gamma[t,i]+gamma1[t,i,j]
        for i in range(N):
          gamma[T-1,i]=alpha[T-1,i]
        #re-estimate phi
        for i in range(N):
          pic[i]+=gamma[0,i]
        #re-estimate A
        for i in range(N):
          denom=0
          for t in range(T-1):
            denom=denom+gamma[t,i]
          for j in range(N):
            numer=0
            for t in range(T-1):
              numer=numer+gamma1[t,i,j]
            Ac[i,j]+=numer/denom
        #re-estimate B
        for i in range(N):
          denom=0
          for t in range (T):
            denom=denom+gamma[t,i]
          for j in range(M):
            numer=0
            for t in range(T):
              if(y[t]==j):
                numer=numer+gamma[t,i]
            Bc[j,i]+=numer/denom
      loglike[k]=sum(loglikec)
      loglikedif=abs(loglike[k]-loglikeold)
      loglikeold=loglike[k]
      A=Ac/np.tile(sum(Ac),(N,1))
      B=Bc/np.tile(sum(Bc),(M,1))
      pi=pic/sum(pic)
    return A,B,pi,loglike,scale