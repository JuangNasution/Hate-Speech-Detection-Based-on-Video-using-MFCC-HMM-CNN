# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import math
import numpy as np
from sklearn.cluster import KMeans
from scipy.fftpack import fft
######################Normalisasi##################
def normalisasi(sinyal):
    dataNorm=[]
    for i in range(len(sinyal)):
        dataNorm.append(sinyal[i]/abs(max(sinyal)))
#    mencari deviasi
    hasil=0
    for i in range(len(dataNorm)):
        hasil+=dataNorm[i]
    rata=hasil/len(dataNorm)
    variance=0
    for i in range(len(dataNorm)):
        variance+=(abs(dataNorm[i]-rata)**2)
    deviasi=math.sqrt((variance/(len(dataNorm)-1))) 
    ########################Remove Silence##################
    while dataNorm[0]<deviasi:
        del dataNorm[0]
    while dataNorm[len(dataNorm)-1]<deviasi:
        del dataNorm[len(dataNorm)-1]
    return dataNorm
##plt.plot(dataNorm)
def framing(samplerate,sinyal,wframe,woverlap):
    pframe=int(samplerate*wframe/1000)
    nilaioverlap=int(samplerate*woverlap/1000)
    jarakframe=pframe-nilaioverlap
    nframe=math.floor((len(sinyal)-pframe)/jarakframe)
    dataFrame=np.zeros((nframe,pframe))
    for i in range(1,nframe):
        start=(i-1)*jarakframe+1
        j=0
        for start in range(start,start+pframe-1):
            dataFrame[i][j]=sinyal[start]
            j+=1
    return dataFrame,nframe,pframe
def MFCC(coefisien,sinyal,samplerate):
    ########################Pre-emphasis##################
    dataPreEmp=[]
    preemph=0.97
    dataPreEmp.append(sinyal[0])
    for i in range(1,len(sinyal)):
        dataPreEmp.append(sinyal[i]-(sinyal[i-1]*preemph))
    ########################Frame Blocking##################
    dataFrame,nframe,pframe=framing(samplerate,dataPreEmp,15,5)
    ########################Windowing##################
    dataWindow=np.zeros((nframe,pframe))
    for i in range(nframe):
        for j in range(pframe):
            hamwindow=0.54-0.46*math.cos(2*math.pi*j/(pframe-1))
            dataWindow[i][j]=dataFrame[i][j]*hamwindow
    ########################FFT##################
    NFFT=512
    y1=0
    dataFFT=np.zeros((nframe,NFFT))
    for i in range(nframe):
        for j in range(NFFT):
            for k in range(pframe):
                y1+=dataWindow[i][k]*np.exp(-2j*math.pi*k*j/NFFT)
            dataFFT[i][j]=(1/pframe)*abs(y1**2)
            y1=0
    ########################Filter Bank##################
    nfilt=20
    datafbank=np.zeros((nframe,int(NFFT/2+1)))
    for i in range(nframe):
        for j in range(1,int(NFFT/2)+1):
                datafbank[i][j]=dataFFT[i][j]
    high_freq_mel =(2595 * np.log10(1 + (samplerate / 2) / 700))
    low_freq_mel=(2595 * np.log10(1 + (300) / 700)) 
    mel_points =np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    hz_points = (700 * (10**(mel_points / 2595) - 1))
    bin = np.floor((NFFT + 1) * hz_points / samplerate)
    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks =np.dot(datafbank, fbank.T)
    ########################DCT##################
    temp=0
    dataMFCC=np.zeros((nframe,coefisien))
    for i in range(nframe):
        for j in range(coefisien):
            for k in range(len(filter_banks[1,:])):
                temp+=filter_banks[i,k]*np.cos((np.pi*j*(2*k+1))/(2*len(filter_banks[0,:])))
            dataMFCC[i,j]=2*temp
            temp=0
    ########################Cepstral##################
    L=22
    cepstral=1 + (L/2)*np.sin(np.pi*coefisien/L)
    dataMFCC*=cepstral
    return dataMFCC
#######################k-means##################
def kmeans(dataMFCC,cluster):
    kmeans=KMeans(n_clusters=cluster, init='k-means++', random_state=0).fit(dataMFCC)
    if(kmeans.labels_[0]!=0):
      temp=kmeans.labels_[0]
      for i in range(len(kmeans.labels_)):
        if(kmeans.labels_[i]==temp):
          kmeans.labels_[i]=0
        elif (kmeans.labels_[i]==0):
          kmeans.labels_[i]=temp
    return kmeans.labels_