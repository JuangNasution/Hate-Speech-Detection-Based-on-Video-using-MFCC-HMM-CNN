from scipy.io import wavfile
import numpy as np
import os
import pickle
import csv
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", DeprecationWarning)
from gensim.models import Word2Vec
def pelatihan_CNN():
    data=[]
    model=Word2Vec.load('model.bin')
    with open('ujaranbaru.txt', 'r') as readFile:
      reader = csv.reader(readFile)
      lines = list(reader)
      for i in range(len(lines)):
        str1 = ''.join(lines[i])
        kata=str1.split()
        kalimat=[]
        for i in range(len(kata)):
          kalimat.append(model[kata[i]][5:])
        data.append(kalimat)
    for f_size in range(2,7):
        reg_dua=np.round(np.random.rand(f_size,2,5),1)
        reg_tiga=np.round(np.random.rand(f_size,3,5),1)
        reg_empat=np.round(np.random.rand(f_size,4,5),1)
        W=np.round(np.random.rand(12),1)
        ETotalNew=np.inf
        ETotalOld=-np.inf
#        for z in range(len(data)):
        while(ETotalNew>ETotalOld):
            z=0
            activation_maps=[]
            fully=[]
            panjang=len(data[z])
            data[z]=np.array(data[z])
            for i in range(f_size):
                activation_maps=[]
                for j in range(panjang-4+1):
                    temp=0
                    for k in range(4):
                        for l in range(5):
                            temp+=np.dot(reg_empat[i,k,l],data[z][k+j][l])
                    temp=temp/20
                    activation_maps.append(temp)
                fully.append(np.max(activation_maps))
            for i in range(f_size):
                activation_maps=[]
                for j in range(panjang-3+1):
                    temp=0
                    for k in range(3):
                        for l in range(5):
                            temp+=np.dot(reg_tiga[i,k,l],data[z][k+j][l])
                    temp=temp/15
                    activation_maps.append(temp)
                fully.append(np.max(activation_maps))
            for i in range(f_size):
                activation_maps=[]
                for j in range(panjang-2+1):
                    temp=0
                    for k in range(2):
                        for l in range(5):
                            temp+=np.dot(reg_dua[i,k,l],data[z][k+j][l])
                    temp=temp/10
                    activation_maps.append(temp)
                fully.append(np.max(activation_maps))
            BFC=0.35
            M11=0
            M21=0
            for i in range(6):
                M11+=fully[i]*W[i*2]
                M21+=fully[i]*W[i*2+1]
            M11+=BFC*1
            M21+=BFC*1
            SM11=np.exp(M11)/(np.exp(M11)+np.exp(M21))
            SM21=np.exp(M21)/(np.exp(M11)+np.exp(M21))
        #    print(SM11,SM21)
            GT11=0
            GT21=1
            ##HITUNG ERROR
            EM11=0.5*((GT11-SM11)**2)
            EM21=0.5*((GT21-SM21)**2) 
            ##ERROR GRADIENT
            GM11=SM11*(1-M11)*EM11
            GM21=SM21*(1-M21)*EM21
        #    print(GM11,GM21)
            ##UBAH DELTA WEIGHT
            LR=0.2
            DELTA_W=np.zeros((12))
            for i in range(6):
              DELTA_W[i*2]=LR*fully[i]*GM11
              DELTA_W[i*2+1]=LR*fully[i]*GM21
            ##UBAH DELTA FC
            DELTA_FC=np.zeros((6))
            for i in range(6):
              DELTA_FC[i]=fully[i]*(1-fully[i])*GM11*W[i*2]
            ##UBAH DELTA BFC
            DELTA_BFC=LR*BFC*DELTA_FC[0]
            ##UBAH DELTA WEIGHT MATRIKS FILTER
            DELTA_reg_empat=np.zeros((f_size,4,5))
            DELTA_reg_tiga=np.zeros((f_size,3,5))
            DELTA_reg_dua=np.zeros((f_size,2,5))
            DELTA_reg_empat=LR*reg_empat*DELTA_FC[0]
            for i in range(f_size):
              for j in range(4):
                for k in range(5):
                  DELTA_reg_empat[i,j,k]=LR*reg_empat[i,j,k]*DELTA_FC[0]
            for i in range(f_size):
              for j in range(3):
                for k in range(5):
                  DELTA_reg_tiga[i,j,k]=LR*reg_tiga[i,j,k]*DELTA_FC[0]
            for i in range(f_size):
              for j in range(2):
                for k in range(5):
                  DELTA_reg_dua[i,j,k]=LR*reg_dua[i,j,k]*DELTA_FC[0]
            ##UBAH DELTA BMF
            BM=0
            DELTA_BMF=LR*BM*fully[0]
            ##UPDATE WEIGHT TERBARU DARI MATRIKS FILTER
            reg_empat+=DELTA_reg_empat
            reg_tiga+=DELTA_reg_tiga
            reg_dua+=DELTA_reg_dua
            ##UPDATE BIAS TERBARU DARI BMF
            BMF=0
            BMF+=DELTA_BMF
            ##UPDATE BIAS TERBARU DARI BFC
            BFC+=DELTA_BFC
            ##UPDATE WEIGHT TERBARU
            for i in range(len(W)):
              W[i]+=DELTA_W[i]
            z+=1
            matrix=[]
            matrix.append(reg_dua)
            matrix.append(reg_tiga)
            matrix.append(reg_empat)
            matrix.append(W)
            ETotalOld=ETotalNew
            ETotalNew=EM11+EM21
            print(ETotalNew)
#        directory='matrix cnn1'
#        folder=str(f_size)
#        with open(directory + '/'+ folder + '/'+'param matrix.pkl',"wb") as f:
#            pickle.dump(matrix,f)
#        f.close()
def klasifikasi_CNN(kata,f_size):
    directory='matrix cnn'
    model=Word2Vec.load('model.bin')
    kalimat=[]
    data=[]
    for i in range(len(kata)):
          chunk=model[kata[i]][5:]
          kalimat.append(chunk)
    data.append(kalimat)
    folder=str(f_size)
    with open(directory + '/'+ folder + '/'+'param matrix.pkl',"rb") as f:
        matrix= pickle.load(f)
    f.close()
    reg_dua=matrix[0]
    reg_tiga=matrix[1]
    reg_empat=matrix[2]
    W=matrix[3]
    for z in range(len(data)):
        activation_maps=[]
        fully=[]
        panjang=len(data[z])
        data[z]=np.array(data[z])
        for i in range(f_size):
            activation_maps=[]
            for j in range(panjang-4+1):
                temp=0
                for k in range(4):
                    for l in range(5):
                        temp+=np.dot(reg_empat[i,k,l],data[z][k+j][l])
                temp=temp/20
                activation_maps.append(temp)
            fully.append(np.max(activation_maps))
        for i in range(f_size):
            activation_maps=[]
            for j in range(panjang-3+1):
                temp=0
                for k in range(3):
                    for l in range(5):
                        temp+=np.dot(reg_tiga[i,k,l],data[z][k+j][l])
                temp=temp/15
                activation_maps.append(temp)
            fully.append(np.max(activation_maps))
        for i in range(f_size):
            activation_maps=[]
            for j in range(panjang-2+1):
                temp=0
                for k in range(2):
                    for l in range(5):
                        temp+=np.dot(reg_dua[i,k,l],data[z][k+j][l])
                temp=temp/10
                activation_maps.append(temp)
            fully.append(np.max(activation_maps))
        BFC=0.35
        M11=0
        M21=0
        for i in range(6):
            M11+=fully[i]*W[i*2]
            M21+=fully[i]*W[i*2+1]
        M11+=BFC*1
        M21+=BFC*1
        satu=np.exp(M11)/(np.exp(M11)+np.exp(M21))
#        dua=np.exp(M21)/(np.exp(M21)+np.exp(M11))
        if(satu<=0.5):
          hasil="ujaran kebencian"
          label=0
        else:
          hasil="bukan ujaran kebencian"
          label=1
    return hasil,label
kata=['kau','cepat','sembuh','biar','baik']
hasil,label=klasifikasi_CNN(kata,5)