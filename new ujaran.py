import sys
from PyQt5 import QtCore, QtGui, QtWidgets,uic
from PyQt5.QtWidgets import QFileDialog,QMessageBox, QWidget, QTableWidgetItem
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", DeprecationWarning)
import subprocess
import time
import csv
import os
import re
import numpy as np
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from gensim.models import Word2Vec
from pydub.silence import split_on_silence
from pydub import AudioSegment
from scipy.io import wavfile
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import pickle
from DeretanObservasi import normalisasi , MFCC , kmeans
from HMM import forwback, forward
from CNN import klasifikasi_CNN, pelatihan_CNN

class MyWindow(QtWidgets.QMainWindow):
    def __init__(self, parent = None):
        super(MyWindow, self).__init__(parent)
        uic.loadUi('new ujaran.ui',self)
        self.pushButton.clicked.connect(self.explore_video)
        self.pushButton_2.clicked.connect(self.deteksibtn)
        self.pushButton_3.clicked.connect(self.explore_pelatihan_speech)
        self.pushButton_5.clicked.connect(self.explore_pelatihan_text)
        self.pushButton_4.clicked.connect(self.pelatihan_speech)
        self.pushButton_6.clicked.connect(self.pelatihan_text)
        self.pushButton_7.clicked.connect(self.explore_pengujian_speech)
        self.pushButton_9.clicked.connect(self.explore_pengujian_text)
        self.pushButton_8.clicked.connect(self.pengujian_speech)
        self.pushButton_10.clicked.connect(self.pengujian_text)
    def errorbox(self):
        self.error = QtWidgets.QMessageBox(window)
        self.error.setIcon(QMessageBox.Critical)
        self.error.setWindowTitle("Error")
        self.error.setText("Salah Memasukkan Data")
        self.error.show()
    def info(self):
        self.error = QtWidgets.QMessageBox(window)
        self.error.setIcon(QMessageBox.Information)
        self.error.setWindowTitle("Information")
        self.error.setText("Pelatihan sukses")
        self.error.show()
    def explore_pengujian_speech(self):
        data_path=QFileDialog.getExistingDirectory()
        data_path=str(data_path)
        self.textBrowser_10.setText(data_path)
    def explore_pengujian_text(self):
        data_path=QFileDialog.getOpenFileName(None,"Select File","","text file (*.txt)")
        data_path=str(data_path)[2:-23]
        self.textBrowser_18.setText(data_path)
    def explore_video(self):
        data_path=QFileDialog.getOpenFileName(None,"Select File","","video file (*.mp4)")
        data_path=str(data_path)[2:-24]
        self.textBrowser.setText(str(data_path))
    def explore_pelatihan_speech(self):
        data_path=QFileDialog.getExistingDirectory()
        data_path=str(data_path)
        base=os.path.basename(data_path)
        if(base!='Dataset Sample'):
          self.errorbox()
          self.textBrowser_7.setText("")
        else:
          self.textBrowser_7.setText(data_path)
    def explore_pelatihan_text(self):
        data_path=QFileDialog.getOpenFileName(None,"Select File","","text file (*.txt)")
        data_path=str(data_path)[2:-23]
        self.textBrowser_17.setText(data_path)
    def clean_data(self,path):
        path=str(path)
        bye=open('ujaranbaru.txt','w+')
        with open(str(path), 'r') as readFile:
            reader = csv.reader(readFile)
            lines=list(reader)
            for i in range(len(lines)):
                str1 = ''.join(lines[i])
                caseF=str1.casefold()
                Runame=re.sub('@[^\s]+','',caseF)
                Rhashtag=re.sub('#[^\s]+','',Runame)
                CleanNumber= ''.join([i for i in Rhashtag if not i.isdigit()])
                line = re.sub('[(),\'.!$]', '', CleanNumber)
                link=re.sub('https[^\s]+','',line)
                garing=re.sub('\\\[^\s]+','',link)
                removeRT=garing.replace("rt", "")
                removespace=removeRT.lstrip()
                factory = StopWordRemoverFactory()
                stopword = factory.create_stop_word_remover()
                stopw=stopword.remove(removespace)
                factory = StemmerFactory()
                stemmer = factory.create_stemmer()
                steam=stemmer.stem(stopw)
                text=steam.split()
                if(len(text)>=5):
                    bye.write(steam+'\n')
                self.progressBar_6.setValue((i+1)/len(lines)*100)
        bye.close()
    def Word2Vec(self):
        data=[]
        with open('ujaranbaru.txt', 'r') as readFile:
            reader = csv.reader(readFile)
            lines = list(reader)
            for i in range(len(lines)):
                str1 = ''.join(lines[i])
                kata=str1.split()
                data.append(kata)
        model = Word2Vec(data, min_count=1)
        model.save('model.bin')
    def pelatihan_text(self):
        path=self.textBrowser_17.toPlainText()
        self.clean_data(path)
#        self.Word2Vec()
        pelatihan_CNN()
        self.info()
    def pelatihan_speech(self):
        path=self.textBrowser_7.toPlainText()
        source=str(path)
        coefisien=[13]
        state=[10]
        cluster=5
        start=time.time()
        for i in range(len(coefisien)):
            self.textBrowser_16.setText(str(coefisien[i]))
            cnt=0
            self.textBrowser_8.setText("")
            for folder in os.listdir(source):
                self.progressBar_2.setValue(cnt/3*100)
                cnt+=1
                suara=[]
                self.textBrowser_9.setText("")
                textfile=""
                allpelatihan=[]
                for file in os.listdir (source + '/'+ folder + '/'):
                    if(file[-3:]=='wav'):
                        textfile+=str(file)+'\n'
#                        time.sleep(1)
                        self.textBrowser_8.setText(str(folder))
                        self.textBrowser_9.setText(textfile)
                        samplerate,data=wavfile.read(source+ '/' + folder + '/'+file)
                        allpelatihan.append(data)
                self.progressBar_3.setValue(0)
                for k in range(len(allpelatihan)):
                    norm=normalisasi(data)
                    dataMFCC=MFCC(coefisien[i],norm,samplerate)
                    kmeanss=kmeans(dataMFCC,cluster)
                    suara.append(kmeanss)
                    self.progressBar_3.setValue((k+1)/len(allpelatihan)*100)
#                self.progressBar_3.setValue(100)
                for j in range(len(state)):
                    mfccFolder=str(coefisien[i])
                    stateFolder=str(state[j])
                    A,B,pi,loglike,scale=forwback(suara,state[j])
                    np.savetxt(source + '/'+ folder + '/'+mfccFolder +'/'+ stateFolder + '/'+'loglike.txt',loglike,fmt='%f')
                    np.savetxt(source + '/'+ folder + '/'+mfccFolder +'/'+ stateFolder + '/'+'a.txt',A,fmt='%s')
                    np.savetxt(source + '/'+ folder + '/'+mfccFolder +'/'+ stateFolder + '/'+'b.txt',B,fmt='%s')
                    np.savetxt(source + '/'+ folder + '/'+mfccFolder +'/'+ stateFolder + '/'+'pi.txt',pi,fmt='%s')
                self.progressBar_2.setValue(100)
        end=time.time()
        print((end-start)/60)
        self.info()
    def pengujian_text(self):
        data=[]
        tampil=""
        labelx=[]
        HasilText=""
        path=self.textBrowser_18.toPlainText()
        if(self.radioButton_10.isChecked()):
            f_size=2
        elif(self.radioButton_11.isChecked()):
            f_size=3
        elif(self.radioButton_12.isChecked()):
            f_size=4
        elif(self.radioButton_19.isChecked()):
            f_size=5
        else:
            f_size=6
        self.tableWidget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        with open(str(path), 'r') as readFile:
            reader = csv.reader(readFile)
            lines = list(reader)
            for i in range(len(lines)):
                self.progressBar_5.setValue((i+1)/len(lines)*100)
                str1 = ''.join(lines[i])
                kata=str1.split()
                data.append(kata)
                tampil+=' '.join(kata)+'\n'
                textC,lbl=klasifikasi_CNN(kata,f_size)
                labelx.append(lbl)
                HasilText+=textC+'\n'
                self.tableWidget.setItem(i,0, QTableWidgetItem(' '.join(kata)))
                self.tableWidget.setItem(i,1, QTableWidgetItem(textC))
        akurasi=self.calculate_akurasi(labelx)
        self.textBrowser_19.setText(str(akurasi)[:4])
    def pengujian_speech(self):
        self.textBrowser_11.setText("")
        self.textBrowser_12.setText("")
        self.progressBar_4.setValue(0)
        path=self.textBrowser_10.toPlainText()
        if(path==""):
            self.errorbox()
        source='Dataset'
        start=time.time()
        if(self.radioButton_3.isChecked()):
            coef=13
        elif(self.radioButton_4.isChecked()):
            coef=26
        else:
            coef=39
        if(self.radioButton.isChecked()):
            gender='Video LK'
        else:
            gender='Video PR'
        if(self.radioButton_6.isChecked()):
            hmmstate=10
        elif(self.radioButton_7.isChecked()):
            hmmstate=20
        elif(self.radioButton_8.isChecked()):
            hmmstate=30
        elif(self.radioButton_9.isChecked()):
            hmmstate=40
        elif(self.radioButton_13.isChecked()):
            hmmstate=50
        elif(self.radioButton_15.isChecked()):
            hmmstate=60
        elif(self.radioButton_14.isChecked()):
            hmmstate=70
        elif(self.radioButton_16.isChecked()):
            hmmstate=80
        elif(self.radioButton_17.isChecked()):
            hmmstate=90
        else:
            hmmstate=100
        dirc=path + '/' +gender
        mfccFolder=str(coef)
        stateFolder=str(hmmstate)
        kalimat=""
        x=[]
        with open(dirc+'/'+'kmeans{}.pkl'.format(coef),"rb") as f:
            allvideo = pickle.load(f)
            f.close()
        for i in range (len(allvideo)):
            prosi=(i/len(allvideo)*100)
            self.progressBar_4.setValue(prosi)
            kmeanss=allvideo[i]
            kata=[]
            for j in range(len(kmeanss)):
                prosj=prosi+j/len(kmeanss)*10
                self.progressBar_4.setValue(prosj)
                label=[]
                log=[]
                count=0
                for data in os.listdir(source):
                    prosk=prosj+(count/41*1)
#                    time.sleep(0.01)
                    self.progressBar_4.setValue(prosk)
                    A=np.loadtxt(source + '/'+ data + '/'+mfccFolder +'/'+ stateFolder + '/'+'a.txt',dtype=float)
                    B=np.loadtxt(source + '/'+ data + '/'+mfccFolder +'/'+ stateFolder + '/'+'b.txt',dtype=float)
                    pi=np.loadtxt(source + '/'+ data + '/'+mfccFolder +'/'+ stateFolder + '/'+'pi.txt',dtype=float)
                    loglike=forward(kmeanss[j],A,B,pi)
                    log.append(loglike)
                    label.append(data)
                    count+=1
                kata.append(label[np.nanargmax(log)])
            x.append(kata)
            kalimat+=' '.join(kata)+'\n'
            self.textBrowser_11.setText(kalimat)
        WER=self.calculate_WER(x)
        self.textBrowser_12.setText(str(WER)[:5])
        self.progressBar_4.setValue(100)
        end=time.time()
        print((end-start)/60)
    def deteksibtn(self,path):
        path=self.textBrowser.toPlainText()
        for i in reversed(range(self.layoutkata.count())): 
            self.layoutkata.itemAt(i).widget().deleteLater()
        for i in reversed(range(self.layoutnorm.count())): 
            self.layoutnorm.itemAt(i).widget().deleteLater()
        for i in reversed(range(self.layoutekstraksi.count())): 
            self.layoutekstraksi.itemAt(i).widget().deleteLater()
        source='Dataset'
        dirc=os.path.dirname(path)
        base=os.path.basename(path)
        coefisien=13
        state=40
        mfccFolder=str(coefisien)
        stateFolder=str(state)
        start=time.time()
        kata=[]
        allsuara=[]
        sinyalvideo=self.split_audio(path)
        for file in os.listdir (dirc):
            if(file[:5]=='chunk'):
                samplerate,data=wavfile.read(dirc +'/'+ file)
                allsuara.append(data)
                os.remove(dirc +'/'+ file)
        fig1 = Figure()
        ax1f1 = fig1.add_subplot(111)
        ax1f1.plot(sinyalvideo)
        ax1f1.set_title(str(base))
        ax1f1.set_axis_off()
        self.canvas = FigureCanvas(fig1)
        self.layoutkata.addWidget(self.canvas)
        allnorm=[]
        for i in range(len(allsuara)):
            norm=normalisasi(allsuara[i])
            allnorm.append(norm)
            fig1 = Figure()
            ax1f1 = fig1.add_subplot(111)
            ax1f1.plot(norm)
            ax1f1.set_title("kata ke-{}".format(i+1))
            ax1f1.set_axis_off()
            self.canvas = FigureCanvas(fig1)
            self.layoutnorm.addWidget(self.canvas)
        allekstraksi=[]
        for i in range(len(allnorm)):
            print("kata ke-{}".format(i+1))
            proses=i/len(allnorm)*100
            self.progressBar.setValue(proses)
            dataMFCC = MFCC(coefisien,allnorm[i],samplerate)
            allekstraksi.append(dataMFCC)
            fig1 = Figure()
            ax1f1 = fig1.add_subplot(111)
            ax1f1.plot(dataMFCC)
            ax1f1.set_title("kata ke-{}".format(i+1))
            ax1f1.set_axis_off()
            self.canvas = FigureCanvas(fig1)
            self.layoutekstraksi.addWidget(self.canvas)
            kmeanss=kmeans(dataMFCC,5)
            label=[]
            log=[]
            update=1/len(allnorm)*100
            j=0
            for data in os.listdir(source):
                self.progressBar.setValue(proses+((j+1)/41*update))
                j+=1
                A=np.loadtxt(source + '/'+ data + '/'+mfccFolder +'/'+ stateFolder + '/'+'a.txt',dtype=float)
                B=np.loadtxt(source + '/'+ data + '/'+mfccFolder +'/'+ stateFolder + '/'+'b.txt',dtype=float)
                pi=np.loadtxt(source + '/'+ data + '/'+mfccFolder +'/'+ stateFolder + '/'+'pi.txt',dtype=float)
                loglike=forward(kmeanss,A,B,pi)
                log.append(loglike)
                label.append(data)
            kata.append(label[np.nanargmax(log)])
        ujaran,lbl=klasifikasi_CNN(kata,2)
        self.textBrowser_4.setText(str(ujaran))
        self.textBrowser_3.setText(' '.join(kata))
        self.progressBar.setValue(100)
        end=time.time()
        print((end-start)/60)
    def calculate_akurasi(self,x):
        label=[1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0]
        cnt=0
        for i in range(len(label)):
            if(label[i]==x[i]):
                cnt+=1
        return cnt/len(label)*100
    def calculate_WER(self,x):
        data=[]
        with open('data_speech.txt', 'r') as readFile:
            reader = csv.reader(readFile)
            lines = list(reader)
            for i in range(len(lines)):
                str1 = ''.join(lines[i])
                kata=str1.split()
                data.append(kata)
        count=0
        for i in range(len(data)):
            for j in range(len(data[i])):
                if(data[i][j]!=x[i][j]):
                    count+=1
        WER=count/53*100
        return WER
    def split_audio(self,path):
        path=str(path)
        dirc=os.path.dirname(path)
        base=os.path.basename(dirc)
        ########CONVERT VIDEO TO AUDIO#########
        svideo = path
        saudio = dirc +'/'+ 'audio.wav'
        command = 'ffmpeg', '-i',svideo, '-ar','16000', '-ac', '1', saudio
        subprocess.call(command, shell=True)
        ########SPLIT AUDIO#########
        sound = AudioSegment.from_wav(dirc+'/'+'audio.wav')
        if(base=='Video LK'):
            chunks = split_on_silence(sound, min_silence_len=200,silence_thresh=-40)
        else:
            chunks = split_on_silence(sound, min_silence_len=200,silence_thresh=-50)
        for i, chunk in enumerate(chunks):
            chunk.export(dirc +'/'+'chunk{0}.wav'.format(i), format="wav")
        time.sleep(2)
        samplerate,data=wavfile.read(dirc+'/'+'audio.wav')
        os.remove(dirc+'/'+'audio.wav')
        return data
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window=MyWindow()
    window.show()
    sys.exit(app.exec_())