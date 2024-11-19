# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 16:36:26 2024

@author: SOGUTPC
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os
from collections import Counter

veri = pd.read_csv("Modern Olympic Games.csv")
veri.describe(include="all")
veri.head()
veri.info()

veri.rename(columns= {
 "ID":"id","Name":'isim',
 'Gender'   : 'cinsiyet', 
                     'Age'   : 'yas', 
                     'Height': 'boy', 
                     'Weight': 'kilo', 
                     'Team'  : 'takim', 
                     'NOC'   : 'uok', 
                     'Games' : 'oyunlar',
                     'Year'  : 'yil', 
                     'Season': 'sezon', 
                     'City'  : 'sehir',
                     'Sport' : 'spor',
                     'Event' : 'etkinlik',
                     'Medal' : 'madalya'}, inplace=True)

veri= veri.drop(["id","oyunlar"],axis=1)
#boy-kilo eksik verisi etkinlik ortalamasına göre tamamlar
essiz = pd.unique(veri.etkinlik)
print("Eşsiz etkinlik {}".format(len(essiz)))

veri_gecici = veri.copy()
boy_kilo_liste = ["boy", "kilo"]

for e in essiz:
    etkinlik_filtre = veri_gecici.etkinlik == e
    veri_filtreli = veri_gecici[etkinlik_filtre]

for s in boy_kilo_liste:
    ort = np.round(np.mean(veri_filtreli[s]),2)
    if ~np.isnan(ort): # eğer etkinlik özelinde ortalama varsa
            veri_filtreli[s] = veri_filtreli[s].fillna(ort)
    else: # etkinlik özelinde ortalama yoksa tüm veri için ortalama bul
            tum_veri_ortalamasi = np.round(np.mean(veri[s]),2)
            veri_filtreli[s] = veri_filtreli[s].fillna(tum_veri_ortalamasi)
    # etkinlik özelinde kayıp değerleri doldurulmuş veriyi veri geçiciye eşitle            
    veri_gecici[etkinlik_filtre] = veri_filtreli
    
veri = veri_gecici.copy() 
veri.info()
 

#eksik yaş verisi tamamlama
yas_ort = np.round(np.mean(veri.yas),2)
print(f" Yas ortalaması {yas_ort}")
veri["yas"] = veri["yas"].fillna(yas_ort)

 
madalya_degiskeni = veri["madalya"]
pd.isnull(madalya_degiskeni).sum()
madalya_degiskeni_filtresi = ~pd.isnull(madalya_degiskeni)
veri = veri[madalya_degiskeni_filtresi]
veri.head(5)   
  
#veri = veri_gecici.copy() 
#veri.info()   


def barPlot(degisken,n):
   veri2_=veri[degisken]
   veri_sayma = veri2_.value_counts()
   veri_sayma = veri_sayma[:n]
   plt.figure()
   plt.bar(veri_sayma.index,veri_sayma,color="purple")
   plt.xticks(veri_sayma.index,veri_sayma.index.values,rotation=45,)
   plt.ylabel("Frekans")
   plt.show()
   
liste = ["isim","sehir","spor"]
for i in liste:
    barPlot(i,6)
    
barPlot("isim",8) 



#Grafik Fonksiyonları

def plotHistogram(degisken):
    """
        Girdi: Değişken/sütun ismi
        Çıktı: Histogram grafiği
    """
    
    plt.figure()
    plt.hist(veri[degisken], bins = 85, color = "orange")
    plt.xlabel(degisken)
    plt.ylabel("Frekans")
    plt.title("Veri Sıklığı - {}".format(degisken))
    plt.show()

sayisal_degisken = ["yas", "boy", "kilo", "yil"]
for i in sayisal_degisken:
    plotHistogram(i)   
  
def plotScatter(degisken1,degisken2):
    plt.figure()
    sns.histplot(x=degisken1,y=degisken2,color="green",bins=85)
    plt.xlabel(degisken1)
    plt.ylabel(degisken2)
    plt.show()

plotScatter(veri["Height"], veri["Weight"])
    
def boxplot(degisken1,degisken2):
    plt.figure()
    sns.boxplot(data=veri,x=degisken1,y=degisken2)
    plt.show()
    
boxplot(veri["Medal"],veri["Weight"])

def sütungrafigi(degisken,degisken2):
    plt.figure()
    sns.barplot(data=veri,x=degisken,y=degisken2)
    plt.xlabel(degisken)
    plt.ylabel(degisken2)
    plt.title(f"Veri Sıklığı {degisken}")
    plt.xticks(rotation=90)
    plt.show()
    
sütungrafigi(veri["Age"],veri["Height"])

def çizgigrafiği(degisken_1,degisken_2,degisken_3):
    plt.figure()
    sns.lineplot(data=veri,x=degisken_1,y=degisken_2,hue=degisken_3,palette="Set1")
    plt.xlabel(degisken_1)
    plt.ylabel(degisken_2)
    plt.title(f"Veri Sıklığı {degisken_1}")
    plt.show()

çizgigrafiği(veri["Height"],veri["Weight"],veri["Gender"])

def displot(degisken_1,degisken_2,degisken_3):
    sns.displot(data=veri,x=degisken_1,y=degisken_2,hue=degisken_3,palette="rocket",kind="kde")
    plt.xlabel(degisken_1)
    plt.ylabel(degisken_2)
    plt.title("Sporcuların Boy ve Kilo Dağılımı")
    plt.show()
    
displot(veri["Height"], veri["Weight"],veri["Gender"])

#İki Değişkenli Analiz
pd.set_option('display.max_columns', None) #tüm sütunları gösterir
erkek = veri[veri.cinsiyet == "M"]
erkek.head(1)

kadin = veri[veri.cinsiyet == "F"]
kadin.head(1)

plt.figure()
plt.scatter(kadin.boy,kadin.kilo,alpha=0.4,label="Kadin")
plt.scatter(erkek.boy,erkek.kilo,alpha=0.4,label="Erkek")
plt.xlabel("Boy")
plt.ylabel("Kilo")
plt.title("Boy ve Kilo Arasındaki İlişki")
plt.legend()
plt.show()

veri.loc[:,["yas","boy","kilo"]].corr() #korelasyon tablosu

veri_gecici = veri.copy()
veri_gecici = pd.get_dummies(veri_gecici,columns=["madalya"])
veri_gecici.head(2)

veri_gecici.loc[:,["yas","madalya_Bronze", "madalya_Gold","madalya_Silver"]].corr()
veri_gecici[["takim","madalya_Gold", "madalya_Silver", "madalya_Bronze"]].groupby(["takim"], as_index = False).sum().sort_values(by="madalya_Gold",ascending = False)[:10]
veri_gecici[["sehir","madalya_Gold","madalya_Silver","madalya_Bronze"]].groupby(["sehir"],as_index=False).sum().sort_values(by="madalya_Gold",ascending=False)[:15]
veri_gecici[["cinsiyet","madalya_Gold","madalya_Silver"]].groupby(["cinsiyet"],as_index=False).sum().sort_values(by="madalya_Silver",ascending=False)[:10]
veri_gecici[["yas","madalya_Gold", "madalya_Silver", "madalya_Bronze"]].groupby(["yas"],as_index=False).sum().sort_values(by="madalya_Gold",ascending=False)[:10]

veri_pivot = veri.pivot_table(index="madalya", columns = "cinsiyet",
                 values=["boy","kilo","yas"], 
                aggfunc={"boy":np.mean,"kilo":np.mean,"yas":[min, max, np.std]})
veri_pivot.head()


a = veri.pivot_table(index="madalya", columns="etkinlik",
                     values=["boy", "kilo", "yas"],
                     aggfunc={"boy": np.mean, 
                              "kilo": np.mean, 
                              "yas": [np.min, np.max]})
a.head()
def anomaliTespiti(df,ozellik): #sonuç yok
    outlier_indices = []
    
    for c in ozellik:
        # 1. çeyrek
        Q1 = np.percentile(df[c],25)
        # 3. çeyrek
        Q3 = np.percentile(df[c],75)
        # IQR: Çeyrekler açıklığı
        IQR = Q3 - Q1
        # aykırı tespiti için çarpan
        outlier_step = IQR * 1.5
        # aykırıyı ve aykırı indeksini tespit et
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        # indeksleri depola
        outlier_indices.extend(outlier_list_col)
    
    # eşsiz aykırı değerleri bul
    outlier_indices = Counter(outlier_indices)
    # eğer bir örnek (v) 1 farklı sütun için aykırı değerse bunu aykırı olarak kabul et (v>1)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 1)
    
    return multiple_outliers

veri_anomali = veri.loc[anomaliTespiti(veri,["yas","kilo","boy"])]
veri_anomali.spor.value_counts()
#%%
veri_zaman = veri.copy() # verinin orjinali bozulmasın diye kopyalayalım
veri_zaman.head(3)

essiz_yillar = veri_zaman.yil.unique()
essiz_yillar

sirali_yillar = np.sort(veri_zaman.yil.unique())
print(sirali_yillar)

boylar = veri_zaman.boy.unique()
sirali_boylar = np.sort(veri_zaman.boy.unique())

plt.figure()
plt.scatter(range(len(sirali_yillar)),sirali_yillar)
plt.xlabel("frekans")
plt.grid(True)
plt.ylabel("yil")
plt.show()


#%%
sns.set_style("whitegrid")
plt.scatter("Height", "Weight",data=veri)
plt.xlabel("Boy")
plt.ylabel("Ağırlık(kilo")

sns.set_style("darkgrid")
sns.scatterplot(data=veri,x ="Height",y="Weight",hue="Gender",style="Medal",palette="rocket")
plt.xlabel("Boy")
plt.ylabel("Ağırlık(kilo")
plt.title("Sporcuların Cinsiyetlerine Göre Boy ve Ağırlık Dağılımları")

sns.set_style("whitegrid")
sns.histplot(data=veri,x="Height",hue="Gender")
plt.ylabel("Frekans")
plt.xlabel("Sporcu Boy")
plt.title("Sporcuların Cinsiyetlerine Göre Boyları Histogramı")
plt.show()

sns.countplot(x="City", data=veri)
plt.xticks(rotation = 90)
plt.show()

sns.set_style("darkgrid")
sns.kdeplot(data=veri,x="Height",y="Weight",hue="Gender")

sns.catplot(x='Medal', y='Age', hue='Gender', col='Season', data=veri, kind='bar',palette='Set1')
             
sns.set_style("darkgrid")
sns.scatterplot(data=veri,x="Height",y="Weight",hue="Gender",size="Age",style="Medal")
plt.title("Sporcuların Takımlarına Göre Aldıkları Madalyalar")

sns.kdeplot(data=veri,x="Height",y="Weight",hue="Gender")


sns.boxplot(x='Season', y='Weight', hue='Gender', data=veri, palette='Set2')
plt.show()

sns.heatmap(veri.corr(), annot=True, linewidths=0.5, fmt='.1f')
plt.show()

sns.violinplot(data=veri,x="Gender",y="Weight")
plt.show()

sns.countplot(data=veri,x="City")
plt.xticks(rotation=90)
plt.show()

sns.countplot(data=veri,x="Sport")
plt.xticks(rotation=90)
plt.figure(figsize=(20,8))
plt.show()

sns.regplot(data=veri,x='Height',y="Weight",color="orange",marker="+",scatter_kws={'alpha':0.2})
plt.xlabel("Boy")
plt.ylabel("Kilo")

sns.pairplot(veri)
plt.show()


