# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:51:07 2024

@author: SOGUTPC
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt # modül çağır dosyayı oku ve dosyadaki sütunların adını değiştir
veriseti = pd.read_csv("insurance.csv")
veriseti = veriseti.rename(columns = {"age":"yas","sex":"cinsiyet","bmi":"vki","children":"cocukSayisi","smoker":"sigaraDurum","region":"bolge","charges":"odemeMiktari"})

veriseti.dtypes #kategorik verileri kategoriye çevirme dtypes veri tiplerini kontrol etmeyi sağlar
veriseti ["cinsiyet"] = veriseti["cinsiyet"].astype('category')
veriseti["sigaraDurum"] = veriseti["sigaraDurum"].astype('category')
veriseti ["bolge"] = veriseti ["bolge"].astype('category')

veriseti ["cinsiyet"] = veriseti["cinsiyet"].replace(["male","female"],["erkek","kadin"])
veriseti ["sigaraDurum"] = veriseti["sigaraDurum"].replace(["yes","no"],["evet","hayir"])
veriseti ["bolge"] = veriseti["bolge"].replace(["southeast","northeast","southwest","northwest"],["guneydogu","kuzeydogu","guneybati","kuzeybati"])
veriseti.describe() #sayisal niteliklerin özet bilgisi
pd.set_option("display.max_columns",20) #özette max 20 sütun
veriseti.describe(include = "all") #tüm veriler

x = range(1,11)
plt.title("İlk 10 Müşterinin Yaşları")
y = veriseti.iloc[0:10,0]
plt.plot(x,y,"o:r") #çizgi grafiği fonk müşterilerin yaşlarını gösteren
plt.plot(x,y,linestyle = "dashed",color = "green",linewidth = "5") #çizgili yapma

x = range(1,21) #ilk 20 müşterinin yaş dağılım çizgi grafiği
y = veriseti.iloc[0:20,0]
plt.plot(x,y,linestyle = "dashed",color = "purple",linewidth = "3") 

x1 = np.arange(1,21)
plt.title("İlk 20 ve Son 20 Müşteri VKİ Grafiği")
plt.xlabel("ID")
plt.ylabel("VKİ")
y1 = veriseti.iloc[0:20,2]
y2 = veriseti.iloc[20:40,2]
plt.xticks(x1)
plt.plot(x1,y1,x1,y2)


x1 = np.arange(1,30)
y1 = veriseti.iloc[0:29,2]
plt.xticks(x1)
plt.plot(x1,y1)

x1 = np.arange(1,51)
y1 = veriseti.iloc[0:50,0]
plt.title("İlk 50 Müşterinin Yaşı")
plt.xlabel("Müşteri ID'si ")
plt.ylabel("Yaş")
plt.plot(x1,y1)
plt.grid(color = "blue",linestyle ="--",linewidth = "0.5")

#İlk 20 ve Son 20 Müşteri VKİ Grafiği
x1 = np.arange(1,21)
y1 = veriseti.iloc[0:20,2]
#plt.subplot(1,2,1) tek kareye sığdırır
plt.subplot(2,1,1) #grafik alanı 2 satır 1 sütun 1.grafik tanımlaması
plt.plot(x1,y1)
plt.xticks(x1)
plt.title("İlk 20 Müşteri")

y2 = veriseti.iloc[20:40,2]
plt.subplot(2,1,2)
plt.plot(x1,y2)
plt.xticks(x1)
plt.title("İlk 20 Müşteri")
plt.suptitle("VKİ KARŞILAŞTIRMASI")
plt.tight_layout(pad=1) #ikinci başlığı sığdırmak için

ozet = veriseti.groupby("sigaraDurum")["odemeMiktari"].mean() # sigara durumuna göre ortalama ödeme miktarı özet değişkeninde
plt.bar( x = ozet.index, height = ozet.values, color ="r")
plt.xlabel("Sigara İçme Durumu")
plt.ylabel("Odeme Miktari")
plt.title("Sigara İçme-Ödeme Miktarı Sütun Grafiği")
plt.grid(color ="blue",linestyle = "--",linewidth = "0.5")

sns.histplot(data = veriseti ,x = "vki",color = "pink") 
#sns.histplot(data= veriseti, y= "vki",color="lightgreen")
plt.title("VKİ Histogramı")


x = veriseti.yas
y = veriseti.vki
plt.scatter(x, y)
plt.xlabel("Yaş")
plt.ylabel("Vücut Kitle İndeksi")
plt.title("Serpilme Diyagramı")
sns.violinplot(y = "sigaraDurum",x = "odemeMiktari",data = veriseti,palette="coolwarm")
plt.title("Sigara İçme Durumu ve Ödeme Miktarı Violin Grafiği")

ozet = veriseti.groupby("bolge")["odemeMiktari"].sum() #bölgelere ait ödeme özet bilgi
degerler = ozet.values
etiketler = ozet.index
renkler = sns.color_palette("viridis",5)
secim =(0,0,0.2,0)
plt.pie(degerler , explode=secim,labels=etiketler,autopct="%%%4.1f",shadow=True,startangle=360,colors = renkler)
plt.title("Bölgelere Göre Ödeme Bilgisi Pasta Grafiği")


korelasyon = veriseti[["yas","vki","cocukSayisi","odemeMiktari"]].corr() #korelasyon hesaplma
sns.heatmap(korelasyon,annot=True,square=True,cmap = "Reds")
plt.title("Isı Haritası Deneme")