# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 20:46:06 2024

@author: SOGUTPC
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


veriseti = pd.read_csv("mycustomers.csv")
veriseti.info()
veriseti.describe()

veriseti.isnull()
veriseti.fillna(veriseti.mean(),inplace=True)
veriseti.columns

veriseti.columns=['ID','Cinsiyet','Medeni hal','Yaş','Eğitim','Gelir','Meslek','Yerleşim büyüklüğü']

num_cols = ["Yaş","Gelir"] #nümerik
cat_cols =["Cinsiyet","Meslek","Yerleşim büyüklüğü","Eğitim",'Medeni hal'] #kategorik

for i in num_cols:
    fig,axs = plt.subplots(figsize=(8,5))
    sns.histplot(data=veriseti,x=i,color="blue",kde=True)
    plt.show()


yas_kutu_graf = sns.boxplot(data=veriseti,x="Yaş",palette="rocket")
gelir_kutu_graf = sns.boxplot(data=veriseti,x="Gelir")

ust_aykiri = veriseti[veriseti["Gelir"]>600000].ID.tolist()
alt_aykiri = veriseti[veriseti["Gelir"]<25000].ID.tolist()
veriseti = veriseti[~(veriseti["ID"].isin(ust_aykiri+alt_aykiri))] #aykırı değerleri verisetinden çıkar
print(ust_aykiri+alt_aykiri)

for x in cat_cols:  
    fig,axs = plt.subplots(figsize=(8,6))
    sns.countplot(data=veriseti,x=x)
    plt.show()
    
for col in num_cols: #aykırılardan sonra 
    sns.histplot(data=veriseti,x=col,color="purple",kde="True")
    plt.show()

ilk_sutun = len(veriseti)
veriseti.drop_duplicates(inplace=True)
son_sutun = len(veriseti)
print(f"Number of duplicate rows: {ilk_sutun - son_sutun}")
veriseti.info()
veriseti[num_cols].describe()

scaler = MinMaxScaler()
sutunlar = num_cols+cat_cols
X = scaler.fit_transform(veriseti[sutunlar])
X = pd.DataFrame(data=X,columns=sutunlar)
X.describe()
Xdata = X.copy()

n_clusters = range(2,12)

inertia = []
for i in n_clusters:
    kmeans = KMeans(n_clusters=i,random_state=2024)
    _=kmeans.fit_predict(Xdata)
    inertia.append(kmeans.inertia_)
    
fig,axs = plt.subplots(figsize=(8,6))
sns.lineplot(x=n_clusters,y=inertia)
axs.set_xlabel("Küme Sayısı")
axs.set_ylabel("Inertia")
axs.set_title("Dirsek Katsayısı")
plt.show()

#veri boyutu indirgemek,gürültülü veri azaltmak için PCA Analizi
pca = PCA()
pca.fit(Xdata)
plt.figure(figsize=(8,5))
plt.plot(range(1,8),pca.explained_variance_ratio_.cumsum(),marker="*")
plt.title("PCA Analizi")
plt.show()

pca = PCA(n_components=3)
pca.fit(Xdata)
pca_skor = pca.transform(Xdata)

kmeans = KMeans(n_clusters=6)
_ = kmeans.fit_predict(pca_skor)

#PCA sonuçlarını K-means etiketleri ile birleştirip yeni veri oluşturma
pca_birlesimler = pd.DataFrame(data=pca_skor,columns=["component_x","component_y","component_z"])
df_kmeans = pd.concat([veriseti.reset_index(drop=True),pca_birlesimler],axis=1)
df_kmeans["label"] = kmeans.labels_
print(df_kmeans)

fig,axs = plt.subplots(figsize=(8,6))
sns.scatterplot(data=df_kmeans,x="component_x",y="component_y",hue="label",palette="tab10")
plt.xlabel("component_x")
plt.ylabel("component_y")
plt.title("Müşteri Kümeleri (K-Means + PCA)")
plt.show()

#kümeleri analiz etme
cat_cols.append("label")
df_kmeans[cat_cols]=df_kmeans[cat_cols].astype(str)
data = df_kmeans
data.describe()

data.to_csv("customers_labeled4.csv")

#Küme 0 küçük şehirde yaşayan, düşük gelirli, düşük eğitim seviyesine sahip orta yaşlı evli kadınlar
#Küme 1  küçük şehirde yaşayan, bekâr, orta eğitimli ve düşük gelirli genç erkekler
#Küme 2 büyük şehirde yaşayan, evli, vasıflı mesleği olan, orta-yüksek gelirli yaşlı kadınlar
#Küme 3 orta şehirde yaşayan, bekâr, orta gelirli, iyi eğitimli orta yaşlı kadınlar
#Küme 4  küçük şehirde yaşayan, evli, düşük gelirli, orta eğitim seviyesine sahip orta yaşlı erkekler
#Küme 5 büyük şehirde yaşayan, bekâr, vasıflı ve yüksek eğitimli, orta-yüksek gelirli genç-orta yaşlı erkekler


def grafikOlustur(hue,title):
    fig, axs = plt.subplots(figsize=(8,6))
    sns.countplot(data=data, x="label",hue=hue,order=["0", "1", "2", "3", "4", "5"])
    axs.set_title(title)
    plt.show()

grafikOlustur(hue="Cinsiyet",title="Cinsiyete Göre Label")
grafikOlustur("Medeni hal", "Medeni Hale Göre Label")
grafikOlustur("Eğitim", "Eğitime Göre Label")
grafikOlustur("Yerleşim büyüklüğü", "Yerleşim Alanına göre Label")


for i in num_cols:
    fig,axs = plt.subplots(figsize=(8,5))
    sns.boxenplot(data=data,x="label",y=i ,order=["0", "1", "2", "3", "4", "5"])
    plt.hlines(y=data[i].median(), xmin=0, xmax=6, colors="b",linestyles="dashed")
    plt.title(f"{i} - Label Dağılımı")
    plt.show()
    
#%% Decision Tree
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import tree

data_labeled = pd.read_csv("customers_labeled4.csv",sep=",",index_col=False)
data_labeled.info()

data2 = data_labeled.copy()
data2.rename(columns={'ID':'ID',
'Sex':'Cinsiyet','Marital status':'Medeni hal',
'Age':'Yaş',
'Education':'Eğitim',
'Income':'Gelir',
'Occupation':'Meslek','Settlement size':'Yerleşim büyüklüğü'},inplace=True)

num_col = ["Yaş","Gelir"]
cat_col = ["Medeni hal","Eğitim",
           "Meslek","Yerleşim büyüklüğü","Cinsiyet","label" ]
data2=data2[num_col+cat_col]

#standartscaler ile ölçeklendirme
standard_s = StandardScaler()
cols = ["Medeni hal","Eğitim","Meslek","Yerleşim büyüklüğü","Cinsiyet","Yaş","Gelir"]
X_ = standard_s.fit_transform(data2[cols])

X = X_
y = data.label
X_test,X_train,y_test,y_train = train_test_split(X,y,train_size=0.8)
X.shape

model = DecisionTreeClassifier(criterion="gini",max_depth=3)
model = model.fit(X_train,y_train)
y_train_predict = model.predict(X_train)
y_test_predict = model.predict(X_test)

accuracy_train = accuracy_score(y_train, y_train_predict)
accuracy_test = accuracy_score(y_test,y_test_predict)
print('Train data score: {}'.format(accuracy_train))
print('Test data score: {}'.format(accuracy_test))
print(classification_report(y_test,y_test_predict))

confusionMatrixTrain = confusion_matrix(y_train, y_train_predict)
confusionMatrixTest = confusion_matrix(y_test, y_test_predict)

fig = plt.figure(figsize=(15,8))
_ = tree.plot_tree(model, feature_names=cols,class_names=['0','1','2','3','4','5'],filled=True)


#sınıflama
cols = ["Medeni hal","Eğitim","Meslek","Yerleşim büyüklüğü","Cinsiyet","Yaş","Gelir"]
test = pd.read_csv("classification.csv")
test.info()
test.rename(columns={'ID':'ID',
'Sex':'Cinsiyet','Marital status':'Medeni hal',
'Age':'Yaş',
'Education':'Eğitim',
'Income':'Gelir',
'Occupation':'Meslek','Settlement size':'Yerleşim büyüklüğü'},inplace=True)

test_scalled = standard_s.fit_transform(test[cols])
test.shape

print(X_train.shape)  # Eğitim veri setinin boyutu
print(X_test.shape)   # Test veri setinin boyutu

test['label']=model.predict(test_scalled)
test[cols]

df_test = pd.DataFrame(test_scalled)
y_label_predict = model.predict(df_test)
print(y_label_predict)


#%%
from sklearn.linear_model import LinearRegression
model2 = LinearRegression()
model2.fit(X_train.reshape(-1, 1), y_train)
y_pred2 = model2.predict(X_test.reshape(-1,1))
y_pred2_rounded =np.round(y_pred2).astype(int)

accuracy = accuracy_score(y_test, y_pred2_rounded)

print("Model accuracy:", accuracy)

