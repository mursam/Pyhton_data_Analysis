
### PYHTON İLE VERİ ANALİZİ

## --Numpy
## - Pandas
## - Veri Görselleştirme : MatPlotlib & SeaBorn
## - Gelişmiş Fonksiyonel Keşifçi Veri Analizi(Advanced Functional Exploratory Data Analysis)


### NUMPY
## Numerical Pyhtın ın kısaltması

## Matematik ve İstatik

## Lİstelere kıyasla daha hızlı şekilde işlem yapar, sabit tipte veri tuttuğu için hızlı


import  numpy as np
import seaborn

a = [1,2,3,4,5]
b = [2,3,4,5, 6]

ab = []

for i in range(0,len(a)):
    ab.append(a[i]* b[i])

a = np.array([1,2,3,4,5])
b = np.array([2,3,4,5,6])
a*b


np.array([10, 20, 30, 40, 50, 60])
type(np.array([10, 20, 30, 40, 50, 60]))


np.zeros(100, dtype=int)## istediğin kadar 0 oluşturur
np.random.randint(0,100, size= 10 ) ## istenilen aralıkta istenilen kadar arrya oluşturu
np.random.normal(10, 4 , (4,4)) ## ortalaması 10 olan standart sapması 4 olan 4x4 matris oluştur


## Numpy array özellikleri

# ndim: boyut sayısı
# shape : boyut bilgisi
# size : toplam eleman sayısı
# dtype : array veri tipi

a = np.random.randint(10 , size = 6 )
a.ndim
a.shape
a.size
a.dtype


##ReShaping

b = np.random.randint(1, 20 , size= 10)
b.reshape(5,2) ## elaman sayısına bağımlıdır .

## Index Seçimi

c = np.random.randint(100, size = 100)
c[0]
c[0:15]

m = np.random.randint(10,size=(3,5,4))

m.reshape(5,4,3)


## Fancy Index
## İndex in içerisine koşulu yaparak gerçekleştirebiliriz .
v = np.arange(0, 30 , 3)

v[5]
catch = [1,2,3]

v[catch]

## Koşullu İşlemler
v < 5

v % 9 == 0
v[v% 9 == 0 ]

## Matematiksel İşlmler

v / 5

v ** 3
v ** (1/2)

np.subtract(v,1) ## bir çıkar
np.add(v,1 ) # bir ekle
np.mean(v)
np.sum(v)
np.min(v)
np.var(v)
np.max(v)


numpy_list = []


a = np.array([[5,1],[1,3]])
b = np.array([12,10])

np.linalg.solve(a,b)


######### Pandas
import pandas as pd


s = pd.Series([10 ,11,123, 4, 5 ])  ## indexli liste verir. içinde default olarak vardır.

type(s)
s.index
s[4]

s.dtype
s.size
s.ndim
s.values ## numpy array ı olarak çıktı verir.
type(s.values)

s.head(3)
s.tail(2)


## Veri Okuma

df = pd.read_csv(##buraya dizin yazılır "" içerisinde)
## pandas cheatsheet


import seaborn as sns

df = sns.load_dataset("titanic")

df.head()
df.tail()
df.shape
df.info() ## object ve categpry aynı veri tipini kategorik demek aynı anlama gelir genel anlamda

df.columns
df.index
df.describe().T ##  veri hakkında bilgi veren komut, ortalama , sayısı, max ve min değerleri gibi
df.isnull().values.any() ## values numpy arry i getirir . Çünkü index bilgisi getirilmiyor.
##Elimizdeki veride null değer var mı nın cevabıdır .

df.isnull().sum() ## her bir kolonda kaç tane eksik değer olduğunu gösteririr. IsNull True = 1 olarak sayar
df["sex"].head()
df["sex"].value_counts() #belirli bir değişkenin içerisinde kaç adet değer olduğunu ve sınıflarını verir.


## Pandas'ta Seçim İşlemleri

df[0:13]
df.drop(0,axis=0).head() ## veri setindeki istenen alanları ve indexleri siler

#df.drop(0,axis=0, inplace=True) tekrar atama yapmadan yapılan değişikliği kalıcı hale getirir.

## Değişkeni indexe çevirmek

df["age"].head()
df.age.head()
df.index = df["age"]
df.drop("age", axis=1).head()
df.drop("age",axis=1, inplace=True)
df.head()


## Index i value'ya çevirmek

df["age"] = df.index

df.head()

df.reset_index().head()
df = df.reset_index()
df.head()


## Değişkenler üzerinde işlem yapmak


pd.set_option('display.max_columns', None)
df.head
type(df["age"].head()) ##[] Tek parantezli seçim seri olarak çıktı verir.

type(df[["age"]].head()) ## çıktı sonucu dataframe olması için [[]] şeklinde bir seçim yapılmalı



df[["age", "alive"]] ## birden fazla değişken seçebilmek için

col_names = ["age","adult_male", "alive"]
type(df[col_names]) # İçerisinde liste tanımladığımız için çıktı yine dataframe dir.

df["age2"] = df["age"]**2
df["age3"] = df["age"] / df["age2"]

df.head()

df.drop("age3", axis=1).head()

df.drop(col_names,axis=1).head()


df.loc[:,df.columns.str.contains("age")].head() ##loc seçme işlemleri için kullanılır
## dataframe kolonları içierisindeki stringlerde parantez içine girilen string ifadeyi bulur


## Loc & ILOc

#Iloc : integer based selection

df.iloc[1:5] ##genel hali ile çalışır. sondaki index e kadar gider.


## loc :label based selection

df.loc[0:3] ## burada girilen değere  gider.


df.iloc[0:3 , 3:5] ## içerisine str tipinde değişken almaz

df.loc[0:3 , "age"]

df.loc[0:3 , col_names] ## içerisine liste alarak devam eder.

## Koşullu seçim

df[df["age"] > 50].head()

df[df["age"]> 50]["age"].count()


df.loc[df["age"] > 50, "class"].head()

df.loc[df["age"] > 50, ["class", "age"]].head()


df.loc[(df["age"] > 50) & (df["sex"] == "male"), ["class", "age"]].head()
## içeriye istenilen kadar kategori ve koşul girilebiliyor. | (veya anlamına gelir)


## Toplulaştırma ve Gruplama

##count() , first(), last(), min(), max(), std(), mean(), median(), var(), sum(), pivot table




import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()
df["age"].mean()

df.groupby("sex")["age"].mean() ## sql deki gibi çalışır

df.groupby("sex").agg({"age":"mean"}) ## sözlük gibi tanımlıyoruz {} parantezleri ile
agg_functions = ["mean", "sum","max","min","count"]
df.groupby("sex").agg({"age":["mean","sum","max"]})


df.groupby("sex").agg({"age":agg_functions ,
                       "survived" : "mean"})

df.groupby(["sex","embark_town","class"]).agg({"age":["mean","sum",],
                                        "sex":"count",
                                       "survived":"mean"})




### pivot table

df.pivot_table("survived", "sex", "embarked") ## value ilk değerdir yani tabloda göstereline değer
## temel tanımı pibot table da değerlerin ortalamasını (mean) değerini getirir.
df.pivot_table("survived", "sex", "embarked", aggfunc="std")

df.pivot_table("survived","sex",["embarked","class"])
## liste içerisinde ifade edilen değerler ^ aynı grup içerisinde yani aynı sütün içerisibde gruplama yapar.
## yani binilen yer ve sınıf aynı sütun içerisinde ayrım yapar


df["new_age"] = pd.cut(df["age"],[0, 10, 18, 25, 40, 90])
## girilen ilk değer başlangıç ikinci değer aralıktaki üst değeri belirler
##.cut değerleri biliyorsak yaş olarak kendimiz sınıflandırablliyoru örn.
# sayısal değeri kategorik değerlere çevirmek için kullanılır.
# ve .qcut değeleri bilmiyoruz ve çeyreklik oalarak ayırır


df.head()

df.pivot_table("survived","sex", "new_age")

df.pivot_table("survived", "sex", ["new_age","class"])

pd.set_option("display.width", 500)

x="Hello"[0]
print(x)

x=sub("Hello",0,1)



import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns',None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")

df. head()

df["age2"]=df["age"]*2
df["age3"]=df["age"]*3

for col in df.columns:
    if "age" in col:
        print((df[col]/10).head())



for col in df.columns:
    if "age" in col:
        df[col] = df[col]/10


df.head()

df[["age","age2","age3"]].apply(lambda x:x/10).head()

df.loc[:,df.columns.str.contains("age")].apply(lambda x:x/10).head

df.loc[:,df.columns.str.contains("age")].apply(lambda x:(x-x.mean()/x.std()).head()
def standart_scaler(col_name):
    return(col_name - col_name.mean())/ col_name.std()



df.loc[:,df.columns.str.contains("age")]=df.loc[:, df.columns.str.contains("age")].apply(standart_scaler)

df.head()


## birleştirme işlemleri

import numpy as np
import pandas as pd

m= np.random.randint(1,30, size =(5,3))
df1=pd.DataFrame(m,columns=["var1","var2","var3"])
df2 = df1 + 99

pd.concat([df1,df2])  ## iki dataframein birlşetirilmesi için kullanılan concat fonksiyou
#pandas kütüphanesi içinde vardır .

pd.concat([df1,df2], ignore_index=True, axis=0) ## yanyana birşetirmek istersek axis =1 değeri olmalı


pd.merge(df1,df2)

pd.merge(df1,df2,on="employess") ## on istenilen index üzeriden birleştirir.




#### Veri Görselleştirme

## MATLPOTLİB
import matplotlib.pyplot as plt
df = sns.load_dataset("titanic")
pd.set_option("display.width", 500)
df.head()

df["sex"].value_counts().plot(kind="bar")
plt.show()

plt.hist(df["age"])
plt.show()


plt.boxplot(df["fare"])


## Katmanlı şekilde veri görselleştirmeyi sağlar

x = np.array([1,98])
y = np.array([0,150])
plt.plot(x,y)
plt.plot(x,y, "o")
plt.show()

x=np.array([2,4,6,8,10])
y=np.array([1,3,5,7,9])


## Marker

y=np.array([12,345,77,56])

plt.plot(y,marker='o')

## Line

plt.plot(y,linestyle = "dashdot", color = "g")

## Multiple lines

x=np.array([23,18,45,37])
y=np.array([13,19,54,36])

plt.plot(x,y)
plt.plot(y)
plt.show() ## kütüphanein son sürümünde gerek yok ama alışkanlık haline getir .


##Labels

plt.title("Ana Başlık")

plt.ylabel("Y Ekseni")

plt.xlabel("x ekseni")

plt.grid()

plt.show()

## SubPlots birden fazla görselle birlikte grafik oluşturma

plt.subplot(1,3,1) ## ilk index oluşturulacak grafik sayısı aynı anda graikf oluşturma
## ikinci index toplam kaç tane oluşturulacak
## üçüncü index kaçıncı olarak oluşturacak soldan sağa sayarak kaçıncı olduğu




## seaborn görselleştirme


import  pandas as pd
import  seaborn as sns
from matplotlib import pyplot as plt

df=sns.load_dataset("titanic")
df.head()

df["sex"].value_counts()
sns.countplot(x=df["sex"], data = df)

plt.show()


sns.boxplot(x=df["total_bill"])
plt.show(
)

df["total_bill"].hist

plt.show()


### Gelişmiş Fonksiyonel Keşifci Veri Analizi (ADVANCED FUNCTIONAL EDA)

#1. Genel Resim
#2. Ktegorik Değişken Analizi(Analysis of Categorical Variables)
#3.Sayısal Değişken Analizi(Analysis of Numerical Variables)
#4.hedef değişekn analizi (Analysis of Target Variable).
#5. Korelasyon Analizi (Analysis Of Correlation)



## GENEL RESİM

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
## İlk bakış için fonksiyonlar
pd.set_option("display.max_columns", None)
pd. set_option("display.width", 500)
df = sns.load_dataset("tips")
df.head()
df.tail()
df.shape
df.info()  ## data tipleri kaç tane null değer ver koln isimlerini verir
df.columns
df.index
df.describe().T ## ortalama std 1/4 ü yarısı yüzde 75 i ve max değerlerini verir. T transpoze etmiş hali
df.isnull().values.any()
df.isnull().sum()


def check_df(dataframe, head= 5):
    print("#####SHAPE####")
    print(dataframe.shape)
    print("######TYPE#####")
    print(dataframe.dtypes)
    print("HEAADDD")
    print(dataframe.head)
    print("####TAİLLL####")
    print(dataframe.isnull().sum())

check_df(df)

df = sns.load_dataset("titanic")


sns.get_dataset_names()

sns.load_dataset("planets")




















