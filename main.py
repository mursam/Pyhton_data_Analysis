print('Hello worodl')



# def function _name


def say_hi(boolean):
    print(boolean)
    print('Hİ')
    print('Murat')


say_hi(True)



def multiplication(a,b):
    c= a*b
    print(c)

multiplication(6.75,7.55)


list_store=[]

def listeyeekleme(a,b):
    c=a*b
    list_store.append(c)
    print(list_store)

listeyeekleme(15,2)
listeyeekleme(18,23)

#Ön tanımlı argüman

def divide(a,b):
    print(a/b)

divide(8,9)

# Ne zaman fonksiyon yazılır ?


def prediction(wind,speed,moisture):
    rain = ((wind* speed) /moisture) * 100
    print(rain)

prediction(12,10,100)



## Return : Fonksiyonun çıktısını Girdi olarak kullanır


def prediction(heat, windspeed, moisture):
    heat = heat - 273
    windspeed = windspeed * 3600
    moisture = moisture / 100
    rain = ((heat * windspeed) / moisture) * 100
    return heat, windspeed , moisture, rain ## değeri fonksiyonda yeniden kullanılacak hale getirir ve kendindnen sonrak fonksiyon durur


type(prediction(12,14,15))

prediction(12,15,15)

heat , windspeed , moisture, rain = prediction(12,15,15)

## Fonksiyon çağırmak


def prediction(heat, windspeed, moisture):
    rain = ((heat * windspeed) / moisture) * 100
    return  rain
prediction(12,15,13)
def standardization(a,b):
    return a * 10 / 100 * b* b

standardization(12,1)

def all_calculatioan(heat,windspeed,moisture, b):
    a = prediction(heat,windspeed,moisture)
    c = standardization(a,b)
    print(b*10)

all_calculatioan(1,2,3,4)

#Koşullar

# if

if 1==1 :
    print('Healal lan ')

if 1 ==2 :
    print('Helal lan')

number = 11

if number > 12:
    print('Sayı 12 den büyük')


def number_check(number):
    if number == 11:
        print('doğru sayı')
    else :
        print('yanlış sayı')
number_check(112)


### DÖNGÜLER

students = ["Murat", "Ceren", "Fuat", "Kürşad"]

for student in  students:
    print(student)

for student in students :
    print(student.upper())

salaries =[8000,9000,10000,11000]


def new_salary(salary, rate):
    return int(salary*rate/100 + salary)

new_salary(10000,50)

for salary in salaries:
    print(new_salary(salary, 27))

for salary in salaries :
    if salary>=1000:
        print(new_salary(salary,15))
    else :
        print(new_salary(salary,25))

range(len("murat"))

def alternating(string):
    new_string = ""
    for string_index in range(len(string)):
        if string_index % 2 == 0 :
            new_string +=  string[string_index].upper()
        else :
            new_string += string[string_index].lower()
    print(new_string)


alternating("Murat Samancı")


## break & continue & while


for salary in salaries:
    if salary == 10000:
        break
    print(salary)

for salary in salaries :
    if salary == 10000:
        continue
    print(salary)

number = 1
while number < 5 :
    print(number)
    number += 1

# Enumerate : Otomatik Counter/Indexer ile for loop

students = ["Casillas", "Rüştü", "Muslera", "Altay", "Volkan"]

for index, student in enumerate(students):
    print(index, student)


A=[]
B=[]


for index, student in enumerate(students):
    if index % 2 == 0 :
        A.append(student)
    else :
        B.append(student)


print(A)
print(B)


def divide_student(students):
    groups=[[],[]]
    for index, student in enumerate(students):
        if index % 2 == 0 :
            groups[0].append(student)
        else:
            groups[1].append(student)
    print(groups)
    return groups

st = divide_student(students)

st[0]
st[1]


##alternatin fonksiyonun enumerate ile yazılması


def alternating_with_enumerate(string):
    new_string = ""
    for i, letter in enumerate(string):
        if i % 2 == 0 :
            new_string += letter.upper()
        else:
            new_string += letter.lower()
    print(new_string)

alternating_with_enumerate("Şampiyon GalataSaray")


## ZİP


students = ["Casillas", "Rüştü", "Muslera", "Van der Saar", "Oblak"]
salaries =[8000,9000,10000,11000]
country = ["İspanya", "Türkiye", "Uruguay","Hollanda" , "Slovakya"]


list(zip(students,salaries,country))


#### lambda, map, filter, reduce



def new(a,b):
    return a +b

new(3,5) * 9


new_sum = lambda a,b : a**b ## kullan at fonksiyondur değişkenleri tutmaz

new_sum(2,8)


# map -- döngü yazmayı kolaylaştırır , for un içine fonksiyon yazmanın kolay yoludur.
salaries =[8000, 9000, 10000, 11000]

def new_salary(s):
    return s*0.2 + s

new_salary(10000)

list(map(new_salary,salaries))


list(map(lambda x : x * 0.2 + x , salaries))


## FILTER

list_store = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
list(filter(lambda x: x%2 == 0, list_store))

## REDUCE

from functools import reduce
list_store = [1,2,3,4]
reduce(lambda a,b : a+b , list_store)



### COMPREHENSIONS


## List Comprehension

salaries =[8000,9000,10000,11000, 12000, 15000, 20000]


def new_salary (x ):
    return x*0.2 + x
for salary in salaries :
    print(new_salary(salary))

null_list = []

for salary in salaries :
    null_list.append(new_salary(salary))

null_list

for salary in salaries :
    if salary>10000:
        null_list.append(new_salary(salary))
    else:
        null_list.append(new_salary(salary*0.45 + salary))



[salary * 2 for salary in salaries if salary < 10000] ## if tek başına ise en sağda


[salary * 2 if salary<10000 else salary * 0.5 + salary for salary in salaries]
## if ve else birlikte olarak kullanılacaksa sol tarfata olmalıdır


## Dict Comprehension

dictionary = {'a':1, 'b':2, 'c':3, 'd': 4}

dictionary.keys()
dictionary.values()
dictionary.items()

{k:v ** 2 for (k,v) in dictionary.items()}

{k.upper() : v*2 for (k,v) in dictionary.items()}


#### Mülakat soruları

## Amaç : Çift sayıların karesi alınarak bir sözlüğe eklenmek istenmektedir.
## Keyler orjinal valueler işlem görmüş hali olacak

numbers= range(20)
new_dict = {}


for number in numbers :
    if number % 2 == 0 :
        new_dict[number] = number ** 2


{n : n**2 for n in numbers if n %2 == 0 }


### List & Dictironary Compherension


## Bir veri setindeki değişken isimlerini değiştirme


import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

for col in df.columns:
    print(col.upper())


df.columns   = [col.upper() for col in df.columns]
df.columns

## İsminde 'INS ' olanların başına FLAG diğerlerine NO_FLAG eklemek istiyoruz .



[col for col in df.columns if "INS" in col]

["FLAG_" + col for col in df.columns if "INS" in col]

df.columns = ["FLAG_" + col if "INS" in col else "NO_FLAG_" + col  for col in df.columns]


df.columns

### Amaç key'i stirng values u liste olan bir liste oluşturmak
## sadece sayısal değişkenler için yapıalcak

######################
import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns


num_cols = [col for col in df.columns if df[col].dtype != "O"]
soz = {}
agg_list = ["mean", "min", "max", "sum"]

for col in num_cols:
    soz[col] = agg_list

neew_dictionary = {col : agg_list for col in num_cols}

df[num_cols].head()

df[num_cols].agg(neew_dictionary)
## dataframedeki dictonry içinde olan fonksiyonları uygularayarak detay çıkarır.
######################



