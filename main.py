from numpy.core.fromnumeric import var
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.core.construction import array
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import euclidean

# pembuatan data frame
data = {
  'tinggiBadan':['158','158','158','160','160','163','163','160','163','165','165','165','168','168','168','170','170','170'],
  'beratBadan':['58','59','63','59','60','60','61','64','64','61','62','65','62','63','66','63','64','68'],
  'ukuranBaju':['M','M','M','M','M','M','M','L','L','L','L','L','L','L','L','L','L','L'] 
}
dataFrame = pd.DataFrame(data)

# visualisasi data
# fig, ax = plt.subplots()
# for jk,d in dataFrame.groupby('ukuranBaju') :
#   ax.scatter(d['tinggiBadan'], d['beratBadan'], label=jk)

# plt.legend(loc='upper left')
# plt.title('Sebaran Data')
# plt.xlabel('Tinggi badan')
# plt.ylabel('Berat Badan (kg)')
# plt.grid(True)
# plt.show()

# preprocesing data
x_train =np.array(dataFrame[['tinggiBadan','beratBadan']])
y_train =np.array(dataFrame['ukuranBaju'])

lb = LabelBinarizer()
y_train = lb.fit_transform(y_train).flatten()

# print(f'y_train:\n{y_train}')
# print(f'x_train:\n{x_train}')

# 

#pengambilan tetangga terdekat dari nilai eucledian data set 
k = 3
model = KNeighborsClassifier(n_neighbors=k)
model.fit(x_train,y_train)

#input data untuk melakukan prediksi 
tBadan = 155
bBadan = 70
arrayData = np.array([tBadan,bBadan]).reshape(1,-1)
# print(arrayData)
# jarakEuclidean = [euclidean(arrayData,d) for d in x_train] #d = hasil eucleadian pada tiap titik


prediksi = model.predict(arrayData) #Menghasilkan variabel boolean dari data y_train

#print hasil prediksi sepat
print(lb.inverse_transform(prediksi))
