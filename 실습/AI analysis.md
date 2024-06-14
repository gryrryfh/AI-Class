## 대구 날씨 딥러닝 모델
```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.model_selection import  train_test_split
import time
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from numpy import newaxis
import seaborn as sns
train = pd.read_csv("C:/Users/jaegy/pythonProject2/data/daegutemp_train3yrs.csv")
test = pd.read_csv("C:/Users/jaegy/pythonProject2/data/2024daegu.csv")

temp_train = train.iloc[:,2:3]
temp_test = test.iloc[:,2:3]

#값을 0과 1 사이로 Scaling
from sklearn.preprocessing import MinMaxScaler
ss= MinMaxScaler(feature_range=(0,1))
temp_train= ss.fit_transform(temp_train)
temp_test= ss.fit_transform(temp_test)

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)
look_back = 1
trainX, trainY = create_dataset(temp_train, look_back)
testX, testY = create_dataset(temp_test, look_back)
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1],1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1],1))

# create and fit the LSTM network

model_temp = Sequential()
#Adding the first LSTM layer and some Dropout regularisation
model_temp.add(LSTM(units = 100, return_sequences = True, input_shape = (trainX.shape[1], 1)))
model_temp.add(Dropout(0.2))
# Adding a second LSTM layer and some Dropout regularisation
model_temp.add(LSTM(units = 100, return_sequences = True))
model_temp.add(Dropout(0.2))
# Adding a third LSTM layer and some Dropout regularisation
model_temp.add(LSTM(units = 100, return_sequences = True))
model_temp.add(Dropout(0.2))
# Adding a fourth LSTM layer and some Dropout regularisation
model_temp.add(LSTM(units = 50))
model_temp.add(Dropout(0.2))

# Adding the output layer
model_temp.add(Dense(units = 1))
model_temp.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mae', 'mse'])
#학습을 시키는 fit 함수.
#trainX : 입력값, trainY: 입력된 값들이 실제 출력되기를 바라는 값
#epochs: 학습시키는 크기, batch_size: 학습시킬때 묶음. 샘플의 수
#32 묶음 씩 100번을 학습시킴
model_temp.fit(trainX, trainY, epochs = 100, batch_size = 32)





prediction = model_temp.predict(testX)
prediction = ss.inverse_transform(prediction)
temp_test = ss.inverse_transform(temp_test)
plt.figure(figsize=(20,10))
idx_list = list(range(len(test)))
plt.xticks(idx_list)
plt.plot(temp_test, color = 'black', label = 'actual Temperature')
plt.plot(prediction, color = 'green', label = 'Predicted Temperature')
plt.title('Temp Prediction')
plt.xlabel('days')
plt.ylabel('avr temp')
plt.legend()
plt.show()
```

## 실행결과
![ai1](https://github.com/gryrryfh/AI-Class/assets/50912987/d816eb1e-9b48-4cc4-9570-f3d6eff33a9d)

