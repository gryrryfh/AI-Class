![image](https://github.com/gryrryfh/AI-Class/assets/50912987/e98a9db8-3041-42e1-af3c-e055e45b24b6)
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("C:/Users/jaegy/pythonProject2/data/wine.csv")

X = df.iloc[:,0:12]
y = df.iloc[:,12]
#학습셋과 테스트셋으로 나눔
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# 모델 구조(relu, sigmoid 사용)
model = Sequential()
model.add(Dense(30,  input_dim=12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

#모델 컴파일, 실행
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history=model.fit(X_train, y_train, epochs=50, batch_size=500, validation_split=0.25)
score=model.evaluate(X_test, y_test)
print('Test accuracy:', score[1])
```
![image](https://github.com/gryrryfh/AI-Class/assets/50912987/f299a42c-cc4c-4fb2-89d5-58f7a8dbc99b)
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/jaegy/pythonProject2/data/wine.csv", header=None)
X = df.iloc[:,0:12]
y = df.iloc[:,12]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# 모델 구조(relu, sigmoid 사용)
model = Sequential()
model.add(Dense(30,  input_dim=12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

#모델 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 모델 저장
modelpath="C:/Users/jaegy/pythonProject2/data/model/{epoch:02d}-{val_accuracy:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, verbose=1)
# 모델 실행
history=model.fit(X_train, y_train, epochs=50, batch_size=500, validation_split=0.25, verbose=0, callbacks=[checkpointer])

# 결과 출력
score=model.evaluate(X_test, y_test)
print('Test accuracy:', score[1])
```

![image](https://github.com/gryrryfh/AI-Class/assets/50912987/07eb2c44-00d5-4038-af17-45c37d9ebb93)
