## 기본 모델
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

## 모델 업데이트
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
## 그래프로 과적합 확인
![image](https://github.com/gryrryfh/AI-Class/assets/50912987/07eb2c44-00d5-4038-af17-45c37d9ebb93)
![image](https://github.com/gryrryfh/AI-Class/assets/50912987/bc16d9a9-47b7-470d-b189-52a9786ed6b4)
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
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 그래프 확인을 위한 긴 학습
history=model.fit(X_train, y_train, epochs=2000, batch_size=500, validation_split=0.25)

# history에 저장된 학습 결과 확인
hist_df=pd.DataFrame(history.history)
# y_vloss에 테스트셋의 오차를 저장
y_vloss=hist_df['val_loss']

# y_loss에 학습셋의 오차를 저장
y_loss=hist_df['loss']

# x 값을 지정하고 테스트셋의 오차를 빨간색으로, 학습셋의 오차를 파란색으로 표시
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, "o", c="red", markersize=2, label='Testset_loss')
plt.plot(x_len, y_loss, "o", c="blue", markersize=2, label='Trainset_loss')

plt.legend(loc='upper right')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
```
## 학습 자동중단
![image](https://github.com/gryrryfh/AI-Class/assets/50912987/7e55725f-4f2e-40ab-a638-84e06ebf37fb)
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import pandas as pd

df = pd.read_csv('C:/Users/jaegy/pythonProject2/data/wine.csv', header=None)
# 와인의 속성을 X로 와인의 분류를 y로 저장
X = df.iloc[:,0:12]
y = df.iloc[:,12]
#학습셋과 테스트셋으로 나눔
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
# 모델 구조 설정
model = Sequential()
model.add(Dense(30,  input_dim=12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
#모델을 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 학습이 언제 자동 중단 될지를 설정
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20)
modelpath="C:/Users/jaegy/pythonProject2/data/model/Ch14-4-bestmodel.hdf5"
# 최적화 모델 업데이트, 저장
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=0, save_best_only=True)
#모델 실행
history=model.fit(X_train, y_train, epochs=2000, batch_size=500, validation_split=0.25, verbose=1, callbacks=[early_stopping_callback,checkpointer])
# 테스트 결과 출력
score=model.evaluate(X_test, y_test)
print('Test accuracy:', score[1])
```
