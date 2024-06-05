## 데이터 파악
![image](https://github.com/gryrryfh/AI-Class/assets/50912987/08d04782-29bb-4685-8be2-fb82a4b68e1a)

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
#데이터 불러오기
df = pd.read_csv("C:/Users/jaegy/pythonProject2/data/house_train.csv")
#카테고리형 변수를 0과 1로 이루어진 변수로 바꿈
df = pd.get_dummies(df)
#결측치를 전체 칼럼의 평균으로 대체
df = df.fillna(df.mean())
print(df)
```
## 속성별 관련도 추출
![image](https://github.com/gryrryfh/AI-Class/assets/50912987/9fe5ee0a-7b88-4612-a470-6ed5f7fe324e)
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
#데이터 불러오기
df = pd.read_csv("C:/Users/jaegy/pythonProject2/data/house_train.csv")
#카테고리형 변수를 0과 1로 이루어진 변수로 바꿈
df = pd.get_dummies(df)
#결측치를 전체 칼럼의 평균으로 대체
df = df.fillna(df.mean())
#데이터 사이의 상관 관계 저장
df_corr=df.corr()
#집 값과 관련이 큰 것부터 순서대로 저장
df_corr_sort=df_corr.sort_values('SalePrice', ascending=False)
#집 값과 관련도가 가장 높은 속성들을 추출한 상관도 그래프
cols=['SalePrice','OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF']
sns.pairplot(df[cols])
plt.show();
```
## 주택가격 예측 모델
![image](https://github.com/gryrryfh/AI-Class/assets/50912987/82baa21b-9084-467b-a9ee-63ed615d707d)
![image](https://github.com/gryrryfh/AI-Class/assets/50912987/f892fd06-da4f-4a0f-9015-573075734f93)
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
#데이터 불러오기
df = pd.read_csv("C:/Users/jaegy/pythonProject2/data/house_train.csv")
#카테고리형 변수를 0과 1로 이루어진 변수로 바꿈
df = pd.get_dummies(df)
#결측치를 전체 칼럼의 평균으로 대체
df = df.fillna(df.mean())
#데이터 사이의 상관 관계 저장
df_corr=df.corr()
#집 값과 관련이 큰 것부터 순서대로 저장
df_corr_sort=df_corr.sort_values('SalePrice', ascending=False)
#집 값을 제외한 나머지 열을 저장합니다.
cols_train=['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF']
X_train_pre = df[cols_train]

#집 값을 저장합니다.
y = df['SalePrice'].values
#전체의 80%를 학습셋으로, 20%를 테스트셋으로 지정합니다.
X_train, X_test, y_train, y_test = train_test_split(X_train_pre, y, test_size=0.2)
#모델의 구조를 설정합니다.
model = Sequential()
model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(1))
model.summary()

#모델을 실행합니다.
model.compile(optimizer ='adam', loss = 'mean_squared_error')

# 20회 이상 결과가 향상되지 않으면 자동으로 중단되게끔 합니다.
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20)

# 모델의 이름을 정합니다.
modelpath="C:/Users/jaegy/pythonProject2/data/model/Ch15-house.hdf5"

# 최적화 모델을 업데이트하고 저장합니다.
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=0, save_best_only=True)

#실행 관련 설정을 하는 부분입니다. 전체의 20%를 검증셋으로 설정합니다.
history = model.fit(X_train, y_train, validation_split=0.25, epochs=2000, batch_size=32, callbacks=[early_stopping_callback, checkpointer])
# 예측 값과 실제 값, 실행 번호가 들어갈 빈 리스트를 만듭니다.
real_prices =[]
pred_prices = []
X_num = []

# 25개의 샘플을 뽑아 실제 값, 예측 값을 출력해 봅니다.
n_iter = 0
Y_prediction = model.predict(X_test).flatten()
for i in range(25):
    real = y_test[i]
    prediction = Y_prediction[i]
    print("실제가격: {:.2f}, 예상가격: {:.2f}".format(real, prediction))
    real_prices.append(real)
    pred_prices.append(prediction)
    n_iter = n_iter + 1
    X_num.append(n_iter)
    # 그래프를 통해 샘플로 뽑은 25개의 값을 비교해 봅니다.

    plt.plot(X_num, pred_prices, label='predicted price')
    plt.plot(X_num, real_prices, label='real price')
    plt.legend()
    plt.show()
```

