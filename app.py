import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import math, datetime

style.use("ggplot")

api_key = "vsZYB2Gkm3sqnwLi3ohN"
quandl.ApiConfig.api_key = api_key

df = quandl.get('BITFINEx/BTCUSD')
df.dropna(inplace=True)
df['HL_PCT'] = (df['High'] - df['Low']) / df['Last'] * 100.0
df['ASKBID_PCT'] = (df['Ask'] - df['Low']) / df['Ask'] * 100.0

df = df[['High','Low','Last','HL_PCT','ASKBID_PCT','Volume']]

forecast_out = int(math.ceil(len(df) * 0.01))
df['Label'] = df['Last'].shift(-forecast_out+1)

x = df.drop(columns='Label')
x = scale(x)
y = df.iloc[:,-1]

x_toPredict = x[-forecast_out:]
x = x[:-forecast_out]
y = y[:-forecast_out]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=0)

regressor = LinearRegression()
regressor.fit(x_train, y_train)
accuracy = regressor.score(x_test, y_test)

prediction_set = regressor.predict(x_toPredict)
df['Prediction'] = np.nan

last_date = df.iloc[-1].name
lastDatetime = last_date.timestamp()
one_day = 3600 * 24
nextDatetime = lastDatetime + one_day

onceki_veri = len(df)
for i in prediction_set:
    next_date = datetime.datetime.fromtimestamp(nextDatetime)
    nextDatetime += one_day
    df.loc[next_date] = [np.nan for q in range(len(df.columns) - 1 )] + [i]
    last_deger = df.iloc[-1,7]
    df.iloc[-1,2] = last_deger
    


df['Prediction'] = df['Last']
df.iloc[0:onceki_veri+1]['Last'].plot(color='b')
df.iloc[onceki_veri:len(df)+1]['Prediction'].plot(color='r')


plt.xlabel('Date')
plt.ylabel('USD(Price)')
plt.legend(loc=4)
plt.show()

print(accuracy)