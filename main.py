import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
import datetime

def StartARIMAForecasting(actual, p, d, q):
    model = ARIMA(actual, order=(p,d,q))
    model_fit = model.fit(disp=0)
    prediction = model_fit.forecast()[0]
    return prediction


fig,g =  plt.subplots(2,2)

#---------------------------------------------------------------------------------
#Прогноз по России
df = pd.read_csv('RUS.csv', sep=' ')
df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
date = []
infected = []
for i in range(0,len(df)):
    date.append(df['date'][i])
    infected.append(df['infected'][i])

z = df.describe()
a = date[-1] + datetime.timedelta(days=46)

while date[-1] != a:
    predicted = StartARIMAForecasting(infected, 1, 1, 0)
    infected.append(predicted)
    date.append(date[-1] + datetime.timedelta(days=1))

g[0][0].plot(date,infected, 'r')
g[0][0].plot(df['date'],df['infected'])
g[0][0].set_title('Прогноз по России')


#---------------------------------------------------------------------------------
#Прогноз по Новосибирску
df = pd.read_csv('NSK.csv', sep=' ')
df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
date = []
infected = []
for i in range(0,len(df)):
    date.append(df['date'][i])
    infected.append(df['infected'][i])

x = df.describe()
a = date[-1] + datetime.timedelta(days=46)

while date[-1] != a:
    predicted = StartARIMAForecasting(infected, 1, 1, 0)
    infected.append(predicted)
    date.append(date[-1] + datetime.timedelta(days=1))

g[1][0].plot(date,infected, 'r')
g[1][0].plot(df['date'],df['infected'])
g[1][0].set_title('Прогноз по Новосибирску')

#---------------------------------------------------------------------------------
#Прогноз по США
df = pd.read_csv('USA.csv', sep=' ')
df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
date = []
infected = []
for i in range(0,len(df)):
    date.append(df['date'][i])
    infected.append(df['infected'][i])

c = df.describe()
a = date[-1] + datetime.timedelta(days=46)

while date[-1] != a:
    predicted = StartARIMAForecasting(infected, 1, 1, 0)
    infected.append(predicted)
    date.append(date[-1] + datetime.timedelta(days=1))

g[1][1].plot(date,infected, 'r')
g[1][1].plot(df['date'],df['infected'])
g[1][1].set_title('Прогноз по США')
plt.show()
