# stock-price-prediction-using-machine-learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf

#pip install yfinance......for installation of yfinance 

tickers = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]['Symbol']

print(tickers)
for ticker in tickers[0:1]:
    end_date = datetime.now()
    start_date = end_date - timedelta(days = 15 * 365)
    
    history = yf.download(ticker, start = start_date, end = end_date, interval = '1d', prepost = False)
    display(history)
    for ticker in tickers[0:1]:
    end_date = datetime.now()
    start_date = end_date - timedelta(days = 15 * 365)
    
    history = yf.download(ticker, start = start_date, end = end_date, interval = '1d', prepost = False)
    history = history.loc[:,['Open', 'Close', 'Volume']]
    display(history)
    for ticker in tickers[0:1]:
    end_date = datetime.now()
    start_date = end_date - timedelta(days = 15 * 365)
    
    history = yf.download(ticker, start = start_date, end = end_date, interval = '1d', prepost = False)
    history = history.loc[:,['Open', 'Close', 'Volume']]
    
    history['Prev_Close'] = history.loc[:, 'Close'].shift(1)
    history['Prev_Volume'] = history.loc[:, 'Volume'].shift(1)
    
    datetimes = history.index.values
    weekdays = []
    
    for dt in datetimes:
        dt = datetime.strptime(str(dt), '%Y-%m-%dT%H:%M:%S.000000000')
        weekdays.append(dt.weekday())
        
    history['weekday'] = weekdays
    display(history)
    
history['5SMA'] = history['Prev_Close'].rolling(5).mean()
history['10SMA'] = history['Prev_Close'].rolling(10).mean()
history['50SMA'] = history['Prev_Close'].rolling(50).mean()
history['100SMA'] = history['Prev_Close'].rolling(100).mean()
history['200SMA'] = history['Prev_Close'].rolling(200).mean()
    
x = history.index.values


plt.figure(figsize = (15,5))
plt.plot(x, history['Prev_Close'], color = 'blue')
plt.plot(x,history['5SMA'], color = 'Pink')
plt.plot(x,history['10SMA'], color = 'yellow')
plt.plot(x,history['50SMA'], color = 'orange')
plt.plot(x,history['100SMA'], color = 'green')
plt.plot(x, history['200SMA'], color = 'red')
plt.show()    
plt.figure(figsize = (15, 3))
# display(history)



#Importing Linear Regression model
from sklearn.linear_model import LinearRegression


y = history['Close']
X = history.drop(['Close', 'Volume'], axis = 1).values

#Splitting into training and testing
num_test = 365
X_train = X[:-1 * num_test]
y_train = y[:-1 * num_test]
X_test = X[-1 * num_test:]
y_test = y[-1 * num_test:]

model = LinearRegression()
model = model.fit(X_train,y_train)
preds = model.predict(X_test)

print(ticker)
plt.figure(figsize=(15, 5))
plt.plot(range(len(y_test)), y_test, 'blue')
plt.plot(range(len(preds)), preds, 'red')
plt.show()
from sklearn.linear_model import LinearRegression

#Calculating the price of the stock
def test_it(opens, Closes, preds, start_account = 1000, thresh = 0):
    account = start_account
    changes = []
    
    for i in range(len(preds)):
        if(preds[i] - opens[i]) / opens[i] >= thresh:
            account += account * (Closes[i] - opens[i]) / opens[i]
        changes.append(account)
    changes = np.array(changes)
    
    plt.plot(range(len(changes)), changes)
    plt.show()
    
    invest_total = start_account + start_account * (Closes[-1] - opens[0]) / opens[0]
    print('Investing_Total:', invest_total, str(round((invest_total - start_account) / start_account * 100,1))+'%')
    print('Algo-Trading Total:', account, str(round((account - start_account) / start_account * 100,1))+'%')



def calc_macd(date, len1, len2, len3):
    shortEMA = date.ewm(span = len1, adjust = False).mean()
    longEMA = date.ewm(span = len2, adjust = False).mean()
    MACD = shortEMA - longEMA
    signal = MACD.ewm(span = len3, adjust = False).mean()
    return MACD, signal

for ticker in tickers[200:210]:
    end_date = datetime.now()
    start_date = end_date - timedelta(days = 15 * 365)
    
    history = yf.download(ticker, start = start_date, end = end_date, interval = '1d', prepost = False)
    history = history.loc[:,['Open', 'Close', 'Volume']]
    
    history['Prev_Close'] = history.loc[:, 'Close'].shift(1)
    history['Prev_Volume'] = history.loc[:, 'Volume'].shift(1)
    
    datetimes = history.index.values
    weekdays = []
    
    for dt in datetimes:
        dt = datetime.strptime(str(dt), '%Y-%m-%dT%H:%M:%S.000000000')
        weekdays.append(dt.weekday())
        
    history['weekday'] = weekdays
#     display(history)
    #Mean values of Simple moving average 
    history['5SMA'] = history['Prev_Close'].rolling(5).mean()
    history['10SMA'] = history['Prev_Close'].rolling(10).mean()
    history['20SMA'] = history['Prev_Close'].rolling(20).mean()
    history['50SMA'] = history['Prev_Close'].rolling(50).mean()
    history['100SMA'] = history['Prev_Close'].rolling(100).mean()
    history['200SMA'] = history['Prev_Close'].rolling(200).mean()
    
    
    MACD, signal = calc_macd(history['Prev_Close'], 12, 26, 9)
    history['MACD'] = MACD
    history['MACD_signal'] = signal
    history = history.replace(np.inf, np.nan).dropna()
    X = history.drop(['Close', 'Volume'], axis = 1).values
    y = history['Close']

    #Splitting model into training and testing 
    num_test = 365
    X_train = X[:-1 * num_test]
    y_train = y[:-1 * num_test]
    X_test = X[-1 * num_test:]
    y_test = y[-1 * num_test:]

    model = LinearRegression()
    model = model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(ticker)
    test_it(X_test.T[0], y_test, preds, 1000, 0) 
