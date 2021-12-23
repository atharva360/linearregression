import numpy as np
import pandas_datareader as data
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('bmh')
import streamlit as st


st.title('STOCK MARKET PREDICTION USING LINEAR REGRESSION MODEL')
user_input = st.text_input('Enter The Stock Ticker', 'AAPL')
start = '2019-01-01'
end = '2021-12-15'
df = data.DataReader(user_input,'yahoo',start,end)
#set the date as the index
# df = df.set_index(data.DatetimeIndex(df['Date'].values))

#describing data
st.subheader('Data From 2010-2021')
st.write(df.describe())

# df.shape

st.subheader('Closing Price vs Time Chart') 
fig = plt.figure(figsize=(12,6))
plt.title(user_input)
plt.xlabel('Year-Month')
plt.ylabel('Close Price USD ($)')
plt.plot(df['Close'])
st.pyplot(fig)

df = df[['Close']]
# df.head(4)


future_days = 25
df['Prediction'] = df[['Close']].shift(-future_days)
# df.head(4)

X = np.array(df.drop(['Prediction'], 1))[:-future_days]
# print(X)

y = np.array(df['Prediction'])[:-future_days]
# print(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

lr = LinearRegression().fit(x_train, y_train)

x_future = df.drop(['Prediction'], 1)[:-future_days]
x_future = x_future.tail(future_days)
x_future = np.array(x_future)
# x_future


# print()
lr_prediction = lr.predict(x_future)
# print(lr_prediction)



#prediction using linear regression 
st.subheader('Linear Regression Prediction') 
predictions = lr_prediction
valid = df[X.shape[0]:]
valid['Predictions'] = predictions
figlr = plt.figure(figsize = (12,6))
plt.title('Linear Regression')
plt.xlabel('Year-Month')
plt.ylabel('Close Price USD ($)')
plt.plot(df['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Orig', 'Val', 'Pred'])
st.pyplot(figlr)
