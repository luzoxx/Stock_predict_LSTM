import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import date, timedelta
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


from plotly import graph_objs as go


START = "2014-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Trend Prediction")


user_input = st.text_input('Enter Stock Ticker')
stock_info = yf.Ticker(user_input).info
company_name = stock_info['shortName']
st.subheader(company_name)



#load data
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data
data = load_data(user_input)

st.write(data.tail())
#visual data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text=f'Data of {company_name} from 2014', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
plot_raw_data()



st.subheader(f'The moving average of {company_name}')
ma_day = [10, 50, 200]
fig1 = plt.figure(figsize = (12,6))
for ma in ma_day:
      column_name = f"MA for {ma} days"
      data[column_name] = data['Close'].rolling(ma).mean()
      plt.plot(data.index, data[column_name], label=column_name)
plt.legend()
plt.show()
st.pyplot(fig1)


# data process

data['Date'] = pd.to_datetime(data['Date'])
data.index = data.pop('Date')

target = data['Close']
df = pd.DataFrame(target)

scaler = MinMaxScaler(feature_range = (0,1))
# to get a col vector of data
sc_data = scaler.fit_transform(np.array(df).reshape(-1, 1))

train_size = int(len(df)* 0.85)
test_size = len(df) - train_size

date_train = int(len(df.index)* 0.85)
date_test = len(df.index) - date_train

train_data = sc_data[ :train_size, 0: 1]
test_data = sc_data[train_size-3: , 0: 1]

tr_Xs = []
tr_y = []
tr_date = []
for i in range(3, len(train_data)):
    tr_Xs.append(train_data[i-3: i,0 ])
    tr_y.append(train_data[i, 0])
    tr_date.append(df.index[i])

tr_Xs, tr_y = np.array(tr_Xs), np.array(tr_y)
tr_Xs = np.reshape(tr_Xs, (tr_Xs.shape[0], tr_Xs.shape[1], 1))
tr_date = np.array(tr_date)


ts_Xs = []
ts_y = []
ts_date = []
for i in range(3, len(test_data)):
    ts_Xs.append(test_data[i-3: i, 0])
    ts_y.append(test_data[i, 0])
    ts_date.append(df.index[len(train_data)+i-3])
ts_Xs, ts_y = np.array(ts_Xs), np.array(ts_y)
ts_Xs = np.reshape(ts_Xs, (ts_Xs.shape[0], ts_Xs.shape[1], 1))
ts_date = np.array(ts_date)

model = load_model('N225_model.h5')

rs = model.fit(tr_Xs, tr_y, epochs=100)


p = model.predict(ts_Xs)
ps = scaler.inverse_transform(p)

train = df.iloc[ :train_size , 0:1]
test = df.iloc[train_size: , 0:1]
test['Predictions'] = ps




st.subheader("Predictions vs Original")
fig2= plt.figure(figsize = (12,6))
plt.plot(train['Close'], linewidth= 3)
plt.plot(test['Close'], linewidth= 3)
plt.plot(test["Predictions"], linewidth= 3)
plt.xlabel('Date', fontsize= 18)
plt.ylabel('Close Price', fontsize= 18)
plt.legend(['Train', 'Test', 'Predictions'])
st.pyplot(fig2)

# create a function to insert ends into  forcasted data
def insert_end(Xin, new_input):
    timestep = 3 # each timestep include 59 Xs and 1 y
    for i in range(timestep - 1):
        Xin[:, i, :] = Xin[:, i+1, :]
    Xin[:, timestep - 1, :] = new_input
    return Xin


n_months = st.slider(' Months of prediction:', 1, 12)

future = n_months * 30
forcast = []
Xin = ts_Xs[-1 :, :, :] #SLICING INPUT (Xin) like tx_Xs -->> 3Dims
time = []
for i in range(0, future):
    out = model.predict(Xin, batch_size=5) # 
    forcast.append(out[0, 0]) 
    print(forcast)
    Xin = insert_end(Xin, out[0, 0]) 
    time.append(pd.to_datetime(df.index[-1]) + timedelta(days=i)) #Append generated datetime in 'time' list

later = np.array(forcast) # CREATE aN ARRAY OF FORCAST LIST THEN RESHAPE IT TO A VECTOR COLUMN
later = np.array(forcast).reshape(-1, 1)
later = scaler.inverse_transform(later) 

later = pd.DataFrame(later)

# CONVERT TIME LIST TO DATAFRAME
datetime = pd.DataFrame(time)

# Concat two dfs 'later30', 'datetime'
nextmonth = pd.concat([datetime, later], axis = 1)
nextmonth.columns = 'Datetime', 'Close price later'



st.subheader(f'Predictions for {n_months} month(s)')
fig3= plt.figure(figsize = (12,6))
plt.xlabel('Datetime', fontsize=18)
plt.ylabel('Close price' ,fontsize=18)
plt.plot(df['Close'])
plt.plot(nextmonth.set_index('Datetime')[['Close price later']])
st.pyplot(fig3)




st.subheader(f'Zoom of predictions')
fig4= go.Figure()
fig4.add_trace(go.Scatter(x=nextmonth['Datetime'], y=nextmonth['Close price later'], name="stock_close"))
fig4.layout.update(xaxis_rangeslider_visible=True)
st.plotly_chart(fig4)