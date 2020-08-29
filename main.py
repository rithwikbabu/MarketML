from yahoo_fin.stock_info import get_data
import numpy as np
import pandas as pd
import re
from datetime import date
import itertools
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt

 # The Start and End date of the Data #
start = '01/01/2010'
end = '01/01/2020'

def percentagedivider_pos(ticker, start, end):
    data = get_data(ticker, start_date= start, end_date= end)
    
    dates = list(data.index)

    date1 = []
    for x in dates:
        date1.append(x.date())

    open1 = data['open']
    close1 = data['close']
    high1 = data['high']
    low1 = data['low']
    volume1 = data['volume']

    d1 =  {'Date': date1,
      'Open': open1,
      'Close': close1,
      'High': high1,
      'Low' : low1,
      'Volume' : volume1
      }

    odf = pd.DataFrame (d1, columns = ['Date', 'Open', 'Close', 'High', 'Low', 'Volume'])

    perchange = []
    for o,c,d in zip(odf['Open'], odf['Close'], odf['Date']):
        pchange = ( ( o - c ) / o ) * 100 
        perchange.append(round(pchange, 3))

    negperchange = []
    posperchange = []
    negdate = []
    posdate = []
    for p, d in zip(perchange, odf['Date']):
         if p > 0:
           posperchange.append(p)
           posdate.append(d)
         if p < 0:
           negperchange.append(p)
           negdate.append(d)

    posd2 = {'%Change' : posperchange,
             'Date' : posdate
            }

    negd3 = {'%Change' : negperchange,
             'Date' : negdate
            }

    posdf = pd.DataFrame (posd2, columns = ['Date', '%Change'])
    negdf = pd.DataFrame (negd3, columns = ['Date', '%Change'])
    return posdf
def percentagedivider_neg(ticker, start, end):
    data = get_data(ticker, start_date= start, end_date= end)

    dates = list(data.index)

    date1 = []
    for x in dates:
        date1.append(x.date())

    open1 = data['open']
    close1 = data['close']
    high1 = data['high']
    low1 = data['low']
    volume1 = data['volume']

    d1 =  {'Date': date1,
      'Open': open1,
      'Close': close1,
      'High': high1,
      'Low' : low1,
      'Volume' : volume1
      }

    odf = pd.DataFrame (d1, columns = ['Date', 'Open', 'Close', 'High', 'Low', 'Volume'])

    perchange = []
    for o,c,d in zip(odf['Open'], odf['Close'], odf['Date']):
        pchange = ( ( o - c ) / o ) * 100 
        perchange.append(round(pchange, 3))

    negperchange = []
    posperchange = []
    negdate = []
    posdate = []
    for p, d in zip(perchange, odf['Date']):
         if p > 0:
           posperchange.append(p)
           posdate.append(d)
         if p < 0:
           negperchange.append(p)
           negdate.append(d)

    posd2 = {'%Change' : posperchange,
             'Date' : posdate
            }

    negd3 = {'%Change' : negperchange,
             'Date' : negdate
            }

    posdf = pd.DataFrame (posd2, columns = ['Date', '%Change'])
    negdf = pd.DataFrame (negd3, columns = ['Date', '%Change'])
    return negdf
def dayupdown(ticker, start, end):                               
    data = get_data(ticker, start_date= start, end_date= end)
    dates = list(data.index)

    perpos = percentagedivider_pos(ticker, start, end)
    perneg = percentagedivider_neg(ticker, start, end)

    val = []
    dates1 = []
    for x in dates:
        for a, b, a1, b1 in zip(perneg['%Change'], perpos['%Change'], perneg['Date'], perpos['Date']):
            if x == a1:
                val.append(int(1))
                dates1.append(a1)
            if x == b1:
                val.append(int(0))
                dates1.append(b1)
    d1 = {'Date': dates1,
          'PosNeg': val,
         }

    pndf = pd.DataFrame (d1, columns = ['Date', 'PosNeg']).set_index('Date')

    return pndf           #pull dates from this, # 0 = Positive , 1 = Negative
def movingaverage(ticker, days, start, end): 
    data = get_data(ticker, start_date= start, end_date= end)

    dates = list(data.index)

    date1 = [] 
    for x in dates:
        date1.append(x.date())

    open1 = data['open']
    close1 = data['close']
    high1 = data['high']
    low1 = data['low']
    volume1 = data['volume']

    d1 =  {'Date': date1,
           'Close': close1
          }

    odf = pd.DataFrame (d1, columns = ['Date', 'Close'])

    base = 0
    base1 = base+days
    mova = []
    movadate = []
    nameave = str(days) + " DMA"
    for a, b in zip(odf['Date'], odf['Close']):

        numbers = odf['Close'][base-days:base]
        moving_average = (sum(numbers))/base1
        if base > base1:
            mova.append(moving_average)
            movadate.append(a)
        base += 1
    
    d2 = { 'Mav' : mova,
          'Date' : movadate
         }
    madf = pd.DataFrame (d2, columns = ['Date', 'Mav']).set_index('Date')

    return madf
def stockprice(ticker, start, end):
    data = get_data(ticker, start_date= start, end_date= end)

    dates = list(data.index)

    date1 = [] 
    for x in dates:
        date1.append(x.date())

    open1 = data['open']
    close1 = data['close']
    high1 = data['high']
    low1 = data['low']
    volume1 = data['volume']

    d1 =  {'Date': date1,
           'Price': close1
          }

    pdf = pd.DataFrame (d1, columns = ['Date', 'Price'])

    return pdf
def stockvolume(ticker, start, end):
    data = get_data(ticker, start_date= start, end_date= end)

    dates = list(data.index)

    date1 = [] 
    for x in dates:
        date1.append(x.date())

    open1 = data['open']
    close1 = data['close']
    high1 = data['high']
    low1 = data['low']
    volume1 = data['volume']

    d1 =  {'Date': date1,
           'Volume': volume1
          }

    vdf = pd.DataFrame (d1, columns = ['Date', 'Volume'])

    return vdf
def dates(ticker, start, end):
    data = get_data(ticker, start_date= start, end_date= end)

    dates = list(data.index)

    date1 = [] 
    for x in dates:
        date1.append(x.date())

    open1 = data['open']
    close1 = data['close']
    high1 = data['high']
    low1 = data['low']
    volume1 = data['volume']

    d1 =  {'Date': date1,
          }

    ddf = pd.DataFrame (d1, columns = ['Date'])

    return ddf
def tfdfmaker(ticker, start, end):
    vol = stockvolume(ticker, start, end)['Volume']
    pri = stockprice(ticker, start, end)['Price']
    mav55 = movingaverage(ticker, 55, start, end)['Mav']
    mav20 = movingaverage(ticker, 200, start, end)['Mav']
    chng = dayupdown(ticker, start, end)['PosNeg']
    d1 =  {'Price': pri,
           'Volume': vol,
           'Mav55': mav55,
           'Mav20': mav20,
           'Change': chng
          }

    tfdf = pd.DataFrame (d1, columns = ['Price', 'Volume', 'Mav55', 'Mav20', 'Change'])

    return tfdf[start:]
def get_compiled_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dropout(.5, input_shape=(2,)),
    tf.keras.layers.Dense(1)
  ])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
  return model


df = tfdfmaker('TSLA', start, end).dropna()

target = df.pop('Change')
dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))
train_dataset = dataset.shuffle(len(df)).batch(1)

model = get_compiled_model()
model.fit(train_dataset, epochs=10)
