import numpy as np
import pandas as pd
from sequentialModel import SequentialModel
from lstmLayer import LSTMLayer
from flatten import Flatten
from denseLayer import DenseLayer

model = SequentialModel(
    LSTMLayers = [
        LSTMLayer(
            size = 4,
            n_cell = 2
        )
    ],
    flatten = Flatten(),
    denseLayers = [
        DenseLayer(n = 10)
    ]
)

def reshapeForInput(data):
    arr = []
    for d in data:
        arr.append([d])
    return np.array(
        [
            arr
        ]
    )

def preprocess(df):
    dates = np.flip(df['Date'].to_numpy())
    opens = np.flip(df['Open'].to_numpy())
    highs = np.flip(df['High'].to_numpy())
    lows = np.flip(df['Low'].to_numpy())
    closes = np.flip(df['Close'].to_numpy())
    volumes = [int(val.replace(',','').replace('-','0')) for val in np.flip(df['Volume'].to_numpy())]
    market_caps = [int(val.replace(',','').replace('-','0')) for val in np.flip(df['Market Cap'].to_numpy())]

    return dates, opens, highs, lows, closes, volumes, market_caps

df_1 = pd.read_csv('./dataset/bitcoin_price_Training - Training.csv')
dates, opens, highs, lows, closes, volumes, market_caps = preprocess(df_1)
pred1 = model.predict(reshapeForInput(opens[-4:]))
pred2 = model.predict(reshapeForInput(highs[-4:]))
pred3 = model.predict(reshapeForInput(lows[-4:]))
pred4 = model.predict(reshapeForInput(volumes[-4:]))
pred5 = model.predict(reshapeForInput(market_caps[-4:]))

print("PREDICTED Data1: ", pred1, pred2, pred3, pred4, pred5)

df_2 = pd.read_csv('./dataset/bitcoin_price_1week_Test - Test.csv')
dates, opens, highs, lows, closes, volumes, market_caps = preprocess(df_2)
pred1 = model.predict(reshapeForInput(opens[-4:]))
pred2 = model.predict(reshapeForInput(highs[-4:]))
pred3 = model.predict(reshapeForInput(lows[-4:]))
pred4 = model.predict(reshapeForInput(volumes[-4:]))
pred5 = model.predict(reshapeForInput(market_caps[-4:]))

print("PREDICTED Data2: ", pred1, pred2, pred3, pred4, pred5)

model.printSummary()
