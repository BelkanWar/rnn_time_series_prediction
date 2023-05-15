'''
https://peaceful0907.medium.com/time-series-prediction-lstm%E7%9A%84%E5%90%84%E7%A8%AE%E7%94%A8%E6%B3%95-ed36f0370204
'''

import utils
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# parameters
TRAIN_LENGTH = 100
LOOKBACK_LENGTH = 10
EPOCH = 3000

data = pd.read_csv("data/AirPassengers.csv")
data['Month'] = pd.to_datetime(data['Month'], format='%Y-%m')
data['#Passengers'] = data['#Passengers'].astype(float)
data = data.sort_values(by='Month', ascending=True)

data_X, data_Y = [], []

for idx in range(LOOKBACK_LENGTH+1, TRAIN_LENGTH):
    data_X.append(data.loc[idx-LOOKBACK_LENGTH-1:idx-1, '#Passengers'].tolist())
    data_Y.append(data.loc[idx-LOOKBACK_LENGTH:idx, '#Passengers'].tolist())

loader = utils.convert2Loader(data_X, data_Y)
model = utils.continus_Model()
model = utils.training(model, loader, EPOCH)


with torch.inference_mode():
    plot = {
        'value':data['#Passengers'].tolist(),
        'type':['ground truth']*data.shape[0],
        'X':[i for i in range(data.shape[0])]}
    
    series = data['#Passengers'].tolist()[:TRAIN_LENGTH]
    tensor = torch.tensor(series).reshape(1, -1, 1)
    predict = model(tensor).flatten().tolist()
    
    plot['value'] += predict
    plot['type'] += ['fit']*(len(predict)-1) + ['predict']
    plot['X'] += [i for i in range(1, TRAIN_LENGTH+1)]
    series.append(predict[-1])
    
    for X in range(TRAIN_LENGTH+1, data.shape[0]):
        tensor = torch.tensor(series).reshape(1, -1, 1)
        predict = model(tensor).flatten().tolist()

        plot['value'].append(predict[-1])
        plot['type'].append('predict')
        plot['X'].append(X)
        series.append(predict[-1])
        print(X, len(series))
        
plt.plot()
sns.lineplot(data=plot, x='X',y='value',hue='type')
plt.savefig("img/continus_prediction.png")
plt.close()

