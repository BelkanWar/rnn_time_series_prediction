import utils
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

# parameters
TRAIN_LENGTH = 100
LOOKBACK_LENGTH = 30
EPOCH = 3000
loss_fn = torch.nn.BCELoss()

data = pd.read_csv("data/AirPassengers.csv")
data['Month'] = pd.to_datetime(data['Month'], format='%Y-%m')
data['#Passengers'] = data['#Passengers'].astype(float)
peak = [1 if data['#Passengers'][0]>data['#Passengers'][1] else 0] + \
       [1 if data['#Passengers'][i]>data['#Passengers'][i-1] and data['#Passengers'][i]>data['#Passengers'][i+1] else 0 for i in range(1,data.shape[0]-1)] + \
       [1 if data['#Passengers'].tolist()[-1]>data['#Passengers'].tolist()[-2] else 0]
data['peak'] = peak
data['peak'] = data['peak'].astype(float)
data = data.sort_values(by='Month', ascending=True)

data_X, data_Y = [], []

for idx in range(LOOKBACK_LENGTH+1, TRAIN_LENGTH):
    data_X.append(data.loc[idx-LOOKBACK_LENGTH-1:idx-1, 'peak'].tolist())
    data_Y.append(data.loc[idx-LOOKBACK_LENGTH:idx, 'peak'].tolist())

loader = utils.convert2Loader(data_X, data_Y)
model = utils.binary_Model()
model = utils.training(model, loader, loss_fn, EPOCH)
metrics_data = {'y_true':[], 'y_prob':[]}

with torch.inference_mode():
    plot = {
        'binary':data['peak'].tolist(),
        'p-value':[0],
        'type':['fit'],
        'X':[i for i in range(data.shape[0])]}
    
    series = data['peak'].tolist()[:LOOKBACK_LENGTH]
    tensor = torch.tensor(series).reshape(1, -1, 1)
    predict = model(tensor).flatten().tolist()
    
    plot['p-value'] += predict
    plot['type'] += ['fit']*(len(predict)-1) + ['predict']
    series.append(predict[-1])
    
    for X in range(LOOKBACK_LENGTH+1, data.shape[0]):
        tensor = torch.tensor(data['peak'].tolist()[X-LOOKBACK_LENGTH:X]).reshape(1, -1, 1)
        predict = model(tensor).flatten().tolist()

        plot['p-value'].append(predict[-1])
        plot['type'].append('predict')
        series.append(predict[-1])
        metrics_data['y_true'].append(data['peak'].tolist()[X])
        metrics_data['y_prob'].append(predict[-1])

        
plt.plot()
sns.scatterplot(data=plot, x='X',y='binary')
sns.lineplot(data=plot, x='X', y='p-value', hue='type')
plt.savefig("img/binary_prediction.png")
plt.close()

print(roc_auc_score(metrics_data['y_true'], metrics_data['y_prob']))