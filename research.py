import utils
import pandas as pd
import torch

# parameters
TRAIN_LENGTH = 100
LOOKBACK_LENGTH = 10
EPOCH = 20

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
    series = data['#Passengers'].tolist()
    tensor = torch.tensor(series).reshape(1, -1, 1)
    Z = model(tensor)

