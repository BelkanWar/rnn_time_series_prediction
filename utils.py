import torch
import torch.nn.functional as F

class continus_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        RNN_DIM = 50
        LINEAR0_OUT_DIM = 50
        LINEAR1_OUT_DIM = 25
        LINEAR2_OUT_DIM = 12
        LINEAR3_OUT_DIM = 6
        LINEAR4_OUT_DIM = 1
        
        self.rnn = torch.nn.GRU(
            input_size=1, hidden_size=RNN_DIM, num_layers=6, batch_first=True, 
            bidirectional=False)
        # self.linear0 = torch.nn.Linear(RNN_DIM*2, LINEAR0_OUT_DIM)
        self.linear1 = torch.nn.Linear(RNN_DIM, LINEAR1_OUT_DIM)
        self.linear2 = torch.nn.Linear(LINEAR1_OUT_DIM, LINEAR2_OUT_DIM)
        self.linear3 = torch.nn.Linear(LINEAR2_OUT_DIM, LINEAR3_OUT_DIM)
        self.linear4 = torch.nn.Linear(LINEAR3_OUT_DIM, LINEAR4_OUT_DIM)
        
    def forward(self, x_input):
        x_input, ht = self.rnn(x_input)
        # x_input = F.relu(self.linear0(x_input))
        x_input = F.relu(self.linear1(x_input))
        x_input = F.relu(self.linear2(x_input))
        x_input = self.linear3(x_input)
        x_input = self.linear4(x_input)

        return x_input

class binary_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        RNN_DIM = 50
        LINEAR0_OUT_DIM = 50
        LINEAR1_OUT_DIM = 25
        LINEAR2_OUT_DIM = 12
        LINEAR3_OUT_DIM = 6
        LINEAR4_OUT_DIM = 1
        
        self.rnn = torch.nn.GRU(
            input_size=1, hidden_size=RNN_DIM, num_layers=6, batch_first=True, 
            bidirectional=False)
        # self.linear0 = torch.nn.Linear(RNN_DIM*2, LINEAR0_OUT_DIM)
        self.linear1 = torch.nn.Linear(RNN_DIM, LINEAR1_OUT_DIM)
        self.linear2 = torch.nn.Linear(LINEAR1_OUT_DIM, LINEAR2_OUT_DIM)
        self.linear3 = torch.nn.Linear(LINEAR2_OUT_DIM, LINEAR3_OUT_DIM)
        self.linear4 = torch.nn.Linear(LINEAR3_OUT_DIM, LINEAR4_OUT_DIM)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x_input):
        x_input, ht = self.rnn(x_input)
        # x_input = F.relu(self.linear0(x_input))
        x_input = F.relu(self.linear1(x_input))
        x_input = F.relu(self.linear2(x_input))
        x_input = self.linear3(x_input)
        x_input = self.linear4(x_input)
        x_input = self.sigmoid(x_input)

        return x_input

class Encoder(torch.nn.Module):
    def __init__(self, input_size=5, hidden_size=128, n_layers=1, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.lstm = torch.nn.LSTM(input_size, hidden_size, n_layers)
        
    def forward(self, x):
        # x: input batch data, size: [input_seq_len, batch_size, feature_size]
        output, (hidden, cell) = self.lstm(x)
        return hidden, cell


class Decoder(torch.nn.Module):
    def __init__(self, output_size=5, hidden_size=128, n_layers=1, dropout=0.3):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.lstm = torch.nn.LSTM(output_size, hidden_size, n_layers)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        """
        x size = [batch_size, feature_size]
        --> x only has two dimensions since the input is batch of last coordinate of observed trajectory
        so the sequence length has been removed.
        """
        # add sequence dimension to x, to allow use of nn.LSTM
        x = x.unsqueeze(0)  # -->[1, batch_size, feature_size]
        
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        prediction = self.linear(output)

        return prediction, hidden, cell    


def convert2Loader(data_X, data_Y):
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(data_Y).unsqueeze(2), 
            torch.tensor(data_X).unsqueeze(2)), 
        shuffle=True, batch_size=200)
    return loader

def training(model, loader, loss_fn, EPOCH):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    
    for epoch in range(EPOCH):
        for step, (y_batch, x_batch) in enumerate(loader):
            y_predict = model(x_batch)
            loss = loss_fn(y_predict, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print(f"epoch: {epoch} | batch: {step} | loss: {float(loss.data)}")
                # print(f"x: {x_batch.tolist()[0]} | y: {y_batch.tolist()[0]} | predict: {y_predict.tolist()[0]}")

    model.eval()
    return model