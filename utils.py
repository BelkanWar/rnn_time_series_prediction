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

def convert2Loader(data_X, data_Y):
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(data_Y).unsqueeze(2), 
            torch.tensor(data_X).unsqueeze(2)), 
        shuffle=True, batch_size=200)
    return loader

def training(model, loader, EPOCH):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    loss_fn = torch.nn.MSELoss()
    
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