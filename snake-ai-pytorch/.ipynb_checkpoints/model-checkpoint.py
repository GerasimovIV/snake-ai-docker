import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from typing import List
import numpy as np

class BaseLinear(nn.Module):
    def __init__(self, inp_size, out_size, activation=F.relu, dropout=0.):
        super().__init__()        
        self.linear = nn.Linear(inp_size, out_size)   
        self.activation = activation
        print(self.activation)
#         self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        x = self.linear(x)        
        x = self.activation(x)
#         x = self.dropout(x)
        return x


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        
#         self.seq = nn.Sequential(*[BaseLinear(inp_size, out_size, **kwargs) for inp_size, out_size in zip(sizes[:-1], sizes[1:])])
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x): 
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self):
        
        file_name = self.__class__.__name__ + '.pth'
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
        
        
class Linear_QNet_Scanner(nn.Module):
    def __init__(self, *sizes, **kwargs):
        super().__init__()
        
        activations = kwargs['activations'] if 'activations' in kwargs else [nn.Identity()] * (len(sizes) - 1)
#         print(activations)
#         print(list(zip(sizes[:-1], sizes[1:], activations)))
        self.seq = nn.Sequential(*[BaseLinear(inp_size, out_size, act) 
                                   for inp_size, out_size, act in zip(sizes[:-1], sizes[1:], activations)])
#         self.linear1 = nn.Linear(input_size, hidden_size)
#         self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x): 
#         x = F.relu(self.linear1(x))
#         x = self.linear2(x)
        return self.seq(x)

    def save(self):
        
        file_name = self.__class__.__name__ + '.pth'
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
        
        
class WrappedLSTM(torch.nn.LSTM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
            
    def forward(self, x, lengths=None):

        if len(x.shape) == 2:
            x = torch.unsqueeze(x, 0)
        x, (h, c) = super().forward(x)

        if lengths is not None:
            res = []
        
#             print(x.shape, lengths)
            for i, l in enumerate(lengths):
                res.append(x[i:i+1, l-1, :])
            
            return torch.cat(res, 0)
        
        return x[:, -1, :]
            
    def save(self):
        
        file_name = self.__class__.__name__ + '.pth'
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()

        
        
class QTrainerLSTM:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step_batched(self, state, action, reward, next_state, done):
#         print(state, action, reward, next_state, done)
    
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        
        
        lengths = [len(s) for s in state]
        state = nn.utils.rnn.pad_sequence([torch.tensor(s, dtype=torch.float) for s in state], batch_first=True, padding_value=0.0)
        next_state = nn.utils.rnn.pad_sequence([torch.tensor(s, dtype=torch.float) for s in next_state], batch_first=True,padding_value=0.0)
        # (n, s, fx)


        # 1: predicted Q values with current state
        # print(f'==========================={state.shape}')
        
#         print(state.shape, print(done))
        pred = self.model(state, lengths)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
        
    def train_step(self, state, action, reward, next_state, done):
        
#         print(len(state), state[0].shape, type(state))
        
        if isinstance(state, tuple):
#             print('batched')
            self.train_step_batched(state, action, reward, next_state, done)
        
        if isinstance(state, np.ndarray):
#             print('alone')
            state = torch.tensor(state, dtype=torch.float)
            next_state = torch.tensor(next_state, dtype=torch.float)
            action = torch.tensor(action, dtype=torch.long)
            reward = torch.tensor(reward, dtype=torch.float)

            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
            
            self.train_step_batched(state, action, reward, next_state, done)



