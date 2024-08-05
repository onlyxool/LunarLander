import torch
import torch.nn as nn

class DeepQNetwork1(nn.Module):
    def __init__(self, num_actions, num_observation, fc_size=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_observation, 12),
            nn.ReLU(inplace=True),
            nn.Linear(12, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, num_actions)
        )

        for layer in [self.layers]:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')

    def forward(self, x):
        Q = self.layers(x)
        return Q


class DeepQNetwork(torch.nn.Module):
    def __init__(self, action_size=4, state_size=8, hidden_size=64):
        super(DeepQNetwork, self).__init__()
        self.layer1 = torch.nn.Linear(state_size, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer3 = torch.nn.Linear(hidden_size, action_size)


    def forward(self, state):
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)
