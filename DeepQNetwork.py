import torch.nn as nn

class DeepQNetwork(nn.Module):
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
