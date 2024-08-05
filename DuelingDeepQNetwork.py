import torch


class DuelingDeepQNetwork(torch.nn.Module):
    def __init__(self, action_size=4, state_size=8, hidden_size=64):
        super(DuelingDeepQNetwork, self).__init__()
        self.layer1 = torch.nn.Linear(state_size, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)

        self.advantage = torch.nn.Linear(hidden_size, action_size)
        self.value = torch.nn.Linear(hidden_size, 1)


    def forward(self, state):
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        advantage = self.advantage(x)
        value = self.value(x)
        return value + (advantage - advantage.max(dim=1, keepdim=True)[0])
