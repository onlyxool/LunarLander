import torch
import torch.nn as nn
import numpy as np
import gym
import os

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_space):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.logits = nn.Linear(128, action_space)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.logits(x)
        value = self.value(x)
        return logits, value

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")

def test_model(env, model, episodes=20):  # Changed to 20 episodes
    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):  # Handle the case where reset returns a tuple
            state = state[0]
        total_reward = 0
        done = False
        while not done:
            env.render()
            state = np.expand_dims(state, axis=0)
            state = torch.FloatTensor(state)
            logits, _ = model(state)
            action_probabilities = torch.softmax(logits, dim=-1)
            action = np.argmax(action_probabilities.detach().numpy().squeeze())
            result = env.step(action)
            next_state, reward, done, truncated, info = result[0], result[1], result[2], result[3], result[4] if len(result) > 4 else False
            done = done or truncated  # Handle the 'truncated' state
            state = next_state
            total_reward += reward
        
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

def main():
    env = gym.make('LunarLander-v2', render_mode='human')  # Set render_mode to 'human'
    input_dim = env.observation_space.shape[0]
    action_space = env.action_space.n
    model = ActorCritic(input_dim, action_space)

    # Load the model
    model_path = 'C:\\Users\\arcad\\LunarLander_A2C_v2\\models\\a2c_lunar_lander_final.pth'  # Path to the model saved after 2000 episodes
    load_model(model, model_path)

    # Test the model
    test_model(env, model, episodes=20)  # Run 20 test episodes

    env.close()  # Close the environment to clean up resources

if __name__ == "__main__":
    main()


