import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

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

class A2CAgent:
    def __init__(self, env):
        self.env = env
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.recent_rewards = deque(maxlen=5)  # Keep track of the last 5 episode rewards
        log_dir = f'runs/lunar_lander_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        self.writer = SummaryWriter(log_dir=log_dir)  # Specify the log directory with a unique name

    def get_action(self, state):
        state = np.expand_dims(state, axis=0)
        state = torch.FloatTensor(state)
        logits, _ = self.model(state)
        action_probabilities = torch.softmax(logits, dim=-1)
        action = np.random.choice(self.env.action_space.n, p=action_probabilities.detach().numpy().squeeze())
        return action

    def compute_loss(self, states, actions, rewards, dones, next_states):
        logits, values = self.model(states)
        _, next_values = self.model(next_states)
        advantages = rewards + self.gamma * next_values * (1 - dones) - values
        value_loss = advantages.pow(2).mean()
        action_masks = torch.nn.functional.one_hot(actions, self.env.action_space.n)
        policy = torch.softmax(logits, dim=-1)
        log_policy = torch.log_softmax(logits, dim=-1)
        entropy = -(policy * log_policy).sum(dim=1)
        policy_loss = -(action_masks * log_policy).sum(dim=1) * advantages.detach()
        total_loss = (value_loss + policy_loss.mean() - 0.01 * entropy.mean())
        return total_loss

    def train_step(self, states, actions, rewards, dones, next_states):
        self.optimizer.zero_grad()
        loss = self.compute_loss(states, actions, rewards, dones, next_states)
        loss.backward()
        self.optimizer.step()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def train(self, episodes=2000, batch_size=32, save_interval=10):  # Changed episodes to 2000
        for episode in range(episodes):
            state = self.env.reset()
            if isinstance(state, tuple):  # Handle the case where reset returns a tuple
                state = state[0]
            total_reward = 0
            states, actions, rewards, dones, next_states = [], [], [], [], []
            done = False
            while not done:
                self.env.render()  # Render the environment
                action = self.get_action(state)
                result = self.env.step(action)
                next_state, reward, done, truncated, info = result[0], result[1], result[2], result[3], result[4] if len(result) > 4 else False
                done = done or truncated  # Handle the 'truncated' state
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                next_states.append(next_state)
                state = next_state
                total_reward += reward
                if len(states) >= batch_size:
                    self.train_step(
                        torch.FloatTensor(np.array(states)), 
                        torch.LongTensor(np.array(actions)), 
                        torch.FloatTensor(np.array(rewards)), 
                        torch.FloatTensor(np.array(dones, dtype=np.float32)), 
                        torch.FloatTensor(np.array(next_states))
                    )
                    states, actions, rewards, dones, next_states = [], [], [], [], []
            
            self.recent_rewards.append(total_reward)
            average_reward = np.mean(self.recent_rewards)
            
            # Log metrics to TensorBoard
            self.writer.add_scalar('Total Reward per Episode', total_reward, episode + 1)
            self.writer.add_scalar('Average Reward (last 5 episodes)', average_reward, episode + 1)
            
            print(f"Episode {episode + 1}, Total Reward: {total_reward}, Average Reward (last 5 episodes): {average_reward:.2f}")

            # Save the model periodically
            if (episode + 1) % save_interval == 0:
                self.save_model(f'models/a2c_lunar_lander_{episode + 1}.pth')

        # Save the final model
        self.save_model('models/a2c_lunar_lander_final.pth')
        self.writer.close()  # Close the TensorBoard writer

def main():
    env = gym.make('LunarLander-v2', render_mode='human')  # Set render_mode to 'human'
    agent = A2CAgent(env)
    if not os.path.exists('models'):
        os.makedirs('models')
    agent.train(episodes=2000)  # Changed episodes to 2000
    env.close()  # Close the environment to clean up resources

if __name__ == "__main__":
    main()
