import sys
import torch
import pygame
import random
import argparse
import warnings
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.optim as optim
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayMemory():
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)


    def store(self, experience):
        self.buffer.append(experience)


    def sample(self, batch_size):
        states = list()
        actions = list()
        next_states = list()
        rewards = list()
        dones = list()
        indices = np.random.choice(len(self), size=batch_size, replace=False)
        for index in indices:
            states.append(self.buffer[index][0])
            actions.append(self.buffer[index][1])
            next_states.append(self.buffer[index][2])
            rewards.append(self.buffer[index][3])
            dones.append(self.buffer[index][4])

        states = torch.as_tensor(np.array(states), dtype=torch.float32, device=device)
        actions = torch.as_tensor(np.array(actions), dtype=torch.long, device=device) 
        next_states = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=device)
        rewards = torch.as_tensor(np.array(rewards), dtype=torch.float32, device=device)
        dones = torch.as_tensor(np.array(dones), dtype=torch.bool, device=device)

        return (states, actions, next_states, rewards, dones)

    def __len__(self):
        return len(self.buffer)


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

#        for layer in [self.layers]:
#            for module in layer:
#                if isinstance(module, nn.Linear):
#                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')

    def forward(self, x):
        Q = self.layers(x)
        return Q


class DQNAgent():
    def __init__(self, env, episodes_num, training_mode):
        self.env = env
        self.seed = 2024
        self.discount = 1
        self.batch_size = 64
        self.running_loss = 0
        self.clip_grad_norm = 5
        self.learned_counts = 0
        self.learning_rate = 1e-3
        self.episodes_num = episodes_num
        self.epsilon_max = 0.999 if training_mode else -1
        self.replay_memory = ReplayMemory(10000)

        self.model = DeepQNetwork(env.action_space.n, env.observation_space.shape[0]).to(device)
        self.target_model = DeepQNetwork(env.action_space.n, env.observation_space.shape[0]).to(device).eval()
        self.critertion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)


    def train(self):
        step_num = 0
        for episode in range(1, self.episodes_num + 1):
            state, _ = self.env.reset(seed=self.seed)
            stop = False
            done = False
            episode_reward = 0
            while not stop and not done:
                action = self.action(state)
                next_state, reward, done, stop, _ = self.env.step(action)

                self.replay_memory.store([state, action, next_state, reward, (stop or done)])
                if len(self.replay_memory) >= self.batch_size:
                    states, actions, next_states, rewards, dones = self.replay_memory.sample(self.batch_size)

                    actions = actions.unsqueeze(1)
                    rewards = rewards.unsqueeze(1)
                    dones = dones.unsqueeze(1)

                    predictedQ = self.model(states)
                    predictedQ = predictedQ.gather(dim=1, index=actions)

                    # Compute the Qt(s', argmax(Q(s', a')) for the next states using the target network
                    with torch.no_grad():
                        next_target_q_value = self.target_model(next_states)
                        selected_actions = self.model(next_states)
                        selected_actions = selected_actions.argmax(dim=1, keepdims=True)
                        next_target_q_value = next_target_q_value.gather(dim=1, index=selected_actions)
                    next_target_q_value[dones] = 0  # Set the Q-value for terminal states to zero
                    y_js = rewards + (self.discount * next_target_q_value)  # Compute the target Q-values
                    loss = self.critertion(predictedQ, y_js)  # Compute the loss

                    # Update the running loss and learned counts for logging and plotting
                    self.running_loss += loss.item()
                    self.learned_counts += 1

                    if done or stop:
                        episode_loss = self.running_loss / self.learned_counts  # The average loss for the episode
                #self.loss_history.append(episode_loss)  # Append the episode loss to the loss history for plotting
                        # Reset the running loss and learned counts
                        self.running_loss = 0
                        self.learned_counts = 0

                    self.optimizer.zero_grad()  # Zero the gradients
                    loss.backward()  # Perform backward pass and update the gradients
                    # Clip the gradients to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.optimizer.step()  # Update the parameters of the main network using the optimizer 

                if done or stop:
                    self.update_weight()
                    break
                state = next_state
                episode_reward += reward
                step_num += 1


    def demo(self):
        """
        Reinforcement learning policy evaluation.
        """

        # Load the weights of the test_network
        self.model.load_state_dict(torch.load('model/final.pth'))
        self.target_model.eval()

        # Testing loop over episodes
        for episode in range(1, self.episodes_num + 1):
            state, _ = self.env.reset(seed=self.seed)
            done = False
            truncation = False
            step_size = 0
            episode_reward = 0
            reward = 0
            while not done and not truncation:
                #print(f'y speed: {state[3]}, y pos: {state[1]}, reward : {reward}')
                action = self.action(state)
                next_state, reward, done, truncation, _ = self.env.step(action)
                state = next_state
                episode_reward += reward
                step_size += 1

            # Print log
            result = (f"Episode: {episode}, "f"Steps: {step_size:}, "f"Reward: {episode_reward:.2f}, ")
            print(result)
        pygame.quit()  # close the rendering window 


    def action(self, state):
        # Exploration: epsilon-greedy
        if np.random.random() < self.epsilon_max:
            return self.env.action_space.sample()

        state = torch.as_tensor(state, dtype=torch.float32, device=device)

        with torch.no_grad():
            Q_values = self.model(state)
            action = torch.argmax(Q_values).item()

        return action


    def update_weight(self):
        self.target_model.load_state_dict(self.model.state_dict())


    def save(self, path):
        torch.save(self.model.state_dict(), path)


def main():
    training_mode = True if len(sys.argv) >= 2 and sys.argv[1] == 'train' else False
    episodes_num = int(sys.argv[2]) if training_mode else 5
    env = gym.make('LunarLander-v2', max_episode_steps=1000, render_mode="human" if not training_mode else None)
    env.metadata['render_fps'] = 60
    warnings.filterwarnings("ignore", category=UserWarning)

    agent = DQNAgent(env, episodes_num, training_mode)
    if training_mode:
        agent.train()
        agent.save('model/final.pth')
    else:
        agent.demo()

    env.close()


if __name__ == '__main__':
    sys.exit(main())
