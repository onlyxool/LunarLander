import torch
import pygame
import torch.nn
import numpy as np
import torch.optim as optim
from collections import deque
from DeepQNetwork import DeepQNetwork

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




class DQNAgent():
    def __init__(self, env, episodes_num, training_mode):
        self.env = env
        self.discount = 1
        self.batch_size = 64
        self.running_loss = 0
        self.clip_grad_norm = 5
        self.learned_counts = 0
        self.learning_rate = 1e-3
        self.episodes_num = episodes_num
        self.epsilon_max = 0.999 if training_mode else -1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.996
        self.replay_memory = ReplayMemory(1000)

        self.final_learning_rate = 75e-5
        self.initial_learning_rate = 1e-3
        self.learning_rate_coefficient = 0.995

        self.discount_factor = 1
        self.max_discount_factor = 0.97
        self.discount_factor_coefficient = 0.9966

        self.model = DeepQNetwork(env.action_space.n, env.observation_space.shape[0]).to(device)
        self.target_model = DeepQNetwork(env.action_space.n, env.observation_space.shape[0]).to(device).eval()
        self.critertion = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)


    def train(self):
        steps_num = 0
        gama_t = 0
        for episode in range(1, self.episodes_num + 1):
            state, _ = self.env.reset()
            stop = False
            done = False
            episode_reward = 0
            step_size = 0
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
                    self.optimizer.step()

                    if done:
                        self.update_weight()
                        break
                state = next_state
                episode_reward += reward
                steps_num += 1
                step_size += 1

            self.update_epsilon()

            result = (f"Episode: {episode}, "
                          f"Total Steps: {steps_num}, "
                          f"Ep Step: {step_size}, "
                          f"Raw Reward: {episode_reward:.2f}, "
                          f"Discount Factor: {gama_t:.2f}, "
                          f"Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}, "
                          f"Epsilon: {self.epsilon_max:.2f}, ")
#f"Choose good mem: {self.choose_good_mem}")

            self.initial_learning_rate = max(self.final_learning_rate, self.initial_learning_rate*self.learning_rate_coefficient)
            for group in self.optimizer.param_groups:
                group['lr'] = self.initial_learning_rate
            gama_t = min(self.discount_factor - self.discount_factor_coefficient * (1 - gama_t), self.max_discount_factor)
            self.discount = gama_t
            print(result)


    def demo(self):
        # Load the weights of the test_network
        self.model.load_state_dict(torch.load('model/final.pth'))
        self.target_model.eval()

        # Testing loop over episodes
        for episode in range(1, self.episodes_num + 1):
            state, _ = self.env.reset()
            done = False
            stop = False
            step_size = 0
            episode_reward = 0
            reward = 0
            while not done and not stop:
                action = self.action(state)
                next_state, reward, done, stop, _ = self.env.step(action)
                state = next_state
                episode_reward += reward
                step_size += 1

            result = (f"Episode: {episode}, "f"Steps: {step_size:}, "f"Reward: {episode_reward:.2f}, ")
            print(result)
        pygame.quit()


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


    def update_epsilon(self):
        self.epsilon_max = max(self.epsilon_min, self.epsilon_max * self.epsilon_decay)

    def save(self, path):
        torch.save(self.model.state_dict(), path)
