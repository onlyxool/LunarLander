import torch
import numpy as np
import torch.optim as optim
from collections import deque, namedtuple
from torch.utils.tensorboard import SummaryWriter

from DeepQNetwork import DeepQNetwork
from DuelingDeepQNetwork import DuelingDeepQNetwork


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class History:
    def __init__(self, plot):
        self.ave_rewards = list()
        self.episode_rewards = list()
        self.writer = SummaryWriter() if plot else None


    def __len__(self):
        return len(self.episode_rewards)


    def append(self, reward):
        self.episode_rewards.append(reward)
        mean_reward = round(np.mean(self.episode_rewards), 3)
        self.ave_rewards.append(mean_reward)

        if self.writer:
            self.writer.add_scalar('Reward/Episode', reward, len(self.episode_rewards))
            self.writer.add_scalar('Average_Reward/Last_5_Episodes', mean_reward, len(self.ave_rewards))


    def restore(self):
        if self.writer:
            for i, reward in enumerate(self.episode_rewards):
                self.writer.add_scalar('Reward/Episode', reward, len(self.episode_rewards))
                self.writer.add_scalar('Average_Reward/Last_5_Episodes', self.ave_rewards[i], len(self.ave_rewards))


    def flush(self):
        if self.writer:
            self.writer.flush()



Replay = namedtuple('Replay', field_names=['state', 'action', 'reward', 'next_state', 'done'])

class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)


    def __len__(self):
        return len(self.buffer)


    def append(self, experience):
        self.buffer.append(experience)


    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones  = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions, dtype=np.int64), np.array(rewards, dtype=np.float32), np.array(next_states, dtype=np.float32), np.array(dones, dtype=np.uint8)



class DQNAgent():
    def __init__(self, env, hyper_parameters, training_mode, algorithms='DQN', network='Dueling', plot=True):
        self.env = env
        self.alg = algorithms
        self.episode = 0
        self.training_mode = training_mode
        self.batch_size = hyper_parameters['batch_size']

        self.history = History(plot) if training_mode else None
        self.replay_memory = ReplayMemory(10000) if training_mode else None

        self.gamma = hyper_parameters['gamma']
        self.epsilon = hyper_parameters['epsilon'] if training_mode else -1
        self.epsilon_min = hyper_parameters['epsilon_min']
        self.epsilon_decay = hyper_parameters['epsilon_decay']

        if network == 'Dueling':
            self.model = DuelingDeepQNetwork(env.action_space.n, env.observation_space.shape[0]).to(device)
            self.target_model = DuelingDeepQNetwork(env.action_space.n, env.observation_space.shape[0]).to(device).eval()
        else:
            self.model = DeepQNetwork(env.action_space.n, env.observation_space.shape[0]).to(device)
            self.target_model = DeepQNetwork(env.action_space.n, env.observation_space.shape[0]).to(device).eval()

        self.clip_grad_norm = 5
        self.learning_rate = hyper_parameters['learning_rate']
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.reset()


    def reset(self):
        self.state, _ = self.env.reset(seed=42)
        self.steps = 0
        self.episode_reward = 0


    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


    def act(self):
        if self.training_mode and np.random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            state = torch.as_tensor(self.state, dtype=torch.float32, device=device).to(device)
            with torch.no_grad():
                q_value = self.model(state)
            self.model.train()
            action = torch.argmax(q_value).item()

        return action


    def step(self):
        action = self.act()
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = (terminated or truncated)

        self.replay_memory.append(Replay(self.state, action, reward, next_state, done))

        self.steps += 1
        self.state = next_state
        self.episode_reward += reward

        if done:
            self.episode += 1
            self.history.append(self.episode_reward)
            print(f'episode: {self.episode}, steps {self.steps}, reward: {round(self.episode_reward,2)} mean reward: {self.history.ave_rewards[-1]}\n')
            self.reset()

        return done


    def update_weights(self):
        if len(self.replay_memory) >= self.batch_size:
            states, actions, rewards, next_states, dones = self.replay_memory.sample(self.batch_size)

            states = torch.tensor(states, dtype=torch.float32).to(device)
            next_states = torch.tensor(next_states).to(device)
            actions = torch.tensor(actions).to(device)
            rewards = torch.tensor(rewards).to(device)
            dones = torch.BoolTensor(dones).to(device)

            q_value = self.model(states).gather(dim=1, index=actions.unsqueeze(-1)).squeeze(-1)

            if self.alg == 'DQN':
                next_q_value = self.target_model(torch.as_tensor(next_states, dtype=torch.float32, device=device)).max(1)[0]
                next_q_value = next_q_value.detach()
                next_q_value[dones] = 0.0
            else: # Double DQN
                next_action_values = self.model(next_states).max(1)[1].unsqueeze(-1)
                next_q_value = self.target_model(next_states).gather(1, next_action_values).detach().squeeze(-1)

            expected_q_value = rewards + next_q_value * self.gamma

            loss = torch.nn.MSELoss()(q_value, expected_q_value)

            self.optimizer.zero_grad()
            loss.backward()

            # Clip the gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.optimizer.step()


    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
        print(f"episode {self.episode}: target model weights updated")


    def demo(self, model_path):
        # Load the weights of the test_network
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        self.reset()
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action = self.act()
            self.state, reward, terminated, truncated, _ = self.env.step(action)
            self.episode_reward += reward
            self.steps += 1

        print(f'Steps: {self.steps}, Reward: {self.episode_reward:.2f}, ')


    def save_checkpoint(self, episode):
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.model.state_dict(),
            'target_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'replay_memory': self.replay_memory,
            'epsilon': self.epsilon,
            'episode_rewards': np.array(self.history.episode_rewards),
            'ave_rewards': np.array(self.history.ave_rewards)
        }
        torch.save(checkpoint, 'model/checkpoint.pt')
        print(f'Checkpoint saved at episode {episode}')


    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.replay_memory = checkpoint['replay_memory']
        self.epsilon = checkpoint['epsilon']
        self.history.ave_rewards = checkpoint['ave_rewards'].tolist()
        self.history.episode_rewards = checkpoint['episode_rewards'].tolist()
        self.history.restore()

        return checkpoint['episode']


    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f'Save Model file to {path}')


    def flush(self):
        self.history.flush()
