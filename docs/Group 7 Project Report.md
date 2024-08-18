# Group 7: Project Report

### Reinforcement Learning Programming - CSCN8020  

*Applied Artificial Intelligence and Machine Learning Program, Conestoga College Sec2*



> Ashraf, Hamna               ID: 8826836
>
> Fernandez, Arcadio       ID: 8951215
>
> Chen, Kun (Kyle)             ID: 8977010
>
> 
>
> Professor: Mahmoud Nasr
>
> August 17, 2024



 <img src="image\1_TBp5k_85gauhZSh7eS1YlA.webp" style="zoom: 67%;" />

#### 1. Introduction

The problem we aim to solve is the LunarLander-v2 environment, a classic control problem provided by the Gymnasium library. The LunarLander-v2 environment simulates a scenario where an agent, represented as a spacecraft, must learn to land on a designated landing pad in Box2D. The spacecraft is affected by gravity, and its movement can be controlled by firing its main and side thrusters. The challenge is to land the spacecraft safely and precisely on the landing pad while minimizing fuel consumption.

<img src="image\lunar_lander.gif" style="zoom:50%;" />

This environment is part of the Box2D environments which contains general information about the environment

| Action Space      | `Discrete(4)`                                                |
| ----------------- | ------------------------------------------------------------ |
| Observation Space | `Box([-1.5 -1.5 -5. -5. -3.1415927 -5. -0. -0. ], [1.5 1.5 5. 5. 3.1415927 5. 1. 1. ], (8,), float32)` |
| import            | `gymnasium.make("LunarLander-v2")`                           |



**Action Space**

There are four discrete actions available:

- 0: do nothing
- 1: fire left orientation engine
- 2: fire main engine
- 3: fire right orientation engine



**Observation Space**

The state is an 8-dimensional vector: the coordinates of the lander in `x` & `y`, its linear velocities in `x` & `y`, its angle, its angular velocity, and two booleans that represent whether each leg is in contact with the ground or not.



**Rewards**

After every step a reward is granted. The total reward of an episode is the sum of the rewards for all the steps within that episode.

For each step, the reward:

- is increased/decreased the closer/further the lander is to the landing pad.
- is increased/decreased the slower/faster the lander is moving.
- is decreased the more the lander is tilted (angle not horizontal).
- is increased by 10 points for each leg that is in contact with the ground.
- is decreased by 0.03 points each frame a side engine is firing.
- is decreased by 0.3 points each frame the main engine is firing.

The episode receive an additional reward of -100 or +100 points for crashing or landing safely respectively.

An episode is considered a solution if it scores at least 200 points.



**Starting State**

The lander starts at the top center of the viewport with a random initial force applied to its center of mass.

To init Lunarland env:

```python
import gymnasium as gym
env = gym.make(
    "LunarLander-v2",
    continuous: bool = False,
    gravity: float = -10.0,
    enable_wind: bool = False,
    wind_power: float = 15.0,
    turbulence_power: float = 1.5,
)
```



**Episode Termination**

The episode finishes if:

1. the lander crashes (the lander body gets in contact with the moon);
2. the lander gets outside of the viewport (`x` coordinate is greater than 1);



This problem is significant because it represents a complex decision-making task under uncertainty, which is common in various real-world applications, such as autonomous vehicle navigation, robotics, and aerospace engineering. Solving this problem efficiently requires an agent to learn and adapt to the environment's dynamics, making it an ideal testbed for evaluating reinforcement learning algorithms.

To address this problem, we will implement and compare three reinforcement learning algorithms: Deep Q-Network (DQN), Double Deep Q-Network (DDQN), and Advantage Actor-Critic (A2C). DQN is a foundational algorithm that uses deep neural networks to approximate the Q-value function, enabling the agent to make decisions based on state-action pairs. DDQN improves upon DQN by reducing the overestimation bias commonly associated with Q-learning, leading to more stable learning. A2C, on the other hand, is a policy-based algorithm that combines the benefits of both value-based and policy-based approaches, leveraging parallel actors to update the policy in a more stable and efficient manner.

By applying and comparing these algorithms, we aim to determine which approach is most effective for solving the LunarLander-v2 environment and to gain insights into the strengths and weaknesses of each method.







#### 2. Background information

##### **Deep Q-Network (DQN)**

DQN is a value-based reinforcement learning algorithm that extends the classical Q-learning approach by incorporating deep neural networks. In traditional Q-learning, a Q-table is used to store the value of state-action pairs, which becomes infeasible in environments with large or continuous state spaces. DQN addresses this issue by using a neural network to approximate the Q-value function, enabling the agent to learn directly from high-dimensional sensory inputs.

The key innovations of DQN is the use of a target network, which helps stabilize the training process. In standard Q-learning, the target values are updated directly from the same network, which leading to unstable reward in the training process. By maintaining a separate target network that is updated less frequently, DQN reduces the risk of instability during training.

Another important feature of DQN is the use of experience replay. Instead of updating the network after each action, experience replay stores the agent’s experiences (state, action, reward, next state) in a buffer. During training, mini-batches of these experiences are sampled randomly to update the network. This approach breaks the correlation between consecutive samples, which improves the efficiency and stability of the learning process.



##### **Double Deep Q-Network (DDQN)**

DDQN is an extension of DQN that addresses the issue of overestimation bias in Q-learning. In standard DQN, the algorithm tends to overestimate the value of certain actions due to the maximization step in the Q-value update. This overestimation can lead to suboptimal policies and slower learning.

DDQN mitigates this problem by decoupling the action selection from the action evaluation. Specifically, DDQN uses the main network to select the best action and the target network to evaluate the value of that action. This simple modification significantly reduces the bias in the value estimates, leading to more accurate and stable learning outcomes.



##### **Advantage Actor-Critic (A2C)**

A2C is a policy-based algorithm that combines the benefits of both value-based and policy-based methods. Unlike DQN and DDQN, which directly approximate the Q-value function, A2C uses an actor-critic framework where the actor is responsible for selecting actions based on a policy, and the critic evaluates the action by computing the value function.

The key components of A2C is the advantage function, which helps the agent focus on actions that are better than the expected outcome. The advantage function is defined as the difference between the actual return and the value function, providing a more informative signal for updating the policy.

A2C also leverages parallel environments to update the policy in a more stable. By running multiple environments simultaneously, A2C can collect a diverse set of experiences, leading to more robust policy updates. This parallelism not only accelerates the learning process but also helps in achieving better generalization across different scenarios.







#### 3. Implementation & experiments



##### **Computational Resources**

1. Macbook Pro without GPU

   **Processor**: Intel Core i9- 2.9Ghz 6-Core

   **Memory**: 32GB RAM.

   **GPU**: None GPU

   **Operating System**: macOS Sonoma

   **Runtime Environment**: Python 3.11 with Miniconda

2. Yoga Laptop with RTX1060 GPU

   **Processor**: Intel Core i7

   **Memory**: 32GB RAM.

   **GPU**: NVIDIA GeForce RTX1060 GPU

   **Operating System**: Windows11

   **Runtime Environment**: Python 3.11

3. Lenovo Laptop with GTX1060 GPU

   **Processor**: Intel Core i7

   **Memory**: 32GB RAM.

   **GPU**: NVIDIA GeForce RTX1060 GPU

   **Operating System**: Windows11 with WSL Ubuntu 22.04

   **Runtime Environment**: Python 3.11  with Miniconda

*Before we were about to start this project, one of our laptops with an RTX 3070 GPU broke down, the screen went black. We had to spend more time on model training, significantly slowing down our progress.*



##### **Dependent libraries**

pytorch, gymnasium, gymnasium, swig, gymnasium[box2d], pygame, tensorboard



Below are the specific implementation details for each algorithm:

##### **1. Deep Q-Network (DQN)** 

  **Neural Network Architecture**:

The neural network architecture used for the Deep Q-Network (DQN) and Double Deep Q-Network (DDQN) algorithms is as follows:

- **Input Layer**: The input layer consists of 8 neurons, which correspond to the state space dimensions of the LunarLander-v2 environment.

- **Hidden Layers**: There are two fully connected hidden layers, each with 64 neurons. These layers use ReLU activation functions to introduce non-linearity and help the network capture complex patterns in the data.

- **Output Layer**: The output layer consists of 4 neurons, corresponding to the action space, where each neuron represents a possible action that the agent can take in the environment.

  <img src="image\DQN_ep2000.onnx.png" style="zoom:33%;" />

  Check the *Neural Network Architecture* : [graphic image](https://netron.app/?url=https://github.com/onlyxool/LunarLander/blob/main/model/DQN_ep2000.onnx)  [Source Code](https://github.com/onlyxool/LunarLander/blob/main/DeepQNetwork.py)

```python
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
```



   **Hyper-Parameters**

```python
hyper_parameters = {
  'batch_size': 64,
  'epsilon': 1,
  'epsilon_min': 0.01,
  'epsilon_decay': 0.995,
  'update_rate': 10,
  'learning_rate': 0.001,
  'gamma': 0.99,
}
```



   **Target Network Update Frequency**: Every 64 steps.

   **Experience Replay Buffer Size**: 10,000 transitions.

   **Number of Episodes**: 2000

```python
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
```



##### **2. Double Deep Q-Network (DDQN)** 

The same architecture and hyperparameters as DQN were used.



##### **3. Advantage Actor-Critic (A2C)** 

To run Actor-Critic Methods (A2C) to file were created, one to run the model and one to test: 

[a2c_lunar_lander.py](https://github.com/onlyxool/LunarLander/blob/main/a2c_lunar_lander.py) & [test_a2c_lunar_lander.py](https://github.com/onlyxool/LunarLander/blob/main/test_a2c_lunar_lander.py)



The ActorCritic class defines the neural network model for the Actor-Critic algorithm. It consists of two fully connected layers (fc1 and fc2) followed by two separate output layers: logits for the policy (actor) and value for the value function (critic).

Forward pass through the neural network. It takes an input state x and returns the output of the actor and critic networks:

```python
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
```



The A2CAgent class defines the A2C agent. It initializes the environment, sets hyperparameters (gamma, learning rate), and creates an instance of the ActorCritic model. It also sets up a TensorBoard writer for logging metrics: 

```python
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
```



**A2CAgent class has several methods:**

**get_action:** Selects an action based on the current state using the policy (actor) network.

```python
def get_action(self, state):
	state = np.expand_dims(state, axis=0)
    state = torch.FloatTensor(state)
    logits, _ = self.model(state)
    action_probabilities = torch.softmax(logits, dim=-1)
    action = np.random.choice(self.env.action_space.n, p=action_probabilities.detach().numpy().squeeze())
    return action
```

**train_step:** Performs a single training step using the computed loss.

**save_model:** Saves the model to a file.

**compute_loss:** Computes the loss function for the Actor-Critic algorithm, which includes the policy loss, value loss, and entropy bonus.

**train:** Trains the agent for a specified number of episodes, logging metrics to TensorBoard and saving the model periodically.

```python
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
```



##### **Problems Encountered**

In the early stages of our project we put a lot of effort into solving a development environment problem with one of the laptop. Initially, when we executed ```pip install gymnasium[box2d]```, we always had the problem of not being able to install it successfully, and then solved the problem after setting the environment variables of swig correctly. After solving the first problem, we again found that we could not successfully install pytorch, the error message showed that we could not load the fbgemm.dll file. Finally we decided not to use window, we switched to WSL Ubuntu and successfully installed the development environment.







#### 4. Results & Discussion

##### **1. Deep Q-Network (DQN)** 

Average Reward / Episode

<img src="image\DQN2000ep_ave.png" style="zoom:80%;" />

Reward / Episode

<img src="image\DQN2000ep_reward.png" style="zoom:80%;" />

This plot shows the reward obtained per episode throughout the training process.

- **Fluctuations in Reward**:  The early stages of training (0-550 episodes). the agent alternates between exploration and exploitation.

- **Stabilization**:  Around 600 episodes, the reward starts to stabilize, with both runs showing a consistent upward trend in rewards. The lander learned that slowly approaching the ground will get higher reward value. Each episode gets closer to the ground as it learns.

- **Fluctuation Again:** After 1200 episodes, the reward values begin to fluctuate strongly again, this is because the lander begins to learn to land on the ground, sometimes successfully and sometimes not. Successful landings get a reward of 100, failures get a reward of -100.

- **Final Stage**:  Towards the end of training. After 1800 episode, most episodes can be successfully landed.



##### **2. Double Deep Q-Network (DDQN)** 

Average Reward / Episode

<img src="image\DDQN2000_avg.png" style="zoom:80%;" />

Reward / Episode

<img src="image\DDQN2000ep_reward.png" style="zoom:80%;" />

We trained it several times without success, two of which are shown in the figure.

Performs similarly to DQN in early and intermediate stages. But in the final stages of training, we see a peak in the average rewards, followed by a noticeable decrease in rewards. Both runs (denoted by different colors) show similar behavior, that could be addressed by further tuning hyperparameters such as the learning rate or exploration strategy, or by introducing techniques like early stopping to prevent overfitting. 



##### **Discussion: Compare the DQN and DDQN**

The results indicate that the DDQN algorithm is more effective than DQN in learning a policy for the LunarLander-v2 environment.  The fluctuation and eventual stabilization of rewards suggest that DDQN enters the stabilization and learning to land phase much earlier.





##### **3. Advantage Actor-Critic (A2C)** 

Average Reward / Episode

<img src="image\a2c1.jpg" style="zoom:80%;" />

Reward / Episode

<img src="image\a2c2.png" style="zoom:80%;" />

Although the results were good for A2C, the lunar lander managed to land safely several times. The erratic behaviour we observed suggests that A2C might not be fully optimized for this specific environment. However, with the right adjustments, it can still potentially yield smoother learning curves and better overall performance.

<img src="image\a2c3.png" style="zoom:80%;" />

The rewards for each episode are mostly negative in test (test_a2c_lunar_lander.py), suggesting that the lunar lander is failing to land successfully on the pad. The variability in rewards suggests that the model hasn't stabilized or learned an optimal policy. Fine-tuning hyperparameters like the learning rate or extending the training duration could help improve performance.







#### 5. Conclusion & Future work



##### **Deep Q-Network (DQN)**

<img src="image\lunar_lander_dqn.gif" style="zoom: 67%;" />

##### **Double Deep Q-Network (DDQN)**

<img src="image\lunar_lander_ddqn.gif" style="zoom:67%;" />

In this project, we implemented and analyzed the performance of Deep Q-Networks (DQN) and Double DQN (DDQN) on the LunarLander-v2 environment. Through extensive training and experimentation, we observed that both DQN and DDQN were able to successfully learn policies that resulted in high average rewards, indicating their effectiveness in solving this reinforcement learning problem.  Although DDQN was ultimately unsuccessful in solving this problem, we believe it is possible to achieve ultimate success after tuning the hyperparameters or reducing the number of episodes.



##### **Advantage Actor-Critic (A2C)**

A2C typically should have offered the best performance due to its ability to combine value-based and policy-based learning. However, as observed in your results, A2C could also have exhibited erratic learning and high variability if not well-tuned. With proper tuning, A2C should have provided smoother and more optimal landings, but if hyperparameters like the learning rate were not carefully adjusted, the model might have struggled, resulting in inconsistent or suboptimal performance.



##### **Future work**

The comparison of A2C, DQN, and DDQN on the Lunar Lander environment provides valuable insights into the strengths and weaknesses of each approach. However, to advance this work, it is crucial to focus on hyperparameter tuning, exploring algorithmic variants, improving exploration strategies, and ensuring computational efficiency. 

Exploring algorithmic variants, such as: 



**Dueling DQN or Double Dueling DQN**
Description: Dueling DQN is a variant of the standard DQN algorithm that introduces two separate estimators within the network: one for the state value function and another for the advantage function. This architecture helps the agent to learn which states are (or are not) valuable, independent of the action taken, leading to more stable learning. Double Dueling DQN further improves this by addressing the overestimation bias in Q-learning.

 

**Soft Actor-Critic (SAC)**

Description: SAC is an off-policy actor-critic algorithm that maximizes a trade-off between expected reward and entropy, encouraging exploration.

 

**Asynchronous Advantage Actor-Critic (A3C)**

Description: A3C is an asynchronous, on-policy actor-critic algorithm that runs multiple instances of the environment in parallel to stabilize training.

Comprehensive benchmarking and documentation are essential to ensure reproducibility and facilitate further research in this area.







#### 6.  Our Code on GitHub

https://github.com/onlyxool/LunarLander


