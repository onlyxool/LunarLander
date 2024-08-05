import os
import sys
import time
import warnings
import numpy as np
import gymnasium as gym
from DQNAgent import DQNAgent


hyper_parameters = {
  'batch_size': 64,
  'epsilon': 1,
  'epsilon_min': 0.01,
  'epsilon_decay': 0.995,
  'update_rate': 10,
  'learning_rate': 0.001,
  'gamma': 0.99,
}


warnings.filterwarnings("ignore", category=UserWarning)
update_rate = 10
algorithms = 'DDQN'

def main():
    training_mode = True if len(sys.argv) >= 2 and sys.argv[1] == 'train' else False
    episodes_num = int(sys.argv[2]) if training_mode else 5

    env = gym.make('LunarLander-v2', max_episode_steps=500, render_mode="human")# if not training_mode else None)
    env.metadata['render_fps'] = 60

    agent = DQNAgent(env, hyper_parameters, algorithms, training_mode, True)

    if training_mode:
        # Record Training episode
        episode_end = episodes_num

        # Record Training Start Time
        start=time.time()

        # Whether is a resume training
        if training_mode and len(sys.argv) >= 4 and os.path.exists(sys.argv[3]):
            episode_start = agent.load_checkpoint(sys.argv[3])
            agent.episode = episode_start
        else:
            episode_start = 1

        for episode in range(episode_start, episodes_num):
            terminate = False
            while not terminate:
                agent.update_epsilon()
                terminate = agent.step()
                agent.update_weights()
            if episode % update_rate == 0:
                agent.update_target_network()

            # Save CheckPoint
            if episode % 50 == 0:
                agent.save_checkpoint(episode)

            # If the average score is greater than 200 for 10 consecutive times the training is terminated
            if np.all(np.array(agent.history.ave_rewards[-10:])>200):
                episode_end = episode
                break

        # Record Training End Time
        end = time.time()
        print(f'The training lasted for: {round(end-start, 2)} Seconds')

        # Flush History to Plot Graph 
        agent.flush()

        # Save Final Model File
        agent.save(f'model/{algorithms}_ep{episode_end}.pth')
    else:
        if len(sys.argv) >= 2 and os.path.exists(sys.argv[1]):
            agent.demo(sys.argv[1])
        else:
            sys.exit('Please specify the model file.')

    env.close()


if __name__ == '__main__':
    sys.exit(main())
