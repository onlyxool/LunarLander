import sys
import warnings
import gymnasium as gym
from DQNAgent import DQNAgent


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
