<div align="center">

# LunarLander
</div>
&nbsp;

## About the Project
LunarLander is a classical problem in reinforcement learning, the goal is to control a lunar lander to safely land it on a designated landing pad. The environment provides continuous control over the lander's engines and rewards the agent based on its landing success and fuel usage.

&nbsp;

![DQN](https://github.com/user-attachments/assets/575005ad-b614-4476-aeb6-02207332b814)

&nbsp;

`States`
- Lander Co-ordinates (x,y)
- Lander Velocity in x and y
- Angle (theta)
- Angular Velocity (theta velocity)
- Two booleans - left leg contact, right leg contact on ground or not

`Actions`

0: do nothing\
1: fire left orientation engine\
2: fire main engine\
3: fire right orientation engine

`Reward`
- Increase if lander is closer to landing pad and decrease if its farther
- Increase if lander is moving slowly and decrease if its moving too fast
- Decrease the more lander tilts
- Increase 10 points when each leg lands the ground
- Decrease 0.3 points each frame the main engine is firing
- Decrease 0.03 points each frame the side engine is firing
- Grant 100 points for safe landing
- Deduct 100 points for crashing 

&nbsp;

## Approaches used:

- DQN
- DDQN
- Advantage Actor-Critic (A2C)

&nbsp;

Install library

```bash
pip install torch torchvision torchaudio gymnasium
pip install wheel setuptools pip --upgrade
pip install swig gymnasium[box2d] pygame
```


Train Model

```bash
python lunar_lander.py train 1000
```



Test Model

```bash
python lunar_lander.py model/[file_name]
```
&nbsp;

<!-- CONTACT -->
## Contact

Arcadio - [@Arcadio Arcadio de Paula Fernandez](https://www.linkedin.com/in/arcadio-de-paula-fernandez-b9b43a194/?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)\
Hamna - [@Hamna Ashraf](https://www.linkedin.com/in/hamna-ashraf/)\
Kyle - [@Kun Chen](https://www.linkedin.com/in/kyle-chen-aa6bb130/)

Project Link: [Lunar Lander](https://github.com/onlyxool/LunarLander)

