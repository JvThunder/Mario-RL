import numpy as np
import gym
from gym import spaces
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation 
from stable_baselines3 import PPO
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game
        over. Done by DeepMind for the DQN and co. since it helps value
        estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped._life
        if self.lives > lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few fr
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped._life
        return obs

# class TimeLimitWrapper(gym.Wrapper):
#   """
#   :param env: (gym.Env) Gym environment that will be wrapped
#   :param max_steps: (int) Max number of steps per episode
#   """
#   def __init__(self, env, max_steps=10000):
#     # Call the parent constructor, so we can access self.env later
#     super(TimeLimitWrapper, self).__init__(env)
#     self.max_steps = max_steps
#     # Counter of steps per episode
#     self.current_step = 0
  
#   def reset(self):
#     """
#     Reset the environment 
#     """
#     # Reset the counter
#     self.current_step = 0
#     return self.env.reset()

#   def step(self, action):
#     """
#     :param action: ([float] or int) Action taken by the agent
#     :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
#     """
#     self.current_step += 1
#     obs, reward, done, info = self.env.step(action)
#     # Overwrite the done signal when 
#     if self.current_step >= self.max_steps:
#       done = True
#       # Update the info dict to signal that the limit was exceeded
#       info['time_limit_reached'] = True
#     info['Current_Step'] = self.current_step
#     return obs, reward, done, info

# class CustomReward(gym.Wrapper):
#     def __init__(self, env):
#         super(CustomReward, self).__init__(env)
#         self._current_score = 0
#         self._current_time = 400

#     def step(self, action):
#         state, reward, done, info = self.env.step(action)
#         reward += (info['score'] - self._current_score) / 20
#         reward -= (info['time'] - self._current_time)
#         self._current_score = info['score']
#         self._current_time = info['time']
#         if done:
#             if info['flag_get']:
#                 reward += 100.0
#             else:
#                 reward -= 10.0
#         return state, reward, done, info

def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = ResizeObservation(env, (84, 84))
    env = GrayScaleObservation(env)
    env = FrameStack(env, 4)
    env = MaxAndSkipEnv(env, 4)
    RIGHT_ONLY = [
        ['NOOP'],
        ['right'],
        ['right', 'A'],
    ]
    env = JoypadSpace(env, RIGHT_ONLY)
    env = EpisodicLifeEnv(env)
    return env

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        # print(observation_space.shape)
        num_inputs = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=num_inputs, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim)
        )

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
                nn.init.constant_(module.bias, 0)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=512),
)

env = make_env()
env.reset()
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log="./mario_tensorboard/",
            policy_kwargs=policy_kwargs,
            n_steps=2048,
            batch_size=32, 
            gamma=0.9,
            learning_rate=0.00025,
            ent_coef=0.01,
            n_epochs=20,
            )
N = 6
TIMESTEPS = 50_000

for i in range(1,N+1):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
    model.save(f"models/PPO2/{TIMESTEPS*i}")