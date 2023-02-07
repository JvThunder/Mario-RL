import numpy as np
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
from stable_baselines3 import PPO

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

env = make_env()
env.reset()

model = PPO.load("models/PPO/300000", env)

import imageio
from PIL import Image

sum_rewards = 0
N = 3
for i in range(1,N+1):
    frames = []
    total_reward = 0
    state = env.reset()
    done = False       
    while not done:
        action, _ = model.predict(np.array(state).copy())
        state, reward, done, info = env.step(int(action))

        frame = env.render(mode='rgb_array')
        frames.append(Image.fromarray(frame))

        total_reward += reward

        if done:
            break

    print(f"Total Reward {i}: {total_reward}")            
    imageio.mimwrite(f'demo/mario_{i}.gif', frames, fps=30)
    sum_rewards += total_reward

print(f"Average {N} Episodes: {sum_rewards/N}")         
env.close()