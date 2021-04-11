import cv2
import gym
import gym.spaces
import numpy as np
import collections

class FireResetEnv(gym.Wrapper):
    """
    Leaving it to the network to learn to press start will take too long,
    so, this wrapper to the environment presses start for it at the beginning
    """
    def __init__(self, env=None):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):# tries to press start, which is done in the atari grames by action 1 and some games action 2
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs

class MaxAndSkipEnv(gym.Wrapper):
    """
    This wrapper solves two issues:
        1- most likely, the consecutive frames are the same without a need for a new action decision,
        so, this wrapper uses the same action decision for skip=4 consecutive frames
        2- the atari games frames flicker, so, it takes every two consequtive frames and takes the max of their pixels,
        so, while skipping over the skip frames, it maxes out the last observed two frames

        
    """
    def __init__(self, env=None, skip=4):
        """Return only every 'skip'-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ProcessFrame84(gym.ObservationWrapper):
    """
    for preprocessing the observed frames,
    1- Changing the dimentions to 84x84x1 by resizing it and cropping the top and bottom parts
    2- Converting the color channels to grayscale using a colorimetric grayscale conversion 
    (which is closer to human color perception than a simple averaging of color channels)
    """
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)

class BufferWrapper(gym.ObservationWrapper):
    """
    In order to get an indication of the observed frame's dynamics, e.g. direction and trajectory of projectiles
     we need to accumulate a small sequence of the observed frames.
     This trick also solves the issue with that the observations consisting of single frames on their own
      would violate the markovian property of our MDP
    so, instead of single frames as observations, we have a buffer of frames in a queue [a new frame gets instered at the end kicking the oldest frame]
    and our observation space is modified to be of bufferSize x old space
    """
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
        old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer



class ImageToPyTorch(gym.ObservationWrapper):
    """
    PyTorch assumes the color channel to be at dimension 0 instead of 2, and that is done through this wrapper 
    also, this is reflected on the observation space
    """
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]), dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)



class ScaledFloatFrame(gym.ObservationWrapper):
    """
    Scaling the pixels values to be between 0 and 1 and converting their datatype to float32
    """
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0



def make_env(env_name):
    env = gym.make(env_name)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    return ScaledFloatFrame(env)