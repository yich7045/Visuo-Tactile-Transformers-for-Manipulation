import gym
import numpy as np
import wandb
import os
from moviepy.editor import ImageSequenceClip


class VideoWrapper(gym.Wrapper):
    def __init__(self, env):
        super(VideoWrapper, self).__init__(env)
        self.nb_observation = env.observation_space[0].shape[0]

        self.episode_states = [[] for _ in range(self.nb_observation)]

    def reset(self, **kwargs):
        self.episode_states[0].clear()
        state = self.env.reset()
        state_visual, state_vector = state
        for i in range(self.nb_observation):
            # import pdb; pdb.set_trace()
            # import pdb; pdb.set_trace()
            # self.episode_states[i].append(state_visual[np.newaxis, i, :, :])
            self.episode_states[i].append(state_visual[np.newaxis, np.newaxis, i, :, :])

        return state

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        state_visual, state_vector = state
        for i in range(self.nb_observation):
            # self.episode_states[i].append(state_visual[np.newaxis, i, :, :])
            # import pdb;pdb.set_trace()
            self.episode_states[i].append(state_visual[np.newaxis, np.newaxis, i, :, :])

        return state, reward, done, info

    def send_wandb_video(self, prefix=""):
        for i in range(self.nb_observation):
            wandb.log(
                {prefix + "video_" + str(i): wandb.Video(np.concatenate(self.episode_states[i]), fps=12, format="gif")})


class PandaGymVideo(gym.Wrapper):
    def __init__(self, env, episode_log_frequency=2):
        super(PandaGymVideo, self).__init__(env)
        self.episode_states = []
        self.episode_log_frequency = episode_log_frequency
        self.episode_counter = 0

    def reset(self, **kwargs):
        if self.episode_counter % self.episode_log_frequency == 0 and len(self.episode_states) > 0:
            self.send_panda_video()
        self.episode_states.clear()
        state = self.env.reset()
        self.episode_counter += 1

        if self.episode_counter % self.episode_log_frequency == 0:
            state_visual = self.env.get_all_sides_image(256, 256)
            self.episode_states.append(state_visual[np.newaxis, :, :, :])
        return state

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        if (self.episode_counter) % self.episode_log_frequency == 0:
            state_visual = self.env.get_all_sides_image(256, 256)
            self.episode_states.append(state_visual[np.newaxis, :, :, :])
        return state, reward, done, info

    def send_panda_video(self):
        wandb.log({"high_res": wandb.Video(np.concatenate(self.episode_states), fps=12, format="gif")})


class VideoWrapperCreateGif(PandaGymVideo):
    def send_panda_video(self):
        np.save('pushVidHighRes.npy', np.concatenate(self.episode_states))
        # gif("pushVidHighRes.gif", np.concatenate(self.episode_states)[:, 0])


def gif(filename, array, fps=12, scale=1.0):
    """Creates a gif given a stack of images using moviepy
    Notes
    -----
    works with current Github version of moviepy (not the pip version)
    https://github.com/Zulko/moviepy/commit/d4c9c37bc88261d8ed8b5d9b7c317d13b2cdf62e
    Usage
    -----
     X = randn(100, 64, 64)
     gif('test.gif', X)
    Parameters
    ----------
    filename : string
        The filename of the gif to write to
    array : array_like
        A numpy array that contains a sequence of images
    fps : int
        frames per second (default: 10)
    scale : float
        how much to rescale each image by (default: 1.0)
    """

    # ensure that the file has the .gif extension
    fname, _ = os.path.splitext(filename)
    filename = fname + '.gif'

    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # make the moviepy clip
    clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
    clip.write_gif(filename, fps=fps, logger=None)
    return clip
