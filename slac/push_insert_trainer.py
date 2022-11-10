import os
from collections import deque
from datetime import timedelta
from time import sleep, time
import pickle
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class SlacObservation:
    """
    Observation for SLAC.
    """
    def __init__(self, state_shape, tactile_shape,  action_shape, num_sequences):
        self.state_shape = state_shape
        self.tactile_shape = tactile_shape
        self.action_shape = action_shape
        self.num_sequences = num_sequences

    def reset_episode(self, state, tactile):
        self._state = deque(maxlen=self.num_sequences)
        self._tactile = deque(maxlen=self.num_sequences)
        self._action = deque(maxlen=self.num_sequences - 1)
        for _ in range(self.num_sequences - 1):
            self._state.append(np.zeros(self.state_shape, dtype=np.uint8))
            self._tactile.append(np.zeros(self.tactile_shape, dtype=np.float32))
            self._action.append(np.zeros(self.action_shape, dtype=np.float32))
        self._state.append(state)
        self._tactile.append(tactile)

    def append(self, state, tactile, action):
        self._state.append(state)
        self._tactile.append(tactile)
        self._action.append(action)

    @property
    def state(self):
        return np.array(self._state)[None, ...]

    @property
    def tactile(self):
        return np.array(self._tactile)[None, ...]

    @property
    def action(self):
        return np.array(self._action).reshape(1, -1)


class pushing_Trainer:
    """
    Trainer for SLAC.
    """
    def __init__(
        self,
        env,
        algo,
        log_dir,
        seed=0,
        action_shape=3,
        num_steps=3 * 10 ** 7,
        initial_collection_steps=50000,
        initial_learning_steps=50000,
        num_sequences=8,
        eval_interval=10,
        num_eval_episodes=10,
    ):
        # Env to collect samples.
        self.env = env
        self.env.seed(seed)
        self.num_eval_episodes = num_eval_episodes

        # Observations for training and evaluation.
        self.ob = SlacObservation((3, 84, 84), (6,), (action_shape,),  num_sequences)
        self.ob_test = SlacObservation((3, 84, 84), (6,), (action_shape,), num_sequences)

        # Algorithm to learn.
        self.algo = algo

        # Log setting.
        self.log = {"step": [], "return": []}
        self.csv_path = os.path.join(log_dir, "log.csv")
        self.log_dir = log_dir
        self.summary_dir = os.path.join(log_dir, "summary")
        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.model_dir = os.path.join(log_dir, "model")
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Other parameters.
        self.action_repeat = 1
        self.num_steps = num_steps
        self.initial_collection_steps = initial_collection_steps
        self.initial_learning_steps = initial_learning_steps
        self.eval_interval = eval_interval
        self.evaluation_reward = []
        self.evaluation_steps = []

    def train(self):
        # Time to start training.
        self.start_time = time()
        # Episode's timestep.
        episodes = 0
        state = self.env.reset()
        img = state[0]
        tactile = state[1]
        self.ob.reset_episode(img, tactile)
        self.algo.buffer.reset_episode(img, tactile)

        # Collect trajectories using random policy.
        for step in range(1, self.initial_collection_steps + 1):
            episodes, _ = self.algo.step(self.env, self.ob, episodes, step <= self.initial_collection_steps)

        # Update latent variable model first so that SLAC can learn well using (learned) latent dynamics.
        bar = tqdm(range(self.initial_learning_steps))
        for _ in bar:
            bar.set_description("Updating latent variable model.")
            self.algo.update_latent(self.writer)
            self.algo.update_latent_align(self.writer)
        # Iterate collection, update and evaluation.

        for step in range(self.initial_collection_steps + 1, self.num_steps // self.action_repeat + 1):
            episodes, run_episode = self.algo.step(self.env, self.ob, episodes, False)

            # Update the algorithm.
            self.algo.update_latent(self.writer)
            self.algo.update_sac(self.writer)
            self.algo.update_latent_align(self.writer)

            # Evaluate regularly.
            step_env = step * self.action_repeat
            if run_episode % self.eval_interval == 0:
                mean_return = self.evaluate(step_env)
                self.evaluation_reward.append(mean_return)
                self.evaluation_steps.append(step_env)
                save_pickle(self.evaluation_reward, "evaluation_rewards.pkl")
                save_pickle(self.evaluation_steps, "evaluation_steps.pkl")
                # uncomment if wants to save model
                # self.algo.save_model(os.path.join(self.model_dir, f"step{step_env}"))


    def evaluate(self, step_env):
        mean_return = 0.0
        for i in range(self.num_eval_episodes):
            state = self.env.reset()
            img = state[0]
            tactile = state[1]
            self.ob_test.reset_episode(img, tactile)
            episode_return = 0.0
            done = False

            while not done:
                action = self.algo.exploit(self.ob_test)
                action = np.append(action, -0.3)
                state, reward, done, _ = self.env.step(action)
                img = state[0]
                tactile = state[1]
                self.ob_test.append(img, tactile, action[0:3])
            if reward != 25:
                reward = 0
            episode_return += reward

            mean_return += episode_return / self.num_eval_episodes
        return mean_return

    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))

def save_pickle(data, myfile):
    with open(myfile, "wb") as f:
        pickle.dump(data, f)
