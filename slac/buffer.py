from collections import deque

import numpy as np
import torch


class LazyFrames:
    """
    Stacked frames which never allocate memory to the same frame.
    """

    def __init__(self, frames):
        self._frames = list(frames)

    def __array__(self, dtype):
        return np.array(self._frames, dtype=dtype)

    def __len__(self):
        return len(self._frames)


class SequenceBuffer:
    """
    Buffer for storing sequence data.
    """

    def __init__(self, num_sequences=8):
        self.num_sequences = num_sequences
        self._reset_episode = False
        self.state_ = deque(maxlen=self.num_sequences + 1)
        self.tactile_ = deque(maxlen=self.num_sequences + 1)
        self.action_ = deque(maxlen=self.num_sequences)
        self.reward_ = deque(maxlen=self.num_sequences)
        self.done_ = deque(maxlen=self.num_sequences)

    def reset(self):
        self._reset_episode = False
        self.state_.clear()
        self.tactile_.clear()
        self.action_.clear()
        self.reward_.clear()
        self.done_.clear()

    def reset_episode(self, state, tactile,):
        assert not self._reset_episode
        self._reset_episode = True
        self.state_.append(state)
        self.tactile_.append(tactile)

    def append(self, action, reward, done, next_state, next_tactile):
        assert self._reset_episode
        self.action_.append(action)
        self.reward_.append([reward])
        self.done_.append([done])
        self.state_.append(next_state)
        self.tactile_.append(next_tactile)

    def get(self):
        state_ = LazyFrames(self.state_)
        tactile_ = np.array(self.tactile_, dtype=np.float32)
        action_ = np.array(self.action_, dtype=np.float32)
        reward_ = np.array(self.reward_, dtype=np.float32)
        done_ = np.array(self.done_, dtype=np.float32)
        return state_, tactile_,  action_, reward_, done_

    def is_empty(self):
        return len(self.reward_) == 0

    def is_full(self):
        return len(self.reward_) == self.num_sequences

    def __len__(self):
        return len(self.reward_)


class ReplayBuffer:
    """
    Replay Buffer.
    """

    def __init__(self, buffer_size, num_sequences, state_shape, tactile_shape, action_shape, device, force_norm):
        self._n = 0
        self._p = 0
        self.buffer_size = buffer_size
        self.num_sequences = num_sequences
        self.state_shape = state_shape
        self.tactile_shape = tactile_shape
        self.action_shape = action_shape
        self.device = device
        self.force_norm = force_norm

        # Store the sequence of images as a list of LazyFrames on CPU. It can store images with 9 times less memory.
        self.state_ = [None] * buffer_size
        # Store other data on GPU to reduce workloads.
        self.tactile_ = [None] * buffer_size
        self.action_ = torch.empty(buffer_size, num_sequences, *action_shape, device=device)
        self.reward_ = torch.empty(buffer_size, num_sequences, 1, device=device)
        self.done_ = torch.empty(buffer_size, num_sequences, 1, device=device)
        # Buffer to store a sequence of trajectories.
        self.buff = SequenceBuffer(num_sequences=num_sequences)

    def reset_episode(self, state, tactile):
        """
        Reset the buffer and set the initial observation. This has to be done before every episode starts.
        """
        self.buff.reset_episode(state, tactile)

    def append(self, action, reward, done, next_state, next_tactile, episode_done):
        """
        Store trajectory in the buffer. If the buffer is full, the sequence of trajectories is stored in replay buffer.
        Please pass 'masked' and 'true' done so that we can assert if the start/end of an episode is handled properly.
        """
        self.buff.append(action, reward, done, next_state, next_tactile)

        if self.buff.is_full():
            state_, tactile_, action_, reward_, done_ = self.buff.get()
            self._append(state_, tactile_, action_, reward_, done_)
        if episode_done:
            self.buff.reset()

    def _append(self, state_, tactile_, action_, reward_, done_):
        self.state_[self._p] = state_
        self.tactile_[self._p] = tactile_
        self.action_[self._p].copy_(torch.from_numpy(action_))
        self.reward_[self._p].copy_(torch.from_numpy(reward_))
        self.done_[self._p].copy_(torch.from_numpy(done_))
        self._n = min(self._n + 1, self.buffer_size)
        self._p = (self._p + 1) % self.buffer_size

    def sample_latent(self, batch_size):
        """
        Sample trajectories for updating latent variable model.
        """
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        state_ = np.empty((batch_size, self.num_sequences + 1, *self.state_shape), dtype=np.uint8)
        tactile_ = np.empty((batch_size, self.num_sequences + 1, *self.tactile_shape), dtype=np.float32)
        for i, idx in enumerate(idxes):
            state_[i, ...] = self.state_[idx]
            tactile_[i, ...] = self.tactile_[idx]
        state_ = torch.tensor(state_, dtype=torch.uint8, device=self.device).float().div_(255.0)
        tactile_ = torch.tensor(tactile_, dtype=torch.float16, device=self.device).float().div(self.force_norm)
        return state_, tactile_, self.action_[idxes], self.reward_[idxes], self.done_[idxes]

    def misalign_sample_latent(self, batch_size):
        """
        Sample trajectories for updating latent variable model.
        """
        idxes_state = np.random.randint(low=0, high=self._n, size=batch_size)
        idxes_tactile = np.random.randint(low=0, high=self._n, size=batch_size)

        while len(np.intersect1d(idxes_state, idxes_tactile)) != 0:
            idxes_state = np.random.randint(low=0, high=self._n, size=batch_size)
            idxes_tactile = np.random.randint(low=0, high=self._n, size=batch_size)

        state_ = np.empty((batch_size, self.num_sequences + 1, *self.state_shape), dtype=np.uint8)
        tactile_ = np.empty((batch_size, self.num_sequences + 1, *self.tactile_shape), dtype=np.float32)
        for i, idx in enumerate(idxes_state):
            state_[i, ...] = self.state_[idx]
        for i, idx in enumerate(idxes_tactile):
            tactile_[i, ...] = self.tactile_[idx]

        state_ = torch.tensor(state_, dtype=torch.uint8, device=self.device).float().div_(255.0)
        tactile_ = torch.tensor(tactile_, dtype=torch.float16, device=self.device).float().div(self.force_norm)
        return state_, tactile_, self.action_[idxes_state], self.reward_[idxes_state], self.done_[idxes_state]

    def sample_sac(self, batch_size):
        """
        Sample trajectories for updating SAC.
        """
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        state_ = np.empty((batch_size, self.num_sequences + 1, *self.state_shape), dtype=np.uint8)
        tactile_ = np.empty((batch_size, self.num_sequences + 1, *self.tactile_shape), dtype=np.float32)
        for i, idx in enumerate(idxes):
            state_[i, ...] = self.state_[idx]
            tactile_[i, ...] = self.tactile_[idx]

        state_ = torch.tensor(state_, dtype=torch.uint8, device=self.device).float().div_(255.0)
        tactile_ = torch.tensor(tactile_, dtype=torch.float16, device=self.device).float().div(self.force_norm)
        return state_, tactile_, self.action_[idxes], self.reward_[idxes, -1], self.done_[idxes, -1]

    def __len__(self):
        return self._n
