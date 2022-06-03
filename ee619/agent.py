"""Agent for DMControl Walker-Run task."""
from os.path import abspath, dirname, realpath
from typing import Dict

from dm_env import TimeStep
import numpy as np


ROOT = dirname(abspath(realpath(__file__)))  # path to the ee619 directory


def flatten_and_concat(dmc_observation: Dict[str, np.ndarray]) -> np.ndarray:
    """Convert a DMControl observation (OrderedDict of NumPy arrays)
    into a single NumPy array.

    """
    return np.concatenate([[obs] if np.isscalar(obs) else obs.ravel()
                           for obs in dmc_observation.values()])


class Agent:
    """Agent for a Walker2DBullet environment."""
    def __init__(self) -> None:
        # Create class variables here if you need to.
        # Example:
        #     self.policy = torch.nn.Sequential(...)
        pass

    def act(self, time_step: TimeStep) -> np.ndarray:
        """Returns the action to take for the current time-step.

        Args:
            time_step: a namedtuple with four fields step_type, reward,
                discount, and observation.
        """
        # You can access each member of time_step by time_step.[name], a
        # for example, time_step.reward or time_step.observation.
        # You may also check if the current time-step is the last one
        # of the episode, by calling the method time_step.last().
        # The return value will be True if it is the last time-step,
        # and False otherwise.
        # Note that the observation is given as an OrderedDict of
        # NumPy arrays, so you would need to convert it into a
        # single NumPy array before you feed it into your network.
        # It can be done by using the `flatten_and_concat` function, e.g.,
        #   observation = flatten_and_concat(time_step.observation)
        #   logits = self.policy(torch.as_tensor(observation))
        return np.ones(6)

    def load(self):
        """Loads network parameters if there are any."""
        # Example:
        #   path = join(ROOT, 'model.pth')
        #   self.policy.load_state_dict(torch.load(path))
