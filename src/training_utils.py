import random
from collections import namedtuple, deque
import torch

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):
    """A simple replay buffer to store playing experience."""
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def copy_params(target_network: torch.nn.Module, value_network: torch.nn.Module, soft_tau: float) -> None:
    """Softly copying params from a `value network` to a `target_network` (torch.nn.Module) with `soft_tau` (from 0.0 to 1.0 rate).
    """
    for target_param, param in zip(target_network.parameters(), value_network.parameters()):
        target_param.data.copy_(
        target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )