import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
    
    def add(self, input, action, reward):
        if len(self.buffer) >= self.max_size:
            # Remove a random entry to make space
            random_idx = np.random.randint(0, len(self.buffer))
            self.buffer.pop(random_idx)
        self.buffer.append((input, action, reward))
    
    def sample(self, batch_size):
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        inputs, actions, rewards = zip(*[self.buffer[idx] for idx in idxs])
        return torch.stack(inputs), torch.stack(actions), torch.tensor(rewards)
    
    def size(self):
        return len(self.buffer)