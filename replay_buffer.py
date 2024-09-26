import torch
import numpy as np
import os

class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
    
    def add(self, input, action, old_log_prob, reward):
        if len(self.buffer) >= self.max_size:
            # Remove a random entry to make space
            random_idx = np.random.randint(0, len(self.buffer))
            self.buffer.pop(random_idx)
        self.buffer.append((input, action, old_log_prob, reward))
    
    def sample(self, batch_size, device="cpu"):
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        inputs, actions, old_log_probs, rewards = zip(*[self.buffer[idx] for idx in idxs])

        # Move tensors to the desired device
        inputs = torch.stack(inputs).to("cuda")
        actions = torch.tensor(actions, dtype=torch.float32).to("cuda")
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).to("cuda")
        rewards = torch.tensor(rewards, dtype=torch.float32).to("cuda")

        return inputs, actions, old_log_probs, rewards
    
    def size(self):
        return len(self.buffer)
    
    # Save the replay buffer to a file
    def save(self, file_name):
        # Save buffer to disk using torch.save
        print(f"Saving replay buffer to {file_name}")
        torch.save(self.buffer, file_name)
    
    # Load the replay buffer from a file if it exists
    def load(self, file_name, device="cpu"):
        # Check if the file exists
        if os.path.exists(file_name):
            print(f"Loading replay buffer from {file_name}")
            # Load buffer from disk using torch.load
            self.buffer = torch.load(file_name)
        else:
            print(f"Replay buffer file {file_name} does not exist. Starting with an empty buffer.")