import abc

import torch
from torch import nn
from torch.distributions import Normal
import torch.nn.functional as F

class ActionSelector(nn.Module, abc.ABC):
    def __init__(self, k):
        """
        :param k: Number of candidates that will be created
        """
        super().__init__()
        self.k = k

    @abc.abstractmethod
    def forward(self, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """
        returns
            - actions: Shape [k x N]
                0 = NOOP
                1 = Clone
                2 = Split
                3 = Prune (does not work in old implementation)
            - log_probabilities for each candidate: Shape [k]
        """
        pass


class OldActionSelector(ActionSelector):
    def __init__(self, *, k=1, grad_threshold=0.0002, min_opacity=0.005, opacity_reset_interval=3000):
        super().__init__(k=k)
        self.grad_threshold = grad_threshold

        self.min_opacity = min_opacity  # Currently not used due to prune problems
        self.opacity_reset_interval = opacity_reset_interval  # Currently not used due to prune problems

    def forward(self, gaussians, *, iteration, scene_extent):
        grads = gaussians.xyz_gradient_accum / gaussians.denom
        grads[grads.isnan()] = 0.0
        grad_norms = torch.norm(grads, dim=-1, p=2)
        max_scalings = torch.max(gaussians.get_scaling, dim=1).values

        # Clone
        # Set all gaussians with grad_norm > threshold to 1
        op_mask = torch.as_tensor(grad_norms > self.grad_threshold, dtype=torch.int)

        # Split
        # Increase all gaussians which should be cloned by 1 if scaling > threshold
        op_mask += (op_mask == 1) & (max_scalings >= gaussians.percent_dense * scene_extent)

        # # Prune (does not work here before we clone/split apparently)
        # size_threshold = 20 if iteration > self.opacity_reset_interval else None
        # prune_mask = (gaussians.opacities < self.min_opacity).squeeze()
        # prune_opacity = torch.sum(prune_mask)
        #
        # if size_threshold:
        #     big_points_vs = gaussians.max_radii2D > size_threshold
        #     big_points_ws = max_scalings > 0.1 * scene_extent
        #     prune_mask = prune_mask | big_points_vs | big_points_ws
        #     print(f"Pruning: Opac={prune_opacity}, vs={torch.sum(big_points_vs)}, ws={torch.sum(big_points_vs)}, total={torch.sum(prune_mask)}, total_noop={torch.sum(prune_mask & (op_mask == 0))}")
        # else:
        #     print(f"Pruning: Opac={prune_opacity}, total={torch.sum(prune_mask)}, total_noop={torch.sum(prune_mask & (op_mask == 0))}")
        # # Only prune if not cloned or splitted
        # prune_mask = prune_mask & (op_mask == 0)
        #
        # op_mask[prune_mask] = 3

        op_mask = op_mask.unsqueeze(0).repeat(self.k, 1)
        return op_mask, torch.ones_like(op_mask, dtype=torch.float32)


class GradNormThresholdSelector(ActionSelector):
    def __init__(self, *, k=2, init_threshold=0.0002, sigma=0.0001, parameter_scale_factor=1000):
        """
        :param parameter_scale_factor:
            As the threshold is very small, the samples only vary between [0.0001, 0.0002, 0.0003].
            The threshold and the grad_norms are multiplied by this value to allow for numbers inbetween.
        """
        super().__init__(k=k)
        self.parameter_scale_factor = parameter_scale_factor

        self.register_parameter(
            "mu", nn.Parameter(torch.tensor(init_threshold * self.parameter_scale_factor, dtype=torch.float32))
        )

        self.register_buffer("sigma", torch.tensor(sigma * self.parameter_scale_factor, dtype=torch.float32))

    def forward(self, gaussians, *, iteration, scene_extent):
        grads = gaussians.xyz_gradient_accum / gaussians.denom
        grads[grads.isnan()] = 0.0
        grad_norms = torch.norm(grads, dim=-1, p=2)
        max_scalings = torch.max(gaussians.get_scaling, dim=1).values
        # num_points = gaussians.num_points

        # First decision: Do we split/clone a point
        # Set all gaussians with grad_norm > threshold to 1
        threshold_dist = Normal(self.mu, self.sigma)
        threshold_samples = threshold_dist.rsample((self.k, 1)).to("cuda")  # sample multiple thresholds
        threshold_logprobs = threshold_dist.log_prob(threshold_samples)
        op_mask = (
                torch.sigmoid((
                                grad_norms.detach() * self.parameter_scale_factor - threshold_samples) / self.parameter_scale_factor) > 0.5
        ).int()
        print(op_mask.shape, torch.sum(op_mask))
        # op_mask = torch.as_tensor(grad_norms > self.grad_threshold, dtype=torch.int)
        print(f"op_mask shape: {op_mask.shape}")
        print(f"max_scalings shape: {max_scalings.shape}")
        print(f"gaussians.percent_dense shape: {gaussians.percent_dense}")
        print(f"scene_extent shape: {scene_extent.shape}")
        # Second decision: Do we split or clone a point
        # Increase all gaussians which should be splitted by 1 if scaling > threshold
        op_mask += (op_mask == 1) & (max_scalings >= gaussians.percent_dense * scene_extent)

        return op_mask.detach(), threshold_logprobs.squeeze()


class ParamNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=3):
        super(ParamNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        #nn.init.zeros_(self.fc3.bias)
        
        # Initialize fc3 weights and biases for changed probabilities
        nn.init.zeros_(self.fc3.weight)
        nn.init.constant_(self.fc3.bias, 0.0)  # Initialize biases to 0
        
        # Example: Increase the bias for a specific action to increase its initial probability
        # Assuming you want to favor the first action initially:
        self.fc3.bias.data[0] = 2.0  # Increase bias for the first action
        self.fc3.bias.data[1] = -2.0  # Decrease bias for the second action
        self.fc3.bias.data[2] = -2.0  # Decrease bias for the third action
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ParamBasedActionSelector(ActionSelector):
    def __init__(self, input_size=3, k=2, hidden_size=128):
        super().__init__(k=k)
        self.param_network = ParamNetwork(input_size, hidden_size)

    def forward(self, gaussians, *, iteration, scene_extent):
        grads = gaussians.xyz_gradient_accum / gaussians.denom
        grads[grads.isnan()] = 0.0
        grad_norms = torch.norm(grads, dim=-1, p=2)
        max_scalings = torch.max(gaussians.get_scaling, dim=1).values
        opacities = gaussians.get_opacity.squeeze(-1)
        
        # Normalize the inputs
        #grad_norms = (grad_norms - grad_norms.mean()) / (grad_norms.std() + 1e-8)
        max_scalings = (max_scalings - max_scalings.mean()) / (max_scalings.std() + 1e-8)
        opacities = (opacities - opacities.mean()) / (opacities.std() + 1e-8)

        # Prepare input for the parameter network
        inputs = torch.cat([
            grad_norms.unsqueeze(-1),
            max_scalings.unsqueeze(-1),
            opacities.unsqueeze(-1)
        ], dim=-1)
        
        # Get action probabilities from the parameter network
        logits = self.param_network(inputs)
        action_probs = torch.softmax(logits, dim=-1)
        # Print logits and action probabilities for debugging
        #print("Logits: ", logits)
        #print("Action probabilities: ", action_probs)
        
        # Probabilistically sample k action candidates using rsample
        action_dist = torch.distributions.Categorical(action_probs)
        actions = action_dist.sample((self.k,)).to("cuda")   # using sample for discrete actions
        log_probs = action_dist.log_prob(actions)

        return actions, log_probs.squeeze()
