import torch
import numpy as np
from torchdyn.datasets import generate_moons
import math

from tensorclass import TensorMultiModal

class NGaussians:
    def __init__(self, dim=2, vocab_size=8):

        self.dim = dim
        self.num_gaussians = vocab_size


    def __call__(self, num_points, device='cpu') -> TensorMultiModal:

        num_points_per_gaussian = num_points // self.num_gaussians

        angle_step = 2 * np.pi / self.num_gaussians
        positions = []
        labels = []

        for i in range(self.num_gaussians):
            angle = i * angle_step
            center_x = np.cos(angle)
            center_y = np.sin(angle)
            normal = torch.distributions.multivariate_normal.MultivariateNormal(
                torch.zeros(self.dim), math.sqrt(0.1) * torch.eye(self.dim)
            )
            points = normal.sample((num_points_per_gaussian,))
            points += np.array([center_x, center_y]) * 5
            positions.append(points)
            labels += [i % self.num_gaussians] * num_points_per_gaussian

        positions = np.concatenate(positions, axis=0)
        positions = torch.tensor(positions, dtype=torch.float32)
        labels = np.array(labels)
        labels = torch.tensor(labels)
        idx = torch.randperm(len(labels))
        positions = positions[idx]
        colors = labels[idx][:,None].long()
        multimodal_state = TensorMultiModal(continuous=positions.unsqueeze(1), discrete=colors.unsqueeze(1))
        return multimodal_state.to(device)

class TwoMoons:
    def __init__(self, dim=2):
        self.dim = dim

    def __call__(self, num_points, device='cpu') -> TensorMultiModal:
        positions, labels = generate_moons(num_points, noise=0.2)
        idx = torch.randperm(len(labels))
        positions = positions[idx] * 3 - 1
        colors = torch.tensor(labels[idx], dtype=torch.long)[:,None]
        multimodal_state = TensorMultiModal(continuous=positions.unsqueeze(1), discrete=colors.unsqueeze(1))
        return multimodal_state.to(device)
