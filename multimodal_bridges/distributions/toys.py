import torch
import os
import json
import torch
import numpy as np
from torchdyn.datasets import generate_moons
import matplotlib.pyplot as plt
import math

from tensorclass import TensorMultiModal

class NGaussians:
    def __init__(
        self,
        dim=2,
        num_gaussians=8,
        num_colors=8,
        num_points_per_gaussian=1000,
        std_dev=0.1,
        scale=5,
        labels_as_state=False,
        labels_as_context=False,
        random_colors=False,
    ):
        self.dim = dim
        
        self.num_points_per_gaussian = num_points_per_gaussian
        self.num_gaussians = num_gaussians
        self.N = num_gaussians * num_points_per_gaussian

        self.num_colors = num_colors if num_colors > 0 else 1


        self.std_dev = std_dev
        self.scale = scale
        self.random_colors = random_colors


    def __call__(self, device) -> TensorMultiModal:

        angle_step = 2 * np.pi / self.num_gaussians
        positions = []
        labels = []

        for i in range(self.num_gaussians):
            angle = i * angle_step
            center_x = np.cos(angle)
            center_y = np.sin(angle)
            normal = torch.distributions.multivariate_normal.MultivariateNormal(
                torch.zeros(self.dim), math.sqrt(self.std_dev) * torch.eye(self.dim)
            )
            points = normal.sample((self.num_points_per_gaussian,))
            points += np.array([center_x, center_y]) * self.scale
            positions.append(points)
            labels += [i % self.num_colors] * self.num_points_per_gaussian

        positions = np.concatenate(positions, axis=0)
        positions = torch.tensor(positions, dtype=torch.float32)
        labels = np.array(labels)
        labels = torch.tensor(labels)
        idx = torch.randperm(len(labels))
        continuous = positions[idx]
        discrete = (
            np.random.randint(0, self.num_gaussians, self.N)
            if self.random_colors
            else labels[idx]
        )
        
        time, mask = None, None
        multimodal_state = TensorMultiModal(time, continuous, discrete.long(), mask)
        return multimodal_state.to(device)