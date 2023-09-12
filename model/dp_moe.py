import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import deepspeed
from deepspeed.accelerator import get_accelerator
import torch.nn as nn
import torch.nn.functional as F

# TODO Add dp's moe
ep_size = 2
num_experts = [2]
top_k = 1
noisy_gate_policy = 'RSample'
mlp_type = None
min_capacity = None


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        if True:
            fc3 = nn.Linear(84, 84)
            self.moe_layer_list = []
            for n_e in num_experts:
                # create moe layers based on the number of experts
                self.moe_layer_list.append(
                    deepspeed.moe.layer.MoE(
                        hidden_size=84,
                        expert=fc3,
                        num_experts=n_e,
                        ep_size=ep_size,
                        use_residual=mlp_type == 'residual',
                        k=top_k,
                        # min_capacity=min_capacity,
                        noisy_gate_policy=noisy_gate_policy))
            self.moe_layer_list = nn.ModuleList(self.moe_layer_list)
            self.fc4 = nn.Linear(84, 10)
        else:
            self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if True:
            for layer in self.moe_layer_list:
                x, _, _ = layer(x)
            x = self.fc4(x)
        else:
            x = self.fc3(x)
        return x
