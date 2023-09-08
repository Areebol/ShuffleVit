import torch.nn as nn
import torch
from tutel import moe
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.moe = moe.moe_layer(
            gate_type={'type': 'top', 'k': 2},
            model_dim=84,
            experts={
                'count_per_node': 2,
                'type': 'ffn', 'hidden_size_per_expert': 2048, 'activation_fn': lambda x: torch.nn.functional.relu(x)
            },
            scan_expert_func=lambda name, param: setattr(
                param, 'skip_allreduce', True),
        )
        # create moe layers based on the number of experts
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.moe(x)
        x = self.fc3(x)
        return x
