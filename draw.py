"""
Draw the model graph 
"""
from model.vit import ViT
from tensorboardX import SummaryWriter
import torch

feats = 8
mlp_hidden = 10
dummy_input = torch.rand(2,3,32,32)
# Set vit model
model = ViT()
with SummaryWriter(comment='Transformer block') as w:
    w.add_graph(model,(dummy_input,))