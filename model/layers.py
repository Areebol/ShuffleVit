import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import deepspeed
from tutel import moe


class TransformerEncoder(nn.Module):
    def __init__(self, feats: int, mlp_hidden: int, head: int = 8, dropout: float = 0.):
        super(TransformerEncoder, self).__init__()
        self.la1 = nn.LayerNorm(feats)
        self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout)
        self.la2 = nn.LayerNorm(feats)
        self.mlp = nn.Sequential(
            nn.Linear(feats, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, feats),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.msa(self.la1(x)) + x
        out = self.mlp(self.la2(out)) + out
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feats: int, head: int = 8, dropout: float = 0.):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.feats = feats
        self.sqrt_d = self.feats**0.5

        self.q = nn.Linear(feats, feats)
        self.k = nn.Linear(feats, feats)
        self.v = nn.Linear(feats, feats)

        self.o = nn.Linear(feats, feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, f = x.size()
        q = self.q(x).view(b, n, self.head, self.feats //
                           self.head).transpose(1, 2)
        k = self.k(x).view(b, n, self.head, self.feats //
                           self.head).transpose(1, 2)
        v = self.v(x).view(b, n, self.head, self.feats //
                           self.head).transpose(1, 2)

        score = F.softmax(torch.einsum("bhif, bhjf->bhij", q,
                          k)/self.sqrt_d, dim=-1)  # (b,h,n,n)
        attn = torch.einsum("bhij, bhjf->bihf", score, v)  # (b,n,h,f//h)
        o = self.dropout(self.o(attn.flatten(2)))
        return o


class MultiHeadDepthwiseSelfAttention(nn.Module):
    def __init__(self, feats: int, head: int = 8, dropout: float = 0):
        super(MultiHeadDepthwiseSelfAttention, self).__init__()
        ...

    def forward(self, x):

        ...


class ShuffleAndRetrieve(nn.Module):
    def __init__(self, shuffle_channel: int = 64) -> None:
        super(ShuffleAndRetrieve, self).__init__()
        self.shuffle_channel = shuffle_channel

    def forward(self, input: torch.Tensor):
        # feature 2
        total = input.size(2)
        assert (int(total) >= self.shuffle_channel)

        # 随机打乱shuffle_channel个channel
        random_sort = torch.randperm(total)[:self.shuffle_channel]
        # 获取原来的位置
        random_index, _ = random_sort.sort()

        index = torch.arange(0, total, dtype=torch.long)

        # 填充打乱后的index
        index[random_index] = random_sort
        # 打乱返回
        return input[:, :, index]


class ShuffleTransformerEncoder(nn.Module):
    def __init__(self, feats: int, mlp_hidden: int, head: int = 8, dropout: float = 0., shuffle_channel: int = 64):
        super(ShuffleTransformerEncoder, self).__init__()
        self.la1 = nn.LayerNorm(feats)
        self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout)
        self.la2 = nn.LayerNorm(feats)
        self.mlp = nn.Sequential(
            nn.Linear(feats, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, feats),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Shuffle only fixed num
        self.shuffle = ShuffleAndRetrieve(shuffle_channel)

    def forward(self, x, is_shuffle: bool = True):
        out = self.msa(self.la1(x)) + x
        out = self.mlp(self.la2(out)) + out
        if is_shuffle:
            out = self.shuffle(out)
        return out


class TutelMoETransformerEncoder(nn.Module):
    def __init__(self, feats: int, mlp_hidden: int, head: int = 8, dropout: float = 0., args=None, num_experts: int = 2,
                 ep_world_size: int = 1, top_k: int = 1, min_capacity: int = 0, noisy_gate_policy: str = None):
        super(TutelMoETransformerEncoder, self).__init__()
        self.la1 = nn.LayerNorm(feats)
        self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout)
        self.la2 = nn.LayerNorm(feats)
        self.mlp = nn.Sequential(
            nn.Linear(feats, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, feats),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.MoE = moe.moe_layer(
            gate_type={'type': 'top', 'k': top_k},
            model_dim=feats,
            experts={
                'count_per_node': num_experts,
                'type': 'ffn', 'hidden_size_per_expert': mlp_hidden, 'activation_fn': lambda x: torch.nn.functional.gelu(x)
            },
            scan_expert_func=lambda name, param: setattr(
                param, 'skip_allreduce', True),
        )

    def forward(self, x):
        out = self.la2(self.msa(self.la1(x)) + x)
        out = self.MoE(out)
        return out
    
# TODO 
# Replace tutel to dp 
class DpMoETransformerEncoder(nn.Module):
    def __init__(self, feats: int, mlp_hidden: int, head: int = 8, dropout: float = 0., args=None, num_experts: int = 2,
                 ep_world_size: int = 1, top_k: int = 1, min_capacity: int = 0, noisy_gate_policy: str = None):
        super(TutelMoETransformerEncoder, self).__init__()
        self.la1 = nn.LayerNorm(feats)
        self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout)
        self.la2 = nn.LayerNorm(feats)
        self.mlp = nn.Sequential(
            nn.Linear(feats, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, feats),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # TODO 
        # Replace tutel moe by dp moe
        # ...

    def forward(self, x):
        out = self.la2(self.msa(self.la1(x)) + x)
        out = self.MoE(out)
        return out  


if __name__ == "__main__":
    b, n, f = 4, 16, 128
    x = torch.randn(b, n, f)
    # net = MultiHeadSelfAttention(f)
    net = TransformerEncoder(f)
    torchsummary.summary(net, (n, f))
    # out = net(x)
    # print(out.shape)
