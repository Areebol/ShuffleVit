'''
Descripttion: 
version: 1.0
Author: Areebol
Date: 2023-07-20 00:10:58
'''
import torch
import torch.nn as nn
import torchsummary

from model.layers import TransformerEncoder, DpMoETransformerEncoder


class MoEViT(nn.Module):
    def __init__(self, args):
        super(MoEViT, self).__init__()
        # hidden=384
        # Set model config
        in_c = args.in_c
        num_classes=args.num_classes
        img_size=args.img_size
        patch=args.patch
        dropout=args.dropout
        num_layers=args.num_layers
        hidden=args.hidden
        mlp_hidden=args.mlp_hidden
        head=args.head
        is_cls_token=args.is_cls_token
        num_experts=args.num_experts
        ep_world_size=args.ep_world_size
        top_k=args.top_k
        min_capacity=args.min_capacity
        noisy_gate_policy=args.noisy_gate_policy
        self.patch = patch  # number of patches in one row(or col)
        self.is_cls_token = is_cls_token
        self.patch_size = img_size//self.patch
        f = (img_size//self.patch)**2*3  # 48 # patch vec length
        num_tokens = (self.patch**2) + \
            1 if self.is_cls_token else (self.patch**2)

        self.emb = nn.Linear(f, hidden)  # (b, n, f)
        self.cls_token = nn.Parameter(torch.randn(
            1, 1, hidden)) if is_cls_token else None
        self.pos_emb = nn.Parameter(torch.randn(1, num_tokens, hidden))
        enc_list = []
        for _ in range(num_layers-len(num_experts)):
            enc_list.append(TransformerEncoder(
                hidden, mlp_hidden=mlp_hidden, dropout=dropout, head=head))
        for n_e in num_experts:
            if (n_e > 1):
                # Change to Dp transformer
                enc_list.append(DpMoETransformerEncoder(args))
            else:
                enc_list.append(TransformerEncoder(
                    hidden, mlp_hidden=mlp_hidden, dropout=dropout, head=head))

        self.enc = nn.Sequential(*enc_list)
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, num_classes)  # for cls_token
        )

    def forward(self, x):
        out = self._to_words(x)
        out = self.emb(out)
        if self.is_cls_token:
            out = torch.cat(
                [self.cls_token.repeat(out.size(0), 1, 1), out], dim=1)
        out = out + self.pos_emb
        out = self.enc(out)
        if self.is_cls_token:
            out = out[:, 0]
        else:
            out = out.mean(1)
        out = self.fc(out)
        return out

    def _to_words(self, x):
        """
        (b, c, h, w) -> (b, n, f)
        """
        out = x.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size).permute(0, 2, 3, 4, 5, 1)
        out = out.reshape(x.size(0), self.patch**2, -1)
        return out


if __name__ == "__main__":
    b, c, h, w = 4, 3, 32, 32
    x = torch.randn(b, c, h, w)

    net = MoEViT(in_c=c, num_classes=10, img_size=h, patch=16, dropout=0.1,
                 num_layers=7, hidden=384, head=12, mlp_hidden=384, is_cls_token=False)
    torchsummary.summary(net, (c, h, w))
