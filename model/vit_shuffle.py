'''
Descripttion: 
version: 1.0
Author: Areebol
Date: 2023-07-19 20:44:53
'''
import torch
import torch.nn as nn
import torchsummary

from model.layers import ShuffleTransformerEncoder, TransformerEncoder


class ShuffleViT(nn.Module):
    def __init__(self, in_c: int = 3, num_classes: int = 10, img_size: int = 32, patch: int = 8, dropout: float = 0.,
                 num_layers: int = 7, hidden: int = 384, mlp_hidden: int = 384*4, head: int = 8, is_cls_token: bool = True,
                 shuffle_num: int = 1, shuffle_channel: int = 64):
        super(ShuffleViT, self).__init__()
        # hidden=384

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
        enc_list = [TransformerEncoder(
            hidden, mlp_hidden=mlp_hidden, dropout=dropout, head=head) for _ in range(num_layers-shuffle_num)]
        self.enc = nn.Sequential(*enc_list)
        self.shuffle_encs = nn.ModuleList()
        for _ in range(shuffle_num):
            self.shuffle_encs.append(
                ShuffleTransformerEncoder(hidden, mlp_hidden=mlp_hidden, dropout=dropout, head=head, shuffle_channel=shuffle_channel))
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, num_classes)  # for cls_token
        )

    def forward(self, x,  is_shuffle: bool = True):
        # print("x.shape",x.shape)
        out = self._to_words(x)
        # print("x.shape",out.shape)
        out = self.emb(out)
        # print("x.shape",out.shape)
        if self.is_cls_token:
            out = torch.cat(
                [self.cls_token.repeat(out.size(0), 1, 1), out], dim=1)
        # print("x.shape",out.shape)
        out = out + self.pos_emb
        out = self.enc(out)
        # print("x.shape",out.shape)
        for shuffle_enc in self.shuffle_encs:
            out = shuffle_enc(out, is_shuffle)
        # print("x.shape",out.shape)
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
    net = ShuffleViT(in_c=c, num_classes=10, img_size=h, patch=16, dropout=0.1,
                     num_layers=7, hidden=384, head=12, mlp_hidden=384, is_cls_token=False)
    # out = net(x)
    # out.mean().backward()
    torchsummary.summary(net, (c, h, w))
    # print(out.shape)
