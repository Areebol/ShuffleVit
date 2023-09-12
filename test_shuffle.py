
import torch
import torch.nn as nn


class ShuffleAndRetrieve(nn.Module):
    def __init__(self, shuffle_num: int) -> None:
        super(ShuffleAndRetrieve, self).__init__()
        self.shuffle_num = shuffle_num

    def forward(self, input: torch.Tensor):
        # feature 2
        total = input.size(2)

        # 随机打乱shuffle_num个channel
        random_sort = torch.randperm(total)[:self.shuffle_num]
        # 获取原来的位置
        random_index, _ = random_sort.sort()

        index = torch.arange(0, total, dtype=torch.long)

        # 填充打乱后的index
        index[random_index] = random_sort
        # 打乱返回
        return input[:, :, index]


# 示例输入张量，假设batch_size=1，num_channels=10，width=8
input_tensor = torch.arange(1, 11).reshape((1, 1, 10))

# 随机打乱并保留6个通道
shuffle = ShuffleAndRetrieve(4)
# shuffled_tensor = shuffle(input_tensor)

# print("原始张量:")
# print(input_tensor)

# print("随机打乱并保留6个通道后的张量:")
# print(shuffled_tensor)
