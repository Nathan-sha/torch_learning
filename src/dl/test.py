from torch import cuda

from src.dl.data.vision import make_mnist_datasets
from src.dl.models.mlp_custom import MLPCustom
from src.dl.models.cnn import CNNModel
import torch

from torch.utils.data import DataLoader

# dataset类型，每个样本为一个张量和一个label组成的元组tuple
train_set, val_set = make_mnist_datasets()

print("length of train_set", len(train_set))
print("length of test_set", len(val_set))

# x,y = train_set[0]
# print(type(x), x.shape, x.min().item(), x.max().item(), y)

# print(type(train_set[0]), type(val_set[0]))

model_input_dim: int = 28 * 28
label_num: int = 10


# 按batch取样
train_loader = DataLoader(train_set, 64, shuffle=True)
val_loader = DataLoader(val_set, 64, shuffle=False)

# print(type(train_loader.dataset), type(val_loader))

model_mlp_custom = MLPCustom(model_input_dim, label_num).to(cuda)

optimizer = torch.optim.Adam(model_mlp_custom.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(100):
    model_mlp_custom.train()

