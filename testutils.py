import torch

print(torch.cuda.is_available())
print(torch.cuda.device_count())

device = torch.device("cuda:0")

print(device)