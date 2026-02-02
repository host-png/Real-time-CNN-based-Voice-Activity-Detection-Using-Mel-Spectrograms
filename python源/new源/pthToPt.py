import torch

from CNNnetWork import CnnNetVoice

model = CnnNetVoice()
model.load_state_dict(torch.load(r"cnnNetVoice_9.pth", map_location="cpu"))
model.eval()

# 2. 转 TorchScript
example_input = torch.randn(1, 1, 50,2)  #cov2d(1,1,50,2)
traced_model = torch.jit.trace(model, example_input)

# 3. 保存 TorchScript
traced_model.save("cnnNetVoice_9.pt")
