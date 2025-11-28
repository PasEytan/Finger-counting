import torch

# Just to see if cuda (Nvidia gpu drivers for ai training) are working correctly 
# You would need to go to the PyTorch website and install the correct version for your OS
print(torch.cuda.is_available())