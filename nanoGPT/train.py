import torch
import torch.nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences we will process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Device:", device)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("GPU name:", torch.cuda.get_device_name(0))
print("GPU index:", torch.cuda.current_device())
print("PyTorch CUDA version:", torch.version.cuda)
