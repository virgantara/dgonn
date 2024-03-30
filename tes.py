import torch

# Check if CUDA is available
print("CUDA Available:", torch.cuda.is_available())

# Get the number of GPUs available
num_gpus = torch.cuda.device_count()
print("Number of GPUs:", num_gpus)

# List all available GPUs
for i in range(num_gpus):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


x = torch.rand(5, 5).cuda()
print(x)