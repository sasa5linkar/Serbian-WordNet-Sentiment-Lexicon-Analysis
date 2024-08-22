import torch

def check_cuda():
    if torch.cuda.is_available():
        print(f"CUDA is available. GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("CUDA is not available. Please check your installation.")

check_cuda()
