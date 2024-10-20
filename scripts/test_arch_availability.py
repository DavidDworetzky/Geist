import torch

if __name__ == "__main__":
    print(torch.__version__)
    print(f'torch.cuda.is_available(): {torch.cuda.is_available()}')
    print(f'torch.backends.mps.is_available(): {torch.backends.mps.is_available()}')
    print(f'torch.backends.mps.is_built(): {torch.backends.mps.is_built()}')