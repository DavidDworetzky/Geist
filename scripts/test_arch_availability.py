import torch

if __name__ == "__main__":
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.backends.mps.is_available())
    print(torch.backend.mps.is_built())