# CUDA test

import sys
import torch

def main():
    # 1. Wersja Pythona 3.11.9
    print("Python version:", sys.version.split()[0])

    # 2. Wersja PyTorch 2.8.0+cu129
    print("PyTorch version:", torch.__version__)

    # 3. Czy CUDA jest dostępna? True
    cuda_ok = torch.cuda.is_available()
    print("CUDA available:", cuda_ok)

    # 4. Jeśli tak, to dodatkowe info o urządzeniach
    if cuda_ok:
        count = torch.cuda.device_count()
        print(f"Number of CUDA devices: {count}")
        for i in range(count):
            name = torch.cuda.get_device_name(i)
            capability = torch.cuda.get_device_capability(i)
            print(f"  - Device {i}: {name}, Compute Capability: {capability}")

if __name__ == "__main__":
    main()
