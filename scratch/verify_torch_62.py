import sys
sys.stdout.reconfigure(encoding='utf-8')

print('=== TORCH IMPORT TEST ===')
import torch
print('torch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())

# CPU tensor operation
t1 = torch.tensor([1.0, 2.0, 3.0])
t2 = torch.tensor([4.0, 5.0, 6.0])
t3 = t1 + t2
print('CPU tensor op (t1+t2):', t3.tolist())

print()
print('=== TORCHVISION IMPORT TEST ===')
import torchvision
print('torchvision version:', torchvision.__version__)

print()
print('=== NUMPY IMPORT TEST ===')
import numpy as np
print('numpy version:', np.__version__)

# numpy -> torch -> numpy round-trip
arr = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(arr)
arr2 = t.numpy()
print('numpy->torch->numpy round-trip:', arr2.tolist())

print()
print('ALL IMPORT TESTS PASSED')
