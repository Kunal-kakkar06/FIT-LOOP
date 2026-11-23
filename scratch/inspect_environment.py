import os
import sys
import platform
import shutil
import psutil
import json
import subprocess

sys.stdout.reconfigure(encoding='utf-8')

print("="*60)
print("TASK 1: HARDWARE INSPECTION")
print("="*60)

# CPU
print(f"Platform: {platform.platform()}")
print(f"Processor: {platform.processor()}")
print(f"CPU count (logical): {psutil.cpu_count(logical=True)}")
print(f"CPU count (physical): {psutil.cpu_count(logical=False)}")

# RAM
mem = psutil.virtual_memory()
print(f"Total RAM: {mem.total / (1024**3):.2f} GB")
print(f"Available RAM: {mem.available / (1024**3):.2f} GB")

# Disk
disk = shutil.disk_usage("C:\\")
print(f"Drive C Total: {disk.total / (1024**3):.2f} GB")
print(f"Drive C Used: {disk.used / (1024**3):.2f} GB")
print(f"Drive C Free: {disk.free / (1024**3):.2f} GB")

# GPU via nvidia-smi
print("\nChecking GPU / NVIDIA driver:")
try:
    res = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
    if res.returncode == 0:
        print("nvidia-smi output:")
        print(res.stdout)
    else:
        print("nvidia-smi returned non-zero code:", res.returncode)
        print(res.stderr)
except FileNotFoundError:
    print("nvidia-smi NOT found in PATH.")

# Also check WMI / CIM for video controller
try:
    res = subprocess.run(["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM, DriverVersion, Status | Format-List"], capture_output=True, text=True, check=False)
    print("Video Controllers (WMI):")
    print(res.stdout)
except Exception as e:
    print("WMI check error:", e)

print("\n" + "="*60)
print("TASK 2: PYTHON ENVIRONMENT INSPECTION")
print("="*60)
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

packages = [
    ("torch", "PyTorch"),
    ("torchvision", "Torchvision"),
    ("ultralytics", "Ultralytics"),
    ("numpy", "NumPy"),
    ("cv2", "OpenCV"),
    ("pandas", "Pandas"),
    ("PIL", "Pillow"),
    ("scipy", "SciPy"),
    ("matplotlib", "Matplotlib")
]

for mod_name, disp_name in packages:
    try:
        mod = __import__(mod_name)
        ver = getattr(mod, "__version__", "unknown")
        print(f"✅ {disp_name:<15}: {ver} (Import succeeded)")
    except ImportError as e:
        print(f"❌ {disp_name:<15}: NOT INSTALLED ({e})")
    except Exception as e:
        print(f"⚠️ {disp_name:<15}: Import failed with error: {e}")

# Check PyTorch CUDA specifically if torch is imported
try:
    import torch
    print(f"\nPyTorch CUDA details:")
    print(f"  CUDA available   : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  Device count     : {torch.cuda.device_count()}")
        print(f"  Current device   : {torch.cuda.current_device()}")
        print(f"  Device name      : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM allocated   : {torch.cuda.memory_allocated(0)/(1024**2):.2f} MB")
        print(f"  VRAM reserved    : {torch.cuda.memory_reserved(0)/(1024**2):.2f} MB")
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  Total VRAM       : {total_vram:.2f} GB")
        print(f"  CUDA capability  : {torch.cuda.get_device_capability(0)}")
        print(f"  PyTorch CUDA build: {torch.version.cuda}")
    else:
        print(f"  PyTorch built with CUDA: {torch.version.cuda}")
except Exception as e:
    pass
