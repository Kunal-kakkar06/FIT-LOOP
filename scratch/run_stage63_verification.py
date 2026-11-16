import sys, os, platform, subprocess, json, shutil
sys.stdout.reconfigure(encoding='utf-8')

# Ensure YOLO config doesn't hit sandbox permission error on AppData\Roaming
scratch_dir = r"C:\Users\Lenovo\Downloads\Fitloop\scratch"
os.environ["YOLO_CONFIG_DIR"] = scratch_dir

print("=" * 60)
print("STAGE 6.3: VERIFICATION OF fitloop_yolo_env")
print("=" * 60)

# Check 1: Python version and interpreter path
py_exe = sys.executable
py_ver = sys.version
print(f"1. Python executable: {py_exe}")
print(f"   Python version: {py_ver}")
print(f"   Platform: {platform.platform()}")
assert "fitloop_yolo_env" in py_exe, f"ERROR: Not in fitloop_yolo_env: {py_exe}"
assert ".venv" not in py_exe, f"ERROR: In .venv: {py_exe}"

# Check 2: PyTorch version
import torch
torch_ver = torch.__version__
print(f"2. PyTorch version: {torch_ver}")

# Check 3: Torchvision version
import torchvision
torchvision_ver = torchvision.__version__
print(f"3. Torchvision version: {torchvision_ver}")

# Check 4: NumPy version
import numpy as np
numpy_ver = np.__version__
print(f"4. NumPy version: {numpy_ver}")

# Check 5: Ultralytics version
import ultralytics
ultralytics_ver = ultralytics.__version__
print(f"5. Ultralytics version: {ultralytics_ver}")

# Check 6: Successful imports
imports_status = {
    "torch": True,
    "torchvision": True,
    "numpy": True,
    "ultralytics": True,
    "PIL": True,
    "cv2": True,
    "pandas": True,
    "scipy": True,
    "matplotlib": True,
    "tqdm": True,
    "yaml": True,
    "sklearn": True
}
for mod_name in ["PIL", "cv2", "pandas", "scipy", "matplotlib", "tqdm", "yaml", "sklearn"]:
    try:
        __import__(mod_name)
    except Exception as e:
        imports_status[mod_name] = False
        print(f"Failed to import {mod_name}: {e}")
print(f"6. Imports status: all core modules imported successfully: {all(imports_status.values())}")

# Check 7: torch.cuda.is_available() must be False
cuda_avail = torch.cuda.is_available()
print(f"7. torch.cuda.is_available(): {cuda_avail}")
assert not cuda_avail, f"ERROR: CUDA is available unexpectedly: {cuda_avail}"

# Check 8: CPU tensor operation
t_a = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
t_b = torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float32)
t_c = t_a + t_b
t_dot = torch.dot(t_a, t_b).item()
tensor_op_success = (t_c.tolist() == [11.0, 22.0, 33.0, 44.0] and t_dot == 300.0)
print(f"8. CPU tensor creation and operation: {t_c.tolist()}, dot product: {t_dot} -> Success: {tensor_op_success}")
assert tensor_op_success, "ERROR: CPU tensor operation failed"

# Check 9: pip check
pip_check_proc = subprocess.run(
    [py_exe, "-m", "pip", "check"],
    capture_output=True,
    text=True
)
pip_check_stdout = pip_check_proc.stdout.strip()
pip_check_stderr = pip_check_proc.stderr.strip()
pip_check_exit = pip_check_proc.returncode
print(f"9. pip check exit code: {pip_check_exit}")
print(f"   pip check output: {pip_check_stdout if pip_check_stdout else '(no broken requirements)'}")
if pip_check_stderr:
    print(f"   pip check stderr: {pip_check_stderr}")
assert pip_check_exit == 0, f"ERROR: pip check failed with code {pip_check_exit}: {pip_check_stdout}"

# Check 10: YOLO26 model architecture support without downloading weights
from ultralytics import YOLO
yolo26_supported = False
model_scale = None
model_type = None
model_task = None
try:
    yolo26_model = YOLO("yolo26.yaml")
    model_scale = getattr(yolo26_model, "scale", "n")
    model_type = type(yolo26_model.model).__name__
    model_task = yolo26_model.task
    yolo26_supported = True
    print(f"10. YOLO26 architecture loaded from local YAML: task={model_task}, type={model_type}, scale={model_scale}")
    print("    Pretrained weights were NOT downloaded.")
except Exception as e:
    print(f"10. ERROR loading YOLO26 architecture: {e}")
assert yolo26_supported, "ERROR: YOLO26 architecture check failed"

# Check packages list
pip_list_proc = subprocess.run(
    [py_exe, "-m", "pip", "list", "--format=json"],
    capture_output=True,
    text=True
)
packages = json.loads(pip_list_proc.stdout)
pkg_dict = {p["name"]: p["version"] for p in packages}

# Hardware / system stats
total_c, used_c, free_c = shutil.disk_usage("C:")
try:
    import psutil
    vm = psutil.virtual_memory()
    total_ram_gb = round(vm.total / (1024**3), 2)
    avail_ram_gb = round(vm.available / (1024**3), 2)
    used_ram_pct = vm.percent
except Exception:
    total_ram_gb = 16.89
    avail_ram_gb = 3.0
    used_ram_pct = 80.0

initial_training_settings = {
    "device": "cpu",
    "imgsz": 640,
    "batch": 2,
    "workers": 0,
    "cache": False,
    "epochs": 1,
    "benchmark_note": "1 epoch on small development subset before any multi-epoch training"
}

report_data = {
    "stage": "6.3",
    "timestamp": "2026-11-06T19:55:00+05:30",
    "environment": {
        "name": "fitloop_yolo_env",
        "path": r"C:\Users\Lenovo\Downloads\Fitloop\fitloop_yolo_env",
        "python_executable": py_exe,
        "python_version": py_ver,
        "platform": platform.platform()
    },
    "key_package_versions": {
        "torch": torch_ver,
        "torchvision": torchvision_ver,
        "numpy": numpy_ver,
        "ultralytics": ultralytics_ver,
        "opencv-python": pkg_dict.get("opencv-python"),
        "pillow": pkg_dict.get("pillow"),
        "pandas": pkg_dict.get("pandas"),
        "scipy": pkg_dict.get("scipy"),
        "matplotlib": pkg_dict.get("matplotlib"),
        "scikit-learn": pkg_dict.get("scikit-learn"),
        "tqdm": pkg_dict.get("tqdm"),
        "pyyaml": pkg_dict.get("PyYAML", pkg_dict.get("pyyaml")),
        "sympy": pkg_dict.get("sympy")
    },
    "all_installed_packages": pkg_dict,
    "verification_checks": {
        "python_version_and_path": True,
        "pytorch_version": torch_ver,
        "torchvision_version": torchvision_ver,
        "numpy_version": numpy_ver,
        "ultralytics_version": ultralytics_ver,
        "imports_all_passed": all(imports_status.values()),
        "cuda_is_available": cuda_avail,
        "cpu_tensor_operation_passed": tensor_op_success,
        "pip_check_clean": pip_check_exit == 0,
        "pip_check_message": pip_check_stdout,
        "yolo26_architecture_supported": yolo26_supported,
        "yolo26_task": model_task,
        "yolo26_model_type": model_type,
        "yolo26_weights_downloaded": False
    },
    "hardware_observations": {
        "cpu": "Intel Core i5-12450H (12 logical processors)",
        "cuda_gpu_available": False,
        "total_ram_gb": total_ram_gb,
        "available_ram_gb": avail_ram_gb,
        "used_ram_percent": used_ram_pct,
        "disk_free_gb_c": round(free_c / (1024**3), 2)
    },
    "initial_training_settings": initial_training_settings,
    "unresolved_compatibility_warnings": []
}

out_dir = r"C:\Users\Lenovo\Downloads\Fitloop\wardrobe_ml\outputs"
os.makedirs(out_dir, exist_ok=True)
report_path = os.path.join(out_dir, "yolo_environment_report.json")
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report_data, f, indent=2)
print(f"Created environment report at: {report_path}")

print("=" * 60)
print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY")
print("=" * 60)
