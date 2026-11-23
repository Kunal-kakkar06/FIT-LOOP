import sys, platform, os
sys.stdout.reconfigure(encoding='utf-8')

print('=== PYTHON ===')
print('Executable:', sys.executable)
print('Version:', sys.version)
print('Platform:', platform.platform())
print()

print('=== INSTALLED PACKAGES ===')
import importlib
pkgs = [
    ('numpy', 'numpy'),
    ('pandas', 'pandas'),
    ('PIL', 'Pillow'),
    ('cv2', 'opencv-python'),
    ('scipy', 'scipy'),
    ('matplotlib', 'matplotlib'),
    ('torch', 'torch'),
    ('torchvision', 'torchvision'),
    ('torchaudio', 'torchaudio'),
    ('ultralytics', 'ultralytics'),
    ('sklearn', 'scikit-learn'),
    ('tqdm', 'tqdm'),
    ('yaml', 'pyyaml'),
    ('IPython', 'IPython'),
]
for imp, name in pkgs:
    try:
        mod = importlib.import_module(imp)
        ver = getattr(mod, '__version__', 'unknown')
        print(f'  {name}: {ver}')
    except ImportError:
        print(f'  {name}: NOT INSTALLED')

print()
print('=== VIRTUAL ENVIRONMENT ===')
print('VIRTUAL_ENV:', os.environ.get('VIRTUAL_ENV', 'not set'))
print('sys.prefix:', sys.prefix)
print('sys.base_prefix:', sys.base_prefix)
venv_active = sys.prefix != sys.base_prefix
print('venv active:', venv_active)

print()
print('=== REQUIREMENTS FILE ===')
req_path = r'C:\Users\Lenovo\Downloads\Fitloop\requirements.txt'
if os.path.exists(req_path):
    with open(req_path, encoding='utf-8') as f:
        print(f.read())
else:
    print('requirements.txt NOT FOUND')

# Check for other envs
print()
print('=== OTHER VENV DIRECTORIES IN PROJECT ===')
root = r'C:\Users\Lenovo\Downloads\Fitloop'
for name in os.listdir(root):
    p = os.path.join(root, name)
    if os.path.isdir(p) and os.path.exists(os.path.join(p, 'Scripts', 'python.exe')):
        print(f'  Found venv: {p}')

print()
print('=== DISK SPACE ===')
import shutil
total, used, free = shutil.disk_usage('C:')
print(f'  C: total={total/1e9:.1f} GB  used={used/1e9:.1f} GB  free={free/1e9:.1f} GB')

print()
print('=== MEMORY ===')
try:
    import psutil
    vm = psutil.virtual_memory()
    print(f'  total={vm.total/1e9:.2f} GB  available={vm.available/1e9:.2f} GB  used={vm.percent}%')
except ImportError:
    print('  psutil not installed; skipping memory check')
