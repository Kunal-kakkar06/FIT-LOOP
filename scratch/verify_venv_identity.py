import sys, os
sys.stdout.reconfigure(encoding='utf-8')
print('Python executable:', sys.executable)
print('sys.prefix:', sys.prefix)
print('sys.base_prefix:', sys.base_prefix)
print('venv active (prefix != base_prefix):', sys.prefix != sys.base_prefix)
print('VIRTUAL_ENV env var:', os.environ.get('VIRTUAL_ENV', 'not set'))

# Confirm it is NOT the .venv
assert '.venv' not in sys.prefix, f"WRONG: prefix points to .venv: {sys.prefix}"
assert 'fitloop_yolo_env' in sys.prefix, f"WRONG: not in fitloop_yolo_env: {sys.prefix}"
print()
print('PASS: interpreter is inside fitloop_yolo_env, not .venv')
