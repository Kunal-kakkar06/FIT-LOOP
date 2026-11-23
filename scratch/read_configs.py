import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

for p in [
    Path(r"C:\Users\Lenovo\Downloads\Fitloop\requirements.txt"),
    Path(r"C:\Users\Lenovo\Downloads\Fitloop\wardrobe_ml\requirements.txt"),
    Path(r"C:\Users\Lenovo\Downloads\Fitloop\wardrobe_ml\config.py"),
]:
    print(f"==================== {p.name} ({p}) ====================")
    if p.exists():
        print(p.read_text(encoding='utf-8'))
    else:
        print("DOES NOT EXIST")
    print()
