import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
log_path = r"C:\Users\Lenovo\.gemini\antigravity\brain\57b5c7be-d326-40ec-a129-e53bc9ecb41d\.system_generated\logs\transcript_full.jsonl"
with open(log_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if 'STEP 5H.3' in line and 'USER_INPUT' in line:
            data = json.loads(line)
            print(f"Step {i}:")
            print(data.get('content', ''))
            print("="*60)
