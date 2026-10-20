"""Record the repository's known vLLM/transformers metadata exception only."""

import json
import subprocess
import sys
from pathlib import Path

result = subprocess.run([sys.executable, "-m", "pip", "check"], text=True, capture_output=True)
lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
known = "vllm 0.18.0 has requirement transformers<5,>=4.56.0, but you have transformers 5.5.0."
unexpected = [line for line in lines if line != known and line != "No broken requirements found."]
report = {"pip_check_exit_code": result.returncode, "output": lines, "stderr": result.stderr,
          "known_metadata_exception": known in lines, "unexpected_conflicts": unexpected,
          "runtime_compatibility_verified": False}
path = Path(__file__).resolve().parents[1] / "logs/dependency-metadata.json"
path.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
if unexpected or (result.returncode and not lines):
    sys.exit(1)
