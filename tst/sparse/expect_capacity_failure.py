"""Capacity exhaustion must fail loudly instead of silently dropping fields."""
import os
import subprocess
import sys

if os.name == "posix":
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
result = subprocess.run([sys.argv[1], "--overflow", "1"],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=45)
output = result.stdout.decode(errors="replace")
if result.returncode == 0 or "Sparse device allocation capacity exceeded" not in output:
    print(output)
    raise SystemExit("Expected explicit device capacity failure")
print("Device capacity exhaustion failed explicitly as expected")
