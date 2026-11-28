"""
run_all.py -- reproduce every table and figure of the analysis in dependency order.

    python run_all.py          (about 10 s; all outputs go to tables/ and figures/)
"""
import importlib
import subprocess
import sys
import glob
import os

ANALYSIS = ["inventory", "symmetrize", "sdh", "hall", "gate", "temperature_vs_field", "rt_model", "dingle", "summary"]

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    for name in ANALYSIS:
        print(f"[analysis] {name}")
        mod = importlib.import_module(name)
        (mod.run if hasattr(mod, "run") else mod.build)()
    for fig in sorted(glob.glob(os.path.join(here, "fig*.py"))):
        print(f"[figure]   {os.path.basename(fig)}")
        subprocess.run([sys.executable, fig], check=True, cwd=here)
    print("[figure]   fig04_hall_plateaus.py (full-width variant)")
    subprocess.run([sys.executable, os.path.join(here, "fig04_hall_plateaus.py")], check=True, cwd=here,
                   env={**os.environ, "FIG04_WIDE": "1"})
    print("[checks]")
    subprocess.run([sys.executable, os.path.join(here, "checks.py")], check=True, cwd=here)
