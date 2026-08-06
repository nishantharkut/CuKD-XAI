"""
Resume script for CuKD-XAI: runs ONLY Student B seeds 8192 and 9999.

This script:
1. Reads the EXACT executed source snapshot
2. Changes ONLY the SEEDS line to [8192, 9999]
3. Writes a temporary copy in the same directory
4. Runs it via subprocess with correct env vars
5. Verifies checkpoint creation
6. Cleans up the temporary file

NO existing files are modified.
"""
import os
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime

# === PATHS ===
REPO_ROOT = Path(r"C:\N Drive\Research\Cukd-XAI\CuKD-XAI")
EXECUTED_SRC = (
    REPO_ROOT / "results" / "wsnds" / "leakage_free_rerun" / "main_10seed"
    / "executed_source_snapshot" / "run_leakage_free_wsnds.executed.py"
)
OUTPUT_DIR = REPO_ROOT / "results" / "wsnds" / "leakage_free_rerun" / "main_10seed"
PYTHON_EXE = (
    REPO_ROOT / "experiments" / "wsnds" / "leakage_free_rerun"
    / ".venv" / "Scripts" / "python.exe"
)
LOG_FILE = OUTPUT_DIR / "resume_student_B_8192_9999.log"

# === SAFETY CHECKS ===
assert EXECUTED_SRC.exists(), f"Executed source not found: {EXECUTED_SRC}"
assert PYTHON_EXE.exists(), f"Python venv not found: {PYTHON_EXE}"
assert (REPO_ROOT / "data" / "wsnds" / "WSN-DS.csv").exists(), "WSN-DS.csv not found"

for seed in [8192, 9999]:
    existing = OUTPUT_DIR / f"checkpoint_student_B_seed_{seed}.json"
    if existing.exists():
        print(f"WARNING: {existing.name} already exists ({existing.stat().st_size} bytes).")
        print("         It will be overwritten by this run.")

# === READ AND PATCH SEEDS ===
code = EXECUTED_SRC.read_text(encoding="utf-8-sig")  # utf-8-sig handles BOM if present

OLD_SEEDS = "SEEDS = [42, 123, 456, 789, 1001, 2024, 3141, 5678, 8192, 9999]"
NEW_SEEDS = "SEEDS = [8192, 9999]  # RESUME: only missing Student B seeds"

assert OLD_SEEDS in code, (
    f"Cannot find expected SEEDS line in executed source.\n"
    f"Expected: {OLD_SEEDS!r}"
)
assert code.count(OLD_SEEDS) == 1, (
    "SEEDS line appears multiple times in source — unsafe to replace"
)

code = code.replace(OLD_SEEDS, NEW_SEEDS, 1)

# Verify the replacement was made correctly
assert NEW_SEEDS in code, "Replacement failed — NEW_SEEDS not found in patched code"
assert OLD_SEEDS not in code, "Replacement incomplete — OLD_SEEDS still present"

# === WRITE TEMP FILE (in same dir so __file__ paths are similar) ===
temp_script = EXECUTED_SRC.parent / "_resume_B_only.py"
temp_script.write_text(code, encoding="utf-8")
print(f"Temp script written: {temp_script}")

# === ENVIRONMENT ===
env = os.environ.copy()
env["WSNDS_PATH"] = str(REPO_ROOT / "data" / "wsnds" / "WSN-DS.csv")
env["CUKD_OUTPUT_DIR"] = str(OUTPUT_DIR)

# === RUN ===
print(f"{'=' * 60}")
print(f"CuKD-XAI Resume: Student B seeds [8192, 9999]")
print(f"{'=' * 60}")
print(f"Started:    {datetime.now().isoformat()}")
print(f"Python:     {PYTHON_EXE}")
print(f"Script:     {temp_script}")
print(f"Output dir: {OUTPUT_DIR}")
print(f"Log file:   {LOG_FILE}")
print(f"{'=' * 60}")
print()

start_time = time.time()
exit_code = -1

try:
    with LOG_FILE.open("w", encoding="utf-8") as lf:
        process = subprocess.Popen(
            [str(PYTHON_EXE), str(temp_script)],
            env=env,
            cwd=str(OUTPUT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        # Stream output to both console and log file
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
            lf.flush()

        process.wait()
        exit_code = process.returncode

except KeyboardInterrupt:
    print("\nInterrupted by user.")
    if process and process.poll() is None:
        process.terminate()
        process.wait(timeout=10)
    exit_code = -2

except Exception as e:
    print(f"\nERROR: {e}")
    exit_code = -1

finally:
    elapsed = time.time() - start_time
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\n{'=' * 60}")
    print(f"RESUME COMPLETE")
    print(f"{'=' * 60}")
    print(f"Exit code:  {exit_code}")
    print(f"Duration:   {hours}h {minutes}m {seconds}s")
    print(f"Finished:   {datetime.now().isoformat()}")
    print()

    # === VERIFY CHECKPOINTS ===
    print("Checkpoint verification:")
    all_ok = True
    for seed in [8192, 9999]:
        cp = OUTPUT_DIR / f"checkpoint_student_B_seed_{seed}.json"
        if cp.exists():
            size = cp.stat().st_size
            mtime = datetime.fromtimestamp(cp.stat().st_mtime).isoformat()
            print(f"  OK  checkpoint_student_B_seed_{seed}.json ({size:,} bytes, modified {mtime})")
        else:
            print(f"  FAIL  checkpoint_student_B_seed_{seed}.json — MISSING!")
            all_ok = False

    print()
    if all_ok and exit_code == 0:
        print("SUCCESS: All missing Student B checkpoints created.")
    elif all_ok:
        print("WARNING: Checkpoints exist but process exited with non-zero code.")
        print(f"         Check log: {LOG_FILE}")
    else:
        print("FAILURE: Some checkpoints are still missing.")
        print(f"         Check log: {LOG_FILE}")

    # === CLEAN UP TEMP FILE ===
    if temp_script.exists():
        temp_script.unlink()
        print(f"\nCleaned up temp file: {temp_script.name}")

    print(f"{'=' * 60}")
