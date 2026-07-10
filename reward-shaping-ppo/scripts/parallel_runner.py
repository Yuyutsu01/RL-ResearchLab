import subprocess
import concurrent.futures
import os
import sys

def run_config(config_name):
    print(f"[START] Running config: {config_name}")
    workspace_dir = r"c:\Users\shiva\OneDrive\Desktop\projects\RL-ResearchLab\reward-shaping-ppo"
    config_path = os.path.join("configs", config_name)
    
    # Restrict PyTorch thread usage to 1 per process to avoid CPU oversubscription
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    
    result = subprocess.run(
        [sys.executable, "main.py", "--config", config_path, "--mode", "all"],
        cwd=workspace_dir,
        env=env,
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print(f"[SUCCESS] Config finished: {config_name}")
    else:
        print(f"[FAILED] Config failed: {config_name}\nError:\n{result.stderr}\nOutput:\n{result.stdout}")
    return result.returncode

configs = [
    "mountaincar_identity.yaml",
    "mountaincar_dense.yaml",
    "mountaincar_pbrs.yaml",
    "acrobot_identity.yaml",
    "acrobot_dense.yaml",
    "acrobot_pbrs.yaml",
    "lunarlander_identity.yaml",
    "lunarlander_dense.yaml",
    "lunarlander_pbrs.yaml"
]

print("Starting parallel benchmarks runner (Optimized for Ryzen Multi-threading)...")
with concurrent.futures.ThreadPoolExecutor(max_workers=9) as executor:
    futures = [executor.submit(run_config, c) for c in configs]
    concurrent.futures.wait(futures)

print("All parallel training configurations completed.")
