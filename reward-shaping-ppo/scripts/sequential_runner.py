import subprocess
import os
import sys
import time

def run_sequential():
    # Resolve project root dynamically
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.abspath(os.path.join(script_dir, ".."))
    configs_dir = os.path.join(workspace_dir, "configs")
    
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
    
    print("=" * 60)
    print("Starting sequential benchmarks runner on local CPU...")
    print(f"Total configurations to train: {len(configs)}")
    print("=" * 60)
    
    suite_start = time.time()
    
    for i, config_name in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] STARTING: {config_name}")
        config_path = os.path.join("configs", config_name)
        
        start_time = time.time()
        
        # Run main.py sequentially for this config
        result = subprocess.run(
            [sys.executable, "main.py", "--config", config_path, "--mode", "all"],
            cwd=workspace_dir
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"[{i}/{len(configs)}] SUCCESS: {config_name} finished in {duration:.1f}s")
        else:
            print(f"[{i}/{len(configs)}] FAILED: {config_name} failed with exit code {result.returncode}")
            # Ask if user wants to continue or abort
            response = input("Do you want to continue to the next config? (y/n): ")
            if response.lower() not in ["y", "yes", ""]:
                print("Aborting training run.")
                sys.exit(1)
                
    total_duration = time.time() - suite_start
    print("\n" + "=" * 60)
    print(f"All sequential training runs completed in {total_duration/60:.1f} minutes!")
    print("=" * 60)

if __name__ == "__main__":
    run_sequential()
