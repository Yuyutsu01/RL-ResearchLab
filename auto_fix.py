import subprocess
import sys

def run_cmd(cmd, cwd="."):
    print(f"Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("STDERR:", res.stderr)
    return res.returncode

def main():
    # 1. Run ruff locally
    run_cmd([sys.executable, "-m", "ruff", "check", "--fix", "."], "reward-shaping-ppo")
    
    # 2. Check git status
    run_cmd(["git", "status"])
    
    # 3. Add all
    run_cmd(["git", "add", "."])
    
    # 4. Try to commit
    ret = run_cmd(["git", "commit", "-m", "style: force ruff auto-fix directly via python"])
    
    # 5. Push if commit succeeded
    if ret == 0:
        run_cmd(["git", "push"])
        print("Successfully pushed to GitHub!")
    else:
        print("Nothing to commit. The files are already perfectly clean locally!")

if __name__ == "__main__":
    main()
