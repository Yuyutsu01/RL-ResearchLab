import subprocess

print("Running ruff...")
result = subprocess.run(["python", "-m", "ruff", "check", "."], cwd="reward-shaping-ppo", capture_output=True, text=True)
print("STDOUT:")
print(result.stdout)
print("STDERR:")
print(result.stderr)
