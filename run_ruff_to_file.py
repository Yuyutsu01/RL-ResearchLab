import subprocess

with open("ruff_output.txt", "w", encoding="utf-8") as f:
    result = subprocess.run(
        ["python", "-m", "ruff", "check", "."],
        cwd="reward-shaping-ppo",
        capture_output=True,
        text=True,
    )
    f.write("STDOUT:\n")
    f.write(result.stdout)
    f.write("\nSTDERR:\n")
    f.write(result.stderr)
