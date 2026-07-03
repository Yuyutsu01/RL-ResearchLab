with open("reward-shaping-ppo/utils/reproducibility.py", "rb") as f:
    lines = f.readlines()
for i in range(23, 32):
    print(f"Line {i + 1}: {repr(lines[i])}")
