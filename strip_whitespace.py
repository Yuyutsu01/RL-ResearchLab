import os

def strip_trailing_whitespace(directory):
    count = 0
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith('.py'):
                path = os.path.join(root, f)
                with open(path, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                
                changed = False
                new_lines = []
                for line in lines:
                    # Strip spaces and tabs from the right, but keep the newline
                    stripped = line.rstrip(' \t\r\n') + '\n'
                    if stripped != line:
                        changed = True
                    new_lines.append(stripped)
                
                if changed:
                    with open(path, 'w', encoding='utf-8') as file:
                        file.writelines(new_lines)
                    count += 1
                    print(f"Cleaned: {path}")
    
    print(f"Done! Cleaned trailing whitespace in {count} files.")

if __name__ == "__main__":
    strip_trailing_whitespace('reward-shaping-ppo')
