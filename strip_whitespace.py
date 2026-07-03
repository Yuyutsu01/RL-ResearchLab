import os


def strip_trailing_whitespace(directory):
    count = 0
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(root, f)
                with open(path, "rb") as file:
                    content = file.read()

                # Split by newline (handles \r\n and \n)
                lines = content.splitlines()

                changed = False
                new_lines = []
                for line in lines:
                    # decode as utf-8, strip ALL trailing whitespace, encode back
                    try:
                        text_line = line.decode("utf-8")
                    except UnicodeDecodeError:
                        text_line = line.decode("latin-1")

                    stripped = text_line.rstrip()
                    new_lines.append(stripped.encode('utf-8'))

                # Re-join with strict \n
                new_content = b"\n".join(new_lines) + b"\n"

                if new_content != content:
                    with open(path, "wb") as file:
                        file.write(new_content)
                    count += 1
                    print(f"Cleaned: {path}")

    print(f"Done! Cleaned trailing whitespace in {count} files.")


if __name__ == "__main__":
    strip_trailing_whitespace("reward-shaping-ppo")
