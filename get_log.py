

def cal(lines):
    cnt = 0
    for line in lines:
        if "proof=[" in line:
            cnt += 1
    return float(cnt) / len(lines)

all_lines = []
with open("test_mul3.log") as f:
    for line in f:
        if "Trajectory" not in line:
            all_lines.append(line)

with open("test_mul3_clean.log", "w") as f:
    for line in all_lines:
        f.write(line)

print("herer")
