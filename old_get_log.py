
def cal(lines):
    cnt = 0
    for line in lines:
        if "proof=[" in line:
            cnt += 1
    return float(cnt) / len(lines)


all_lines = []
with open("test_mul6.log") as f:
    for line in f:
        if "SearchResult" in line:
            all_lines.append(line)

print(cal(all_lines))
print(len(all_lines))
print(cal(all_lines[:591]))

