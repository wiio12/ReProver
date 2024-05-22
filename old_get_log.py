import json
import re
from pathlib import Path
from multilevel_isabelle.src.main.python.pisa_client import Theorem

def get_name_state_and_proof(lines):
    results = []
    for line in lines:
        name = line[line.index("full_name=") + len("full_name="): line.index(", count=")]
        state = "Status.PROVED" in line
        if state is True:
            proof = eval(line[line.index("proof=[")+len("proof="): line.index(", actor_time")])
            proof = "\n".join(proof)
        else:
            if "FAKE" in line:
                proof = eval(line[line.index("proof=[")+len("proof="): line.index(", actor_time")])
                proof = "\n".join(proof)
            else:
                proof = ""
        results.append([name, state, proof])
    return results

def print_fake(cl):
    adds = []
    for line in cl:
        if "Status.FAKE_PROVED" in line:
            adds.append(line)
    return adds

def compare(al, cl):
    allss = []
    memed_theorem = []
    for line in al:
        for cline in cl:
            if line[0] == cline[0] and line[1] != cline[1]:
                if line[0] not in memed_theorem:
                    allss.append([line, cline])
                    memed_theorem.append(line[0])
    return allss

def compare2(al, cl):
    allss = []
    for line in al:
        for cline in cl:
            if line[0] == cline[0] and "sorry" in line[2]:
                allss.append([line, cline])
    return allss

def cal(lines):
    cnt = 0
    for line in lines:
        if "Status.PROVED" in line:
            cnt += 1
    total_score = float(cnt) / len(lines)

    with open("test_detailed_split.json", "r") as f:
        data = json.load(f)
    normal_theorem, multi_theorem = data
    normal_theorem = [Theorem(file_path=thm["file_path"].replace("/data2/wanghaiming/Isabelle2022", "/hpc2hdd/home/zyang398/Isabelle2022").replace("/data2/wanghaiming/afp-2022-12-06", "/hpc2hdd/home/zyang398/afp-2022-12-06"), full_name=thm["full_name"], count=thm["count"]) for thm in normal_theorem]
    multi_theorem = [Theorem(file_path=thm["file_path"].replace("/data2/wanghaiming/Isabelle2022", "/hpc2hdd/home/zyang398/Isabelle2022").replace("/data2/wanghaiming/afp-2022-12-06", "/hpc2hdd/home/zyang398/afp-2022-12-06"), full_name=thm["full_name"], count=thm["count"]) for thm in multi_theorem]

    normal_total, normal_cnt = 0, 0
    multi_total, multi_cnt = 0, 0
    visited_theorem = []
    for line in lines:
        pattern = r"Theorem\(file_path=PosixPath\('(.*?)'\), full_name=[\'\"](.*?)[\'|\"], count=(\d+)"
        match = re.search(pattern, line)
        assert match
        thm = Theorem(
            file_path=Path(match.group(1)), 
            full_name=match.group(2).encode().decode('unicode_escape'), 
            count=int(match.group(3))
        )
        if thm in visited_theorem:
            continue
        visited_theorem.append(thm)
        if thm in normal_theorem:
            normal_total += 1
            if "Status.PROVED" in line:
                normal_cnt += 1
        elif thm in multi_theorem:
            multi_total += 1
            if "Status.PROVED" in line:
                multi_cnt += 1
        else:
            continue
    normal_score = normal_cnt/float(normal_total) if normal_total != 0 else 0
    multi_score = multi_cnt/float(multi_total) if multi_total != 0 else 0
    return total_score, normal_score, multi_score


all_lines = []
with open("test_mul_all.log", "r") as f:
    for line in f:
        if "SearchResult" in line:
            all_lines.append(line)

compare_lines = []
with open("test_mul_all.log", "r") as f:
    for line in f:
        if "SearchResult" in line:
            compare_lines.append(line)

all_line_full_name = []
new_all_line = []
for line in all_lines:
    pattern = r"Theorem\(file_path=PosixPath\('(.*?)'\), full_name=[\'\"](.*?)[\'|\"], count=(\d+)"
    match = re.search(pattern, line)
    assert match
    full_name=match.group(2).encode().decode('unicode_escape')
    if full_name not in all_line_full_name:
        all_line_full_name.append(full_name)
        new_all_line.append(line)
all_lines = new_all_line

compare_line_full_name = []
new_compare_line = []
new_all_line = []
for line in compare_lines:
    pattern = r"Theorem\(file_path=PosixPath\('(.*?)'\), full_name=[\'\"](.*?)[\'|\"], count=(\d+)"
    match = re.search(pattern, line)
    assert match
    full_name=match.group(2).encode().decode('unicode_escape')
    if full_name not in compare_line_full_name:
        compare_line_full_name.append(full_name)
        new_compare_line.append(line)
        for idx, na in enumerate(all_line_full_name):
            if na == full_name:
                new_all_line.append(all_lines[idx])
compare_lines = new_compare_line
all_lines = new_all_line

# compare_lines = compare_lines[:600]

print("All lines:")
print(cal(all_lines))
print(len(all_lines))
# print(cal(all_lines[:len(compare_lines)]))

print("Compare lines:")
print(cal(compare_lines))
print(len(compare_lines))

al = get_name_state_and_proof(all_lines)
cl = get_name_state_and_proof(compare_lines)

with open("checkcasess3.txt", "w") as f:
    for line in cl:
        if line[1] and len(line[2].split("\n")) > 10:
            f.write(line[0].encode().decode("unicode-escape")[1:-1])
            f.write("\n")
            f.write(line[2])
            f.write("\n\n---------------------------------------------\n\n")


rets = compare(al, cl)

for ret in rets:
    print("theorem: ", ret[0][0].encode().decode("unicode-escape"))
    print("TrueFalse:", ret[0][1], ret[1][1])
    print("Solution:", ret[0][2], ret[1][2])
    print("\n\n")


print("############## FAKE ################")
fl = print_fake(compare_lines)
fl = get_name_state_and_proof(fl)
for ret in fl:
    print("theorem: ", ret[0].encode().decode("unicode-escape"))
    print("TrueFalse:", ret[1])
    print("Solution:", ret[2])
    print("\n\n")
