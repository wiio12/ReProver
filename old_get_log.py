import json
import re
from pathlib import Path
from multilevel_isabelle.src.main.python.pisa_client import Theorem

def cal(lines):
    cnt = 0
    for line in lines:
        if "proof=[" in line:
            cnt += 1
    total_score = float(cnt) / len(lines)

    with open("val_detailed_split.json", "r") as f:
        data = json.load(f)
    normal_theorem, multi_theorem = data
    normal_theorem = [Theorem(file_path=thm["file_path"].replace("/data2/wanghaiming/Isabelle2022", "/hpc2hdd/home/zyang398/Isabelle2022").replace("/data2/wanghaiming/afp-2022-12-06", "/hpc2hdd/home/zyang398/afp-2022-12-06"), full_name=thm["full_name"], count=thm["count"]) for thm in normal_theorem]
    multi_theorem = [Theorem(file_path=thm["file_path"].replace("/data2/wanghaiming/Isabelle2022", "/hpc2hdd/home/zyang398/Isabelle2022").replace("/data2/wanghaiming/afp-2022-12-06", "/hpc2hdd/home/zyang398/afp-2022-12-06"), full_name=thm["full_name"], count=thm["count"]) for thm in multi_theorem]

    normal_total, normal_cnt = 0, 0
    multi_total, multi_cnt = 0, 0
    for line in lines:
        pattern = r"Theorem\(file_path=PosixPath\('(.*?)'\), full_name=[\'\"](.*?)[\'|\"], count=(\d+)"
        match = re.search(pattern, line)
        assert match
        thm = Theorem(
            file_path=Path(match.group(1)), 
            full_name=match.group(2).encode().decode('unicode_escape'), 
            count=int(match.group(3))
        )
        if thm in normal_theorem:
            normal_total += 1
            if "proof=[" in line:
                normal_cnt += 1
        elif thm in multi_theorem:
            multi_total += 1
            if "proof=[" in line:
                multi_cnt += 1
        else:
            raise Exception
    
    return total_score, normal_cnt/float(normal_total), multi_cnt/float(multi_total)


all_lines = []
with open("test_gptf2.log", "r") as f:
    for line in f:
        if "SearchResult" in line:
            all_lines.append(line)

compare_lines = []
with open("tmp.log", "r") as f:
    for line in f:
        if "SearchResult" in line:
            compare_lines.append(line)

print("All lines:")
print(cal(all_lines))
print(len(all_lines))
print(cal(all_lines[:len(compare_lines)]))

print("Compare lines:")
print(cal(compare_lines))
print(len(compare_lines))