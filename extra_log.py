
begin = False
lines = []
with open("multilevel-c1w1s1-miniF2F-test_2.log", "r") as f:
    for line in f:
        if "GpuProver pid=3042866" not in line:
            continue
        if "mathd_algebra_209" in line:
            begin = True
            lines.append(line)
            continue
        if begin:
            lines.append(line)
        if "Proving Theorem" in line and begin:
            break

with open("tmps2.log", "w") as f:
    for line in lines:
        f.write(line)