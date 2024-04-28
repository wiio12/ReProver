

lines = []

with open("multilevel-c1w1s1-pisa_test_1.log", "r") as f:
    for line in f:
        if "ehereher" not in line:
            lines.append(line)

with open("tmps2.log", "w") as f:
    f.write("".join(lines))