

with open("checkss/induction_12dvd4expnp1p20.log", "r") as f:
    lines = f.readlines()

new_liens = []
for line in lines:
    line = line[25:]
    if "-" in line:
        line = line[line.index("-"):]
        new_liens.append(line)

with open("checkss/induction_12dvd4expnp1p20.log", "w") as f:
    for line in new_liens:
        f.write(line)