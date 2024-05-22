
begin = False
lines = []
gpu_line = None
with open("test_mul_all.log", "r") as f:
    for line in f:
        if gpu_line and gpu_line not in line:
            continue
        # else:
        #     lines.append(line)
        if "n_mult_closed" in line:
            begin = True
            lines.append(line)
            # gpu_line = line[line.index("(GpuProver pid="): line.index(")")]
            continue
        if begin:
            lines.append(line)
        if "Proving Theorem" in line and begin:
            break

with open("checkss/new_induction_12dvd4expnp1p20.log", "w") as f:
    for line in lines:
        f.write(line)