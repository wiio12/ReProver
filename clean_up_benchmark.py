import os
import json

with open("/hpc2hdd/home/zyang398/wanghaiming/data/isabelle/isadojo_benchmark_v0.1/test.json", "r") as f:
    data = json.load(f)


new_data = data
# full_names = []

# for line in data:
#     if line["full_name"] not in full_names:
#         new_data.append(line)
#         full_names.append(line["full_name"])

new_a = new_data[:750]
new_b = new_data[750: 1450]
new_c = new_data[1450:]


os.makedirs("/hpc2hdd/home/zyang398/wanghaiming/data/isabelle/isadojo_benchmark_v0.2_a")
with open("/hpc2hdd/home/zyang398/wanghaiming/data/isabelle/isadojo_benchmark_v0.2_a/test.json", "w") as f:
    json.dump(new_a, f)

os.makedirs("/hpc2hdd/home/zyang398/wanghaiming/data/isabelle/isadojo_benchmark_v0.2_b")
with open("/hpc2hdd/home/zyang398/wanghaiming/data/isabelle/isadojo_benchmark_v0.2_b/test.json", "w") as f:
    json.dump(new_b, f)

os.makedirs("/hpc2hdd/home/zyang398/wanghaiming/data/isabelle/isadojo_benchmark_v0.2_c")
with open("/hpc2hdd/home/zyang398/wanghaiming/data/isabelle/isadojo_benchmark_v0.2_c/test.json", "w") as f:
    json.dump(new_c, f)

print("here")