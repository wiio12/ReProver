

def cal(lines):
    cnt = 0
    for line in lines:
        if "proof=[" in line:
            cnt += 1
    return float(cnt) / len(lines)

def compare_liens(linesa, linesb):
    assert len(set(linesa)) == len(linesa)
    assert len(set(linesb)) == len(linesb)

    a_more = set(linesa) - set(linesb)
    print("line a more:")
    for line in a_more:
        print(line)
    
    b_more = set(linesb) - set(linesb)
    print("line b more")
    for line in b_more:
        print(line)

# all_lines = []
# with open("test_mul3.log") as f:
#     for line in f:
#         if "Trajectory" not in line:
#             all_lines.append(line)

# with open("test_mul3_clean.log", "w") as f:
#     for line in all_lines:
#         f.write(line)
def filter_by_name(lines, name):
    for line in lines:
        if name in line:
            return line
    return None

def get_name_state_and_proof(lines):
    results = []
    for line in lines:
        name = line[line.index("full_name=") + len("full_name="): line.index(", count=")]
        state = "proof=[" in line
        if state is True:
            proof = eval(line[line.index("proof=[")+len("proof="): line.index(", actor_time")])
        else:
            proof = []
        results.append([name, state, proof])
    return results

def have_sorry(lines):
    return any(["sorry" in l for l in lines])

gptf_lines = []
with open("test_gptf2.log") as f:
    for line in f:
        if "SearchResult" in line:
            gptf_lines.append(line)
gptf_lines = gptf_lines[:591]

mult_lines = []
with open("test_mul6.log") as f:
    for line in f:
        if "SearchResult" in line:
            mult_lines.append(line)

compare_liens(gptf_lines, mult_lines)
# for line in all_lines:
#     if "lemma forall_coeffs_conv:" in line:
#         print(line)

# print(cal(all_lines))
# print(len(all_lines))
# print(cal(all_lines[:591]))

