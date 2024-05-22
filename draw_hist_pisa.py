
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def get_name_state_and_proof(lines):
    results = []
    for line in lines:
        name = line[line.index("full_name=") + len("full_name="): line.index(", count=")]
        state = "Status.PROVED" in line
        if state is True:
            proof = eval(line[line.index("proof=[")+len("proof="): line.index(", actor_time")])
            # proof = "\n".join(proof)
        else:
            if "FAKE" in line:
                proof = eval(line[line.index("proof=[")+len("proof="): line.index(", actor_time")])
                # proof = "\n".join(proof)
            else:
                proof = []
        results.append([name, state, proof])
    return results

def collect_proofs(log_path):
    all_lines = []
    with open(log_path, "r") as f:
        for line in f:
            if "SearchResult" in line:
                all_lines.append(line)
    all_lines = get_name_state_and_proof(all_lines)

    all_true_lines = [line[2] for line in all_lines if line[1] is True]
    all_true_lines_length = [len(line) for line in all_true_lines]
    return all_true_lines, all_true_lines_length

miniF2F_POETRY, miniF2F_POETRY_length = collect_proofs("PISA_POETRY_all.log")
miniF2F_GPTF, miniF2F_GPTF_length = collect_proofs("PISA_GPTF_all.log")

miniF2F_GPTF_length = np.asarray(miniF2F_GPTF_length)
miniF2F_POETRY_length = np.asarray(miniF2F_POETRY_length)

data = list(zip(list(miniF2F_GPTF_length), ["GPT-f Baseline"]*len(miniF2F_GPTF_length))) + \
        list(zip(list(miniF2F_POETRY_length), ["POETRY"]*len(miniF2F_POETRY_length)))

data = pd.DataFrame(data).rename(columns={1:"method", 0:"Proof length"})

plot = sns.histplot(data=data, x="Proof length", hue="method", kde=False, discrete=True, element="step")
plt.xticks(np.arange(1, 27, 3), fontsize=14)
plt.yticks(fontsize=14)
plt.title("Proof length histogram on PISA", fontsize=20)
plt.yscale("log")
plt.ylabel("Problem counts", fontsize=18)
plt.xlabel("Proof length", fontsize=18)
plt.setp(plot.get_legend().get_texts(), fontsize='15') # for legend text
plt.setp(plot.get_legend().get_title(), fontsize='17') # for legend title
plt.tight_layout() 
# plot = sns.histplot(data=miniF2F_POETRY_length, kde=True, bins=np.asarray([1,2,3,4,5,6,7,8]))
# plot = sns.histplot(data=miniF2F_GPTF_length, kde=False, discrete=True, bins=[0,1,2,3,4])

plot.figure.savefig("PISA_GPTF_and_POETRY_hist.pdf")


print("herer")




    