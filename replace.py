import json

replace_kv = {
    "\<in>":"∈",
    "\<Oplus>":"⨁",
    "\<^bsub>":"⇘",
    "\<^esub>":"⇙",
    "\<lambda>":"λ",
    "\<And>":"⋀",
    "\<le>":"≤",
    "\<Longrightarrow>":"⟹",
    "\<odot>":"⊙",
    
}

with open("example.json", "r") as f:
    data = json.load(f)

for step, proof in data["multilevel_proof"].items():
    for k, v in replace_kv.items():
        proof["state_before"] = proof["state_before"].replace(k,v)
        proof["state_after"] = proof["state_after"].replace(k,v)

print("herer")