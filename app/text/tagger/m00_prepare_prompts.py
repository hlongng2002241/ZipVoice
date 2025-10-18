from collections import defaultdict
import pandas as pd

df = pd.read_csv("data/Generate text normalization - Prompt.csv")


with open("prompts/raw.py", "w") as f:
    tag_index = defaultdict(int)
    for row in df.to_dict("records"):
        tag = row["Thẻ gán"]
        prompt = row["Prompt"]
        if pd.isna(tag) is True or pd.isna(prompt) is True:
            continue

        tag = tag.replace("|", "").replace(" ", "_")

        index = tag_index[tag]
        tag_index[tag] += 1
        tag += "_" + str(index)
        print(tag, "=", '"""', file=f)
        print(prompt.strip(), file=f)
        print('"""', file=f)
        print("\n# ============================================= #\n", file=f)




