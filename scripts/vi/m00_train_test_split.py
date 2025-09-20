import random
from collections import defaultdict

data_path = "data/vi/metadata.tsv"
speakers = ["minh_chau", "minh_ngoc"]
split_by_speakers = defaultdict(list)

with open(data_path) as f:
    for line in f.readlines():
        found = False
        for spk in speakers:
            if line.startswith(spk):
                found = True
                split_by_speakers[spk].append(line.strip())

S = 200
random.seed(8686)
train = []
test = []

for k, vs in split_by_speakers.items():
    print(k, "=", len(vs))
    random.shuffle(vs)
    train.extend(vs[S :])
    test.extend(vs[: S])

print("train =", len(train))
print("test =", len(test))

with open("data/vi/train.tsv", "w") as f:
    for l in train:
        print(l, file=f)
with open("data/vi/test.tsv", "w") as f:
    for l in test:
        print(l, file=f)
