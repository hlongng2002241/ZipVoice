import os, sys
sys.path.append(".")
import jsonlines

from data_synthesis import Client


client = Client("gemini"); model ="gemini-2.5-flash-0520"
working_dir = "data/batches"
total_cost = 0
num_requests = 0

for fn in os.listdir(working_dir):
    if fn.endswith("_responses.jsonl") is False:
        continue
    with jsonlines.open(os.path.join(working_dir, fn)) as f:
        for item in f:
            total_cost += client.cost(model=model, **item["usage"])
            num_requests += 1

print("num_requests =", num_requests)
print("total_cost =", total_cost)
print("cost per request =", total_cost / num_requests)