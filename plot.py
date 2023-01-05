import matplotlib.pyplot as plt
import numpy as np
import json

mean_data = {}

for platform in ["cpu", "cuda", "xpu"]:
    with open(f"report_{platform}.json", "r") as f:
        data = json.load(f)

    means = [data[key]["mean"] for key in data]
    mean_data[platform] = means

width = 0.2
max_len = max([len(mean_data[key]) for key in mean_data.keys()])
x = np.arange(max_len)

for platform in mean_data:
    while len(mean_data[platform]) < max_len:
        mean_data[platform].append(0)

# plot data in grouped manner of bar type
for i, platform in enumerate(mean_data.keys()):
    plt.bar(x - (width - (i * width)), mean_data[platform], width)

plt.title("Mean Time")
plt.xticks(x, x + 1)
plt.xlabel("# Rows(M)")
plt.ylabel("Time (s)")
plt.legend(["CPU - i13900k", "GPU - 1070 GTX", "GPU - Arc A770"])
plt.savefig("results.png")
