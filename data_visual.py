import json
import matplotlib.pyplot as plt

with open("fpsb_results.json", "r") as f:
    data = json.load(f)

values = []
bids = []
for row in data:
    values.append(row["value"])
    bids.append(row["bid"])

plt.scatter(values, bids, alpha=0.7, label="LLM bids")

# plot the expected symmetric equilibrium for 3 bidders
import numpy as np
x = np.linspace(0, 99, 100)
y = (2.0/3.0)*x
plt.plot(x, y, label="2/3 * value (BNE)", linestyle="dashed")

plt.xlabel("Value")
plt.ylabel("Bid")
plt.title("First-Price Sealed-Bid: LLM Agent Bids vs. Values")
plt.legend()
plt.show()
