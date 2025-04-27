import json
import matplotlib.pyplot as plt
import numpy as np

with open("results/results.json", "r") as f:
    data = json.load(f)

values_nonwinner = []
bids_nonwinner = []
values_winner = []
bids_winner = []

for row in data:
    # Check if the agent is the winner for the round
    if row["agent"] == row["winner"]:
        values_winner.append(row["value"])
        bids_winner.append(row["bid"])
    else:
        values_nonwinner.append(row["value"])
        bids_nonwinner.append(row["bid"])

# Plot non-winner bids
plt.scatter(values_nonwinner, bids_nonwinner, alpha=0.7, label="LLM bids (non-winner)")

# Plot winner bids with a different shape (e.g., diamond)
plt.scatter(values_winner, bids_winner, alpha=0.8, label="Winner bids", marker="D", color="red")

# Plot the expected symmetric equilibrium for 3 bidders
x = np.linspace(0, 99, 100)
y = x
plt.plot(x, y, label="Value (DSE)", linestyle="dashed")

plt.xlabel("Value")
plt.ylabel("Bid")
plt.title("Second-Price Sealed-Bid: LLM Agent Bids vs. Values")
plt.legend()
plt.show()