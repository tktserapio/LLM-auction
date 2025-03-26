import os
from dotenv import load_dotenv
import openai
from openai import OpenAI
import json
import random

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

NUM_BIDDERS = 3
NUM_ROUNDS = 3
VALUE_LOW, VALUE_HIGH = 0, 99 
MODEL_NAME = "gpt-3.5-turbo"
TEMPERATURE = 1.5
MAX_TOKENS_PLAN = 200
MAX_TOKENS_BID = 30
MAX_TOKENS_REFLECT = 120

BID_INCREMENT = 1

AGENT_NAMES = ["Andy", "Betty", "Charles"]

# just explaining the rules for first-price sealed-bid auction
# modified from the the paper's Appendix A.3
RULES_EXPLANATION = """
In this game, you will participate in an auction for a prize against {num_bidders} other bidders.
At the start of each round, bidders will see their value for the prize, randomly drawn between
$0 and ${private}, with all values equally likely.

After learning your value, you will submit a bid privately at the same time as the other bidders.
Bids must be between $0 and ${private} in $1 increments.

The highest bidder wins the prize and pays their bid amount. This means that if you win, we add
to your earnings the value of the prize, and then subtract your own bid. If you do not win,
your earnings remain unchanged for that round.

After each auction, we will display all bids and the winner’s profits. Ties for the highest bid
are resolved randomly.

Before locking that bid in, consider if other bidders bid much lower than you—or much higher
than you. Would you regret your bid?
"""

# persona / user objective
PERSONA_PROMPT = """
Your TOP PRIORITY is to place bids which maximize your profit in the long run.
To do this, you should explore different bidding strategies, including possibly risky or
aggressive options for data-gathering purposes. Learn from the history of previous rounds
in order to maximize your total profit. Remember that each round’s value is redrawn independently.
"""

# kind of chain-of-thought-ish prompts for each step

# step 1: plan
PLAN_PROMPT = """
Write your plans for what bidding strategies to test next.
Be detailed but keep it within 100 words.
Try not to repeat yourself.
"""

# step 2: bid
BID_INSTRUCTION = """
Your value is {value}. Your plan is: {plan}
FOLLOW YOUR PLAN. 
How much would you like to bid?
Give your response with a single number and no other text, e.g. '36'.
"""

# step 3: reflection
REFLECT_PROMPT = """
The previous round history is:
{history}

Do a counterfactual analysis. 
Remember your goal is to win the bid and make higher profits.
Limit your output to 100 words. 
Start your reflection with:
'If I bid down by .., I could...
 If I bid up by ..., I could...'
"""


def chat_completion(system_text, user_text, max_tokens=200):
    """Utility function to call the OpenAI ChatCompletion API with given system & user text."""
    response = openai.chat.completions.create(model=MODEL_NAME,
    temperature=TEMPERATURE,
    messages=[
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ],
    max_tokens=max_tokens)
    return response.choices[0].message.content.strip()


# SIMULATE THE AUCTION BELOW

def run_fpsb_auction():
    agent_plans = {name: "" for name in AGENT_NAMES} # store agent plans
    agent_reflections = {name: "" for name in AGENT_NAMES} # store agent reflections
    round_history = []  # store round history

    # for final analysis, we store a list of dicts: {round, agent, value, bid, is_winner, profit}
    results_data = []

    for rnd in range(1, NUM_ROUNDS+1):
        print(f"--- Starting Round {rnd} ---")

        # notes:
        # 1. each agent sees full public history and their own reflections, 
        # 2. maintain IPV --> agents don't see other agents' reflections directly. 

        # build a summary of everything that happened so far:
        # "Round 1: Andy bid 20, Betty bid 30, Charles bid 25. Winner was Betty at profit 70 - 30=40"
        public_history_text = "\n".join(round_history)

        for name in AGENT_NAMES:
            # user message for planning
            system_base = f"You are Bidder {name}.\nYou are bidding against {', '.join(n for n in AGENT_NAMES if n!=name)}.\n"
            system_base += RULES_EXPLANATION.format(num_bidders=NUM_BIDDERS-1, private=VALUE_HIGH)
            system_base += PERSONA_PROMPT

            user_plan_text = ""
            user_plan_text += f"Here is the history so far:\n{public_history_text}\n\n"
            user_plan_text += PLAN_PROMPT

            plan_response = chat_completion(system_base, user_plan_text, max_tokens=MAX_TOKENS_PLAN)
            agent_plans[name] = plan_response

        
        # random values and then prompt for a bid
        agent_values = {}
        agent_bids = {}
        for name in AGENT_NAMES:
            value = random.randint(VALUE_LOW, VALUE_HIGH)
            agent_values[name] = value

        
        for name in AGENT_NAMES:
            # system text again with rules
            system_text = f"You are Bidder {name}.\nYou are bidding against {', '.join(n for n in AGENT_NAMES if n!=name)}.\n"
            system_text += RULES_EXPLANATION.format(num_bidders=NUM_BIDDERS-1, private=VALUE_HIGH)
            system_text += PERSONA_PROMPT

            user_text = BID_INSTRUCTION.format(value=agent_values[name], plan=agent_plans[name])
            bid_str = chat_completion(system_text, user_text, max_tokens=MAX_TOKENS_BID)

            # parse the bid
            try:
                bid_float = float(bid_str.split()[0])
            except:
                # fallback: set to 0 if can't parse
                bid_float = 0.0

            # clipping bid
            if bid_float < 0:
                bid_float = 0.0
            elif bid_float > VALUE_HIGH:
                bid_float = VALUE_HIGH

            bid_final = round(bid_float)
            agent_bids[name] = bid_final

        # winner determination
        sorted_bidders = sorted(agent_bids.items(), key=lambda x: x[1], reverse=True)
        top_bidder, top_bid = sorted_bidders[0]
        
        # tie breaking mechanism
        winners = [top_bidder]
        for i in range(1, len(sorted_bidders)):
            if sorted_bidders[i][1] == top_bid:
                winners.append(sorted_bidders[i][0])
            else:
                break

        if len(winners) > 1:
            winner = random.choice(winners)
        else:
            winner = winners[0]

        # ############ 4.4: Calculate Profits ############
        winner_value = agent_values[winner]
        # first-price sealed-bid => pay own bid
        winner_profit = winner_value - agent_bids[winner]
        # losers profit 0
        # build a textual summary for the public history
        round_summary = f"Round {rnd}:\n"
        for name in AGENT_NAMES:
            round_summary += f"  {name} value={agent_values[name]}, bid={agent_bids[name]}\n"
        round_summary += f"  WINNER: {winner}, profit={winner_value}-{agent_bids[winner]}={winner_profit}\n"

        round_history.append(round_summary)


        # Let each agent see the final outcome. Then we store their reflection.
        # This reflection is agent-specific and doesn't become public for others.
        public_history_text_rnd = round_summary  # the result of this round only
        for name in AGENT_NAMES:
            system_text = f"You are Bidder {name}.\n"
            system_text += f"You are bidding against {', '.join(n for n in AGENT_NAMES if n!=name)}.\n"
            system_text += RULES_EXPLANATION.format(num_bidders=NUM_BIDDERS-1, private=VALUE_HIGH)
            system_text += PERSONA_PROMPT

            user_reflect = REFLECT_PROMPT.format(history=public_history_text_rnd)
            reflection_response = chat_completion(system_text, user_reflect, max_tokens=MAX_TOKENS_REFLECT)
            agent_reflections[name] = reflection_response
            # We won't insert it into public history. Only the agent sees it.

        # store history
        for name in AGENT_NAMES:
            is_winner = (name == winner)
            profit = winner_profit if is_winner else 0
            results_data.append({
                "round": rnd,
                "agent": name,
                "value": agent_values[name],
                "bid": agent_bids[name],
                "winner": winner,
                "profit": profit,
                "plan": agent_plans[name],
                "reflection": agent_reflections[name]
            })

    # All rounds done
    return results_data, round_history


if __name__ == "__main__":
    # run the simulation
    data, history = run_fpsb_auction()

    # Save results to JSON or CSV:
    with open("fpsb_results.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # Print final round history
    print("\n=== Auction Completed ===\n")
    for h in history:
        print(h)
    print("\nResults stored in fpsb_results.json\n")