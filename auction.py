import os
import json
import re
import random
import numpy as np
from dotenv import load_dotenv

import openai
from openai import OpenAI
# from azure.ai.inference import ChatCompletionsClient
# from azure.ai.inference.models import SystemMessage
# from azure.ai.inference.models import UserMessage
# from azure.core.credentials import AzureKeyCredential
from google import genai
from google.genai import types

load_dotenv()

NUM_ROUNDS      = 20
VALUE_HIGH      = 99 # clip it to available capital
AGENT_NAMES     = [
    "WeatherBot1", "WeatherBot2",
    "TrafficBot1", "TrafficBot2",
    "MaintBot", "LocalEventsBot1", "RoadConditionBot1"
]
MAX_TOKENS_PLAN    = 200
MAX_TOKENS_BID     = 30
MAX_TOKENS_REFLECT = 120
p_array = [0.58092181, 0.74994838, 0.53546851, 0.64430303, 0.47282165, 0.58420343,
 0.65763313, 0.73082427, 0.74466788, 0.63594236, 0.57868029, 0.59322022,
 0.69759318, 0.65834205, 0.58448011, 0.53830929, 0.62175658, 0.55001748,
 0.48609097, 0.71235304]

RULES_EXPLANATION = """
In this game, you will participate in an auction to win the rights to implement a particular plan against {num_bidders} other bidders.
At the start of each round, you will see a market signal and then propose a plan and bid for execution rights.
Bids must be between $0 and ${private} in $1 increments.
The winner pays the second-highest bid and earns the reward of their plan.
Ties for the highest bid are broken uniformly at random.
"""

PERSONA_PROMPT = """
Your TOP PRIORITY is to place bids which maximize your profit in the long run.
You should learn from previous rounds to refine your strategy.
"""

ACTION_PROMPT = """
The market-implied probability that route A succeeds is {p:.2f}.

You need to deliver a package.
1) Propose your action plan to maximize the chance of delivery success.
2) Estimate your value for this action plan being implemented. 

Use this JSON schema for your response:
{{"plan":"<your plan>", "value":<float>}}
"""

BID_INSTRUCTION = """
Your value for implementing your plan is {value}.
How much would you like to bid?
Give your response with a single number and no other text, e.g. '36'.
"""

REFLECT_PROMPT = """
The previous round summary was:
{history}
Do a brief counterfactual:
If I had bid lower, what might I have missed?
If I had bid higher, what opportunity would I have gained?
"""

def chat_completion(system_text, user_text, max_tokens=200):
    
    # Gemini API
    # client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # response = client.models.generate_content(
    #     model="gemini-2.0-flash",
    #     config=types.GenerateContentConfig(
    #         system_instruction=system_text, 
    #         max_output_tokens=max_tokens,
    #         temperature=1.0,
    #         ),
    #     contents=user_text
    # )
    
    # OpenAI API
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        temperature=1.0,
        messages=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]
    )

    # AZURE AI
    # client = ChatCompletionsClient(
    #     endpoint="https://models.github.ai/inference",
    #     credential=AzureKeyCredential(os.environ["GITHUB_TOKEN"]),
    # )

    # response = client.complete(
    #     messages=[
    #         SystemMessage(system_text),
    #         UserMessage(user_text),
    #     ],
    #     model="openai/gpt-4o-mini",
    #     temperature=1.0,
    #     max_tokens=4096,
    #     top_p=0.1
    # )
    return response.text

def run_second_price_auction():
    round_history = []
    results = []

    for rnd in range(1, NUM_ROUNDS+1):
        print(f"\n=== Round {rnd} ===")
        public_hist = "\n".join(round_history)

        # 1) Prediction‐market phase
        p = p_array[rnd-1]
        print(f"Market P(success|route A) = {p:.2f}")

        # 2) Action proposals & expected‐value bids
        agent_plans     = {}
        agent_expected  = {}
        agent_bids      = {}

        for name in AGENT_NAMES:
            # a) get plan + `reward` estimates
            sys  = f"You are subagent {name}.\n" + RULES_EXPLANATION.format(num_bidders=len(AGENT_NAMES)-1, private=VALUE_HIGH)
            sys += PERSONA_PROMPT
            usr  = ACTION_PROMPT.format(p=p)
            resp = chat_completion(sys, usr, max_tokens=MAX_TOKENS_PLAN)
            info = json.loads(resp)
            plan    = info["plan"]

            # store
            agent_plans[name]    = plan
            agent_expected[name] = float(info["value"])
        
        # get bids
        for name in AGENT_NAMES:
            # system text again with rules
            system_text = f"You are Bidder {name}.\nYou are bidding against {', '.join(n for n in AGENT_NAMES if n!=name)}.\n"
            system_text += RULES_EXPLANATION.format(num_bidders=len(AGENT_NAMES)-1, private=VALUE_HIGH)
            system_text += PERSONA_PROMPT

            user_text = BID_INSTRUCTION.format(value=agent_expected[name])
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

        # 3) Determine winner & payment
        sorted_bids = sorted(agent_bids.items(), key=lambda x: x[1], reverse=True)
        top_name, top_bid = sorted_bids[0]
        second_bid        = sorted_bids[1][1] if len(sorted_bids) > 1 else 0

        # tie‐break
        top_ties = [n for n,b in sorted_bids if b == top_bid]
        winner   = random.choice(top_ties) if len(top_ties)>1 else top_name

        # 4) Compute profit
        profit = agent_expected[winner] - second_bid

        # 5) Summarize round
        summary = f"Round {rnd} (p={p:.2f}):\n"
        for n in AGENT_NAMES:
            summary += f"  {n}: plan={agent_plans[n]!r}, bid={agent_bids[n]}\n"
        summary += f"  WINNER: {winner}, pays {second_bid}, profit ≈ {profit:.2f}\n"
        print(summary)

        round_history.append(summary)

        # 6) Reflections
        for name in AGENT_NAMES:
            sys = f"You are bidder {name}.\n" + RULES_EXPLANATION.format(num_bidders=len(AGENT_NAMES)-1, private=VALUE_HIGH)
            sys += PERSONA_PROMPT
            usr = REFLECT_PROMPT.format(history=summary)
            refl = chat_completion(sys, usr, max_tokens=MAX_TOKENS_REFLECT)
            results.append({
                "round": rnd,
                "agent": name,
                "market_p": p,
                "plan": agent_plans[name],
                "value": agent_expected[name],
                "bid": agent_bids[name],
                "winner": winner,
                "paid": second_bid,
                "profit": profit if name == winner else 0,
                "reflection": refl
            })

    return results, round_history

if __name__ == "__main__":
    data, history = run_second_price_auction()
    with open("results/results.json", "w") as f:
        json.dump(data, f, indent=2)
    print("\nDone! Results written to results/results.json.")