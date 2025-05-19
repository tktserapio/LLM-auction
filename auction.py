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

p_array_A = np.array([
    0.5367087131860564, 0.6270528911069735, 0.718479467648265, 0.7343158966995874, 0.42505353887660596, 0.6909350599035342, 0.5769743627986648, 0.5405851628145751, 0.7429837678074941, 0.3705068251225533, 0.4395770233786555, 0.7275295978336679, 0.6094379418421041, 0.6665431735242812, 0.6895199792627221, 0.5810268764111848, 0.5478732063109074, 0.4826254681300905, 0.4571623363123326, 0.5450119814945459
])
p_array_B = np.array([
    0.563507078546367, 0.6427203603388921, 0.4438949623146147, 0.6622210432708827, 0.5736197859451999, 0.5748749295306727, 0.6940448347030787, 0.6129842444129404, 0.5921754212432251, 0.3703882932332144, 0.5782049250629524, 0.5880992896782119, 0.5682674654250238, 0.5986645157069062, 0.671489529963227, 0.5851762064505622, 0.6174593512624333, 0.6363576922517016, 0.5318014108065534, 0.5572522226356624
])
p_array_C = np.array([
    0.46821405420229983, 0.5652253795322305, 0.4724795597812008, 0.5239562874643906, 0.5624783319908322, 0.6469573240377386, 0.576766421132828, 0.6825894970866684, 0.5310162276140105, 0.39568078697718223, 0.569616882473477, 0.5627272000585485, 0.6605341433890783, 0.66448206152783, 0.5461109678281807, 0.5900713407012408, 0.7029085502822965, 0.47514507501503594, 0.5775555302551074, 0.44010648195988966
])

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
Market probabilities:
  • Route A → {pA:.2f}
  • Route B → {pB:.2f}
  • Route C → {pC:.2f}

You need to deliver a package.
1) Propose which route to take and why.
2) Estimate your value for that plan.

Use this JSON schema for your response:
{{"plan":"<your chosen route>", "value":<float>}}
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

    return response.choices[0].message.content

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
    # return response.text

def run_second_price_auction():
    round_history = []
    results = []

    for rnd in range(1, NUM_ROUNDS+1):
        print(f"\n=== Round {rnd} ===")
        public_hist = "\n".join(round_history)

        # 1) Prediction‐market phase
        pA = float(p_array_A[rnd-1])
        pB = float(p_array_B[rnd-1])
        pC = float(p_array_C[rnd-1])
        print(f"\n=== Round {rnd} ===")
        print(f"Market: A={pA:.2f}, B={pB:.2f}, C={pC:.2f}")

        # 2) Action proposals & expected‐value bids
        agent_plans     = {}
        agent_values = {}

        for name in AGENT_NAMES:
            # a) get plan + `reward` estimates
            sys  = f"You are subagent {name}.\n" + RULES_EXPLANATION.format(num_bidders=len(AGENT_NAMES)-1, private=VALUE_HIGH)
            sys += PERSONA_PROMPT
            usr  = ACTION_PROMPT.format(pA=pA, pB=pB, pC=pC)
            resp = chat_completion(sys, usr, max_tokens=MAX_TOKENS_PLAN)
            info = json.loads(resp)

            # store
            agent_plans[name]  = info.get("plan","Route A")
            agent_values[name] = float(info.get("value",0.0))
        
        # get bids
        agent_bids = {}
        for name in AGENT_NAMES:
            # system text again with rules
            system_text = f"You are Bidder {name}.\nYou are bidding against {', '.join(n for n in AGENT_NAMES if n!=name)}.\n"
            system_text += RULES_EXPLANATION.format(num_bidders=len(AGENT_NAMES)-1, private=VALUE_HIGH)
            system_text += PERSONA_PROMPT

            user_text = BID_INSTRUCTION.format(value=agent_values[name])
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

        # Auction winner selection
        triples = [(name, agent_plans[name], agent_bids[name]) for name in AGENT_NAMES]
        # find top
        top_bid = max(b for _,_,b in triples)
        top_candidates = [(n,r,b) for n,r,b in triples if b==top_bid]
        winner, route_win, _ = random.choice(top_candidates)
        # second price among same route
        same = [b for n,r,b in triples if r==route_win and n!=winner]
        second_bid = max(same) if same else 0
        profit = agent_values[winner] - second_bid

        # 5) Summarize round
        summary = f"Round {rnd} summary:\n"
        for n in AGENT_NAMES:
            summary += f"  {n}: plan={agent_plans[n]!r}, bid={agent_bids[n]}\n"
        summary += f"Winner: {winner} on {route_win}, pays {second_bid}, profit≈{profit:.2f}"
        print(summary)
        round_history.append(summary)

        # Reflections & results
        for name in AGENT_NAMES:
            sys_txt = f"You are bidder {name}.\n" \
                      + RULES_EXPLANATION.format(num_bidders=len(AGENT_NAMES)-1, private=VALUE_HIGH) \
                      + PERSONA_PROMPT
            refl = chat_completion(sys_txt, REFLECT_PROMPT.format(history=summary), MAX_TOKENS_REFLECT)
            results.append({
                "round":rnd,
                "agent":name,
                "market_pA":pA,
                "market_pB":pB,
                "market_pC":pC,
                "plan":agent_plans[name],
                "value":agent_values[name],
                "bid":agent_bids[name],
                "winner":winner,
                "route_win":route_win,
                "paid":second_bid,
                "profit":profit if name==winner else 0,
                "reflection":refl
            })

    return results, round_history

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    data, hist = run_second_price_auction()
    with open("results/results.json","w") as f:
        json.dump(data,f,indent=2)
    print("\nDone! Results saved to results/results.json")