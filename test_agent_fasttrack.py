from models.guard_agent_local import LocalGuardAgent

agent = LocalGuardAgent()
print("1. Testing get_high_risk_transactions")
res = agent.analyze("Show the most dangerous transactions")
print("ANSWER:")
print(res["answer"])

print("\n2. Testing query_rag")
res2 = agent.analyze("What is credit fraud?")
print("ANSWER:")
print(res2["answer"])
