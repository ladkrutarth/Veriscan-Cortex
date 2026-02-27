from models.guard_agent_local import LocalGuardAgent

def evaluate_agent():
    print("Initializing GuardAgent...")
    agent = LocalGuardAgent()
    
    scenarios = [
        {
            "user_query": "Investigate USER_123 for potential fraud.",
            "expected_tool": "get_user_risk_profile"
        },
        {
            "user_query": "What are the latest CFPB trends for credit card disputes?",
            "expected_tool": "query_rag"
        },
        {
            "user_query": "Does USER_456 have a high risk score?",
            "expected_tool": "get_user_risk_profile"
        },
        {
            "user_query": "Explain what 1h velocity means in fraud detection.",
            "expected_tool": "query_rag"
        }
    ]
    
    success_count = 0
    print("\n--- Agentic AI Tool-Selection Evaluation ---")
    
    for i, scenario in enumerate(scenarios):
        query = scenario["user_query"]
        expected = scenario["expected_tool"]
        print(f"\nScenario [{i+1}]: '{query}'")
        
        result = agent.analyze(query)
        actions = result.get("actions", [])
        
        # Check if the expected tool was called in any step
        found_tool = any(a["tool"] == expected for a in actions)
        
        if found_tool:
            print(f"  ✅ Correct Tool Called: {expected}")
            success_count += 1
        else:
            print(f"  ❌ Expected Tool '{expected}' not called.")
            if actions:
                print(f"  Agent called: {[a['tool'] for a in actions]}")
            else:
                print("  Agent did not call any tools.")
            
    accuracy = success_count / len(scenarios)
    print(f"\nTool Selection Accuracy: {accuracy:.2f}")
    
    return accuracy

if __name__ == "__main__":
    evaluate_agent()
