#!/usr/bin/env python3
"""
Demonstration script to show the system prompts for each agent.
This script can be used to show your judge what each agent is designed to do.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from safehive.agents.orchestrator import OrchestratorAgent
from safehive.agents.user_twin import UserTwinAgent
from safehive.agents.honest_vendor import HonestVendorAgent
from safehive.agents.malicious_vendor import MaliciousVendorAgent

def show_agent_prompts():
    """Display the system prompts for all agents."""
    
    print("=" * 80)
    print("SAFEHIVE AI SECURITY SANDBOX - AGENT SYSTEM PROMPTS")
    print("=" * 80)
    
    # Create instances of each agent (minimal initialization for demo)
    try:
        orchestrator = OrchestratorAgent("demo_orchestrator", {})
        user_twin = UserTwinAgent("demo_user_twin", {})
        honest_vendor = HonestVendorAgent("demo_honest", {})
        malicious_vendor = MaliciousVendorAgent("demo_malicious", {})
        
        agents = [
            ("OrchestratorAgent", orchestrator),
            ("UserTwinAgent", user_twin),
            ("HonestVendorAgent", honest_vendor),
            ("MaliciousVendorAgent", malicious_vendor)
        ]
        
        for agent_name, agent in agents:
            print(f"\n{'=' * 20} {agent_name} {'=' * 20}")
            print(agent.get_system_prompt_description())
            print("\n" + "-" * 80)
            
    except Exception as e:
        print(f"Error creating agents: {e}")
        print("\nThis is expected in a demo environment.")
        print("The methods are available and will work when the agents are properly initialized.")

if __name__ == "__main__":
    show_agent_prompts()
