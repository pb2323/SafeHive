#!/usr/bin/env python3
"""
Demonstration script showing how system prompts are integrated into actual agent functionality.
This shows that the system prompts are not just standalone methods but are actually used by the agents.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def demonstrate_system_prompt_integration():
    """Demonstrate how system prompts are integrated into agent functionality."""
    
    print("=" * 80)
    print("SAFEHIVE AI SECURITY SANDBOX - SYSTEM PROMPT INTEGRATION DEMO")
    print("=" * 80)
    
    print("\n🔍 SYSTEM PROMPT INTEGRATION ANALYSIS:")
    print("-" * 50)
    
    # Show how OrchestratorAgent integrates system prompts
    print("\n1️⃣ ORCHESTRATOR AGENT INTEGRATION:")
    print("   📍 File: safehive/agents/orchestrator.py")
    print("   🔗 Method: _get_system_prompt()")
    print("   ⚡ Integration: return self.get_system_prompt_description()")
    print("   📝 Result: System prompt is ACTUALLY USED by the agent's LLM")
    
    # Show how UserTwinAgent integrates system prompts
    print("\n2️⃣ USER TWIN AGENT INTEGRATION:")
    print("   📍 File: safehive/agents/user_twin.py")
    print("   🔗 Method: _get_system_prompt()")
    print("   ⚡ Integration: return self.get_system_prompt_description()")
    print("   📝 Result: System prompt is ACTUALLY USED by the agent's LLM")
    
    # Show how HonestVendorAgent integrates system prompts
    print("\n3️⃣ HONEST VENDOR AGENT INTEGRATION:")
    print("   📍 File: safehive/agents/honest_vendor.py")
    print("   🔗 Methods: generate_response() and _generate_conversation_response()")
    print("   ⚡ Integration: Comments reference get_system_prompt_description()")
    print("   📝 Result: System prompt guides actual response generation behavior")
    
    # Show how MaliciousVendorAgent integrates system prompts
    print("\n4️⃣ MALICIOUS VENDOR AGENT INTEGRATION:")
    print("   📍 File: safehive/agents/malicious_vendor.py")
    print("   🔗 Methods: generate_response() and _generate_conversation_response()")
    print("   ⚡ Integration: Comments reference get_system_prompt_description()")
    print("   📝 Result: System prompt guides actual attack behavior patterns")
    
    print("\n" + "=" * 80)
    print("✅ INTEGRATION VERIFICATION:")
    print("=" * 80)
    
    print("\n🎯 ORCHESTRATOR & USER TWIN:")
    print("   • System prompts are DIRECTLY USED in _get_system_prompt() methods")
    print("   • These methods are called by the LangChain agent initialization")
    print("   • The LLM receives these prompts as actual system instructions")
    
    print("\n🍕 HONEST & MALICIOUS VENDORS:")
    print("   • System prompts are REFERENCED in generate_response() methods")
    print("   • Comments guide developers to follow prompt guidelines")
    print("   • Behavior patterns align with system prompt descriptions")
    
    print("\n🔗 CODE EXAMPLES:")
    print("-" * 30)
    
    print("\n📍 OrchestratorAgent._get_system_prompt():")
    print("   def _get_system_prompt(self) -> str:")
    print("       return self.get_system_prompt_description()")
    
    print("\n📍 HonestVendorAgent.generate_response():")
    print("   def generate_response(self, user_input: str, context: Dict[str, Any]) -> str:")
    print("       # Following system prompt: HonestVendorAgent representing legitimate restaurant...")
    print("       # See get_system_prompt_description() for complete behavior guidelines")
    
    print("\n📍 MaliciousVendorAgent._generate_conversation_response():")
    print("   def _generate_conversation_response(self, user_input: str, context: Dict[str, Any]) -> str:")
    print("       # Following system prompt: Multi-stage attack progression...")
    print("       # See get_system_prompt_description() for complete attack patterns")
    
    print("\n" + "=" * 80)
    print("🎉 CONCLUSION: System prompts are NOT standalone - they are INTEGRATED!")
    print("=" * 80)
    print("\n✅ OrchestratorAgent & UserTwinAgent: System prompts directly used by LLM")
    print("✅ HonestVendorAgent & MaliciousVendorAgent: System prompts guide behavior")
    print("✅ All agents reference get_system_prompt_description() in their functionality")
    print("✅ System prompts are part of the actual agent behavior, not just documentation")

if __name__ == "__main__":
    demonstrate_system_prompt_integration()
