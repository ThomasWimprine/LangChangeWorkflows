#!/usr/bin/env python3
"""Test delegation extraction logic to verify the fix works correctly."""

import logging

# Setup logging to see debug output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_delegation_extraction():
    """Simulate the delegation extraction logic from process_draft_responses."""

    # Simulate what Claude actually returns (from user's debug data)
    test_cases = [
        {
            "name": "Dict-key format (actual Claude format)",
            "agent_name": "devops-engineer",
            "delegations": [
                {"web-developer": "Create HTML/CSS implementation for dashboard"},
                {"docker-specialist": "Dockerfile optimization and security"},
                {"security-reviewer": "Review authentication flow"}
            ]
        },
        {
            "name": "Structured format (if agents return this)",
            "agent_name": "architect-reviewer",
            "delegations": [
                {"agent": "python-developer", "reason": "Implement backend API"},
                {"agent": "test-automation", "reason": "Write integration tests"}
            ]
        },
        {
            "name": "String format (legacy support)",
            "agent_name": "business-analyst",
            "delegations": [
                "data-scientist: Analyze user behavior patterns",
                "ml-specialist: Build recommendation engine"
            ]
        },
        {
            "name": "Mixed formats",
            "agent_name": "product-manager",
            "delegations": [
                {"react-developer": "Frontend components"},
                {"agent": "nodejs-developer", "reason": "Backend services"},
                "database-administrator: Schema design"
            ]
        }
    ]

    # Run extraction for each test case
    all_delegation_suggestions = []

    for test in test_cases:
        print(f"\n{'='*70}")
        print(f"TEST: {test['name']}")
        print(f"{'='*70}")

        agent_name = test["agent_name"]
        delegations = test["delegations"]

        logger.debug(f"Agent {agent_name} raw delegations: {delegations}")

        # This is the actual extraction logic from the fix
        for delegation in delegations:
            if isinstance(delegation, dict):
                # Check if it has "agent" key (structured format)
                if "agent" in delegation:
                    # Format: {"agent": "web-developer", "reason": "..."}
                    all_delegation_suggestions.append({
                        **delegation,
                        "suggested_by": agent_name
                    })
                    logger.debug(f"  Extracted (structured): {delegation.get('agent')}")
                else:
                    # Dict-key format: {"web-developer": "reason text"}
                    # Each key is an agent name, value is the reason
                    for agent_key, reason in delegation.items():
                        all_delegation_suggestions.append({
                            "agent": agent_key,
                            "reason": reason,
                            "suggested_by": agent_name
                        })
                        logger.debug(f"  Extracted (dict-key): {agent_key}")
            elif isinstance(delegation, str):
                # String format: "web-developer: reason text"
                suggested_agent = delegation.split(":")[0].strip()
                all_delegation_suggestions.append({
                    "agent": suggested_agent,
                    "reason": delegation.split(":", 1)[1].strip() if ":" in delegation else "",
                    "suggested_by": agent_name
                })
                logger.debug(f"  Extracted (string): {suggested_agent}")

    # Summary
    print(f"\n{'='*70}")
    print("EXTRACTION SUMMARY")
    print(f"{'='*70}")

    # Extract unique agent names (this is what the feedback loop does)
    suggested_agents = set()
    for suggestion in all_delegation_suggestions:
        agent_name = suggestion.get("agent", "").strip()
        if agent_name:
            suggested_agents.add(agent_name)

    logger.info(f"Total delegation suggestions: {len(all_delegation_suggestions)}")
    logger.info(f"Unique agents suggested: {len(suggested_agents)}")
    logger.info(f"Agent names: {sorted(suggested_agents)}")

    print(f"\n{'='*70}")
    print("DETAILED RESULTS")
    print(f"{'='*70}")
    for i, suggestion in enumerate(all_delegation_suggestions, 1):
        print(f"{i}. {suggestion['agent']:25s} <- suggested by {suggestion['suggested_by']}")
        if suggestion.get('reason'):
            print(f"   Reason: {suggestion['reason'][:60]}...")

    # Verify the critical case (dict-key format from Claude)
    print(f"\n{'='*70}")
    print("VERIFICATION")
    print(f"{'='*70}")
    dict_key_agents = [s['agent'] for s in all_delegation_suggestions
                       if s['suggested_by'] == 'devops-engineer']
    expected = {'web-developer', 'docker-specialist', 'security-reviewer'}
    actual = set(dict_key_agents)

    if actual == expected:
        print("✅ Dict-key extraction PASSED - all agents correctly extracted")
    else:
        print(f"❌ Dict-key extraction FAILED")
        print(f"   Expected: {expected}")
        print(f"   Got: {actual}")
        return False

    # Verify no empty agent names
    empty_agents = [s for s in all_delegation_suggestions if not s.get('agent', '').strip()]
    if empty_agents:
        print(f"❌ Found {len(empty_agents)} suggestions with empty agent names")
        return False
    else:
        print("✅ No empty agent names - all extractions successful")

    print(f"\n{'='*70}")
    print("ALL TESTS PASSED ✅")
    print(f"{'='*70}")
    return True

if __name__ == "__main__":
    success = test_delegation_extraction()
    exit(0 if success else 1)
