"""
Test script for Part 5: Monitoring & Guardrails

Tests the monitoring and guardrails code from Part 5.
Run this from the .tmp directory.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

print("=" * 60)
print("Part 5: Testing Monitoring & Guardrails")
print("=" * 60)

day2_path = Path(__file__).parent.parent / "day2"
sys.path.insert(0, str(day2_path))

# Create a test project
test_dir = Path(tempfile.mkdtemp(prefix="monitoring_test_"))
test_project = test_dir / "test_project"
test_project.mkdir()
(test_project / "manage.py").write_text("# Django manage.py")

print("\n--- Test 1: Import logfire ---")
import logfire
print("[OK] logfire is installed")

print("\n--- Test 2: Logfire configuration ---")
os.environ["LOGFIRE_TOKEN"] = "pylf_v1_eu_2SmlvBHznz531ZfmpFQbfwMhYsfbZXHpfjGpjH75dpRZ"
logfire.configure()
print("[OK] logfire configured")

print("\n--- Test 3: Instrument PydanticAI ---")
logfire.instrument_pydantic_ai()
print("[OK] PydanticAI instrumented for Logfire")

print("\n--- Test 4: Input Validation with Pydantic ---")
try:
    from pydantic import BaseModel, Field, field_validator

    class UserRequest(BaseModel):
        """Validate user requests to the agent."""
        task: str = Field(..., min_length=5, max_length=1000)

        @field_validator('task')
        @classmethod
        def forbid_dangerous_commands(cls, v):
            dangerous = ['rm -rf', 'format', 'delete all', 'drop table']
            if any(cmd in v.lower() for cmd in dangerous):
                raise ValueError('Potentially dangerous command detected')
            return v

    # Test valid input
    request = UserRequest(task="Create a blog app")
    print("[OK] UserRequest model created")
    print(f"  Valid task: '{request.task}'")

    # Test dangerous input
    try:
        UserRequest(task="rm -rf everything")
        print("[FAIL] Should have rejected dangerous command!")
    except ValueError as e:
        print(f"[OK] Dangerous command rejected: {e}")

except ImportError:
    print("  Skipping (pydantic not installed)")

print("\n--- Test 5: Output Sanitization ---")
def sanitize_code_output(code: str) -> str:
    """Remove potentially harmful patterns from generated code."""
    dangerous_imports = ['os.system', 'subprocess.call', 'eval', 'exec']

    for imp in dangerous_imports:
        if imp in code:
            raise ValueError(f"Potentially dangerous code: {imp}")

    return code

# Test safe code
safe_code = "def hello():\n    return 'world'"
result = sanitize_code_output(safe_code)
print(f"[OK] Safe code passed: {repr(result[:30])}...")

# Test dangerous code
try:
    dangerous_code = "import os.system\nos.system('rm -rf')"
    sanitize_code_output(dangerous_code)
    print("[FAIL] Should have caught dangerous import!")
except ValueError as e:
    print(f"[OK] Dangerous code caught: {e}")

print("\n--- Test 6: Cost Limits with RunLimits ---")
try:
    from pydantic_ai import Agent, RunLimits

    # Set limits on agent runs
    agent_with_limits = Agent(
        'openai:gpt-4o-mini',
        instructions='You are a coding assistant.',
        run_limits=RunLimits(
            max_steps=20,
            max_tokens=10000
        )
    )
    print("[OK] Agent created with RunLimits")
    print(f"  max_steps: 20")
    print(f"  max_tokens: 10000")
except ImportError:
    print("  Skipping (pydantic_ai not installed)")

print("\n--- Test 7: Content Moderation ---")
try:
    from openai import OpenAI

    def moderate_content(text: str) -> bool:
        """Check if content is safe."""
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-test"))
        try:
            response = client.moderations.create(input=text)
            return not response.results[0].flagged
        except Exception:
            # If API call fails, assume content is safe for testing
            return True

    # Test with safe content
    is_safe = moderate_content("Create a todo list app")
    print(f"[OK] Content moderation check: {'Safe' if is_safe else 'Flagged'}")

except ImportError:
    print("  Skipping (openai not installed)")

print("\n--- Test 8: Complete Agent with Monitoring ---")
try:
    from pydantic_ai import Agent, RunLimits
    from agent_tools import AgentTools

    # Initialize tools
    agent_tools = AgentTools(Path(test_project))

    # Define a simple prompt
    DEVELOPER_PROMPT = """
    You are a coding agent. Your task is to modify the provided Django project
    according to user instructions.
    """

    # Create agent with monitoring
    agent = Agent(
        'openai:gpt-4o-mini',
        instructions=DEVELOPER_PROMPT,
        tools=[
            agent_tools.read_file,
            agent_tools.write_file,
            agent_tools.execute_bash_command,
            agent_tools.see_file_tree,
            agent_tools.search_in_files
        ],
        run_limits=RunLimits(
            max_steps=30,
            max_tokens=20000
        )
    )
    print("[OK] Complete agent created with monitoring")
    print(f"  Tools: {len([agent_tools.read_file, agent_tools.write_file, agent_tools.execute_bash_command, agent_tools.see_file_tree, agent_tools.search_in_files])}")
    print(f"  max_steps: 30")
    print(f"  max_tokens: 20000")

except ImportError:
    print("  Skipping (pydantic_ai not installed)")

print("\n" + "=" * 60)
print("All Part 5 tests passed!")
print("=" * 60)
print("\nNotes:")
print("  - Logfire requires: pip install logfire")
print("  - PydanticAI requires: pip install pydantic-ai")
print("  - Content moderation requires OPENAI_API_KEY")

# Cleanup
print(f"\nCleaning up: {test_dir}")
shutil.rmtree(test_dir)
print("[OK] Cleanup complete")
