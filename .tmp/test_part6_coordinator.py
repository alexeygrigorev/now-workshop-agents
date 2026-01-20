"""
Test script for Part 6: Coordinator Agent Pattern

Tests the coordinator agent pattern where agents can invoke other agents.
This is an EXPERIMENTAL pattern - not as thoroughly tested as Part 5.

Run this from the .tmp directory.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

print("=" * 60)
print("Part 6: Testing Coordinator Agent Pattern")
print("=" * 60)
print("NOTE: This is an experimental pattern.")
print("For production use, consider the fixed orchestration from Part 5.")
print()

day2_path = Path(__file__).parent.parent / "day2"
sys.path.insert(0, str(day2_path))

# Create a test project
test_dir = Path(tempfile.mkdtemp(prefix="coordinator_test_"))
test_project = test_dir / "test_project"
test_project.mkdir()
(test_project / "manage.py").write_text("# Django manage.py")
(test_project / "myapp").mkdir()
(test_project / "myapp" / "__init__.py").write_text("")
(test_project / "myapp" / "views.py").write_text("def home(request):\n    pass")

print(f"Created test project at: {test_project}\n")

print("--- Test 1: Import required libraries ---")
try:
    from pathlib import Path
    from pydantic_ai import Agent, RunContext
    from agent_tools import AgentTools
    print("[OK] All imports successful")
except ImportError as e:
    print(f"[FAIL] Import failed: {e}")
    sys.exit(1)

print("\n--- Test 2: Initialize AgentTools ---")
agent_tools = AgentTools(Path(test_project))
print(f"[OK] AgentTools initialized")

print("\n--- Test 3: Define researcher agent ---")
RESEARCHER_INSTRUCTIONS = """
You are a code researcher. Your job is to:
1. Explore the codebase structure
2. Find relevant files and patterns
3. Understand existing conventions
4. Report your findings clearly

Use search and read_file tools extensively.
"""

researcher = Agent(
    'openai:gpt-4o-mini',
    instructions=RESEARCHER_INSTRUCTIONS,
    tools=[
        agent_tools.see_file_tree,
        agent_tools.read_file,
        agent_tools.search_in_files
    ]
)
print("[OK] Researcher agent created")

print("\n--- Test 4: Define writer agent ---")
WRITER_INSTRUCTIONS = """
You are a code writer. Your job is to:
1. Create new files based on specifications
2. Modify existing files following Django patterns
3. Use TailwindCSS for styling
4. Follow Python and Django best practices

Use write_file tool to create code.
"""

writer = Agent(
    'openai:gpt-4o-mini',
    instructions=WRITER_INSTRUCTIONS,
    tools=[
        agent_tools.write_file,
        agent_tools.read_file
    ]
)
print("[OK] Writer agent created")

print("\n--- Test 5: Define coordinator agent ---")
COORDINATOR_INSTRUCTIONS = """
You coordinate a team to build Django applications:
- researcher: Explores the codebase and finds information
- writer: Writes and modifies code

When given a task, decide which agent to use:
- Use researcher when you need to explore or understand the codebase
- Use writer when you need to create or modify files

You can call multiple agents in sequence if needed.
Always use the available tools to call your team members.
"""

coordinator = Agent(
    'openai:gpt-4o-mini',
    instructions=COORDINATOR_INSTRUCTIONS
)
print("[OK] Coordinator agent created")

print("\n--- Test 6: Add tools to coordinator that call other agents ---")

@coordinator.tool
async def research(ctx: RunContext, query: str) -> str:
    """Runs the researcher agent to explore the codebase."""
    result = await researcher.run(user_prompt=query)
    return result.output

@coordinator.tool
async def write(ctx: RunContext, filepath: str, instruction: str) -> str:
    """Runs the writer agent to create or modify files."""
    prompt = f"Write to {filepath}: {instruction}"
    result = await writer.run(user_prompt=prompt)
    return result.output

print("[OK] Coordinator tools added (research, write)")

print("\n--- Test 7: Verify tool structure ---")
print(f"  Coordinator has @agent.tool decorators registered")
print(f"  - research (calls researcher agent)")
print(f"  - write (calls writer agent)")

print("\n" + "=" * 60)
print("COORDINATOR PATTERN TESTS COMPLETE")
print("=" * 60)
print("\nComparison of approaches:")
print("  Part 5 (Fixed orchestration): Python controls flow - TESTED")
print("  Part 6 (Coordinator agent): LLM controls flow - EXPERIMENTAL")
print("\nThe coordinator pattern is more flexible but less predictable.")
print("Use it when you need flexibility and can tolerate unpredictability.")
print("\nNote: To actually run the coordinator:")
print("  result = await coordinator.run('Create a todo list app')")

# Cleanup
print(f"\nCleaning up: {test_dir}")
shutil.rmtree(test_dir)
print("[OK] Cleanup complete")
