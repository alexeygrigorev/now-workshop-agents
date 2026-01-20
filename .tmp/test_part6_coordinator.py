"""
Test script for Part 6: Coordinator Agent Pattern

Tests the coordinator agent pattern with planner, researcher, writer, and validator.
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
    from pydantic import BaseModel
    from pydantic_ai import Agent, RunContext
    from agent_tools import AgentTools
    print("[OK] All imports successful")
except ImportError as e:
    print(f"[FAIL] Import failed: {e}")
    sys.exit(1)

print("\n--- Test 2: Initialize AgentTools ---")
agent_tools = AgentTools(Path(test_project))
print(f"[OK] AgentTools initialized")

print("\n--- Test 3: Define structured output models ---")
class PlanStep(BaseModel):
    name: str
    detailed_description: str

class Plan(BaseModel):
    overview: str
    steps: list[PlanStep]

class ResearchFindings(BaseModel):
    summary: str
    relevant_files: list[str]
    existing_patterns: list[str]
    recommendations: list[str]

class ValidationReport(BaseModel):
    status: str
    issues_found: list[str]
    suggestions: list[str]
print("[OK] Structured output models defined (Plan, ResearchFindings, ValidationReport)")

print("\n--- Test 4: Create planner agent ---")
PLANNER_INSTRUCTIONS = """
You are a planning agent. Create a step-by-step plan for implementing Django features.

Your output must be a structured Plan with:
- overview: Brief description of what will be built
- steps: List of concrete steps to implement

Focus on clarity and modularity. Each step should be self-contained.
"""

planner = Agent(
    'openai:gpt-4o-mini',
    instructions=PLANNER_INSTRUCTIONS,
    tools=[
        agent_tools.see_file_tree,
        agent_tools.read_file,
        agent_tools.search_in_files
    ]
)
print("[OK] Planner agent created")

print("\n--- Test 5: Create researcher agent ---")
RESEARCHER_INSTRUCTIONS = """
You are a code researcher. Explore the codebase and provide structured findings.

Your output must include:
- summary: What you found
- relevant_files: Files that are relevant to the task
- existing_patterns: Patterns or conventions used in the codebase
- recommendations: Suggestions for implementation

Use see_file_tree, read_file, and search_in_files tools.
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

print("\n--- Test 6: Create writer agent ---")
WRITER_INSTRUCTIONS = """
You are a code writer. Your job is to:
1. Create new files based on specifications
2. Modify existing files following Django patterns
3. Use TailwindCSS for styling
4. Follow Python and Django best practices

Use write_file and read_file tools. After making changes, provide a brief summary.
"""

writer = Agent(
    'openai:gpt-4o-mini',
    instructions=WRITER_INSTRUCTIONS,
    tools=[
        agent_tools.write_file,
        agent_tools.read_file,
        agent_tools.execute_bash_command,
        agent_tools.see_file_tree,
        agent_tools.search_in_files
    ]
)
print("[OK] Writer agent created")

print("\n--- Test 7: Create validator agent ---")
VALIDATOR_INSTRUCTIONS = """
You are a code validator. Review code for quality and correctness.

Your output must include:
- status: "pass" if code looks good, "fail" if there are critical issues, "warning" for minor issues
- issues_found: List of any problems detected
- suggestions: List of improvements

Check for:
- Django best practices
- Security issues
- Code quality
- Missing tests
"""

validator = Agent(
    'openai:gpt-4o-mini',
    instructions=VALIDATOR_INSTRUCTIONS,
    tools=[
        agent_tools.read_file,
        agent_tools.search_in_files
    ]
)
print("[OK] Validator agent created")

print("\n--- Test 8: Create coordinator agent ---")
COORDINATOR_INSTRUCTIONS = """
You coordinate a team of agents to build Django applications:
- planner: Creates a structured plan with steps
- researcher: Explores the codebase and provides structured findings
- writer: Writes and modifies code
- validator: Reviews code for quality

When given a task, decide which agents to call and in what order:

1. For new tasks, start with researcher to understand the codebase
2. Then call planner to create a plan
3. For each step in the plan, call writer to implement
4. After implementation, call validator to check quality

You can call agents multiple times if needed. Always use the available tools.
"""

coordinator = Agent(
    'openai:gpt-4o-mini',
    instructions=COORDINATOR_INSTRUCTIONS
)
print("[OK] Coordinator agent created")

print("\n--- Test 9: Add tools to coordinator ---")

@coordinator.tool
async def plan(ctx: RunContext, task: str) -> str:
    """Runs the planner agent to create a structured plan."""
    result = await planner.run(user_prompt=task)
    return result.output

@coordinator.tool
async def research(ctx: RunContext, question: str) -> str:
    """Runs the researcher agent to explore the codebase."""
    result = await researcher.run(user_prompt=question)
    return result.output

@coordinator.tool
async def write(ctx: RunContext, instruction: str) -> str:
    """Runs the writer agent to create or modify code."""
    result = await writer.run(user_prompt=instruction)
    return result.output

@coordinator.tool
async def validate(ctx: RunContext, filepath: str) -> str:
    """Runs the validator agent to check code quality."""
    result = await validator.run(user_prompt=f"Review {filepath}")
    return result.output

print("[OK] Coordinator tools registered:")
print("  - plan (calls planner agent)")
print("  - research (calls researcher agent)")
print("  - write (calls writer agent)")
print("  - validate (calls validator agent)")

print("\n" + "=" * 60)
print("COORDINATOR PATTERN TESTS COMPLETE")
print("=" * 60)
print("\nAgent setup:")
print("  1. planner -> creates structured Plan (PlanStep[], overview)")
print("  2. researcher -> provides ResearchFindings (structured output)")
print("  3. writer -> reads and writes code")
print("  4. validator -> provides ValidationReport (status, issues, suggestions)")
print("\nComparison of approaches:")
print("  Part 5 (Fixed orchestration): Python controls flow - TESTED")
print("  Part 6 (Coordinator agent): LLM controls flow - EXPERIMENTAL")
print("\nThe coordinator pattern is more flexible but less predictable.")
print("\nNote: To actually run the coordinator:")
print("  result = await coordinator.run('Create a todo list app')")

# Cleanup
print(f"\nCleaning up: {test_dir}")
shutil.rmtree(test_dir)
print("[OK] Cleanup complete")
