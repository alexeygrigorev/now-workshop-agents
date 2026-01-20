"""
Test script for Part 5: Multi-Agent Systems

Tests the multi-agent setup code from Part 5.
Run this from the .tmp directory.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

print("=" * 60)
print("Part 5: Testing Multi-Agent Systems")
print("=" * 60)

day2_path = Path(__file__).parent.parent / "day2"
sys.path.insert(0, str(day2_path))

# Create a test project
test_dir = Path(tempfile.mkdtemp(prefix="multi_agent_test_"))
test_project = test_dir / "test_project"
test_project.mkdir()
(test_project / "manage.py").write_text("# Django manage.py")
(test_project / "myapp").mkdir()
(test_project / "myapp" / "__init__.py").write_text("")
(test_project / "myapp" / "views.py").write_text("def home(request):\n    pass")
project_name = "test_project"

print(f"\nCreated test project at: {test_project}")

print("\n--- Test 1: Import required libraries ---")
try:
    from pathlib import Path
    from pydantic import BaseModel
    from pydantic_ai import Agent
    from agent_tools import AgentTools
    print("[OK] All imports successful (pydantic_ai available)")
except ImportError as e:
    print(f"[FAIL] Import failed: {e}")
    print("  Note: pydantic-ai may not be installed")
    print("  Install with: pip install pydantic-ai")
    sys.exit(1)

print("\n--- Test 2: Initialize AgentTools ---")
agent_tools = AgentTools(Path(test_project))
print(f"[OK] AgentTools initialized with project_path: {agent_tools.project_path}")

print("\n--- Test 3: Define PlanStep and Plan models ---")
class PlanStep(BaseModel):
    name: str
    detailed_description: str

class Plan(BaseModel):
    overview: str
    steps: list[PlanStep]
print("[OK] PlanStep and Plan models defined")

print("\n--- Test 4: Define planner instructions ---")
PLANNER_INSTRUCTIONS = """
You are a planning agent responsible for designing the application based on functional requirements.

Your Role:

- You get a set of functional requirements from the user
- You do not modify the codebase directly, but you must describe precisely the changes
  and actions that should be taken
- Your goal is to translate the user requirements into a clear step-by-step plan

Instructions:

- Check the file structure to better plan your work
- Always include styling in the plan
- Data processing should happen in backend, not templates
- Focus on clarity, modularity, and maintainability in your plan
- Don't include the exact code in the output, focus on the instructions
- Don't overcomplicate. The output should be an MVP
"""
print(f"[OK] PLANNER_INSTRUCTIONS defined ({len(PLANNER_INSTRUCTIONS)} chars)")

print("\n--- Test 5: Create planner agent ---")
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
print(f"  Model: openai:gpt-4o-mini")
print(f"  Tools: see_file_tree, read_file, search_in_files")

print("\n--- Test 6: Define executor instructions ---")
EXECUTOR_INSTRUCTIONS = """
You are a coding agent responsible for applying specific modifications to
a Django project based on instructions from a planning agent.

You are given a plan and you need to execute it step by step.
Follow the instructions precisely and modify the codebase directly
using the tools available to you.

Each time you execute a step, give a short explanation of what you did and which
step number you finished.

You do not ask questions. You do not suggest. You execute.

## Project Context

This is a Django 5.2.4 web application scaffolded with common conventions and clean separation of concerns.

Key technologies and constraints:

- Django 5.2.4
- SQLite for local database
- TailwindCSS for all styling
- HTML templates used for rendering

## Execution rules

- Do not run the development server, but you may run diagnostic or validation commands
- Use TailwindCSS to create clean and user-friendly UI components
- Do not place logic in templates. Use views or models for data processing
- You can create, edit, or delete any files as needed to complete your task
- Read relevant files before making edits to make sure your work is correct

Act with precision. Implement the plan.
"""
print(f"[OK] EXECUTOR_INSTRUCTIONS defined ({len(EXECUTOR_INSTRUCTIONS)} chars)")

print("\n--- Test 7: Create executor agent ---")
executor = Agent(
    'openai:gpt-4o-mini',
    instructions=EXECUTOR_INSTRUCTIONS,
    tools=[
        agent_tools.read_file,
        agent_tools.write_file,
        agent_tools.execute_bash_command,
        agent_tools.see_file_tree,
        agent_tools.search_in_files
    ]
)
print("[OK] Executor agent created")
print(f"  Model: openai:gpt-4o-mini")
print(f"  Tools: read_file, write_file, execute_bash_command, see_file_tree, search_in_files")

print("\n--- Test 8: Create mock plan for testing ---")
mock_plan = Plan(
    overview="Create a simple todo list app",
    steps=[
        PlanStep(name="Create todo model", detailed_description="Create a Todo model in myapp/models.py"),
        PlanStep(name="Create todo view", detailed_description="Create a view in myapp/views.py to display todos"),
        PlanStep(name="Create template", detailed_description="Create a template for the todo list"),
    ]
)
print(f"[OK] Mock plan created with {len(mock_plan.steps)} steps")

print("\n--- Test 9: Test execution loop structure ---")
for i, step in enumerate(mock_plan.steps):
    print(f"  Step {i+1}/{len(mock_plan.steps)}: {step.name}")
print("[OK] Execution loop structure verified")

print("\n" + "=" * 60)
print("CORE MULTI-AGENT TESTS COMPLETE")
print("=" * 60)
print("\nCore multi-agent setup (planner + executor): tested")
print("\nNote: To actually run the multi-agent system:")
print("  1. Set OPENAI_API_KEY environment variable")
print("  2. Use: plan_result = await planner.run(prompt)")
print("  3. Execute steps: for step in plan.steps: await executor.run(...)")
print("  4. PydanticAI uses async, so run in async context")

# Cleanup
print(f"\nCleaning up: {test_dir}")
shutil.rmtree(test_dir)
print("[OK] Cleanup complete")
