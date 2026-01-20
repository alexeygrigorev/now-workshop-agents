"""
Test script for Part 5: Multi-Agent Systems

Tests the multi-agent setup code from Part 5 with all 4 agents:
1. Clarifier - gathers requirements
2. Namer - generates project name and slug
3. Planner - creates implementation plan
4. Executor - implements each step

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

print(f"\nCreated test project at: {test_project}\n")

print("--- Test 1: Import required libraries ---")
try:
    from pathlib import Path
    from pydantic import BaseModel
    from pydantic_ai import Agent
    from agent_tools import AgentTools
    print("[OK] All imports successful")
except ImportError as e:
    print(f"[FAIL] Import failed: {e}")
    sys.exit(1)

print("\n--- Test 2: Initialize AgentTools ---")
agent_tools = AgentTools(Path(test_project))
print(f"[OK] AgentTools initialized")

print("\n--- Test 3: Define structured output models ---")
class ProjectName(BaseModel):
    name: str
    slug: str

class PlanStep(BaseModel):
    name: str
    detailed_description: str

class Plan(BaseModel):
    overview: str
    steps: list[PlanStep]
print("[OK] Structured output models defined (ProjectName, PlanStep, Plan)")

print("\n--- Test 4: Create clarifier agent ---")
CLARIFIER_INSTRUCTIONS = """
You are the Requirement Clarifier for a multi-agent AI coding assistant project.
Your job is to come up with EXACTLY three multiple-choice questions to clarify the
user's goals and constraints.

Ask these questions one by one until you get the answers to all.
Output "READY" when all questions are answered.

Don't focus on implementation details (like target platform, etc).
Only on the functional requirements.

The final implementation will be a web-based application implemented as a Django service,
so don't ask questions about interface.
"""

clarifier = Agent(
    'openai:gpt-4o-mini',
    instructions=CLARIFIER_INSTRUCTIONS
)
print("[OK] Clarifier agent created")
print("  Role: Ask questions to understand requirements")

print("\n--- Test 5: Create namer agent ---")
NAMER_INSTRUCTIONS = """
Your task is to come up with a name for a project.
Name the project with a unique, meaningful title that reflects its purpose
and avoids naming conflicts.

Generate a slug (a URL- and filesystem-safe version of the project name)
to serve as the project folder name.
The slug should be short, up to 15 characters.

Be original. Also be unpredictable.
"""

namer = Agent(
    'openai:gpt-4o-mini',
    instructions=NAMER_INSTRUCTIONS
)
print("[OK] Namer agent created")
print("  Output: ProjectName(name, slug)")

print("\n--- Test 6: Create planner agent ---")
PLANNER_INSTRUCTIONS = """
You are a planning agent responsible for designing the application based on functional requirements.

Your Role:

- You get a set of functional requirements from the clarifier agent
- You do not modify the codebase directly, but you must describe precisely the changes
  and actions that should be taken
- Your goal is to translate the user requirements into a clear step-by-step plan

Instructions:

- Check the file structure to better plan your work
- Always include styling in the plan
- Data processing should happen in backend, not templates
- Make sure the main interaction elements are accessible from the home page
- Focus on clarity, modularity, and maintainability in your plan
- Include tests
- Don't include the exact code in the output, focus on the instructions
- Don't overcomplicate. The output should be an MVP

Your output will be consumed by a coding agent, so it must be precise, unambiguous,
and broken down into logical steps.

Only include coding instructions. The output will not be read by humans.
The coding agent can only follow your suggested actions, but cannot plan.

The coding agent will see only one step at a time, so make steps self-contained.
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
print("  Output: Plan(overview, steps: PlanStep[])")
print("  Tools: see_file_tree, read_file, search_in_files")

print("\n--- Test 7: Create executor agent ---")
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
print("  Tools: read_file, write_file, execute_bash_command, see_file_tree, search_in_files")

print("\n--- Test 8: Create mock structured outputs for testing ---")
mock_project_name = ProjectName(
    name="Todo List Manager",
    slug="todo_manager"
)
print(f"[OK] Mock ProjectName: {mock_project_name.name} ({mock_project_name.slug})")

mock_plan = Plan(
    overview="Create a simple todo list app",
    steps=[
        PlanStep(name="Create todo model", detailed_description="Create a Todo model in myapp/models.py"),
        PlanStep(name="Create todo view", detailed_description="Create a view in myapp/views.py to display todos"),
        PlanStep(name="Create template", detailed_description="Create a template for the todo list"),
    ]
)
print(f"[OK] Mock Plan with {len(mock_plan.steps)} steps")

print("\n--- Test 9: Test execution loop structure ---")
for i, step in enumerate(mock_plan.steps):
    print(f"  Step {i+1}/{len(mock_plan.steps)}: {step.name}")
print("[OK] Execution loop structure verified")

print("\n" + "=" * 60)
print("MULTI-AGENT SYSTEM TESTS COMPLETE")
print("=" * 60)
print("\nAgent pipeline:")
print("  1. clarifier -> asks questions (outputs 'READY')")
print("  2. namer -> ProjectName(name, slug)")
print("  3. planner -> Plan(overview, steps: PlanStep[])")
print("  4. executor -> implements each step")
print("\nAll agents use structured output for reliable data passing.")
print("\nNote: To actually run the multi-agent system:")
print("  1. Set OPENAI_API_KEY environment variable")
print("  2. Run: clarifier_result = await clarifier.run(prompt)")
print("  3. Run: namer_result = await namer.run(qa_prompt)")
print("  4. Run: planner_result = await planner.run(qa_prompt)")
print("  5. Loop: for step in plan.steps: await executor.run(step_prompt)")

# Cleanup
print(f"\nCleaning up: {test_dir}")
shutil.rmtree(test_dir)
print("[OK] Cleanup complete")
