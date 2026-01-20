"""
Test script for Part 3: Single Agent with ToyAIKit

Tests the single agent setup code from Part 3.
Run this from the .tmp directory.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

print("=" * 60)
print("Part 3: Testing Single Agent with ToyAIKit")
print("=" * 60)

day2_path = Path(__file__).parent.parent / "day2"
sys.path.insert(0, str(day2_path))

# Create a test project
test_dir = Path(tempfile.mkdtemp(prefix="agent_test_"))
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
    from openai import OpenAI
    from toyaikit.tools import Tools
    from toyaikit.chat import IPythonChatInterface
    from toyaikit.chat.runners import OpenAIResponsesRunner
    from toyaikit.llm import OpenAIClient
    from agent_tools import AgentTools
    print("[OK] All imports successful")
except ImportError as e:
    print(f"[FAIL] Import failed: {e}")
    print("  Note: toyaikit or openai may not be installed")
    print("  Install with: pip install toyaikit openai")
    sys.exit(1)

print("\n--- Test 2: Define DEVELOPER_PROMPT ---")
DEVELOPER_PROMPT = """
You are a coding agent. Your task is to modify the provided Django project
according to user instructions.
"""
print(f"[OK] DEVELOPER_PROMPT defined ({len(DEVELOPER_PROMPT)} chars)")

print("\n--- Test 3: Initialize AgentTools ---")
agent_tools = AgentTools(Path(test_project))
print(f"[OK] AgentTools initialized with project_path: {agent_tools.project_path}")

print("\n--- Test 4: Create Tools object and add agent_tools ---")
tools_obj = Tools()
tools_obj.add_tools(agent_tools)
print("[OK] Tools object created and agent_tools added")
print(f"  Number of tools: {len([attr for attr in dir(agent_tools) if not attr.startswith('_')])}")

print("\n--- Test 5: Create OpenAI client (mock mode) ---")
# Check for API key
api_key = os.environ.get("OPENAI_API_KEY")
if api_key:
    print(f"[OK] OPENAI_API_KEY found (sk-...{api_key[-4:]})")
    llm_client = OpenAIClient(client=OpenAI(), model='gpt-4o-mini')
else:
    print("[WARN] No OPENAI_API_KEY found, creating client anyway (will fail on actual API call)")
    llm_client = OpenAIClient(client=OpenAI(api_key="sk-test"), model='gpt-4o-mini')

print("\n--- Test 6: Create IPythonChatInterface ---")
chat_interface = IPythonChatInterface()
print("[OK] IPythonChatInterface created")

print("\n--- Test 7: Create OpenAIResponsesRunner ---")
runner = OpenAIResponsesRunner(
    tools=tools_obj,
    developer_prompt=DEVELOPER_PROMPT,
    chat_interface=chat_interface,
    llm_client=llm_client
)
print("[OK] OpenAIResponsesRunner created")

print("\n--- Test 8: Test result saving (without running) ---")
result_file = test_dir / "agent_result.txt"
Path(result_file).write_text("Mock result - would contain actual agent output")
print(f"[OK] Result saved to: {result_file}")
content = Path(result_file).read_text()
print(f"  Content: {repr(content[:50])}...")

print("\n--- Test 9: Better DEVELOPER_PROMPT ---")
DEVELOPER_PROMPT_BETTER = """
You are a coding agent. Your task is to modify the provided Django project
according to user instructions. You don't tell the user what to do; you do it
yourself using the available tools.

## Project Overview

- Django 5.2.4 with uv for environment management
- SQLite database
- Custom app called 'myapp'
- HTML templates with TailwindCSS styling

## Instructions

1. First, explore the project structure to understand what exists
2. Think about the sequence of changes needed
3. Make changes step by step
4. Use TailwindCSS classes for styling
5. Don't execute 'runserver' - use other commands to verify
"""
print(f"[OK] Better DEVELOPER_PROMPT defined ({len(DEVELOPER_PROMPT_BETTER)} chars)")

print("\n--- Test 10: Final DEVELOPER_PROMPT ---")
DEVELOPER_PROMPT_FINAL = """
You are a coding agent. Your task is to modify the provided Django project
according to user instructions. You don't tell the user what to do; you do it
yourself using the available tools.

## Project Overview

The project is a Django 5.2.4 web application scaffolded with best practices:
- Python 3.10+
- Django 5.2.4
- uv for environment management
- SQLite database
- Standard Django apps with a custom app called 'myapp'
- HTML templates with TailwindCSS styling

## File Tree

├── manage.py
├── pyproject.toml
├── myapp/
│   ├── views.py
│   ├── models.py
│   ├── templates/home.html
│   └── ...
├── myproject/
│   ├── settings.py
│   ├── urls.py
│   └── ...
└── templates/base.html

## Instructions

1. First, explore the project structure to understand what exists
2. Think about the sequence of changes needed
3. Make changes step by step
4. Use TailwindCSS classes for styling
5. Keep server-side logic in views, minimal logic in templates
6. Don't execute 'runserver' - use other commands to verify
7. After changes, suggest how to test the application
"""
print(f"[OK] Final DEVELOPER_PROMPT defined ({len(DEVELOPER_PROMPT_FINAL)} chars)")

print("\n" + "=" * 60)
print("All Part 3 tests passed!")
print("=" * 60)
print("\nNote: To actually run the agent with an API call:")
print("  1. Set OPENAI_API_KEY environment variable")
print("  2. Call: result = runner.run()")
print("  3. The agent will prompt for input or use runner.run(input='...')")

# Cleanup
print(f"\nCleaning up: {test_dir}")
shutil.rmtree(test_dir)
print("[OK] Cleanup complete")
