# Day 2: Building a Custom Coding Agent

Learning Objectives:
 - Set up a Django project template
 - Implement tools for file access and manipulation
 - Design and develop a coding agent
 - Coordinate multiple agents
 - Add monitoring and guardrails with Pydantic Logfire


## Prerequisites

 - Completed Day 1
 - Django project template (provided)
 - All API keys configured


### Environment Setup

If you haven't set up `dirdotenv` from Day 1:

```bash
pip install dirdotenv
echo 'eval "$(dirdotenv hook bash)"' >> ~/.bashrc
```

Alternatively, use `python-dotenv` to load environment variables in your code:

```bash
uv add python-dotenv
```

Create a `.env` file with your API keys:

```bash
# .env
OPENAI_API_KEY='sk-...'
ANTHROPIC_API_KEY='sk-ant-...'
GROQ_API_KEY='gsk_...'
GEMINI_API_KEY='...'
```

Load environment variables in your code:

```python
from dotenv import load_dotenv
load_dotenv()
```

# Part 1: Setting Up the Django Project Template

## What We're Building

We're building a "Project Bootstrapper" agent - an AI agent that can:
 - Read files in a codebase
 - Write new files
 - Execute bash commands
 - Modify a Django template to build applications

This is the foundation for tools like [Lovable](https://lovable.dev) - an AI coding assistant that uses a project template approach similar to what we're building here.

## Creating the Template

Alternatively, create the Django template from scratch:

```bash
mkdir django_template
cd django_template/
uv init
rm main.py

uv add django
uv run django-admin startproject myproject .
uv run python manage.py startapp myapp
```

Add the new app (`myapp`) into `myproject/settings.py`'s `INSTALLED_APPS`:

```python
INSTALLED_APPS = [
    # ... other apps
    'myapp',
]
```

Create a `Makefile` with useful commands:

```makefile
.PHONY: install migrate run

install:
    uv sync --dev

migrate:
    uv run python manage.py migrate

run:
    uv run python manage.py runserver
```

Create the base html template in `templates/base.html`:

```html
{% block title %}{% endblock %}
{% block content %}{% endblock %}
```

Add this templates directory to the settings file:

```python
TEMPLATES = [{
    'DIRS': [BASE_DIR / 'templates'],
    # ...
}]
```

Create the home view:

```python
# myapp/views.py
def home(request):
    return render(request, 'home.html')

# myproject/urls.py
from myapp import views

urlpatterns = [
    # ...
    path('', views.home, name='home'),
]
```

HTML code for `myapp/templates/home.html`:

```html
{% extends 'base.html' %}
{% block content %}

## Home

{% endblock %}
```

Add TailwindCSS and Font-Awesome to `base.html`. See the pre-built template for the complete setup.

## Getting the Template

Download the pre-built Django template:

```bash
git clone https://github.com/alexeygrigorev/django_template.git
cd django_template
```

### Install Dependencies and Run

```bash
uv sync

make migrate
make run
```

Open http://localhost:8000 to see the template running.

## Understanding the Template Structure

```
django_template/
├── .python-version
├── README.md
├── manage.py
├── pyproject.toml
├── uv.lock
├── myapp/
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── migrations/
│   ├── models.py
│   ├── templates/
│   │   └── home.html
│   └── views.py
├── myproject/
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
└── templates/
    └── base.html
```

 - Django 5.2.4: Web framework
 - SQLite: Default database
 - TailwindCSS: Styling via CDN
 - uv: Python environment management

# Part 2: Implementing Tools for File Access

## Agent Tools Overview

Our coding agent needs these capabilities:
1. Read file - View existing code
2. Write file - Create or modify files
3. Execute bash - Run commands (migrate, test, etc.)
4. See file tree - Navigate the project
5. Search in files - Find specific code

## Why a Class Instead of Functions?

We organize tools into a class rather than standalone functions for several reasons:

- Shared state: The `project_path` is stored once in `__init__` and used by all methods
- Encapsulation: Related functionality is grouped together logically
- Agent compatibility: AI agent frameworks expect tools as class methods that can be passed around
- Easier integration: The entire `AgentTools` instance can be passed to an agent with all methods available


## AgentTools Class Stub

Create `agent_tools.py` with the class stub:

```python
class AgentTools:

    def __init__(self, project_path):
        pass

    def read_file(self, filepath):
        pass

    def write_file(self, filepath, content):
        pass

    def execute_bash_command(self, command, cwd=None):
        pass

    def see_file_tree(self, root_dir="."):
        pass

    def search_in_files(self, pattern, root_dir="."):
        pass
```

## Prompt to Add Type Hints and Implement

Use this prompt with ChatGPT to add type hints, docstrings, and implement the methods:

```
Please enhance this AgentTools class with type hints, docstrings, and simple implementations
for file operations, bash execution, and search.
```

If you see a lot files when looking at the tree, ask AI to exlude .venv and other files:

```
Exlude .git, .venv and other technical files and folders from see_file_tree and search_in_files
```

## Testing the Tools

Initialize the tools with your Django template:

```python
from pathlib import Path
from agent_tools import AgentTools

project_path = Path("django_template")
tools = AgentTools(project_path)
```

List all files in the project:

```python
files = tools.see_file_tree()
print("Files in project:")
for f in files[:10]:
    print(f"  {f}")
```

Read a specific file:

```python
views_content = tools.read_file("myapp/views.py")
print("Views.py content:")
print(views_content)
```

Search for a pattern across files:

```python
matches = tools.search_in_files("def home")
print("Searching for 'def home':")
for path, line_num, line in matches:
    print(f"  {path}:{line_num} - {line.strip()}")
```

## Implementation

See [agent_tools.py](agent_tools.py) for the complete implementation with type hints, docstrings, and full method implementations.


## Copying the Template and Initializing Tools

Create a function to copy the template into a new project directory:

```python
import os
import shutil

def start(project_name):
    if os.path.exists(project_name):
        print(f"Directory '{project_name}' already exists.")
        return

    shutil.copytree('django_template', project_name)
    print(f"Django template copied to '{project_name}' directory.")
```

Run it to create your project:

```python
project_name = "my_project"
start(project_name)
```

Now initialize the tools with your new project path:

```python
from pathlib import Path
from agent_tools import AgentTools

project_path = Path(project_name)
tools = AgentTools(project_path)

# Verify tools are working
files = tools.see_file_tree()
print(f"Found {len(files)} files in project")
```

# Part 3: Designing and Developing the Coding Agent

## Starting with a Simple Prompt

Let's start with a minimal developer prompt:

```python
DEVELOPER_PROMPT = """
You are a coding agent. Your task is to modify the provided Django project
according to user instructions.
"""
```

## Creating the Agent

Use ChatGPT to help set up the agent:

```
I want to create a coding agent using ToyAIKit. Help me with these steps:

1. Import what I need: Path from pathlib, OpenAI from openai, and the toyaikit components (Tools, IPythonChatInterface, OpenAIResponsesRunner, OpenAIClient)
2. Initialize AgentTools with my project path
3. Create a Tools object and add my agent_tools to it
4. Set up the OpenAI client
5. Create the runner with all components
6. Run the agent
```

Import the required libraries:

```python
from pathlib import Path
from openai import OpenAI
from toyaikit.tools import Tools
from toyaikit.chat import IPythonChatInterface
from toyaikit.chat.runners import OpenAIResponsesRunner
from toyaikit.llm import OpenAIClient
from agent_tools import AgentTools
```

Initialize your tools and register them:

```python
agent_tools = AgentTools(Path(project_name))
tools_obj = Tools()
tools_obj.add_tools(agent_tools)
```

Create the LLM client and chat interface:

```python
llm_client = OpenAIClient(client=OpenAI(), model='gpt-4o-mini')
chat_interface = IPythonChatInterface()
```

Create and run the agent:

```python
runner = OpenAIResponsesRunner(
    tools=tools_obj,
    developer_prompt=DEVELOPER_PROMPT,
    chat_interface=chat_interface,
    llm_client=llm_client
)

result = runner.run()
```

Save the result to a file:

```python
Path("agent_result.txt").write_text(str(result))
```

Note: You can pass input directly like `runner.run(input="Create a todo list app")` or let the chat interface prompt you.

## Improving the Prompt with ChatGPT

The simple prompt works, but the agent may struggle with specific details. Use ChatGPT to expand it:

```
I have a Django coding agent with this prompt:

"You are a coding agent. Your task is to modify the provided Django project according to user instructions."

The project uses Django 5.2.4, uv, TailwindCSS, and has an app called 'myapp'.

Please improve my prompt to make the agent more effective. Include:
1. Instructions to explore before making changes
2. Style guidelines (TailwindCSS)
3. Best practices for Django
4. A reminder not to run 'runserver'
```

## Better Version

After iterating with ChatGPT, you might get something like this:

```python
DEVELOPER_PROMPT = """
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
```

## Final Version with ChatGPT Help

For a production-ready agent, ask ChatGPT to add more detail:

```
Please expand this Django coding agent prompt to include:
- A complete file tree showing the project structure
- Detailed descriptions of each main directory and file
- Specific Django best practices to follow
- Instructions for using Font-Awesome icons
- How to handle models, views, URLs, and templates
- Testing and verification steps
```

This will give you a comprehensive prompt:

```python
DEVELOPER_PROMPT = """
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

IMPORTANT: All Python commands MUST be executed via UV.
- Use: `uv run python ...`
- Never run: `python ...`, `pip ...`, or `python -m ...` directly.
- If you need Django management commands, ALWAYS do: `uv run python manage.py <command>`
Examples:
- `uv run python manage.py migrate`
- `uv run python manage.py makemigrations`
- `uv run python manage.py test`
- `uv run python manage.py check`
"""
```

See the full example at: https://github.com/alexeygrigorev/workshops/blob/main/coding-agent/README.md

## Example Session

Try asking the agent to build something:

```
You: Create a todo app with the ability to add, complete, and delete todos
```

The agent will:
1. Explore the project structure
2. Create a Todo model
3. Create migrations
4. Update views with todo logic
5. Create templates for todo UI
6. Update URLs

After the agent finishes, run the app:

```bash
cd my_project
make migrate
make run
```

# Part 4: PydanticAI and Monitoring


We'll progress through two stages:
1. Run agent with PydanticAI directly
2. Add monitoring with Logfire

## Running with PydanticAI

So far we've used ToyAIKit, which is great for learning and experimentation. Now let's switch to PydanticAI.

Why PydanticAI?

- Production-ready: PydanticAI is built for real-world applications with better error handling and type safety
- Logfire integration: Built-in observability through Logfire for monitoring agent behavior, costs, and debugging
- Better async support: Designed for asynchronous operations from the ground up
- Active development: Regularly updated with improvements and new features

Now let's do the same with PydanticAI instead of ToyAIKit.

Import what you need:

```python
from pathlib import Path
from pydantic_ai import Agent
from agent_tools import AgentTools

agent_tools = AgentTools(Path(project_name))
```

Define the developer prompt with complete instructions:

```python
CODING_AGENT_INSTRUCTIONS = """
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
```

Create the agent with tools:

```python
agent = Agent(
    'openai:gpt-4o-mini',
    instructions=CODING_AGENT_INSTRUCTIONS,
    tools=[
        agent_tools.read_file,
        agent_tools.write_file,
        agent_tools.execute_bash_command,
        agent_tools.see_file_tree,
        agent_tools.search_in_files
    ]
)
```

Run the agent directly:

```python
result = await agent.run("Create a todo list app")

print(result.output)
```

## Monitoring Tool Calls with Callbacks

To see what tools the agent is calling, create a callback class:

```python
from pydantic_ai.messages import FunctionToolCallEvent

class ToolCallback:

    async def print_function_calls(self, ctx, event):
        # Handle nested async streams
        if hasattr(event, "__aiter__"):
            async for sub in event:
                await self.print_function_calls(ctx, sub)
            return

        if isinstance(event, FunctionToolCallEvent):
            tool_name = event.part.tool_name
            args = event.part.args
            print(f"TOOL CALL: {tool_name}({args})")

    async def __call__(self, ctx, event):
        return await self.print_function_calls(ctx, event)
```

Use the callback when running the agent:

```python
callback = ToolCallback()

result = await agent.run(
    "Create a todo list app",
    event_stream_handler=callback
)
```

Now you can see exactly which tools the agent uses:

```
TOOL CALL: see_file_tree({'root_dir': '.'})
TOOL CALL: read_file({'filepath': 'myapp/models.py'})
TOOL CALL: write_file({'filepath': 'myapp/models.py', 'content': '...'})
...
```

## Chat-Bot Like Conversation

This is how we implement it with Pydantic AI:

```python
message_history = []

while True:
    user_prompt = input('You:')
    if user_prompt.lower().strip() == 'stop':
        break

    print(user_prompt)

    result = await agent.run(
        user_prompt=user_prompt,
        message_history=message_history,
        event_stream_handler=callback
    )

    print('ASSISTANT:')
    print(result.output)
    message_history.extend(result.new_messages())
    
    print()
```

## Adding Monitoring with Logfire

Install Logfire:

```bash
uv add logfire
```

Add the Logfire token to your `.env` file:

```bash
LOGFIRE_TOKEN='your-logfire-token-here'
```

Configure Logfire and instrument PydanticAI:

```python
import logfire

# Configure Logfire (reads LOGFIRE_TOKEN from .env)
logfire.configure()

# Instrument all PydanticAI calls
logfire.instrument_pydantic_ai()
```

Now create a new agent - all calls will be traced:

```python
agent = Agent(
    'openai:gpt-4o-mini',
    instructions=CODING_AGENT_INSTRUCTIONS,
    tools=[
        agent_tools.read_file,
        agent_tools.write_file,
        agent_tools.execute_bash_command,
        agent_tools.see_file_tree,
        agent_tools.search_in_files
    ]
)

# This run is now traced in Logfire
result = await agent.run("Add a contact page to the app")
```

## Summary

We switched from ToyAIKit to PydanticAI and added monitoring:
- PydanticAI runs agents directly with `await agent.run()`
- Logfire instruments all agent calls automatically
- Tool callbacks let you monitor individual tool calls in real-time

# Part 5: Multi-Agent Systems


## Why Multiple Agents?

So far we've used a single agent with all available tools. For complex tasks, we can split work across specialized agents.

Benefits of multi-agent systems:

- Specialization: Each agent focuses on a specific skill (clarification, naming, planning, coding)
- Separation of concerns: Cleaner architecture with defined roles
- Better control: Python code orchestrates the flow, not an LLM
- Structured communication: Each agent produces predictable output for the next

Disadvantages of multi-agent systems:

- Complexity: More moving parts to understand and debug
- Higher costs: Each agent call consumes tokens
- Harder to test: You need to test each agent AND their interactions

## Multi-Agent Architecture

The workflow consists of four agents:

1. Clarifier: Asks questions to understand requirements
2. Namer: Generates a project name and slug
3. Planner: Creates a step-by-step implementation plan
4. Executor: Implements each step

Execution flow:

```
1. clarifier (asks questions until READY)
2. namer (generates ProjectName with name and slug)
3. planner (creates Plan with PlanStep[])
4. for each step from planner: executor
5. done
```

## Structured Output Models

Define Pydantic models for each agent's output:

```python
from pydantic import BaseModel

# For the Namer agent
class ProjectName(BaseModel):
    name: str  # Human-readable project name
    slug: str  # URL/filesystem-safe folder name

# For the Planner agent
class PlanStep(BaseModel):
    name: str
    detailed_description: str

class Plan(BaseModel):
    overview: str
    steps: list[PlanStep]
```

## Clarifier Agent

The clarifier asks questions to understand the user's requirements:

```python
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
```

The clarifier runs interactively until it outputs "READY":

```python
result = await clarifier.run("Create a todo list app")

# Build Q&A prompt for the next agent
qa_prompt = result.output  # Contains the full conversation
```

## Namer Agent

The namer generates a project name and slug (structured output):

```python
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

# Run with structured output
result = await namer.run(qa_prompt)
project_name = result.output  # ProjectName(name="...", slug="...")
```

## Planner Agent

The planner creates a detailed implementation plan (structured output):

```python
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

# Run with structured output
result = await planner.run(qa_prompt)
plan = result.output  # Plan(overview="...", steps=[PlanStep(...), ...])
```

## Executor Agent

The executor implements each step from the plan:

```python
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
```

## Executing the Plan

First, get the structured outputs from each agent:

```python
# 1. Clarifier gathers requirements
clarifier_result = await clarifier.run("Create a todo list app")
qa_prompt = clarifier_result.output

# 2. Namer generates project name and slug
namer_result = await namer.run(qa_prompt)
project_name: ProjectName = namer_result.output

# 3. Planner creates structured plan
planner_result = await planner.run(qa_prompt)
plan: Plan = planner_result.output
```

Then execute each step sequentially:

```python
for i, step in enumerate(plan.steps):
    print(f"Step {i+1}/{len(plan.steps)}: {step.name}")

    prompt = f"""
Project overview: {plan.overview}

Current step - step #{i+1}:

{step.name}

{step.detailed_description}

File tree:
{chr(10).join(agent_tools.see_file_tree())}
""".strip()

    result = await executor.run(prompt)
    print(result.output)
```

## Summary

Multi-agent systems add specialization at the cost of complexity:

- Clarifier: Gathers requirements through questions
- Namer: Generates `ProjectName` with structured `name` and `slug`
- Planner: Creates structured `Plan` with `PlanStep[]`
- Executor: Implements each step sequentially
- Python code orchestrates the entire flow
- Each agent produces structured output that the next agent can consume reliably

## Advanced: Extending the Pattern

You can extend the multi-agent pattern in several ways:

1. Add a validator agent that checks code quality after each step
2. Add retry logic where executor continues until a step is marked complete

These are ideas for extending the pattern. The core planner-executor pattern described above has been tested and works well. Adding more complexity should be done carefully as it increases costs and makes the system harder to debug.


# Part 6: Coordinator Agent Pattern


In Part 5, we used Python code to orchestrate agents in a fixed sequence. Another approach is to let a coordinator agent decide which agents to call and when.

This pattern is more flexible but less predictable. The coordinator agent decides at runtime which specialized agent to invoke based on the task.

## Why Use a Coordinator Agent?

Benefits:
- Flexible: The coordinator can adapt the flow based on the situation
- Dynamic: Different tasks can trigger different agent combinations
- Declarative: You describe what agents can do, the coordinator figures out how

Drawbacks:
- Less predictable: The LLM decides which agents to call, not your code
- Harder to debug: You need to inspect what the coordinator decided
- More expensive: Each delegation adds another LLM call
- Untested: This is an experimental pattern

## Setting Up the Coordinator

Define the specialized agents:

```python
from pydantic_ai import Agent
from pydantic import BaseModel
```

First, define structured output for the planner:

```python
class PlanStep(BaseModel):
    name: str
    detailed_description: str

class Plan(BaseModel):
    overview: str
    steps: list[PlanStep]
```

Create the planner agent:

```python
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
```

Create the researcher agent that reports findings in a structured way:

```python
class ResearchFindings(BaseModel):
    summary: str
    relevant_files: list[str]
    existing_patterns: list[str]
    recommendations: list[str]

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
```

Create the executor agent:

```python
EXECUTOR_INSTRUCTIONS = """
You are a code executor. Your job is to:
1. Create new files based on specifications
2. Modify existing files following Django patterns
3. Execute bash commands for testing and validation
4. Use TailwindCSS for styling
5. Follow Python and Django best practices

Use all available tools (write_file, read_file, execute_bash_command, etc).
After making changes, provide a brief summary of what was done.
"""

executor = Agent(
    'openai:gpt-4o-mini',
    instructions=EXECUTOR_INSTRUCTIONS,
    tools=[
        agent_tools.write_file,
        agent_tools.read_file,
        agent_tools.execute_bash_command,
        agent_tools.see_file_tree,
        agent_tools.search_in_files
    ]
)
```

Create the validator agent:

```python
class ValidationReport(BaseModel):
    status: str  # "pass", "fail", or "warning"
    issues_found: list[str]
    suggestions: list[str]

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
```

Now create the coordinator with tools that call each agent:

```python
from pydantic_ai import RunContext

COORDINATOR_INSTRUCTIONS = """
You coordinate a team of agents to build Django applications:
- planner: Creates a structured plan with steps
- researcher: Explores the codebase and provides structured findings
- executor: Executes the plan (reads, writes, runs commands)
- validator: Reviews code for quality

When given a task, decide which agents to call and in what order:

1. For new tasks, start with researcher to understand the codebase
2. Then call planner to create a plan
3. For each step in the plan, call executor to implement
4. After implementation, call validator to check quality

You can call agents multiple times if needed. Always use the available tools.
"""

coordinator = Agent(
    'openai:gpt-4o-mini',
    instructions=COORDINATOR_INSTRUCTIONS
)

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
async def execute(ctx: RunContext, instruction: str) -> str:
    """Runs the executor agent to implement code changes."""
    result = await executor.run(user_prompt=instruction)
    return result.output

@coordinator.tool
async def validate(ctx: RunContext, filepath: str) -> str:
    """Runs the validator agent to check code quality."""
    result = await validator.run(user_prompt=f"Review {filepath}")
    return result.output
```

Run the coordinator:

```python
result = await coordinator.run("Create a todo list app with add/delete functionality")

print(result.output)
```

## How It Works

The coordinator agent decides which agents to call and in what order. Each agent produces structured output that other agents can consume:

```
user task
    ↓
coordinator (decides what to do)
    ↓
┌─────────────────────────────────────────────────────────┐
│  1. researcher → ResearchFindings                        │
│     - summary: what was found                           │
│     - relevant_files: list of important files          │
│     - existing_patterns: conventions used               │
│     - recommendations: suggestions for implementation   │
├─────────────────────────────────────────────────────────┤
│  2. planner → Plan                                        │
│     - overview: brief description                       │
│     - steps: list of PlanStep (name, description)      │
├─────────────────────────────────────────────────────────┤
│  3. executor → implements code                           │
│     Uses Plan steps and ResearchFindings as context     │
├─────────────────────────────────────────────────────────┤
│  4. validator → ValidationReport                         │
│     - status: "pass", "fail", or "warning"              │
│     - issues_found: list of problems                    │
│     - suggestions: list of improvements                 │
└─────────────────────────────────────────────────────────┘
```

## Why Structured Output?

Using Pydantic models for agent output is crucial for multi-agent systems:

- Predictable shape: Each agent knows exactly what format to expect
- Type safety: Pydantic validates the output structure
- Composability: Agents can consume other agents' output reliably
- Debugging: Structured data is easier to inspect than free text

For example, the researcher produces `ResearchFindings` with `relevant_files`. The planner can use this list to focus on the right files. The executor receives both the plan AND the research findings, giving it full context.

## Summary

The coordinator agent pattern is an alternative to fixed orchestration:

- Fixed orchestration (Part 5): Python code controls the flow, tested and predictable
- Coordinator agent (Part 6): LLM controls the flow, flexible but experimental

Key difference: In Part 5, Python code orchestrates agents in a fixed sequence. In Part 6, the coordinator LLM decides dynamically which agents to call, using structured outputs to pass data between them.

The coordinator pattern shines when:
- Tasks vary widely in structure
- You need adaptive workflows
- You can tolerate some unpredictability

Use fixed orchestration when you need reliability and control.

Note: This pattern has not been extensively tested. It works technically but may be less reliable than the fixed orchestration approach.

# Part 7: Guardrails


## Why Guardrails Matter

When building agents, you need to ensure safety:
- Block dangerous commands
- Limit costs and token usage
- Validate inputs and outputs

## Input Validation

Validate user input before passing it to the agent:

```python
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

# Use before running agent
request = UserRequest(task=user_input)
```

## Output Sanitization

Check generated code for harmful patterns:

```python
def sanitize_code_output(code: str) -> str:
    """Remove potentially harmful patterns from generated code."""
    dangerous_imports = ['os.system', 'subprocess.call', 'eval', 'exec']

    for imp in dangerous_imports:
        if imp in code:
            raise ValueError(f"Potentially dangerous code: {imp}")

    return code

# Apply to agent output
result = await agent.run(task)
if hasattr(result, 'data') and result.data:
    sanitize_code_output(result.data)
```

## Cost Limits

Limit the number of requests and tokens per run:

```python
from pydantic_ai import Agent, UsageLimits

agent = Agent(
    'openai:gpt-4o-mini',
    instructions='You are a coding assistant.',
)

# UsageLimits are passed when running the agent
result = await agent.run(
    'Create a todo list',
    usage_limits=UsageLimits(
        request_limit=20,  # Maximum number of requests
        total_tokens_limit=10000  # Maximum tokens per run
    )
)
```

## Complete Agent with Guardrails

Putting it all together:

```python
from pathlib import Path
from pydantic_ai import Agent, UsageLimits
from agent_tools import AgentTools
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

# Initialize tools
agent_tools = AgentTools(Path(project_name))

# Create agent
agent = Agent(
    'openai:gpt-4o-mini',
    instructions=CODING_AGENT_INSTRUCTIONS,
    tools=[
        agent_tools.read_file,
        agent_tools.write_file,
        agent_tools.execute_bash_command,
        agent_tools.see_file_tree,
        agent_tools.search_in_files
    ]
)

# Validate input, run agent with limits, check output
user_input = "Create a blog app"
validated = UserRequest(task=user_input)

result = await agent.run(
    validated.task,
    usage_limits=UsageLimits(
        request_limit=30,
        total_tokens_limit=20000
    )
)
```

# Summary: Day 2

Today we built a complete coding agent:

1. Django Template: Set up a working project scaffold
2. Agent Tools: Implemented file access, bash execution, search
3. Single Agent: Built a coding agent with ToyAIKit
4. Multi-Agent: Coordinated specialized agents with PydanticAI
5. Monitoring: Added observability with Logfire
6. Guardrails: Implemented safety measures

## Key Takeaways

 - Agents need clear prompts and well-defined tools
 - File manipulation tools enable coding agents
 - Multi-agent systems handle complexity through specialization
 - Monitoring is essential for production agents
 - Guardrails protect against dangerous operations

## Next Steps

 - Try building different types of applications
 - Add more specialized agents (testing, documentation, etc.)
 - Implement more sophisticated guardrails
 - Explore MCP for external tool integration
 - Deploy your agent as a service

## Resources

 - PydanticAI Docs: https://ai.pydantic.dev
 - Logfire: https://logfire.pydantic.dev
 - Django: https://docs.djangoproject.com
 - ToyAIKit: https://github.com/alexeygrigorev/toyaikit
