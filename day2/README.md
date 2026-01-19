# Day 2: Building a Custom Coding Agent

Learning Objectives:
 - Set up a Django project template
 - Implement tools for file access and manipulation
 - Design and develop a coding agent
 - Coordinate multiple agents
 - Add monitoring and guardrails with Pydantic Logfire

Source: https://github.com/alexeygrigorev/workshops/blob/main/coding-agent/README.md

## Prerequisites

 - Completed Day 1
 - Django project template (provided)
 - All API keys configured

Source: https://github.com/alexeygrigorev/workshops/blob/main/coding-agent/README.md

### Environment Setup

If you haven't set up `dirdotenv` from Day 1:

```bash
pip install dirdotenv
echo 'eval "$(dirdotenv hook bash)"' >> ~/.bashrc
```

Create a `.env` file with your API keys:

```bash
# .env
OPENAI_API_KEY='sk-...'
ANTHROPIC_API_KEY='sk-ant-...'
GROQ_API_KEY='gsk_...'
GEMINI_API_KEY='...'
```

# Part 1: Setting Up the Django Project Template

## What We're Building

We're building a "Project Bootstrapper" agent - an AI agent that can:
 - Read files in a codebase
 - Write new files
 - Execute bash commands
 - Modify a Django template to build applications

This is the foundation for tools like Cursor, Windsurf, and other AI coding assistants.

## Getting the Template

Download the Django template:

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

Source: https://github.com/alexeygrigorev/workshops/blob/main/coding-agent/tools.py

See [agent_tools.py](agent_tools.py) for the `AgentTools` class implementation with methods for file access, bash execution, and search.

## Testing the Tools

```python
from pathlib import Path
from agent_tools import AgentTools

# Initialize with our template
project_path = Path("django_template")
tools = AgentTools(project_path)

# Test file tree
files = tools.see_file_tree()
print("Files in project:")
for f in files[:10]:
    print(f"  {f}")

# Test read file
views_content = tools.read_file("myapp/views.py")
print("\nViews.py content:")
print(views_content)

# Test search
matches = tools.search_in_files("def home")
print("\nSearching for 'def home':")
for path, line_num, line in matches:
    print(f"  {path}:{line_num} - {line.strip()}")
```

# Part 3: Designing and Developing the Coding Agent

## The Developer Prompt

The key to a good coding agent is a clear developer prompt:

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

## Available Tools

- read_file(filepath): Read any file
- write_file(filepath, content): Write or modify files
- execute_bash_command(command, cwd): Run shell commands
- see_file_tree(root_dir): List all files
- search_in_files(pattern, root_dir): Search for code patterns
""".strip()
```

## Creating the Agent with ToyAIKit

```python
from pathlib import Path
from openai import OpenAI
from toyaikit.tools import Tools
from toyaikit.chat import IPythonChatInterface
from toyaikit.chat.runners import OpenAIResponsesRunner
from toyaikit.llm import OpenAIClient
from agent_tools import AgentTools

# Initialize tools
agent_tools = AgentTools(Path("django_template"))
tools_obj = Tools()
tools_obj.add_tools(agent_tools)

# Create the agent
llm_client = OpenAIClient(client=OpenAI())
chat_interface = IPythonChatInterface()

runner = OpenAIResponsesRunner(
    tools=tools_obj,
    developer_prompt=DEVELOPER_PROMPT,
    chat_interface=chat_interface,
    llm_client=llm_client
)

# Run the agent
runner.run()
```

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
cd django_template
make migrate
make run
```

# Part 4: Coordinating Multiple Agents

Source: https://github.com/alexeygrigorev/ai-bootcamp

## Why Multiple Agents?

Complex tasks benefit from specialized agents:
 - Research Agent: Gathers information, explores codebase
 - Writer Agent: Creates code and content
 - Reviewer Agent: Checks for issues
 - Fixer Agent: Applies corrections

## Multi-Agent with PydanticAI

```python
from pathlib import Path
from pydantic_ai import Agent
from agent_tools import AgentTools

# Initialize tools
agent_tools = AgentTools(Path("django_template"))

# Research Agent - explores the codebase
researcher = Agent(
    'openai:gpt-4o-mini',
    instructions="""
    You are a code researcher. Your job is to:
    1. Explore the codebase structure
    2. Find relevant files and patterns
    3. Understand existing conventions
    4. Report your findings clearly

    Use search and read_file tools extensively.
    """,
    tools=[
        agent_tools.search_in_files,
        agent_tools.see_file_tree,
        agent_tools.read_file
    ]
)

# Writer Agent - creates and modifies code
writer = Agent(
    'openai:gpt-4o-mini',
    instructions="""
    You are a code writer. Your job is to:
    1. Create new files based on specifications
    2. Modify existing files following Django patterns
    3. Use TailwindCSS for styling
    4. Follow Python and Django best practices

    Use write_file tool to create code.
    """,
    tools=[
        agent_tools.write_file,
        agent_tools.read_file
    ]
)

# Reviewer Agent - checks for issues
reviewer = Agent(
    'openai:gpt-4o-mini',
    instructions="""
    You are a code reviewer. Your job is to:
    1. Check code for bugs and issues
    2. Verify Django patterns are followed
    3. Suggest improvements
    4. Report specific problems with file locations

    Read files and provide constructive feedback.
    """,
    tools=[
        agent_tools.read_file,
        agent_tools.search_in_files
    ]
)
```

## Coordinator Agent

The coordinator orchestrates other agents:

```python
coordinator = Agent(
    'openai:gpt-4o-mini',
    instructions="""
    You are coordinating a team of coding agents to build Django applications.

    Your team:
    - researcher: Explores codebase and finds information
    - writer: Creates and modifies code
    - reviewer: Checks for issues

    Process:
    1. When given a task, delegate to researcher first
    2. Use research findings to guide the writer
    3. Have reviewer check the work
    4. Apply fixes if reviewer finds issues
    5. Report final status to the user

    Be specific in your delegation - tell agents exactly what to do.
    """,
    tools=[
        researcher,
        writer,
        reviewer
    ]
)
```

### Running Multi-Agent with ToyAIKit

```python
from toyaikit.chat.runners import PydanticAIRunner

runner = PydanticAIRunner(
    chat_interface=IPythonChatInterface(),
    agent=coordinator
)

await runner.run()
```

# Part 5: Monitoring & Guardrails with Logfire

Source: https://ai.pydantic.dev

## Why Monitoring Matters

When building agents, you need to observe:
 - What tools are being called
 - How much each API call costs
 - How long operations take
 - Whether the agent is behaving safely

## Setting Up Logfire

```bash
uv add logfire
```

Configure Logfire:

```python
import os
import logfire

# Log in to Logfire (free for development)
logfire.configure(
    send_to_logfire=True,
    api_key=os.getenv('LOGFIRE_TOKEN')
)
```

## Instrumenting PydanticAI Agents

```python
from pathlib import Path
from pydantic_ai import Agent
from pydantic_ai.logfire import instrument_pydantic_ai
from agent_tools import AgentTools

# Initialize tools
agent_tools = AgentTools(Path("django_template"))

# Instrument all PydanticAI calls
instrument_pydantic_ai()

# Now all agent calls are automatically traced
agent = Agent(
    'openai:gpt-4o-mini',
    instructions='You are a helpful assistant.',
    tools=[agent_tools.read_file, agent_tools.write_file]
)
```

## Viewing Traces

Every agent run creates a trace in Logfire showing:
 - Input prompts
 - Tool calls made
 - LLM responses
 - Token usage
 - Costs
 - Timing

Visit https://logfire.pydantic.dev to view your traces.

## Guardrails

### Input Validation

```python
from pydantic import BaseModel, Field, validator

class UserRequest(BaseModel):
    """Validate user requests to the agent."""
    task: str = Field(..., min_length=5, max_length=1000)

    @validator('task')
    def forbid_dangerous_commands(cls, v):
        dangerous = ['rm -rf', 'format', 'delete all', 'drop table']
        if any(cmd in v.lower() for cmd in dangerous):
            raise ValueError('Potentially dangerous command detected')
        return v

# Use before running agent
request = UserRequest(task=user_input)
```

### Output Sanitization

```python
def sanitize_code_output(code: str) -> str:
    """Remove potentially harmful patterns from generated code."""
    # Check for import blacklists
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

### Cost Limits

```python
from pydantic_ai import Agent, RunLimits

# Set limits on agent runs
agent = Agent(
    'openai:gpt-4o-mini',
    instructions='You are a coding assistant.',
    run_limits=RunLimits(
        max_steps=20,  # Maximum tool calls
        max_tokens=10000  # Maximum tokens per run
    )
)
```

### Content Moderation

```python
from openai import OpenAI

def moderate_content(text: str) -> bool:
    """Check if content is safe."""
    client = OpenAI()
    response = client.moderations.create(input=text)
    return not response.results[0].flagged

# Check agent outputs
result = await agent.run(task)
if moderate_content(result.data):
    print(result.data)
else:
    print("Content flagged as inappropriate")
```

## Complete Agent with Monitoring

```python
from pathlib import Path
from pydantic_ai import Agent, RunLimits
from pydantic_ai.logfire import instrument_pydantic_ai
import logfire
from agent_tools import AgentTools

# Initialize tools
agent_tools = AgentTools(Path("django_template"))

# Configure monitoring
logfire.configure(send_to_logfire=True)
instrument_pydantic_ai()

# Create instrumented agent
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

# Run with full observability
result = await agent.run("Create a blog app with posts and comments")

# Check the Logfire dashboard for traces
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
