# AI Agents Workshop: From LLMs to Agentic RAG

A comprehensive 2-day hands-on workshop covering LLM APIs, RAG systems, agent orchestration with PydanticAI, and Model Context Protocol (MCP).

## Prerequisites

- Python 3.10+
- Jupyter Notebook/Lab
- Docker (for optional components)
- API keys for: OpenAI, Groq, Anthropic, Google Gemini

## Environment Setup

```bash
# Recommended: Use uv for environment management
pip install uv
uv sync

# Or install manually
pip install jupyter openai anthropic google-genai groq toyaikit pydantic-ai minsearch
```

### Setting up API Keys

```bash
# Add to .envrc or export directly
export OPENAI_API_KEY='sk-...'
export ANTHROPIC_API_KEY='sk-ant-...'
export GROQ_API_KEY='gsk_...'
export GEMINI_API_KEY='...'
```

---

# Day 1: Foundations of LLMs and RAG

## Module 1: Introduction to LLM APIs (10-intro.ipynb)

**Learning Objectives:**
- Understand LLM basics and why they matter
- Set up development environment
- Connect to multiple LLM providers (OpenAI, Groq, Anthropic, Gemini)
- Learn the difference between Responses API and Chat Completions API

### Topics Covered:

#### 1.1 OpenAI API (gpt-4o-mini)
- Initialize the OpenAI client
- Send basic requests with Responses API
- Use system and user prompts
- Handle conversation history
- Stream responses
- Parse structured output with Pydantic

```python
from openai import OpenAI

openai_client = OpenAI()

response = openai_client.responses.create(
    model='gpt-4o-mini',
    input=[
        {"role": "developer", "content": "You're a helpful assistant."},
        {"role": "user", "content": "Tell me a joke about coding"}
    ]
)
print(response.output_text)
```

#### 1.2 Groq API (Free alternative)
- Use OpenAI-compatible API with Groq
- Configure base URL and API key
- Select models (Llama, Mixtral, etc.)

```python
from openai import OpenAI
import os

groq_client = OpenAI(
    api_key=os.getenv('GROQ_API_KEY'),
    base_url='https://api.groq.com/openai/v1'
)

response = groq_client.chat.completions.create(
    model='llama-3.3-70b-versatile',
    messages=[...]
)
```

#### 1.3 Anthropic API (Claude)
- Initialize Anthropic client
- Use Messages API
- Handle system prompts separately
- Tool calling format differences

```python
from anthropic import Anthropic

client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

response = client.messages.create(
    model='claude-3-5-haiku-latest',
    max_tokens=1024,
    system="You're a helpful assistant.",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

#### 1.4 Google Gemini API
- Use google-genai SDK
- Generate content with system instructions
- Handle streaming responses
- Structured output with JSON schema

```python
from google import genai
from google.genai import types

client = genai.Client()

response = client.models.generate_content(
    model='models/gemini-2.0-flash-exp',
    config=types.GenerateContentConfig(
        system_instruction="You're a helpful assistant."
    ),
    contents="Tell me a joke about coding"
)
```

#### 1.5 Cost Tracking
- Calculate costs per provider
- Compare pricing across providers
- Track token usage

---

## Module 2: Retrieval-Augmented Generation (20-rag.ipynb)

**Learning Objectives:**
- Understand RAG architecture
- Build a FAQ assistant with RAG
- Learn text search vs vector search
- Implement hybrid search

### Topics Covered:

#### 2.1 What is RAG?
- Search + Generation pattern
- Why LLMs need external context
- RAG pipeline architecture

#### 2.2 Simple RAG with Text Search
- Use minsearch for in-memory search
- Index FAQ documents
- Implement search function
- Build prompt with context

```python
from minsearch import Index

index = Index(
    text_fields=["question", "text", "section"],
    keyword_fields=["course"]
)
index.fit(documents)

def search(query):
    return index.search(query, num_results=5)

def rag(query):
    results = search(query)
    prompt = build_prompt(query, results)
    return llm(prompt, instructions=system_prompt)
```

#### 2.3 Vector Search with Embeddings
- Use sentence-transformers for embeddings
- Encode documents and queries
- Semantic search vs lexical search
- Combine text and vector search (hybrid)

#### 2.4 Document Chunking
- Sliding window approach
- Paragraph-based chunking
- Handling long documents (transcripts, docs)

---

## Module 3: Project - FAQ Assistant (30-faq-assistant.ipynb)

**Learning Objectives:**
- Build a complete FAQ assistant
- Implement multi-source RAG
- Add citations and references

### Features:
- Search course FAQ database
- Provide answers with sources
- Handle follow-up questions
- Support multiple courses

---

## Module 4: Introduction to Agents (40-agents-intro.ipynb)

**Learning Objectives:**
- Understand what makes an LLM an "agent"
- Learn function calling basics
- Build first agent with tools

### Topics Covered:

#### 4.1 From Chatbot to Agent
- What is function calling?
- How agents use tools
- The agentic loop

#### 4.2 Function Calling with OpenAI
```python
import json

def get_weather(location: str) -> str:
    """Get current weather for a location."""
    return f"The weather in {location} is sunny, 22°C"

weather_tool = {
    "type": "function",
    "name": "get_weather",
    "description": "Get current weather for a location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name"
            }
        },
        "required": ["location"]
    }
}

response = openai_client.responses.create(
    model='gpt-4o-mini',
    input=messages,
    tools=[weather_tool]
)
```

#### 4.3 Tool Execution Loop
- Detect tool calls in response
- Execute functions
- Return results to LLM
- Continue until final answer

#### 4.4 Multi-Provider Function Calling
- OpenAI Responses API
- Groq Chat Completions
- Anthropic Messages API
- Gemini Function Calling

---

## Module 5: ToyAIKit - Simplified Agent Framework (50-toyaikit.ipynb)

**Learning Objectives:**
- Use ToyAIKit for faster agent development
- Implement chat interfaces
- Run agentic workflows

### Topics Covered:

#### 5.1 What is ToyAIKit?
- Educational library for agents
- Handles conversation history
- Supports multiple LLM providers
- Built-in runners

#### 5.2 Basic Agent with ToyAIKit
```python
from toyaikit.llm import OpenAIClient
from toyaikit.chat import IPythonChatInterface
from toyaikit.chat.runners import OpenAIResponsesRunner
from toyaikit.tools import Tools

# Define tools
tools_obj = Tools()
tools_obj.add_tool(get_weather, weather_tool)

# Create agent
llm_client = OpenAIClient(client=OpenAI())
chat_interface = IPythonChatInterface()

runner = OpenAIResponsesRunner(
    tools=tools_obj,
    developer_prompt="You're a helpful assistant.",
    chat_interface=chat_interface,
    llm_client=llm_client
)

runner.run()
```

#### 5.3 Using Different Providers
```python
# Groq
from toyaikit.llm import OpenAIChatCompletionsClient

groq_client = OpenAI(api_key=GROQ_API_KEY, base_url='https://api.groq.com/openai/v1')
llm_client = OpenAIChatCompletionsClient(model='llama-3.3-70b-versatile', client=groq_client)

# Anthropic
from toyaikit.llm import AnthropicClient

llm_client = AnthropicClient(model='claude-3-5-haiku-latest')
```

---

## Module 6: Agentic RAG (60-agentic-rag.ipynb)

**Learning Objectives:**
- Combine RAG with agents
- Let LLM decide when to search
- Implement multi-step reasoning

### Topics Covered:

#### 6.1 RAG as a Tool
```python
def search_faq(query: str) -> str:
    """Search the FAQ database for answers."""
    results = index.search(query, num_results=5)
    return json.dumps(results)

search_tool = {
    "type": "function",
    "name": "search_faq",
    "description": "Search the FAQ database",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"}
        },
        "required": ["query"]
    }
}
```

#### 6.2 Agentic FAQ Assistant
- Agent decides when to search
- Follow-up clarifications
- Multi-query expansion

#### 6.3 Beyond Traditional RAG
- Re-ranking results
- Query rewriting
- Reflection and iteration

---

# Day 2: Advanced Agents and Orchestration

## Module 7: PydanticAI - Production Agent Framework (70-pydantic-ai.ipynb)

**Learning Objectives:**
- Use PydanticAI for production agents
- Define agents with proper typing
- Handle structured outputs
- Multi-provider support

### Topics Covered:

#### 7.1 Introduction to PydanticAI
```python
from pydantic_ai import Agent

agent = Agent(
    'openai:gpt-4o-mini',
    instructions="You're a helpful assistant.",
)
```

#### 7.2 Tools with Type Hints
```python
from typing import List

def search_faq(query: str) -> List[dict]:
    """Search FAQ database for relevant entries.

    Args:
        query: The search query text.

    Returns:
        List of matching FAQ entries.
    """
    return index.search(query, num_results=5)

agent = Agent(
    'openai:gpt-4o-mini',
    instructions="You're a FAQ assistant.",
    tools=[search_faq]
)
```

#### 7.3 Switching Providers
```python
# Anthropic
agent = Agent(
    'anthropic:claude-3-5-sonnet-latest',
    instructions=...,
    tools=...
)

# Groq
agent = Agent(
    'groq:llama-3.3-70b-versatile',
    instructions=...,
    tools=...
)
```

#### 7.4 Structured Outputs
```python
from pydantic import BaseModel

class FAQAnswer(BaseModel):
    answer: str
    sources: List[str]
    confidence: float

agent = Agent(
    'openai:gpt-4o-mini',
    instructions=...,
    tools=[...],
    result_type=FAQAnswer
)
```

#### 7.5 Running with ToyAIKit
```python
from toyaikit.chat.runners import PydanticAIRunner

runner = PydanticAIRunner(
    chat_interface=IPythonChatInterface(),
    agent=agent
)
await runner.run()
```

---

## Module 8: Multi-Agent Systems (80-multi-agent.ipynb)

**Learning Objectives:**
- Coordinate multiple specialized agents
- Implement agent handoffs
- Build multi-agent workflows

### Topics Covered:

#### 8.1 Agent Specialization
- Research agent
- Writer agent
- Critic agent
- Coordinator agent

#### 8.2 Multi-Agent with PydanticAI
```python
from pydantic_ai import Agent, RunContext

researcher = Agent(
    'openai:gpt-4o-mini',
    instructions="You research topics and gather information.",
    tools=[search_tool, web_search_tool]
)

writer = Agent(
    'openai:gpt-4o-mini',
    instructions="You write clear, engaging content.",
)

coordinator = Agent(
    'openai:gpt-4o-mini',
    instructions="Coordinate between researchers and writers.",
    tools=[researcher, writer]
)
```

#### 8.3 Agent Handoffs
- Context passing
- Delegation patterns
- Result aggregation

---

## Module 9: Model Context Protocol (MCP) (90-mcp.ipynb)

**Learning Objectives:**
- Understand MCP architecture
- Expose tools via MCP servers
- Connect agents to external tools

### Topics Covered:

#### 9.1 What is MCP?
- Standard for tool exposure
- Client-server model
- Transport independence

#### 9.2 MCP Server Examples
- File system tools
- Database access
- External API integration

#### 9.3 Using MCP with Agents
```python
# Connect to MCP server
# Expose tools to agent
# Handle tool execution
```

---

## Module 10: Monitoring and Guardrails (100-monitoring.ipynb)

**Learning Objectives:**
- Add observability to agents
- Implement safety guardrails
- Track costs and performance

### Topics Covered:

#### 10.1 Logging with Logfire
```python
from pydantic_ai import Agent
from pydantic_logfire import LogfireInstrument

agent = Agent(
    'openai:gpt-4o-mini',
    instrument=LogfireInstrument()
)
```

#### 10.2 Guardrails
- Input validation
- Output sanitization
- PII filtering
- Content moderation

#### 10.3 Cost Tracking
- Token usage per agent
- Cost per query
- Budget limits

---

## Module 11: Project - Coding Agent (110-coding-agent.ipynb)

**Learning Objectives:**
- Build a complete coding agent
- Use Django template project
- Implement file manipulation tools
- Test and iterate

### Topics Covered:

#### 11.1 Project Setup
```bash
git clone https://github.com/alexeygrigorev/django_template.git
cd django_template
uv sync
make migrate
make run
```

#### 11.2 Agent Tools
```python
from pathlib import Path

class AgentTools:
    def __init__(self, project_path: Path):
        self.project_path = project_path

    def read_file(self, filepath: str) -> str:
        """Read file contents."""
        full_path = self.project_path / filepath
        return full_path.read_text()

    def write_file(self, filepath: str, content: str) -> None:
        """Write content to file."""
        full_path = self.project_path / filepath
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)

    def execute_bash_command(self, command: str) -> str:
        """Execute bash command in project directory."""
        result = subprocess.run(
            command,
            cwd=self.project_path,
            capture_output=True,
            text=True
        )
        return result.stdout

    def see_file_tree(self, root_dir: str = ".") -> List[str]:
        """List all files in project."""
        full_path = self.project_path / root_dir
        return [
            str(p.relative_to(self.project_path))
            for p in full_path.rglob("*")
            if p.is_file()
        ]
```

#### 11.3 Building the Agent
```python
from toyaikit.tools import Tools, get_instance_methods
from toyaikit.chat.runners import OpenAIResponsesRunner

DEVELOPER_PROMPT = """
You are a coding agent. Your task is to modify the provided Django project
according to user instructions. Use the available tools to make changes.

## Project Structure
[Describe Django template structure]

## Instructions
- Use TailwindCSS for styling
- Add pictograms and emojis
- Keep logic server-side when possible
"""

agent_tools = AgentTools(Path(project_name))
tools_obj = Tools()
tools_obj.add_tools(agent_tools)

runner = OpenAIResponsesRunner(
    tools=tools_obj,
    developer_prompt=DEVELOPER_PROMPT,
    chat_interface=IPythonChatInterface(),
    llm_client=OpenAIClient(client=OpenAI())
)

runner.run()
```

---

## Module 12: Advanced Topics (120-advanced.ipynb)

**Learning Objectives:**
- Explore advanced agent patterns
- Learn optimization techniques
- Prepare for production

### Topics Covered:

#### 12.1 Streaming Responses
- Real-time output
- Progress indicators
- Early termination

#### 12.2 Batching and Parallelism
- Parallel tool calls
- Batch processing
- Rate limiting

#### 12.3 Caching Strategies
- Semantic caching
- Response caching
- Embedding caching

#### 12.4 Production Considerations
- Error handling
- Retries and timeouts
- Fallback strategies

---

## Module 13: Workshop Summary & Next Steps (130-summary.ipynb)

**Learning Objectives:**
- Review key concepts
- Explore further resources
- Plan continued learning

### Topics:
- Recap: APIs → RAG → Agents → MCP
- Framework comparison
- When to use each approach
- Further reading and resources

---

## Appendix: Quick Reference

### Provider SDK Installation

```bash
# OpenAI
pip install openai

# Anthropic
pip install anthropic

# Google Gemini
pip install google-genai

# Groq (uses OpenAI SDK)
# pip install openai

# Frameworks
pip install toyaikit pydantic-ai minsearch
```

### Client Initialization Quick Reference

```python
# OpenAI
from openai import OpenAI
client = OpenAI()

# Anthropic
from anthropic import Anthropic
client = Anthropic(api_key=...)

# Gemini
from google import genai
client = genai.Client()

# Groq
from openai import OpenAI
client = OpenAI(api_key=..., base_url='https://api.groq.com/openai/v1')
```

### PydanticAI Provider Strings

```python
'openai:gpt-4o-mini'
'openai:gpt-4o'
'anthropic:claude-3-5-sonnet-latest'
'anthropic:claude-3-5-haiku-latest'
'groq:llama-3.3-70b-versatile'
'groq:mixtral-8x7b-32768'
'gemini:gemini-2.0-flash-exp'
'gemini:gemini-1.5-pro'
```

---

## Project Structure

```
now-workshop/
├── plan.md                    # This file
├── outline.md                 # Original outline
├── 10-intro.ipynb            # LLM APIs overview
├── 20-rag.ipynb              # RAG basics
├── 30-faq-assistant.ipynb    # FAQ project
├── 40-agents-intro.ipynb     # Function calling
├── 50-toyaikit.ipynb         # ToyAIKit framework
├── 60-agentic-rag.ipynb      # Agentic RAG
├── 70-pydantic-ai.ipynb      # PydanticAI framework
├── 80-multi-agent.ipynb      # Multi-agent systems
├── 90-mcp.ipynb              # Model Context Protocol
├── 100-monitoring.ipynb      # Monitoring & guardrails
├── 110-coding-agent.ipynb    # Coding agent project
├── 120-advanced.ipynb        # Advanced topics
└── 130-summary.ipynb         # Summary & next steps
```

---

## Homework Assignments

### Day 1 Homework
1. Build a RAG system for your own documents
2. Compare results across 2 different LLM providers
3. Implement chunking for a long document

### Day 2 Homework
1. Build a multi-agent system with 3 specialized agents
2. Add monitoring and cost tracking
3. Create a custom tool and integrate with an agent

---

## Resources

- **OpenAI Docs**: https://platform.openai.com/docs
- **Anthropic Docs**: https://docs.anthropic.com
- **Google Gemini**: https://ai.google.dev/gemini-api/docs
- **Groq Docs**: https://console.groq.com/docs
- **PydanticAI**: https://ai.pydantic.dev
- **ToyAIKit**: https://github.com/alexeygrigorev/toyaikit
- **MCP**: https://modelcontextprotocol.io
