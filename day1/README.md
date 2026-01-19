# Day 1: Foundations of AI Agents

Learning Objectives:
 - Learn Large Language Models and Retrieval-Augmented Generation
 - Build conversational agents using the OpenAI SDK
 - Create data-processing pipelines
 - Add agentic behavior with function calling
 - Use PydanticAI as the agents orchestrator framework
 - Expose tools via MCP

## Prerequisites

 - Python 3.10+
 - Visual Studio Code (or another Python editor)
 - Jupyter Notebook
 - API keys for LLM providers

## Environment Setup

For this workshop, we use `uv` for package management:

```bash
pip install uv
```

Create a `.env` file for environment variables (recommended):

```bash
# .env
OPENAI_API_KEY='sk-...'
ANTHROPIC_API_KEY='sk-ant-...'
GROQ_API_KEY='gsk_...'
GEMINI_API_KEY='...'
ZAI_API_KEY='sk-...'
```

Add `.env` to `.gitignore` to avoid committing API keys:

```bash
echo .env >> .gitignore
```

I use `dirdotenv` to get access to the env variables from `.env` and `.envrc` to my terminal:

```bash
pip install dirdotenv
echo 'eval "$(dirdotenv hook bash)"' >> ~/.bashrc
```

Alternatively, export keys directly in your shell or use dotenv library (I'll show it later too).

## Project Initialization

Initialize the project and install core dependencies:

```bash
uv init
uv add jupyter openai
```

If you didn't have `dirdotenv`, you can also install `python-dotenv` to load `.env` files:

```bash
uv add python-dotenv
```

Start Jupyter:

```bash
uv run jupyter notebook
```

# Part 1: Accessing LLMs with OpenAI API

## Introduction to LLM APIs

Large Language Models (LLMs) are accessed via APIs. You send a prompt and receive text back.

### OpenAI Client Setup

If you used python-dotenv, import it to load .env files:

```python
import dotenv
dotenv.load_dotenv()
```

Now import OpenAI:

```python
from openai import OpenAI

openai_client = OpenAI()
```

If your API key is configured correctly, this completes without errors.

### Basic Request

Let's send our first request:

```python
messages = [
    {"role": "user", "content": "tell me a joke about programming"}
]

response = openai_client.responses.create(
    model='gpt-4o-mini',
    input=messages
)

print(response.output_text)
```

### Understanding the Response

Let's look at what we get back:

```python
import json

print(json.dumps(response.model_dump(), indent=2))
```

Key fields:
 - `output` - everything the LLM returned
 - `usage` - token counts (input, output, total)
 - `id` - request identifier

### Getting the Text

```python
# Full text
response.output_text

# Or access via structure
response.output[0].content[0].text
```

### Streaming Responses

For long responses, streaming provides better UX:

```python
stream = openai_client.responses.create(
    model='gpt-4o-mini',
    input="write a story about programmers",
    stream=True
)

for event in stream:
    if hasattr(event, 'delta'):
        print(event.delta, end='')
```

### System Prompts

System prompts configure how the assistant behaves. They set the persona, tone, output format, and any constraints for the model. System prompts are sent once at the start of a conversation and implicitly apply to all subsequent messages.

Common uses: specifying a role (teacher, code reviewer), setting output styles (concise, verbose), defining rules (no markdown, always cite sources), or providing domain expertise.

```python
system_prompt = """
You are a funny assistant. Always make your responses humorous
and include programming jokes when possible.
""".strip()

messages = [
    {"role": "developer", "content": system_prompt},
    {"role": "user", "content": "How do I fix a bug?"}
]

response = openai_client.responses.create(
    model='gpt-4o-mini',
    input=messages
)

print(response.output_text)
```

### Conversation History

LLMs are stateless - each request is independent and the model doesn't remember previous interactions. You must send the full conversation history with every request.

Practical implications: you need to manage the messages array yourself, appending each user message and assistant response. As conversations grow, you'll hit token limits, requiring strategies like summarization, sliding windows, or selectively dropping old messages. This also means every request re-sends the entire context, which affects latency and cost.

First, introduce yourself:

```python
messages = [
    {"role": "developer", "content": "You're a helpful assistant."},
    {"role": "user", "content": "My name is Alice"}
]

response = openai_client.responses.create(
    model='gpt-4o-mini',
    input=messages
)

print(response.output_text)
```

Output:

```
Nice to meet you, Alice! How can I help you today?
```

Now ask for the name without sending history:

```python
# NEW request - no history
response = openai_client.responses.create(
    model='gpt-4o-mini',
    input="what's my name?"
)

print(response.output_text)
```

Output:

```
I don't know your name. You haven't told me yet.
```

The model doesn't remember because we didn't send the previous conversation. Now let's do it correctly by including history:

```python
# First request
messages = [
    {"role": "developer", "content": "You're a helpful assistant."},
    {"role": "user", "content": "My name is Alice"}
]

response = openai_client.responses.create(
    model='gpt-4o-mini',
    input=messages
)

# Accumulate history
messages.extend(response.output)

# Second request
messages.append({"role": "user", "content": "What's my name?"})

response = openai_client.responses.create(
    model='gpt-4o-mini',
    input=messages
)

print(response.output_text)
```

Output:

```
Your name is Alice!
```

Now the model remembers "Alice" because we sent the full conversation history.

## Structured Output with OpenAI

LLMs naturally return unstructured text. Structured output forces the response into a specific format (JSON) that matches your schema, so you can reliably parse and use the data in code. This is essential for building applications - you get strongly-typed objects instead of parsing freeform text with regex or fragile string matching.

```python
from pydantic import BaseModel

class JokeResponse(BaseModel):
    """A joke with metadata."""
    setup: str
    punchline: str
    category: str

messages = [
    {"role": "developer", "content": "Tell me a programming joke."},
    {"role": "user", "content": "Give me a joke"}
]

response = openai_client.responses.parse(
    model='gpt-4o-mini',
    input=messages,
    text_format=JokeResponse
)

joke = response.output[0].content[0].parsed
print(f"Category: {joke.category}")
print(f"Setup: {joke.setup}")
print(f"Punchline: {joke.punchline}")
```

Two ways to access the parsed result:

```python
# Method 1: Navigate the output structure
joke = response.output[0].content[0].parsed

# Method 2: Direct access via output_parsed
joke = response.output_parsed
```

### Adding Descriptions for Better LLM Understanding

You can provide additional context to the LLM about each field using `Field(description=...)`:

```python
from pydantic import Field

class JokeResponse(BaseModel):
    """A joke with metadata."""
    setup: str = Field(description="The setup that builds anticipation for the joke")
    punchline: str = Field(description="The funny conclusion or twist")
    category: str = Field(description="Type of joke: programming, general, or dad joke")
```

You can also use `Literal` to force the LLM to pick from specific options:

```python
from typing import Literal

class JokeResponse(BaseModel):
    """A joke with metadata."""
    setup: str = Field(description="The setup that builds anticipation for the joke")
    punchline: str = Field(description="The funny conclusion or twist")
    category: Literal["programming", "general", "dad"] = Field(
        description="The type of joke based on the target audience and theme"
    )
```

These descriptions are sent to the LLM and help it understand what each field should contain, resulting in more accurate outputs.

### Why Structured Output Matters

It gives type safety and validation. Without structured output, you'd need to parse JSON from text and hope it's valid:

```python
# Old way - fragile
response = openai_client.responses.create(...)
text = response.output_text
data = json.loads(text)  # Could fail if LLM returns invalid JSON
```

With structured output, the LLM guarantees valid data matching your schema:

```python
# New way - type safe
response = openai_client.responses.parse(...)
joke = response.output[0].content[0].parsed
# joke is a validated JokeResponse object
```

### Video Summarization Example

Let's look at a practical example: summarizing a YouTube transcript with structured output.

First, define the output structure:

```python
class Chapter(BaseModel):
    timestamp: str = Field(description="Timestamp in the video (e.g. 10:25)")
    title: str = Field(description="Title of this chapter section")

class VideoSummary(BaseModel):
    summary: str = Field(description="Overall summary of the video content")
    chapters: list[Chapter] = Field(description="List of chapters with timestamps")
```

Note: `VideoSummary` contains a `list[Chapter]` - this is composability, where one class is nested inside another.

### Fetching a Transcript

We'll use a transcript from the Temporal.io workshop series:

```python
import requests

transcript_url = 'https://raw.githubusercontent.com/alexeygrigorev/workshops/main/temporal.io/data/_fbe1QyJ1PY.txt'
response = requests.get(transcript_url)
transcript = response.text
```

### Generating the Summary

```python
developer_prompt = """
Summarize the transcript and describe the main purpose of the video
and the main ideas.

Also output chapters with time. Use usual sentence case, not Title Case for chapters.
More chapters is better than fewer chapters. Have a chapter at least every 3-5 minutes.
""".strip()

response = openai_client.responses.parse(
    model='gpt-4o-mini',
    input=[
        {"role": "developer", "content": developer_prompt},
        {"role": "user", "content": transcript}
    ],
    text_format=VideoSummary
)

summary = response.output_parsed
```

Let's print the output:

```python
print(summary.summary)

for chapter in summary.chapters:
    print(f"{chapter.timestamp}: {chapter.title}")
```

This structured output is ready to use - perfect for building video navigation features,
search indexing, or content recommendation systems.

# Part 2: Alternatives to OpenAI

Beyond OpenAI, there are several excellent LLM providers. We'll start with Groq since it's OpenAI-compatible, then explore Anthropic and Google Gemini.

## Groq API

Groq provides free access to open-source models. It's OpenAI-compatible:

### Groq Client Setup

Use the OpenAI client with Groq's base URL:

```python
from openai import OpenAI

groq_client = OpenAI(
    api_key=os.getenv('GROQ_API_KEY'),
    base_url='https://api.groq.com/openai/v1'
)
```

### Basic Request

```python
response = groq_client.responses.create(
    model='openai/gpt-oss-20b',
    input=[
        {"role": "system", "content": "You're a helpful assistant."},
        {"role": "user", "content": "Tell me a joke about programming"}
    ]
)

print(response.output_text)
```

## Anthropic (Claude) API

Install the Anthropic SDK:

```bash
uv add anthropic
```

### Anthropic Client Setup

```python
from anthropic import Anthropic

anthropic_client = Anthropic()
```

The Anthropic client automatically reads the `ANTHROPIC_API_KEY` environment variable. You typically don't need to specify it.

Let's test it:

```python
message = anthropic_client.messages.create(
    model='claude-haiku-4-5',
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "tell me a joke about programming"}
    ]
)

print(message.content[0].text)
```

If you need to use a different API key than the one in your environment:

```python
anthropic_client = Anthropic(api_key='sk-ant-your-key-here')
```

### Accessing Anthropic via AWS Bedrock

You can also use Claude models through AWS Bedrock.

Install boto3:

```bash
uv add boto3
```

Use `AnthropicBedrock` instread of `Anthropic`:

```python
from anthropic import AnthropicBedrock

bedrock_client = AnthropicBedrock(
    aws_region='eu-west-1',
    # Credentials are loaded from AWS environment or ~/.aws/credentials
)

message = bedrock_client.messages.create(
    model="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    max_tokens=256,
    messages=[{"role": "user", "content": "Hello, world"}]
)

print(message.content[0].text)
```

Bedrock uses your AWS credentials automatically. Make sure your credentials have access to Bedrock:

```bash
aws bedrock get-foundation-model-list
```

### Accessing Anthropic via z.ai

Z.ai provides an API compatible with Anthropic's SDK:

```python
import os
from anthropic import Anthropic

zai_client = Anthropic(
    base_url=os.getenv('ZAI_BASE_URL', 'https://api.z.ai/api/anthropic'),
    api_key=os.getenv('ZAI_API_KEY')
)
```

z.ai offers the same Claude models with potentially different pricing and latency characteristics.


### Streaming

```python
with anthropic_client.messages.stream(
    model='claude-haiku-4-5',
    max_tokens=1024,
    messages=[{"role": "user", "content": "Tell me a story about programmers"}]
) as stream:
    for text in stream.text_stream:
        print(text, end='', flush=True)
```

### Structured Output with Anthropic

Anthropic supports structured output with Pydantic models. Note: Only Sonnet and Opus models support structured output, not Haiku.

```python
from anthropic import transform_schema
from typing import Literal

class JokeResponse(BaseModel):
    """A joke with metadata."""
    setup: str
    punchline: str
    category: Literal["programming", "general", "dad"]
```

With `.create()` - requires `transform_schema()`:

```python
response = anthropic_client.beta.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    betas=["structured-outputs-2025-11-13"],
    messages=[
        {
            "role": "user",
            "content": "Tell me a programming joke about bugs."
        }
    ],
    output_format={
        "type": "json_schema",
        "schema": transform_schema(JokeResponse),
    }
)

print(response.content[0].text)
```

With `.parse()` - can pass Pydantic model directly:

```python
response = anthropic_client.beta.messages.parse(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    betas=["structured-outputs-2025-11-13"],
    messages=[
        {
            "role": "user",
            "content": "Tell me a programming joke about bugs."
        }
    ],
    output_format=JokeResponse
)

joke = response.parsed
print(joke)
```

Note: Z.ai client doesn't currently support structured output


## Google Gemini API

Install the Google GenAI SDK:

```bash
uv add google-genai
```

Setup the client:

```python
from google import genai

gemini_client = genai.Client()
```

Basic request:

```python
from google.genai import types

response = gemini_client.models.generate_content(
    model='models/gemini-2.0-flash-exp',
    config=types.GenerateContentConfig(
        system_instruction="You're a helpful assistant."
    ),
    contents="Tell me a joke about programming"
)

print(response.text)
```

Streaming:

```python
response = gemini_client.models.generate_content_stream(
    model='models/gemini-2.0-flash-exp',
    config=types.GenerateContentConfig(
        system_instruction="You're a helpful assistant."
    ),
    contents="Tell me a story about programming"
)

for chunk in response:
    if hasattr(chunk, 'text'):
        print(chunk.text, end='', flush=True)
```

### Structured Output with Gemini

Gemini supports structured output via `response_mime_type`:

```python
class JokeResponse(BaseModel):
    """A joke with metadata."""
    setup: str
    punchline: str
    category: str

response = gemini_client.models.generate_content(
    model='models/gemini-2.0-flash-exp',
    config=types.GenerateContentConfig(
        system_instruction="You're a helpful assistant.",
        response_mime_type='application/json',
        response_schema=JokeResponse.model_json_schema()
    ),
    contents="Tell me a programming joke"
)

joke_data = json.loads(response.text)
joke = JokeResponse.model_validate(joke_data)
print(joke)
```

# Part 3: Introduction to Retrieval-Augmented Generation (RAG)

## What is RAG?

LLMs are trained on public data, but they don't know your private data - your company documents, codebase, product manuals, course materials, etc.

RAG (Retrieval-Augmented Generation) solves this by combining two steps:

1. Retrieval: Search your knowledge base for relevant information
2. Generation: Pass the retrieved information to the LLM as context

This way, the LLM can answer questions using your specific data without being retrained.

### The RAG Flow

```python
def rag(query):
    # 1. Retrieve relevant documents
    search_results = search(query)

    # 2. Build prompt with context
    prompt = build_prompt(query, search_results)

    # 3. Generate answer
    answer = llm(prompt)
    return answer
```

## Search: The Retrieval Component

For RAG to work, we need a way to search through documents. We'll use minsearch, a simple in-memory search engine.

GitHub: [github.com/alexeygrigorev/minsearch](https://github.com/alexeygrigorev/minsearch)

This library resulted from implementing search from scratch in the [Build Your Own Search Engine](https://github.com/alexeygrigorev/build-your-own-search-engine/) workshop.

Install minsearch:

```bash
uv add minsearch
```

## Setting Up Data

We'll use a FAQ dataset from a data engineering course:

```python
import requests

docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

documents = []
for course in documents_raw:
    course_name = course['course']
    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)

print(f"Loaded {len(documents)} documents")
```

## Text Search with minsearch

minsearch provides two index types. `Index` uses Scikit-Learn and NumPy for faster search, but doesn't support adding documents after fitting. `AppendableIndex` is slightly slower but allows adding new documents dynamically.

We'll use `AppendableIndex` because later we'll add documents to the index with the `add_entry` tool. If you don't need to append documents, use the regular `Index` for better performance:

```python
from minsearch import AppendableIndex

index = AppendableIndex(
    text_fields=["question", "text", "section"],
    keyword_fields=["course"]
)

index.fit(documents)
```

Search for relevant documents:

```python
question = "I just found the course. Can I join now?"

results = index.search(
    question,
    filter_dict={'course': 'data-engineering-zoomcamp'},
    num_results=5,
    boost_dict={'question': 3.0, 'section': 0.5}
)

for result in results:
    print(result['question'])
```

## Building the RAG System

You can use any LLM provider we covered earlier (OpenAI, Anthropic, Gemini, Groq). Just replace the content of your `llm` function with the proper library code. We'll use OpenAI for this example.

### Define the Search Function

```python
def search(query):
    boost = {'question': 3.0, 'section': 0.5}

    results = index.search(
        query=query,
        filter_dict={'course': 'data-engineering-zoomcamp'},
        boost_dict=boost,
        num_results=5
    )

    return results
```

### Build the Prompt

```python
prompt_template = """
<QUESTION>
{question}
</QUESTION>

<CONTEXT>
{context}
</CONTEXT>
""".strip()

def build_prompt(question, search_results):
    context = json.dumps(search_results, indent=2)
    return prompt_template.format(question=question, context=context).strip()
```

### Define the LLM Function

You can swap this function with any provider (Anthropic, Gemini, Groq) by using their API call instead:

```python
instructions = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT
from the FAQ database. Use only the facts from the CONTEXT.
""".strip()

def llm(prompt):
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": prompt}
    ]

    response = openai_client.responses.create(
        model='gpt-4o-mini',
        input=messages
    )
    return response.output_text
```

### Put It Together

```python
def rag(query):
    search_results = search(query)
    user_prompt = build_prompt(query, search_results)
    return llm(user_prompt)

# Test it
answer = rag("Can I join the course after it started?")
print(answer)
```

## Vector Search (Semantic Search)

Install sentence-transformers for embeddings and tqdm for progress bars:

```bash
uv add sentence-transformers tqdm
```

Text search only finds exact word matches. Vector search finds semantic meaning:

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm.auto import tqdm

# Load embedding model
embedding_model = SentenceTransformer('multi-qa-distilbert-cos-v1')

# Create embeddings for documents
embeddings = []
for d in tqdm(documents):
    text = d['question'] + ' ' + d['text']
    v = embedding_model.encode(text)
    embeddings.append(v)

embeddings = np.array(embeddings)
```

### Using Vector Search with minsearch

```python
from minsearch import VectorSearch

vindex = VectorSearch(keyword_fields=['course'])
vindex.fit(embeddings, documents)

def vector_search(question):
    q = embedding_model.encode(question)
    return vindex.search(
        q,
        filter_dict={'course': 'data-engineering-zoomcamp'},
        num_results=5
    )
```

Use it with RAG:

```python
def rag_vector(query):
    search_results = vector_search(query)
    user_prompt = build_prompt(query, search_results)
    return llm(user_prompt)

answer = rag_vector("Can I join the course after it started?")
print(answer)
```

### Hybrid Search (Text + Vector)

Combine both methods for better results:

```python
def hybrid_search(question):
    r1 = search(question)      # text search
    r2 = vector_search(question)  # vector search
    return r1 + r2

def rag_hybrid(query):
    search_results = hybrid_search(query)
    user_prompt = build_prompt(query, search_results)
    return llm(user_prompt)

answer = rag_hybrid("Can I join the course after it started?")
print(answer)
```

# Part 4: Agents

## Why Agents?

The RAG system we built always searches, then generates. This works for simple Q&A, but real applications need more flexibility:

- Sometimes the user's question doesn't require searching (general knowledge, greetings)
- Sometimes you need to search multiple sources or databases
- Sometimes you need to search, then search again based on the results
- Sometimes you need to call other tools (APIs, databases, calculations)

Hardcoding the flow ("always search first") limits what your application can do. Agents solve this by letting the LLM decide which tools to call and in what order.

## Agentic RAG

In static RAG, we always search first. In agentic RAG, the LLM decides whether and what to search.

First, we need to define the search function and tool schema.

The LLM doesn't know about your functions. The tool schema describes what the function does, what parameters it accepts, and what each parameter means. This is how the LLM knows when and how to call your function.

```python
def search_faq(query: str) -> list:
    results = index.search(
        query=query,
        filter_dict={'course': 'data-engineering-zoomcamp'},
        num_results=5
    )
    return results

search_tool = {
    "type": "function",
    "name": "search_faq",
    "description": "Search the course FAQ database",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query for the FAQ"
            }
        },
        "required": ["query"],
        "additionalProperties": False
    }
}
```

### Implementing with Plain OpenAI SDK

Let's see how the agentic flow works step by step.

```python
instructions = "You're a course teaching assistant. When asked questions, search the FAQ database first."
user_question = "How do I install Kafka?"
```

Send the question with the search tool available:

```python
messages = [
    {"role": "system", "content": instructions},
    {"role": "user", "content": user_question}
]

response = openai_client.responses.create(
    model='gpt-4o-mini',
    input=messages,
    tools=[search_tool]
)
```

Let's see what the LLM decided:

```python
response.output
```

If the LLM decided to invoke our function, we need to execute it and send the result back. Remember, LLMs are stateless - they don't remember previous requests. We need to maintain the conversation history ourselves. First, add the LLM's response to our messages:

```python
messages.extend(response.output)
```

Then execute the function and add the result:

```python
call = response.output[0]
args = json.loads(call.arguments)
results = search_faq(**args)
result_json = json.dumps(results, indent=2)

messages.append({
    "type": "function_call_output",
    "call_id": call.call_id,
    "output": result_json,
})
```

Now invoke the LLM again with the search results:

```python
response = openai_client.responses.create(
    model='gpt-4o-mini',
    input=messages,
    tools=[search_tool]
)

print(response.output_text)
```

### The Tool-Calling Loop

The code above handles one function call. But what if the LLM needs to call multiple tools? After seeing the result of one tool call, the LLM may decide to call another tool, or even the same tool again with different parameters.

This is why we need a loop - we keep calling the API until there are no more tool calls. This is called the "tool-calling loop" or "agentic loop".

```python
user_question = "How do I install Kafka?"

messages = [
    {"role": "system", "content": instructions},
    {"role": "user", "content": user_question}
]

while True:
    response = openai_client.responses.create(
        model='gpt-4o-mini',
        input=messages,
        tools=[search_tool]
    )

    has_tool_calls = False

    for entry in response.output:
        messages.append(entry)

        if entry.type == 'function_call':
            args = json.loads(entry.arguments)
            print(f"Calling search_faq with: {args}")
            results = search_faq(**args)
            result_json = json.dumps(results, indent=2)

            messages.append({
                "type": "function_call_output",
                "call_id": entry.call_id,
                "output": result_json,
            })
            has_tool_calls = True

        elif entry.type == 'message':
            print(entry.content[0].text)

    if not has_tool_calls:
        break
```

This is the tool-calling loop. Every agentic framework implements this loop internally.

### Simplifying with ToyAIKit

ToyAIKit is a framework we implemented in my workshops to illustrate how agents work. It's designed for learning, debugging, and exploring - not for production use.

It makes agent behavior very visible, which helps you understand what's happening under the hood. Once you've figured out the right prompts and tools, you can move to a production-ready framework with the same instructions and tools.

GitHub: [github.com/alexeygrigorev/toyaikit](https://github.com/alexeygrigorev/toyaikit)

See it in action: [agents-mcp workshop](https://github.com/alexeygrigorev/workshops/tree/main/agents-mcp) | [agent-skills workshop](https://github.com/alexeygrigorev/workshops/tree/main/agent-skills)

Install toyaikit:

```bash
uv add toyaikit
```

For a single question, use `loop()`:

Start by importing:

```python
from toyaikit.tools import Tools
from toyaikit.chat.runners import OpenAIResponsesRunner
from toyaikit.llm import OpenAIClient
```

Create the tools object and add your function:

```python
tools_obj = Tools()
tools_obj.add_tool(search_faq, search_tool)
```

ToyAIKit can also infer the schema from your function's docstring and type hints:

```python
def search_faq(query: str) -> list:
    """Search the FAQ database for relevant entries.

    Args:
        query: The search query text
    """
    results = index.search(
        query=query,
        filter_dict={'course': 'data-engineering-zoomcamp'},
        num_results=5
    )
    return results

tools_obj.add_tool(search_faq)  # Schema inferred from docstring
```

This isn't unique to ToyAIKit - other agent frameworks do the same, including PydanticAI which we cover later.

Define the system prompt:

```python
developer_prompt = """
You're a course teaching assistant for the Data Engineering Zoomcamp.
Search the FAQ database and provide sources.
""".strip()
```

Create the client with a specific model:

```python
from openai import OpenAI

client = OpenAI()
llm_client = OpenAIClient(client=client, model='gpt-4o-mini')
```

Create the runner:

```python
runner = OpenAIResponsesRunner(
    tools=tools_obj,
    developer_prompt=developer_prompt,
    llm_client=llm_client
)
```

All that loop code we wrote? Now it's just one line:

```python
result = runner.loop(prompt='How do I install Kafka?')
```

Access the final response and cost information:

```python
print(result.last_message)

result.cost  # CostInfo(input_cost=..., output_cost=..., total_cost=...)
```

### Adding More Tools

Adding more tools is straightforward. Define another function with type hints and a docstring, then add it to the tools object:

```python
def add_entry(question: str, answer: str) -> None:
    """Add a new entry to the FAQ database.
    Returns "OK" if it's successful

    Args:
        question: The question to be added to the FAQ database.
        answer: The corresponding answer to the question.
    """
    doc = {
        'question': question,
        'text': answer,
        'section': 'user added',
        'course': 'data-engineering-zoomcamp'
    }

    index.append(doc)
    return "OK"
```

Add this to our tools collection:

```python
tools_obj.add_tool(add_entry)
```

Now the agent has access to both `search` and `add_entry`. Run the agent:

```python
result = runner.loop()
```

Try prompts like:
- "How do I do well in module 1?"
- "Add this back to FAQ"

```python
result_1 = runner.loop(
    prompt='How do I do well in module 1?'
)

result_2 = runner.loop(
    prompt='add this back to FAQ',
    previous_messages=result_1.new_messages,
)

print(result_2.last_message)
```

Check that it was added to the index:

```python
index.docs[-1]
```

### Callbacks for Visibility

The loop hides all the details. What if you want to see what's happening inside - the function calls, the results, the intermediate steps? Use callbacks.

Create a callback handler and pass it to `loop()`:

```python
from toyaikit.chat.runners import DisplayingRunnerCallback
from toyaikit.chat import IPythonChatInterface

chat_interface = IPythonChatInterface()
callback = DisplayingRunnerCallback(chat_interface)

messages = runner.loop(
    prompt='How do I install Kafka?',
    callback=callback
)
```

Now each step is printed as it happens - you can see the LLM's decision to call the tool, the tool execution, and the final answer. This visibility is what makes ToyAIKit great for learning and debugging.

### Q&A Loop

For continuous conversation, use `run()`. This adds the Q&A loop - an outer loop that keeps taking user input until they say stop. Combined with the tool-calling loop inside, you get a full conversational agent:

For this to work, the runner needs to access to `chat_interface` that displays the results:

```python
runner = OpenAIResponsesRunner(
    tools=tools_obj,
    developer_prompt=instructions,
    llm_client=llm_client,
    chat_interface=chat_interface
)
```

Now run it:

```python
result = runner.run()
```

Type "stop" to end the conversation.


### Organizing Tools in a Separate File

For better organization, we can put our tools in a separate file. This makes it easier to share code between different notebooks and scripts.

See [search_tools.py](search_tools.py) - it contains a `SearchTools` class with `search` and `add_entry` methods, along with helper functions to initialize the index.

Download the file:

```bash
wget https://raw.githubusercontent.com/alexeygrigorev/now-workshop-agents/refs/heads/main/day1/search_tools.py
```

Or with curl:

```bash
curl -O https://raw.githubusercontent.com/alexeygrigorev/now-workshop-agents/refs/heads/main/day1/search_tools.py
```

Use the `SearchTools` class with ToyAIKit:

```python
from search_tools import init_tools
search_tools = init_tools()

agent_tools = Tools()
agent_tools.add_tools(search_tools)

messages = runner.run()
```


### ToyAIKit with Groq

Groq provides an OpenAI-compatible API. You can use the same `OpenAIResponsesRunner` by just changing the client:

```python
import os
from openai import OpenAI
from toyaikit.chat.runners import OpenAIResponsesRunner
from toyaikit.llm import OpenAIClient

groq_client = OpenAI(
    api_key=os.getenv('GROQ_API_KEY'),
    base_url='https://api.groq.com/openai/v1'
)

llm_client = OpenAIClient(
    client=groq_client,
    model='openai/gpt-oss-20b'
)

runner = OpenAIResponsesRunner(
    tools=tools_obj,
    developer_prompt=developer_prompt,
    llm_client=llm_client
)

messages = runner.loop(prompt='How do I install Kafka?')
```

Groq also supports the chat completions API:

```python
from toyaikit.chat.runners import OpenAIChatCompletionsRunner
from toyaikit.llm import OpenAIChatCompletionsClient

groq_llm_client = OpenAIChatCompletionsClient(
    model='openai/gpt-oss-20b',
    client=groq_client
)

groq_runner = OpenAIChatCompletionsRunner(
    tools=tools_obj,
    developer_prompt=developer_prompt,
    llm_client=groq_llm_client
)

messages = groq_runner.loop(prompt='How do I install Kafka?')
```

### ToyAIKit with Anthropic

Anthropic uses a different interface, so you need the `AnthropicMessagesRunner` and `AnthropicClient`:

```python
from anthropic import Anthropic
from toyaikit.tools import Tools
from toyaikit.chat.runners import AnthropicMessagesRunner
from toyaikit.llm import AnthropicClient

tools_obj = Tools()
tools_obj.add_tool(search_faq)

runner = AnthropicMessagesRunner(
    tools=tools_obj,
    developer_prompt=developer_prompt,
    llm_client=AnthropicClient(client=Anthropic())
)

messages = runner.loop(prompt='How do I install Kafka?')
```


Note: ToyAIKit is designed for experimentation and fast iteration in Jupyter notebooks, not for production use. For production applications, use PydanticAI.


See ToyAIKit in action:
- Course: [AI Bootcamp: From RAG to Agents](https://maven.com/alexey-grigorev/from-rag-to-agents)
- [Agents and MCP](https://github.com/alexeygrigorev/workshops/tree/main/agents-mcp) - Video: https://www.youtube.com/watch?v=W2EDdZplLcU
- [Create a Coding Agent](https://github.com/alexeygrigorev/workshops/tree/main/coding-agent) - Video: https://www.youtube.com/watch?v=Sue_mn0JCsY
- [Coding Agent with Skills](https://github.com/alexeygrigorev/workshops/tree/main/agent-skills) - Video: https://youtu.be/OhgDEZfHsvg


# Part 5: PydanticAI as Agent Orchestrator

PydanticAI is a production-ready agent framework from the Pydantic team.


Install PydanticAI:

```bash
uv add pydantic-ai
```


## Basic PydanticAI Agent

```python
from pydantic_ai import Agent

agent = Agent(
    'openai:gpt-4o-mini',
    instructions='You are a helpful assistant.',
)

result = await agent.run('tell me a joke')

print(result.output)
```

PydanticAI uses async by design. In Jupyter, top-level await works. In scripts, use `asyncio.run()`.

## Multiple Queries with Message History

For multi-turn conversations, maintain a message history and pass it to each call:

```python
message_history = []

result = await agent.run('tell me a joke', message_history=message_history)
message_history.extend(result.new_messages())

result = await agent.run('tell me another one', message_history=message_history)
message_history.extend(result.new_messages())

print(result.output)
```

## Q&A Loop

For continuous conversation, wrap the agent call in a while loop:

```python
message_history = []

while True:
    user_input = input('You: ')
    if user_input.lower() == 'stop':
        break

    result = await agent.run(user_input, message_history=message_history)
    message_history.extend(result.new_messages())
    print(f'Agent: {result.output}')
```

ToyAIKit provides `PydanticAIRunner` for interactive chat in Jupyter - see the ToyAIKit section for details.

## Adding Tools

Use the `SearchTools` class from [search_tools.py](search_tools.py):

```python
from search_tools import init_tools
from pydantic_ai import Agent

search_tools = init_tools()

agent = Agent(
    'openai:gpt-4o-mini',
    instructions='You are a course teaching assistant.',
    tools=[search_tools.search, search_tools.add_entry]
)
```

Run the agent:

```python
result = await agent.run('How do I install Kafka?')
print(result.output)
```

## Switching Providers

PydanticAI makes it easy to switch LLM providers:

```python
# Anthropic
agent = Agent(
    'anthropic:claude-3-5-sonnet-latest',
    instructions='You are a course teaching assistant.',
    tools=[search_faq]
)

# Groq
agent = Agent(
    'groq:openai/gpt-oss-20b',
    instructions='You are a course teaching assistant.',
    tools=[search_faq]
)

# Gemini
agent = Agent(
    'gemini:gemini-2.0-flash-exp',
    instructions='You are a course teaching assistant.',
    tools=[search_faq]
)
```

## Structured Outputs

```python
class FAQAnswer(BaseModel):
    """Structured FAQ response."""
    answer: str = Field(description="The answer to the user's question")
    sources: List[str] = Field(description="FAQ entries or documents used")
    confidence: float = Field(description="Confidence score from 0 to 1")
    follow_up_question: str | None = Field(default=None, description="Suggested follow-up question if relevant")

agent = Agent(
    'openai:gpt-4o-mini',
    instructions='You are a course teaching assistant.',
    tools=[search_faq],
    result_type=FAQAnswer  # Returns structured output
)
```

### Using Structured Output

```python
result = await agent.run('How do I install Kafka?')

# Access structured fields
print(result.answer)
print(result.sources)
print(result.confidence)
if result.follow_up_question:
    print(f"Follow-up: {result.follow_up_question}")
```

# Part 6: Model Context Protocol (MCP)

## What is MCP?

Anthropic describes [MCP as "like USB but for agent tools"](https://modelcontextprotocol.io/docs/getting-started/intro).

Instead of each developer implementing database access, API integrations, or file operations separately, a service provider releases an MCP server. Anyone using that service connects to the server and gets tools automatically.

MCP consists of:
- MCP Server: Exposes tools and resources via stdio or SSE
- MCP Client: Connects to servers and makes tools available to agents
- Transport: Communication layer (stdio is most common)

## Creating an MCP Server

In this section, we'll build an MCP server that exposes FAQ search tools. The server will provide two tools:
- `search`: Search the FAQ database for relevant entries
- `add_entry`: Add new entries to the FAQ database

We'll then connect to this server from an agent using ToyAIKit and PydanticAI.

Create a folder and initialize it:

```bash
mkdir faq-mcp
cd faq-mcp
uv init
```

Install FastMCP and dependencies:

```bash
uv add fastmcp minsearch requests toyaikit
```

Create `search_tools.py`:

```python
from typing import List, Dict, Any
import requests
from minsearch import AppendableIndex


class SearchTools:

    def __init__(self, index):
        self.index = index

    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search the FAQ database for entries matching the given query.

        Args:
            query (str): Search query text to look up in the course FAQ.

        Returns:
            List[Dict[str, Any]]: A list of search result entries, each containing relevant metadata.
        """
        boost = {'question': 3.0, 'section': 0.5}

        results = self.index.search(
            query=query,
            filter_dict={'course': 'data-engineering-zoomcamp'},
            boost_dict=boost,
            num_results=5,
        )

        return results

    def add_entry(self, question: str, answer: str) -> None:
        """
        Add a new entry to the FAQ database.

        Args:
            question (str): The question to be added to the FAQ database.
            answer (str): The corresponding answer to the question.
        """
        doc = {
            'question': question,
            'text': answer,
            'section': 'user added',
            'course': 'data-engineering-zoomcamp'
        }
        self.index.append(doc)


def init_index():
    docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
    docs_response = requests.get(docs_url)
    documents_raw = docs_response.json()

    documents = []

    for course in documents_raw:
        course_name = course['course']

        for doc in course['documents']:
            doc['course'] = course_name
            documents.append(doc)

    index = AppendableIndex(
        text_fields=["question", "text", "section"],
        keyword_fields=["course"]
    )

    index.fit(documents)
    return index


def init_tools():
    index = init_index()
    return SearchTools(index)
```

Create `main.py`:

```python
from fastmcp import FastMCP
from toyaikit.tools import wrap_instance_methods
from search_tools import init_tools


def init_mcp():
    mcp = FastMCP("Demo")
    agent_tools = init_tools()
    wrap_instance_methods(mcp.tool, agent_tools)
    return mcp


if __name__ == "__main__":
    mcp = init_mcp()
    mcp.run()
```

Install dependencies:

```bash
uv add minsearch requests fastmcp toyaikit
```

Run the server:

```bash
uv run python main.py
```

## Testing MCP from Command Line

The server runs over stdio (standard input/output). You can test it by sending JSON-RPC messages directly to the server. This is the same protocol that MCP clients use internally.

Start the server in one terminal:

```bash
uv run python main.py
```

The server waits for JSON-RPC messages on stdin. Each message has:
- `jsonrpc`: Protocol version (always "2.0")
- `id`: Request identifier for matching responses
- `method`: The MCP method to call
- `params`: Method-specific parameters

In another terminal, send the handshake sequence. First, initialize the connection:

```json
{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test-client", "version": "1.0"}}}
```

The server responds with its capabilities and supported protocol version.

Confirm initialization:

```json
{"jsonrpc": "2.0", "method": "notifications/initialized"}
```

This tells the server the handshake is complete. Note this is a notification, not a request, so there's no `id` field and no response.

List available tools:

```json
{"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
```

The server responds with a list of available tools (`search` and `add_entry`), including their names, descriptions, and parameter schemas.

Call the search tool:

```json
{"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "search", "arguments": {"query": "how do I install kafka?"}}}
```

The server executes the search function and returns the results formatted as a JSON-RPC response.

## Using MCP with ToyAIKit

ToyAIKit includes a simple sync MCP client for testing. This is useful for development in Jupyter where async clients can be tricky.

In Jupyter, create the MCP client and initialize it:

```python
from toyaikit.mcp import MCPClient, SubprocessMCPTransport

command = "uv run python main.py".split()
workdir = "faq-mcp"

client = MCPClient(
    transport=SubprocessMCPTransport(
        server_command=command,
        workdir=workdir
    )
)

client.full_initialize()  # Handles handshake automatically
```

MCP tools have their own schema format. To use them with OpenAI's function-calling API, we need to convert them. ToyAIKit's `MCPTools` wrapper handles this conversion automatically:

```python
from toyaikit.llm import OpenAIClient
from toyaikit.mcp import MCPTools
from toyaikit.chat import IPythonChatInterface
from toyaikit.chat.runners import OpenAIResponsesRunner

mcp_tools = MCPTools(client)

developer_prompt = """
You're a course teaching assistant.
You're given a question from a course student and your task is to answer it.

If you want to look up the answer, explain why before making the call. Use as many
keywords from the user question as possible when making first requests.

Make multiple searches. Try to expand your search by using new keywords based on the results you
get from the search.

At the end, make a clarifying question based on what you presented and ask if there are
other areas that the user wants to explore.
""".strip()

chat_interface = IPythonChatInterface()

runner = OpenAIResponsesRunner(
    tools=mcp_tools,
    developer_prompt=developer_prompt,
    chat_interface=chat_interface,
    llm_client=OpenAIClient(model='gpt-4o-mini')
)

runner.run()
```

## Using MCP with PydanticAI

PydanticAI has built-in MCP support via `MCPServerStdio`. Create a separate project folder for the client:

```bash
mkdir faq-client
cd faq-client
uv init
uv add pydantic-ai openai toyaikit
```

Create `test.py`:

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from toyaikit.chat.interface import StdOutputInterface
from toyaikit.chat.runners import PydanticAIRunner

mcp_client = MCPServerStdio(
    command="uv",
    args=["run", "python", "main.py"],
    cwd="faq-mcp"
)

developer_prompt = """
You're a course teaching assistant.
You're given a question from a course student and your task is to answer it.

If you want to look up the answer, explain why before making the call. Use as many
keywords from the user question as possible when making first requests.

Make multiple searches. Try to expand your search by using new keywords based on the results you
get from the search.

At the end, make a clarifying question based on what you presented and ask if there are
other areas that the user wants to explore.
""".strip()

agent = Agent(
    name="faq_agent",
    instructions=developer_prompt,
    toolsets=[mcp_client],
    model='gpt-4o-mini'
)

chat_interface = StdOutputInterface()
runner = PydanticAIRunner(
    chat_interface=chat_interface,
    agent=agent
)

if __name__ == "__main__":
    import asyncio
    asyncio.run(runner.run())
```

Run:

```bash
uv run python test.py
```

The agent automatically gets tools from the MCP server - no manual tool definitions needed.

## SSE Transport

MCP servers can also use HTTP (SSE) instead of stdio:

```python
# Server
if __name__ == "__main__":
    mcp.run(transport="sse")
```

The server is now available at http://localhost:8000/sse. For PydanticAI, use `MCPServerSSE`:

```python
from pydantic_ai.mcp import MCPServerSSE

mcp_server = MCPServerSSE(
    url='http://localhost:8000/sse'
)

agent = Agent(
    'openai:gpt-4o-mini',
    instructions='You are a helpful assistant.',
    mcp_servers=[mcp_server]
)
```

## Adding MCP to Cursor

Configure MCP servers in Cursor by editing `.cursor/mcp.json`:

For stdio transport, Cursor runs the MCP server as a subprocess. The `uv run --project` flag tells uv to run the command within the `faq-mcp` project directory:

```json
{
  "mcpServers": {
    "faqmcp": {
      "command": "uv",
      "args": [
        "run",
        "--project",
        "faq-mcp",
        "python",
        "main.py"
      ]
    }
  }
}
```

For SSE transport:

```json
{
  "mcpServers": {
    "faqmcp": {
      "url": "http://localhost:8000/sse"
    }
  }
}
```

After restarting Cursor, enable the server in Preferences - Cursor Settings - MCP - Integrations.

## Adding MCP to VS Code

Configure MCP servers for VS Code Copilot in `.vscode/mcp.json`:

For stdio transport:

```json
{
  "servers": {
    "faq-mcp": {
      "command": "uv",
      "args": [
        "run",
        "--project",
        "faq-mcp",
        "python",
        "main.py"
      ]
    }
  }
}
```

For SSE transport:

```json
{
  "servers": {
    "faq-mcp": {
      "type": "http",
      "url": "http://127.0.0.1:8000/sse"
    }
  }
}
```

See [VS Code MCP documentation](https://code.visualstudio.com/docs/copilot/customization/mcp-servers) for more details.

# Summary: Day 1

Today we covered:

1. LLM APIs: OpenAI, Anthropic, Gemini, Groq
2. Structured Output: Getting validated JSON from LLMs
3. RAG: Text search and vector search for retrieval-augmented generation
4. Agents: Tool-calling loop and agentic RAG
5. ToyAIKit: Learning framework for agents with OpenAI, Groq, and Anthropic
6. PydanticAI: Production-ready agent framework with tools and structured outputs
7. MCP: Creating servers, testing, and integrating with agents

## Key Takeaways

- LLMs are accessed via APIs with a consistent chat pattern
- Structured output ensures type-safe responses from LLMs
- RAG combines search with generation for knowledge-grounded answers
- The tool-calling loop lets LLMs decide when and how to use tools
- ToyAIKit is great for learning and debugging with visible internals
- PydanticAI is production-ready with async support and multiple providers
- MCP standardizes tool exposure across different systems and clients

## Tomorrow: Day 2

Building a custom coding agent with:
 - Django project template
 - File manipulation tools
 - Multi-agent coordination
 - Monitoring and guardrails
