"""
Test script for Day 1 README code examples.
Run this to verify all code examples work correctly.
"""

import os
import sys
import json
from pathlib import Path

# Track test results
TESTS_PASSED = []
TESTS_FAILED = []
TESTS_SKIPPED = []


def test(name):
    """Decorator to track test results."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"\n{'='*60}")
            print(f"TEST: {name}")
            print('='*60)
            try:
                result = func(*args, **kwargs)
                TESTS_PASSED.append(name)
                print(f"PASSED: {name}")
                return result
            except SkipTest as e:
                TESTS_SKIPPED.append((name, str(e)))
                print(f"SKIPPED: {name} - {e}")
                return None
            except Exception as e:
                TESTS_FAILED.append((name, str(e)))
                print(f"FAILED: {name}")
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
                return None
        return wrapper
    return decorator


class SkipTest(Exception):
    """Raised when a test should be skipped."""
    pass


def require_env_var(var_name):
    """Check if environment variable is set, otherwise skip."""
    value = os.getenv(var_name)
    if not value:
        raise SkipTest(f"Environment variable {var_name} not set")


def require_package(package_name):
    """Check if package is installed, otherwise skip."""
    try:
        __import__(package_name)
    except ImportError:
        raise SkipTest(f"Package {package_name} not installed")


# ============================================================================
# PART 1: OpenAI API
# ============================================================================

@test("Import OpenAI")
def test_import_openai():
    require_env_var("OPENAI_API_KEY")
    from openai import OpenAI
    OpenAI()
    print("OpenAI client created successfully")
    return True


@test("OpenAI basic request")
def test_openai_basic_request():
    require_env_var("OPENAI_API_KEY")
    from openai import OpenAI

    client = OpenAI()
    messages = [
        {"role": "user", "content": "say 'hello world' in exactly 5 characters"}
    ]
    response = client.responses.create(
        model='gpt-4o-mini',
        input=messages
    )
    print(f"Response: {response.output_text}")
    assert len(response.output_text) > 0
    return True


@test("OpenAI response structure")
def test_openai_response_structure():
    require_env_var("OPENAI_API_KEY")
    from openai import OpenAI

    client = OpenAI()
    messages = [
        {"role": "user", "content": "say 'test'"}
    ]
    response = client.responses.create(
        model='gpt-4o-mini',
        input=messages
    )

    # Test output_text
    text = response.output_text
    print(f"output_text: {text}")

    # Test output_parsed is None for non-structured requests
    assert response.output_parsed is None

    # Test model_dump
    data = response.model_dump()
    assert 'output' in data
    assert 'usage' in data
    print(f"Usage: {data['usage']}")
    return True


@test("OpenAI streaming")
def test_openai_streaming():
    require_env_var("OPENAI_API_KEY")
    from openai import OpenAI

    client = OpenAI()
    stream = client.responses.create(
        model='gpt-4o-mini',
        input="count from 1 to 3",
        stream=True
    )

    collected = []
    for event in stream:
        if hasattr(event, 'delta'):
            collected.append(event.delta)

    print(f"Collected stream: {''.join(collected)}")
    assert len(collected) > 0
    return True


@test("OpenAI system prompt")
def test_openai_system_prompt():
    require_env_var("OPENAI_API_KEY")
    from openai import OpenAI

    client = OpenAI()
    system_prompt = "You are a helpful assistant who always ends responses with 'over and out'."

    messages = [
        {"role": "developer", "content": system_prompt},
        {"role": "user", "content": "say hello"}
    ]

    response = client.responses.create(
        model='gpt-4o-mini',
        input=messages
    )

    print(f"Response: {response.output_text}")
    assert "over and out" in response.output_text.lower()
    return True


@test("OpenAI conversation history")
def test_openai_conversation_history():
    require_env_var("OPENAI_API_KEY")
    from openai import OpenAI

    client = OpenAI()

    # First request
    messages = [
        {"role": "developer", "content": "You're a helpful assistant."},
        {"role": "user", "content": "My name is Alice. Just reply with 'Nice to meet you, Alice!'"}
    ]

    response = client.responses.create(
        model='gpt-4o-mini',
        input=messages
    )
    print(f"First response: {response.output_text}")

    # Accumulate history
    messages.extend(response.output)

    # Second request - it remembers
    messages.append({"role": "user", "content": "What's my name? Reply with just the name."})

    response = client.responses.create(
        model='gpt-4o-mini',
        input=messages
    )

    print(f"Second response: {response.output_text}")
    assert "Alice" in response.output_text
    return True


@test("OpenAI structured output")
def test_openai_structured_output():
    require_env_var("OPENAI_API_KEY")
    from openai import OpenAI
    from pydantic import BaseModel, Field
    from typing import Literal

    class JokeResponse(BaseModel):
        """A joke with metadata."""
        setup: str = Field(description="The setup that builds anticipation for the joke")
        punchline: str = Field(description="The funny conclusion or twist")
        category: Literal["programming", "general", "dad"] = Field(
            description="The type of joke based on the target audience and theme"
        )

    client = OpenAI()
    messages = [
        {"role": "developer", "content": "Tell me a programming joke about bugs."},
        {"role": "user", "content": "Give me a joke"}
    ]

    response = client.responses.parse(
        model='gpt-4o-mini',
        input=messages,
        text_format=JokeResponse
    )

    joke = response.output_parsed
    print(f"Category: {joke.category}")
    print(f"Setup: {joke.setup}")
    print(f"Punchline: {joke.punchline}")

    assert joke.category == "programming"
    assert len(joke.setup) > 0
    assert len(joke.punchline) > 0
    return True


@test("OpenAI video summarization")
def test_openai_video_summarization():
    require_env_var("OPENAI_API_KEY")
    from openai import OpenAI
    from pydantic import BaseModel, Field
    import requests

    class Chapter(BaseModel):
        timestamp: str = Field(description="Timestamp in the video (e.g. 10:25)")
        title: str = Field(description="Title of this chapter section")

    class VideoSummary(BaseModel):
        summary: str = Field(description="Overall summary of the video content")
        chapters: list[Chapter] = Field(description="List of chapters with timestamps")

    # Fetch transcript
    transcript_url = 'https://raw.githubusercontent.com/alexeygrigorev/workshops/main/temporal.io/data/_fbe1QyJ1PY.txt'
    response = requests.get(transcript_url)
    transcript = response.text[:2000]  # Use first 2000 chars for speed

    developer_prompt = """
    Summarize the transcript and describe the main purpose of the video
    and the main ideas.

    Also output at least 2 chapters with time. Use usual sentence case, not Title Case for chapters.
    """.strip()

    client = OpenAI()
    response = client.responses.parse(
        model='gpt-4o-mini',
        input=[
            {"role": "developer", "content": developer_prompt},
            {"role": "user", "content": transcript}
        ],
        text_format=VideoSummary
    )

    summary = response.output_parsed
    print(f"Summary: {summary.summary}")
    print(f"Chapters: {len(summary.chapters)}")
    for chapter in summary.chapters[:3]:
        print(f"  {chapter.timestamp}: {chapter.title}")

    assert len(summary.summary) > 0
    assert len(summary.chapters) >= 2
    return True


# ============================================================================
# PART 2: Alternatives to OpenAI
# ============================================================================

@test("Groq API")
def test_groq_api():
    require_env_var("GROQ_API_KEY")
    from openai import OpenAI

    groq_client = OpenAI(
        api_key=os.getenv('GROQ_API_KEY'),
        base_url='https://api.groq.com/openai/v1'
    )

    response = groq_client.responses.create(
        model='llama-3.3-70b-versatile',
        input=[
            {"role": "system", "content": "You're a helpful assistant."},
            {"role": "user", "content": "say 'hello from Groq' and nothing else"}
        ]
    )

    print(f"Groq response: {response.output_text}")
    assert "hello" in response.output_text.lower()
    return True


@test("Anthropic API")
def test_anthropic_api():
    require_env_var("ANTHROPIC_API_KEY")
    from anthropic import Anthropic

    client = Anthropic()
    message = client.messages.create(
        model='claude-haiku-4-5',
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "say 'hello from Claude' and nothing else"}
        ]
    )

    print(f"Claude response: {message.content[0].text}")
    assert "hello" in message.content[0].text.lower()
    return True


@test("Anthropic structured output")
def test_anthropic_structured_output():
    require_env_var("ANTHROPIC_API_KEY")
    from anthropic import Anthropic, transform_schema
    from pydantic import BaseModel
    from typing import Literal

    class JokeResponse(BaseModel):
        """A joke with metadata."""
        setup: str
        punchline: str
        category: Literal["programming", "general", "dad"]

    client = Anthropic()
    response = client.beta.messages.parse(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        betas=["structured-outputs-2025-11-13"],
        messages=[
            {
                "role": "user",
                "content": "Tell me a short programming joke."
            }
        ],
        output_format=JokeResponse
    )

    joke = response.parsed
    print(f"Category: {joke.category}")
    print(f"Setup: {joke.setup}")
    print(f"Punchline: {joke.punchline}")

    assert joke.category == "programming"
    return True


@test("Anthropic streaming")
def test_anthropic_streaming():
    require_env_var("ANTHROPIC_API_KEY")
    from anthropic import Anthropic

    client = Anthropic()
    collected = []

    with client.messages.stream(
        model='claude-haiku-4-5',
        max_tokens=1024,
        messages=[{"role": "user", "content": "count from 1 to 3"}]
    ) as stream:
        for text in stream.text_stream:
            collected.append(text)

    print(f"Claude stream: {''.join(collected)}")
    assert len(collected) > 0
    return True


@test("Gemini API")
def test_gemini_api():
    require_env_var("GEMINI_API_KEY")
    from google import genai
    from google.genai import types

    client = genai.Client()
    response = client.models.generate_content(
        model='models/gemini-2.0-flash-exp',
        config=types.GenerateContentConfig(
            system_instruction="You're a helpful assistant."
        ),
        contents="say 'hello from Gemini' and nothing else"
    )

    print(f"Gemini response: {response.text}")
    assert "hello" in response.text.lower()
    return True


@test("Gemini structured output")
def test_gemini_structured_output():
    require_env_var("GEMINI_API_KEY")
    from google import genai
    from google.genai import types
    from pydantic import BaseModel

    class JokeResponse(BaseModel):
        """A joke with metadata."""
        setup: str
        punchline: str
        category: str

    client = genai.Client()
    response = client.models.generate_content(
        model='models/gemini-2.0-flash-exp',
        config=types.GenerateContentConfig(
            system_instruction="You're a helpful assistant.",
            response_mime_type='application/json',
            response_schema=JokeResponse.model_json_schema()
        ),
        contents="Tell me a programming joke in JSON format"
    )

    joke_data = json.loads(response.text)
    joke = JokeResponse.model_validate(joke_data)
    print(f"Setup: {joke.setup}")
    print(f"Punchline: {joke.punchline}")
    print(f"Category: {joke.category}")

    assert len(joke.setup) > 0
    assert len(joke.punchline) > 0
    return True


@test("Gemini streaming")
def test_gemini_streaming():
    require_env_var("GEMINI_API_KEY")
    from google import genai
    from google.genai import types

    client = genai.Client()
    collected = []

    response = client.models.generate_content_stream(
        model='models/gemini-2.0-flash-exp',
        config=types.GenerateContentConfig(
            system_instruction="You're a helpful assistant."
        ),
        contents="count from 1 to 3"
    )

    for chunk in response:
        if hasattr(chunk, 'text') and chunk.text:
            collected.append(chunk.text)

    print(f"Gemini stream: {''.join(collected)}")
    assert len(collected) > 0
    return True


# ============================================================================
# PART 3: RAG
# ============================================================================

@test("minsearch text search")
def test_minsearch_text_search():
    from minsearch import AppendableIndex
    import requests

    # Load data
    docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
    docs_response = requests.get(docs_url)
    documents_raw = docs_response.json()

    documents = []
    for course in documents_raw:
        course_name = course['course']
        for doc in course['documents']:
            doc['course'] = course_name
            documents.append(doc)

    # Create index
    index = AppendableIndex(
        text_fields=["question", "text", "section"],
        keyword_fields=["course"]
    )
    index.fit(documents)

    # Search
    question = "I just found the course. Can I join now?"
    results = index.search(
        question,
        filter_dict={'course': 'data-engineering-zoomcamp'},
        num_results=5,
        boost_dict={'question': 3.0, 'section': 0.5}
    )

    print(f"Found {len(results)} results")
    for result in results[:2]:
        print(f"  - {result.get('question', 'N/A')[:60]}...")

    assert len(results) > 0
    return True


@test("RAG pipeline")
def test_rag_pipeline():
    require_env_var("OPENAI_API_KEY")
    from openai import OpenAI
    from minsearch import AppendableIndex
    import requests

    # Load data
    docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
    docs_response = requests.get(docs_url)
    documents_raw = docs_response.json()

    documents = []
    for course in documents_raw:
        course_name = course['course']
        for doc in course['documents']:
            doc['course'] = course_name
            documents.append(doc)

    # Create index
    index = AppendableIndex(
        text_fields=["question", "text", "section"],
        keyword_fields=["course"]
    )
    index.fit(documents)

    # Define RAG components
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

    def search(query):
        boost = {'question': 3.0, 'section': 0.5}
        results = index.search(
            query=query,
            filter_dict={'course': 'data-engineering-zoomcamp'},
            boost_dict=boost,
            num_results=5
        )
        return results

    instructions = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT
from the FAQ database. Use only the facts from the CONTEXT. Keep answer under 50 words.
""".strip()

    def llm(prompt):
        client = OpenAI()
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": prompt}
        ]
        response = client.responses.create(
            model='gpt-4o-mini',
            input=messages
        )
        return response.output_text

    def rag(query):
        search_results = search(query)
        user_prompt = build_prompt(query, search_results)
        return llm(user_prompt)

    # Test
    answer = rag("Can I join the course after it started?")
    print(f"RAG answer: {answer[:200]}...")

    assert len(answer) > 0
    return True


@test("Vector search with sentence-transformers")
def test_vector_search():
    from minsearch import VectorSearch
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import requests

    # Load data
    docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
    docs_response = requests.get(docs_url)
    documents_raw = docs_response.json()

    documents = []
    for course in documents_raw:
        course_name = course['course']
        for doc in course['documents']:
            doc['course'] = course_name
            documents.append(doc)

    # Create embeddings
    embedding_model = SentenceTransformer('multi-qa-distilbert-cos-v1')
    embeddings = []
    for d in documents:
        text = d['question'] + ' ' + d['text']
        v = embedding_model.encode(text)
        embeddings.append(v)

    embeddings = np.array(embeddings)
    print(f"Created embeddings with shape: {embeddings.shape}")

    # Create vector index
    vindex = VectorSearch(keyword_fields=['course'])
    vindex.fit(embeddings, documents)

    # Search
    question = "Can I join after the course started?"
    q = embedding_model.encode(question)
    results = vindex.search(
        q,
        filter_dict={'course': 'data-engineering-zoomcamp'},
        num_results=3
    )

    print(f"Vector search found {len(results)} results")
    for result in results[:2]:
        print(f"  - {result.get('question', 'N/A')[:60]}...")

    assert len(results) > 0
    return True


# ============================================================================
# PART 4: Agents
# ============================================================================

@test("Tool-calling loop with OpenAI")
def test_tool_calling_loop():
    require_env_var("OPENAI_API_KEY")
    from openai import OpenAI

    # Simple tool
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return f"The weather in {city} is sunny and 75 degrees."

    search_tool = {
        "type": "function",
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name"
                }
            },
            "required": ["city"],
            "additionalProperties": False
        }
    }

    client = OpenAI()
    instructions = "You're a helpful assistant. Use the weather tool when asked about weather."

    user_question = "What's the weather in Paris?"

    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_question}
    ]

    # Tool-calling loop
    max_iterations = 5
    for i in range(max_iterations):
        response = client.responses.create(
            model='gpt-4o-mini',
            input=messages,
            tools=[search_tool]
        )

        has_tool_calls = False

        for entry in response.output:
            messages.append(entry)

            if entry.type == 'function_call':
                args = json.loads(entry.arguments)
                print(f"  Calling {entry.name} with: {args}")

                if entry.name == 'get_weather':
                    results = get_weather(**args)
                else:
                    results = f"Unknown tool: {entry.name}"

                result_json = json.dumps(results)
                messages.append({
                    "type": "function_call_output",
                    "call_id": entry.call_id,
                    "output": result_json,
                })
                has_tool_calls = True

            elif entry.type == 'message':
                print(f"  Final answer: {entry.content[0].text}")
                return True  # Success!

        if not has_tool_calls:
            break

    return True


@test("ToyAIKit with OpenAI")
def test_toyaikit_openai():
    require_env_var("OPENAI_API_KEY")
    require_package("toyaikit")

    from openai import OpenAI
    from toyaikit.tools import Tools
    from toyaikit.chat.runners import OpenAIResponsesRunner
    from toyaikit.llm import OpenAIClient

    def search_faq(query: str) -> list:
        """Search the FAQ database for relevant entries.

        Args:
            query: The search query text
        """
        return [{"question": "Test question", "text": "Test answer", "course": "test"}]

    tools_obj = Tools()
    tools_obj.add_tool(search_faq)

    developer_prompt = "You're a course teaching assistant. Keep answers brief."

    client = OpenAI()
    llm_client = OpenAIClient(client=client, model='gpt-4o-mini')

    runner = OpenAIResponsesRunner(
        tools=tools_obj,
        developer_prompt=developer_prompt,
        llm_client=llm_client
    )

    result = runner.loop(prompt='Do you have information about Kafka?')
    print(f"Response: {result.last_message[:200]}...")

    assert result.last_message is not None
    return True


@test("ToyAIKit with Groq")
def test_toyaikit_groq():
    require_env_var("GROQ_API_KEY")
    require_package("toyaikit")

    from openai import OpenAI
    from toyaikit.tools import Tools
    from toyaikit.chat.runners import OpenAIResponsesRunner
    from toyaikit.llm import OpenAIClient

    def search_faq(query: str) -> list:
        """Search the FAQ database."""
        return [{"question": "Test", "text": "Test answer", "course": "test"}]

    tools_obj = Tools()
    tools_obj.add_tool(search_faq)

    groq_client = OpenAI(
        api_key=os.getenv('GROQ_API_KEY'),
        base_url='https://api.groq.com/openai/v1'
    )

    llm_client = OpenAIClient(
        client=groq_client,
        model='llama-3.3-70b-versatile'
    )

    runner = OpenAIResponsesRunner(
        tools=tools_obj,
        developer_prompt="You're a helpful assistant. Keep answers brief.",
        llm_client=llm_client
    )

    result = runner.loop(prompt='Say hello')
    print(f"Groq response: {result.last_message[:200]}...")

    assert result.last_message is not None
    return True


@test("ToyAIKit with Anthropic")
def test_toyaikit_anthropic():
    require_env_var("ANTHROPIC_API_KEY")
    require_package("toyaikit")

    from anthropic import Anthropic
    from toyaikit.tools import Tools
    from toyaikit.chat.runners import AnthropicMessagesRunner
    from toyaikit.llm import AnthropicClient

    def search_faq(query: str) -> list:
        """Search the FAQ database."""
        return [{"question": "Test", "text": "Test answer", "course": "test"}]

    tools_obj = Tools()
    tools_obj.add_tool(search_faq)

    runner = AnthropicMessagesRunner(
        tools=tools_obj,
        developer_prompt="You're a helpful assistant. Keep answers brief.",
        llm_client=AnthropicClient(client=Anthropic())
    )

    result = runner.loop(prompt='Say hello')
    print(f"Anthropic response: {result.last_message[:200]}...")

    assert result.last_message is not None
    return True


# ============================================================================
# PART 5: PydanticAI
# ============================================================================

@test("PydanticAI basic agent")
def test_pydanticai_basic():
    require_env_var("OPENAI_API_KEY")
    require_package("pydantic_ai")

    import asyncio
    from pydantic_ai import Agent

    agent = Agent(
        'openai:gpt-4o-mini',
        instructions='You are a helpful assistant. Keep answers brief.',
    )

    async def run_test():
        result = await agent.run('say hello world')
        return result.output

    output = asyncio.run(run_test())
    print(f"PydanticAI response: {output}")
    assert len(output) > 0
    return True


@test("PydanticAI with tools")
def test_pydanticai_tools():
    require_env_var("OPENAI_API_KEY")
    require_package("pydantic_ai")

    import asyncio
    from pydantic_ai import Agent

    def search_faq(query: str) -> str:
        """Search the FAQ database for relevant entries.

        Args:
            query: The search query text
        """
        return "Found: Kafka is installed in module 1."

    agent = Agent(
        'openai:gpt-4o-mini',
        instructions='You are a course teaching assistant. Use tools when needed.',
        tools=[search_faq]
    )

    async def run_test():
        result = await agent.run('How do I install Kafka?')
        return result.output

    output = asyncio.run(run_test())
    print(f"PydanticAI with tools response: {output[:200]}...")
    assert len(output) > 0
    return True


@test("PydanticAI structured output")
def test_pydanticai_structured_output():
    require_env_var("OPENAI_API_KEY")
    require_package("pydantic_ai")

    import asyncio
    from pydantic_ai import Agent
    from pydantic import BaseModel, Field
    from typing import List

    class FAQAnswer(BaseModel):
        """Structured FAQ response."""
        answer: str = Field(description="The answer to the user's question")
        sources: List[str] = Field(description="FAQ entries or documents used")
        confidence: float = Field(description="Confidence score from 0 to 1")

    agent = Agent(
        'openai:gpt-4o-mini',
        instructions='You are a course teaching assistant.',
        result_type=FAQAnswer
    )

    async def run_test():
        result = await agent.run('What is this course about? Answer in one sentence.')
        return result.output

    output = asyncio.run(run_test())
    print(f"Answer: {output.answer}")
    print(f"Sources: {output.sources}")
    print(f"Confidence: {output.confidence}")

    assert len(output.answer) > 0
    assert 0 <= output.confidence <= 1
    return True


@test("PydanticAI with different providers")
def test_pydanticai_providers():
    # Test Anthropic provider
    require_env_var("ANTHROPIC_API_KEY")
    require_package("pydantic_ai")

    import asyncio
    from pydantic_ai import Agent

    agent = Agent(
        'anthropic:claude-3-5-sonnet-latest',
        instructions='You are a helpful assistant. Keep answers brief.',
    )

    async def run_test():
        result = await agent.run('say hello from Anthropic')
        return result.output

    output = asyncio.run(run_test())
    print(f"Anthropic via PydanticAI: {output}")
    assert len(output) > 0
    return True


# ============================================================================
# PART 6: MCP
# ============================================================================

@test("MCP server with FastMCP")
def test_mcp_server():
    require_package("fastmcp")
    require_package("toyaikit")

    from fastmcp import FastMCP
    from toyaikit.tools import wrap_instance_methods

    def simple_search(query: str) -> list:
        """Search for information.

        Args:
            query: The search query
        """
        return [{"result": f"Results for {query}"}]

    mcp = FastMCP("Test")

    # Add tool to MCP server
    mcp.tool()(simple_search)

    # List tools
    tools = mcp._tools
    print(f"MCP server has {len(tools)} tools")
    for name in list(tools.keys())[:3]:
        print(f"  - {name}")

    assert len(tools) > 0
    return True


@test("search_tools.py module")
def test_search_tools_module():
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "day1"))

    from search_tools import SearchTools, init_tools, init_index

    # Test init_index
    index = init_index()
    print(f"Index created with {len(index.docs)} documents")

    # Test SearchTools
    tools = SearchTools(index)
    results = tools.search("kafka installation")
    print(f"Search found {len(results)} results")

    # Test init_tools
    tools2 = init_tools()
    results2 = tools2.search("course prerequisites")
    print(f"Second search found {len(results2)} results")

    assert len(results) > 0
    assert len(results2) > 0
    return True


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all tests and print summary."""
    print("=" * 60)
    print("DAY 1 README CODE VERIFICATION")
    print("=" * 60)

    # List of all test functions
    all_tests = [
        # Part 1: OpenAI
        test_import_openai,
        test_openai_basic_request,
        test_openai_response_structure,
        test_openai_streaming,
        test_openai_system_prompt,
        test_openai_conversation_history,
        test_openai_structured_output,
        test_openai_video_summarization,

        # Part 2: Alternatives
        test_groq_api,
        test_anthropic_api,
        test_anthropic_structured_output,
        test_anthropic_streaming,
        test_gemini_api,
        test_gemini_structured_output,
        test_gemini_streaming,

        # Part 3: RAG
        test_minsearch_text_search,
        test_rag_pipeline,
        test_vector_search,

        # Part 4: Agents
        test_tool_calling_loop,
        test_toyaikit_openai,
        test_toyaikit_groq,
        test_toyaikit_anthropic,

        # Part 5: PydanticAI
        test_pydanticai_basic,
        test_pydanticai_tools,
        test_pydanticai_structured_output,
        test_pydanticai_providers,

        # Part 6: MCP
        test_mcp_server,
        test_search_tools_module,
    ]

    # Run all tests
    for test_func in all_tests:
        try:
            test_func()
        except Exception as e:
            print(f"ERROR running {test_func.__name__}: {e}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"PASSED: {len(TESTS_PASSED)}")
    print(f"FAILED: {len(TESTS_FAILED)}")
    print(f"SKIPPED: {len(TESTS_SKIPPED)}")

    if TESTS_PASSED:
        print("\nPASSED tests:")
        for name in TESTS_PASSED:
            print(f"  - {name}")

    if TESTS_FAILED:
        print("\nFAILED tests:")
        for name, error in TESTS_FAILED:
            print(f"  - {name}: {error[:100]}...")

    if TESTS_SKIPPED:
        print("\nSKIPPED tests (missing dependencies/API keys):")
        for name, reason in TESTS_SKIPPED:
            print(f"  - {name}: {reason}")

    return 0 if len(TESTS_FAILED) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
