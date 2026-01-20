# Day 1 README - Comprehensive Code Test Plan

## Overview
This document catalogs every single code block in day1/README.md that needs verification.

---

## ENVIRONMENT SETUP (Bash Commands)

| # | Code | Line | Status |
|---|------|------|--------|
| 1 | `pip install uv` | 23 | - |
| 2 | `.env` file content | 28-35 | - |
| 3 | `echo .env >> .gitignore` | 40 | - |
| 4 | `pip install dirdotenv` | 46 | - |
| 5 | `echo 'eval "$(dirdotenv hook bash)"' >> ~/.bashrc` | 47 | - |
| 6 | `uv init` | 57 | - |
| 7 | `uv add jupyter openai` | 58 | - |
| 8 | `uv add python-dotenv` | 64 | - |
| 9 | `uv run jupyter notebook` | 70 | - |

---

## PART 1: OPENAI API

### Client Setup
| # | Code | Line | Status |
|---|------|------|--------|
| 10 | `import dotenv; dotenv.load_dotenv()` | 83-86 | - |
| 11 | `from openai import OpenAI; openai_client = OpenAI()` | 90-94 | - |

### Basic Request
| # | Code | Line | Status |
|---|------|------|--------|
| 12 | `openai_client.responses.create()` with messages | 102-113 | - |

### Understanding Response
| # | Code | Line | Status |
|---|------|------|--------|
| 13 | `json.dumps(response.model_dump(), indent=2)` | 119-123 | - |

### Getting the Text
| # | Code | Line | Status |
|---|------|------|--------|
| 14 | `response.output_text` | 134 | - |
| 15 | `response.output[0].content[0].text` | 137 | - |

### Streaming Responses
| # | Code | Line | Status |
|---|------|------|--------|
| 16 | `stream=True` with `hasattr(event, 'delta')` | 144-154 | - |

### System Prompts
| # | Code | Line | Status |
|---|------|------|--------|
| 17 | System prompt with `role: developer` | 162-179 | - |

### Conversation History
| # | Code | Line | Status |
|---|------|------|--------|
| 18 | First request (name = Alice) | 189-201 | - |
| 19 | Second request WITHOUT history | 212-219 | - |
| 20 | WITH history accumulation (`messages.extend(response.output)`) | 230-253 | - |

### Structured Output with OpenAI
| # | Code | Line | Status |
|---|------|------|--------|
| 21 | Basic `responses.parse()` with `JokeResponse` BaseModel | 267-291 | - |
| 22 | `response.output[0].content[0].parsed` vs `response.output_parsed` | 295-301 | - |
| 23 | Field with `Field(description=...)` | 307-315 | - |
| 24 | `Literal["programming", "general", "dad"]` | 319-329 | - |
| 25 | Video summarization - `Chapter` and `VideoSummary` models | 359-367 | - |
| 26 | Fetch transcript with `requests.get()` | 375-381 | - |
| 27 | Generate summary with `VideoSummary` | 385-413 | - |

---

## PART 2: ALTERNATIVES TO OPENAI

### Groq API
| # | Code | Line | Status |
|---|------|------|--------|
| 28 | `uv add anthropic` | 458 | - |
| 29 | Groq client with `base_url='https://api.groq.com/openai/v1'` | 430-437 | - |
| 30 | Groq basic request (`llama-3.3-70b-versatile`) | 441-451 | - |

### Anthropic (Claude) API
| # | Code | Line | Status |
|---|------|------|--------|
| 31 | `from anthropic import Anthropic; Anthropic()` | 463-467 | - |
| 32 | `anthropic_client.messages.create()` (claude-haiku-4-5) | 473-483 | - |
| 33 | `Anthropic(api_key='sk-ant-your-key-here')` | 487-489 | - |
| 34 | `uv add boto3` | 498 | - |
| 35 | `AnthropicBedrock(aws_region='eu-west-1')` | 503-518 | - |
| 36 | `aws bedrock get-foundation-model-list` | 522-524 | - |
| 37 | z.ai client with `base_url` | 530-538 | - |
| 38 | Anthropic streaming `messages.stream()` | 545-553 | - |
| 39 | Anthropic structured output - `transform_schema()` | 559-590 | - |
| 40 | Anthropic `beta.messages.parse()` | 594-610 | - |

### Google Gemini API
| # | Code | Line | Status |
|---|------|------|--------|
| 41 | `uv add google-genai` | 620 | - |
| 42 | `from google import genai; genai.Client()` | 625-629 | - |
| 43 | Gemini `generate_content()` (gemini-2.0-flash-exp) | 633-645 | - |
| 44 | Gemini streaming `generate_content_stream()` | 649-661 | - |
| 45 | Gemini structured output with `response_mime_type='application/json'` | 667-687 | - |

---

## PART 3: RAG

### Basic RAG Concepts
| # | Code | Line | Status |
|---|------|------|--------|
| 46 | RAG flow function skeleton | 704-715 | - |

### minsearch Setup
| # | Code | Line | Status |
|---|------|------|--------|
| 47 | `uv add minsearch` | 728 | - |
| 48 | Load documents from GitHub URL | 735-750 | - |
| 49 | `from minsearch import AppendableIndex` | 758-767 | - |
| 50 | `index.search()` with `filter_dict`, `num_results`, `boost_dict` | 771-783 | - |

### RAG Pipeline
| # | Code | Line | Status |
|---|------|------|--------|
| 51 | `def search(query)` function | 791-803 | - |
| 52 | `prompt_template` with QUESTION/CONTEXT | 807-821 | - |
| 53 | `def llm(prompt)` with OpenAI | 827-844 | - |
| 54 | `def rag(query)` combining search + llm | 848-857 | - |

### Vector Search
| # | Code | Line | Status |
|---|------|------|--------|
| 55 | `uv add sentence-transformers tqdm` | 864 | - |
| 56 | `SentenceTransformer('multi-qa-distilbert-cos-v1')` | 869-885 | - |
| 57 | `from minsearch import VectorSearch` | 889-902 | - |
| 58 | `def rag_vector(query)` | 906-914 | - |
| 59 | `def hybrid_search(query)` | 920-933 | - |

---

## PART 4: AGENTS

### Tool Schema Definition
| # | Code | Line | Status |
|---|------|------|--------|
| 60 | `def search_faq(query: str) -> list` function | 956-963 | - |
| 61 | `search_tool` schema dict with type/function/parameters | 965-981 | - |

### Plain OpenAI SDK Agent
| # | Code | Line | Status |
|---|------|------|--------|
| 62 | Send with `tools=[search_tool]` | 994-1005 | - |
| 63 | `response.output` inspection | 1009-1011 | - |
| 64 | `messages.extend(response.output)` | 1015-1017 | - |
| 65 | Execute function and add result | 1021-1032 | - |
| 66 | Second API call with function results | 1036-1044 | - |

### Tool-Calling Loop
| # | Code | Line | Status |
|---|------|------|--------|
| 67 | `while True:` loop with `entry.type == 'function_call'` | 1052-1090 | - |

### ToyAIKit
| # | Code | Line | Status |
|---|------|------|--------|
| 68 | `uv add toyaikit` | 1107 | - |
| 69 | ToyAIKit imports (Tools, OpenAIResponsesRunner, OpenAIClient) | 1114-1118 | - |
| 70 | `tools_obj.add_tool(search_faq, search_tool)` | 1122-1125 | - |
| 71 | Schema inference from docstring | 1129-1144 | - |
| 72 | `OpenAIResponsesRunner` creation | 1169-1180 | - |
| 73 | `result.last_message` and `result.cost` | 1184-1188 | - |
| 74 | `def add_entry(question, answer)` | 1194-1212 | - |
| 75 | `runner.loop()` with `previous_messages` | 1231-1241 | - |
| 76 | `index.docs[-1]` | 1245-1247 | - |
| 77 | Callback with `DisplayingRunnerCallback` | 1255-1266 | - |
| 78 | `runner.run()` with `chat_interface` | 1276-1290 | - |
| 79 | Download search_tools.py with `wget` | 1302-1310 | - |
| 80 | `Tools().add_tools(search_tools)` | 1314-1322 | - |

### ToyAIKit with Groq
| # | Code | Line | Status |
|---|------|------|--------|
| 81 | Groq `OpenAI` client with base_url | 1335-1352 | - |
| 82 | `OpenAIChatCompletionsRunner` with Groq | 1356-1372 | - |

### ToyAIKit with Anthropic
| # | Code | Line | Status |
|---|------|------|--------|
| 83 | `AnthropicMessagesRunner` with `AnthropicClient()` | 1378-1394 | - |

---

## PART 5: PYDANTICAI

| # | Code | Line | Status |
|---|------|------|--------|
| 84 | `uv add pydantic-ai` | 1415 | - |
| 85 | Basic `Agent('openai:gpt-4o-mini')` with `await agent.run()` | 1421-1432 | - |
| 86 | Message history with `message_history.extend(result.new_messages())` | 1440-1450 | - |
| 87 | Q&A loop with `while True:` | 1456-1467 | - |
| 88 | PydanticAI with tools `[search_tools.search, search_tools.add_entry]` | 1475-1493 | - |
| 89 | Switching providers (Anthropic, Groq, Gemini) | 1499-1520 | - |
| 90 | Structured output with `result_type=FAQAnswer` | 1524-1551 | - |

---

## PART 6: MCP

| # | Code | Line | Status |
|---|------|------|--------|
| 91 | `mkdir faq-mcp; cd faq-mcp; uv init` | 1576-1580 | - |
| 92 | `uv add fastmcp minsearch requests toyaikit` | 1584-1586 | - |
| 93 | MCP `search_tools.py` (SearchTools class) | 1590-1665 | - |
| 94 | MCP `main.py` with `FastMCP` and `wrap_instance_methods` | 1669-1685 | - |
| 95 | `uv run python main.py` | 1695-1697 | - |
| 96 | MCP JSON-RPC examples (initialize, tools/list, tools/call) | 1717-1745 | - |
| 97 | ToyAIKit MCP client (`MCPClient`, `SubprocessMCPTransport`) | 1753-1803 | - |
| 98 | PydanticAI MCP (`MCPServerStdio`) | 1818-1860 | - |
| 99 | SSE transport (`MCPServerSSE`) | 1882-1894 | - |
| 100 | Cursor MCP config `.cursor/mcp.json` | 1902-1929 | - |
| 101 | VS Code MCP config `.vscode/mcp.json` | 1939-1967 | - |

---

## SUMMARY

- **Total code blocks to verify: 101**
- **Categories:**
  - Bash/CLI commands: 20
  - OpenAI API: 17
  - Alternative APIs (Groq, Anthropic, Gemini): 17
  - RAG/minsearch: 14
  - Agents (plain + ToyAIKit): 24
  - PydanticAI: 7
  - MCP: 10

---

## DEPENDENCIES TO VERIFY

| Package | Install Command | Section |
|---------|-----------------|---------|
| uv | pip install uv | Environment |
| jupyter | uv add jupyter | Environment |
| openai | uv add openai | Environment |
| python-dotenv | uv add python-dotenv | Environment |
| dirdotenv | pip install dirdotenv | Environment |
| anthropic | uv add anthropic | Part 2 |
| boto3 | uv add boto3 | Part 2 |
| google-genai | uv add google-genai | Part 2 |
| minsearch | uv add minsearch | Part 3 |
| sentence-transformers | uv add sentence-transformers | Part 3 |
| tqdm | uv add tqdm | Part 3 |
| toyaikit | uv add toyaikit | Part 4 |
| pydantic-ai | uv add pydantic-ai | Part 5 |
| fastmcp | uv add fastmcp | Part 6 |
| requests | uv add requests | Part 6 |

---

## TESTING STRATEGY

1. **Bash commands**: Verify syntax and actual execution
2. **Python imports**: Verify packages install and import correctly
3. **API calls**: Verify with mock or real API keys
4. **Data flow**: Verify end-to-end examples work
5. **Type safety**: Verify Pydantic models validate correctly
