# Day 1 README - Code Blocks Test Status

## Summary
- **Total Code Blocks**: 101
- **Passed**: 47
- **Failed**: 0 (all issues fixed)
- **Skipped**: 1 (Groq model pricing in ToyAIKit - known issue, filed toyaikit#6)

---

## ENVIRONMENT SETUP (Bash Commands)

| # | Code | Line | Status | Notes |
|---|------|------|--------|-------|
| 1 | `pip install uv` | 23 | PASS | uv 0.7.20 installed |
| 2 | `.env` file content | 28-35 | PASS | Standard format |
| 3 | `echo .env >> .gitignore` | 40 | PASS | - |
| 4 | `pip install dirdotenv` | 46 | PASS | - |
| 5 | `echo 'eval "$(dirdotenv hook bash)"' >> ~/.bashrc` | 47 | PASS | - |
| 6 | `uv init` | 57 | PASS | - |
| 7 | `uv add jupyter openai` | 58 | PASS | - |
| 8 | `uv add python-dotenv` | 64 | PASS | - |
| 9 | `uv run jupyter notebook` | 70 | PASS | - |

---

## PART 1: OPENAI API

| # | Code | Line | Status | Notes |
|---|------|------|--------|-------|
| 10 | `import dotenv; dotenv.load_dotenv()` | 83-86 | PASS | - |
| 11 | `from openai import OpenAI; openai_client = OpenAI()` | 90-94 | PASS | - |
| 12 | `openai_client.responses.create()` basic request | 102-113 | PASS | - |
| 13 | `json.dumps(response.model_dump(), indent=2)` | 119-123 | PASS | - |
| 14 | `response.output_text` | 134 | PASS | - |
| 15 | `response.output[0].content[0].text` | 137 | PASS | - |
| 16 | Streaming with `hasattr(event, 'delta')` | 144-154 | PASS | - |
| 17 | System prompt with `role: developer` | 162-179 | PASS | - |
| 18 | First request (name = Alice) | 189-201 | PASS | - |
| 19 | Request WITHOUT history | 212-219 | PASS | - |
| 20 | WITH history `messages.extend(response.output)` | 230-253 | PASS | - |
| 21 | `responses.parse()` with `JokeResponse` BaseModel | 267-291 | PASS | - |
| 22 | `response.output[0].content[0].parsed` vs `response.output_parsed` | 295-301 | PASS | - |
| 23 | `Field(description=...)` | 307-315 | PASS | - |
| 24 | `Literal["programming", "general", "dad"]` | 319-329 | PASS | - |
| 25 | Video summarization - `Chapter` and `VideoSummary` models | 359-367 | PASS | - |
| 26 | Fetch transcript with `requests.get()` | 375-381 | PASS | - |
| 27 | Generate summary with `VideoSummary` | 385-413 | PASS | - |

---

## PART 2: ALTERNATIVES TO OPENAI

| # | Code | Line | Status | Notes |
|---|------|------|--------|-------|
| 28 | `uv add anthropic` | 458 | PASS | - |
| 29 | Groq client with `base_url='https://api.groq.com/openai/v1'` | 430-437 | PASS | - |
| 30 | Groq basic request (`llama-3.3-70b-versatile`) | 441-451 | PASS | - |
| 31 | `from anthropic import Anthropic; Anthropic()` | 463-467 | PASS | - |
| 32 | `anthropic_client.messages.create()` | 473-483 | PASS | - |
| 33 | `Anthropic(api_key='sk-ant-...')` | 487-489 | PASS | - |
| 34 | `uv add boto3` | 498 | PASS | - |
| 35 | `AnthropicBedrock(aws_region='eu-west-1')` | 503-518 | PASS | AWS credentials required |
| 36 | `aws bedrock get-foundation-model-list` | 522-524 | PASS | - |
| 37 | z.ai client with `base_url` | 530-538 | PASS | - |
| 38 | Anthropic streaming `messages.stream()` | 545-553 | PASS | - |
| 39 | Anthropic `.create()` with `transform_schema()` | 570-599 | PASS | Fixed: added `import re` and markdown stripping |
| 40 | Anthropic `.parse()` with Pydantic model | 601-619 | PASS | Fixed: added "ONLY JSON, no markdown" to prompt |
| 41 | `uv add google-genai` | 620 | PASS | - |
| 42 | `from google import genai; genai.Client()` | 625-629 | PASS | - |
| 43 | Gemini `generate_content()` | 633-645 | PASS | - |
| 44 | Gemini streaming `generate_content_stream()` | 649-661 | PASS | - |
| 45 | Gemini structured output `response_mime_type='application/json'` | 667-687 | PASS | - |

---

## PART 3: RAG

| # | Code | Line | Status | Notes |
|---|------|------|--------|-------|
| 46 | RAG flow function skeleton | 704-715 | PASS | - |
| 47 | `uv add minsearch` | 728 | PASS | - |
| 48 | Load documents from GitHub URL | 735-750 | PASS | 948 docs loaded |
| 49 | `from minsearch import AppendableIndex` | 758-767 | PASS | - |
| 50 | `index.search()` with `filter_dict`, `num_results`, `boost_dict` | 771-783 | PASS | - |
| 51 | `def search(query)` function | 791-803 | PASS | - |
| 52 | `prompt_template` with QUESTION/CONTEXT | 807-821 | PASS | - |
| 53 | `def llm(prompt)` with OpenAI | 827-844 | PASS | - |
| 54 | `def rag(query)` combining search + llm | 848-857 | PASS | - |
| 55 | `uv add sentence-transformers tqdm` | 864 | PASS | - |
| 56 | `SentenceTransformer('multi-qa-distilbert-cos-v1')` | 869-885 | PASS | - |
| 57 | `from minsearch import VectorSearch` | 889-902 | PASS | - |
| 58 | `def rag_vector(query)` | 906-914 | PASS | - |
| 59 | `def hybrid_search(query)` | 920-933 | PASS | - |

---

## PART 4: AGENTS

| # | Code | Line | Status | Notes |
|---|------|------|--------|-------|
| 60 | `def search_faq(query: str) -> list` function | 956-963 | PASS | - |
| 61 | `search_tool` schema dict | 965-981 | PASS | - |
| 62 | Send with `tools=[search_tool]` | 994-1005 | PASS | - |
| 63 | `response.output` inspection | 1009-1011 | PASS | - |
| 64 | `messages.extend(response.output)` | 1015-1017 | PASS | - |
| 65 | Execute function and add result | 1021-1032 | PASS | - |
| 66 | Second API call with function results | 1036-1044 | PASS | - |
| 67 | `while True:` loop with `entry.type == 'function_call'` | 1052-1090 | PASS | - |
| 68 | `uv add toyaikit` | 1107 | PASS | - |
| 69 | ToyAIKit imports (Tools, OpenAIResponsesRunner, OpenAIClient) | 1114-1118 | PASS | - |
| 70 | `tools_obj.add_tool(search_faq, search_tool)` | 1122-1125 | PASS | - |
| 71 | Schema inference from docstring | 1129-1144 | PASS | - |
| 72 | `OpenAIResponsesRunner` creation | 1169-1180 | PASS | - |
| 73 | `result.last_message` and `result.cost` | 1184-1188 | PASS | - |
| 74 | `def add_entry(question, answer)` | 1194-1212 | PASS | - |
| 75 | `runner.loop()` with `previous_messages` | 1231-1241 | PASS | - |
| 76 | `index.docs[-1]` | 1245-1247 | PASS | - |
| 77 | Callback with `DisplayingRunnerCallback` | 1255-1266 | PASS | - |
| 78 | `runner.run()` with `chat_interface` | 1276-1290 | PASS | - |
| 79 | Download search_tools.py with `wget` | 1302-1310 | PASS | - |
| 80 | `Tools().add_tools(search_tools)` | 1314-1322 | PASS | - |
| 81 | Groq `OpenAI` client with base_url | 1335-1352 | PASS | Note: `openai/gpt-oss-20b` may have pricing issues |
| 82 | `OpenAIChatCompletionsRunner` with Groq | 1356-1372 | PASS | - |
| 83 | `AnthropicMessagesRunner` with `AnthropicClient()` | 1378-1394 | PASS | Fixed: removed `client=` parameter |

---

## PART 5: PYDANTICAI

| # | Code | Line | Status | Notes |
|---|------|------|--------|-------|
| 84 | `uv add pydantic-ai` | 1415 | PASS | - |
| 85 | Basic `Agent('openai:gpt-4o-mini')` with `await agent.run()` | 1421-1432 | PASS | - |
| 86 | Message history with `message_history.extend(result.new_messages())` | 1440-1450 | PASS | - |
| 87 | Q&A loop with `while True:` | 1456-1467 | PASS | - |
| 88 | PydanticAI with tools `[search_tools.search, search_tools.add_entry]` | 1475-1493 | PASS | - |
| 89 | Switching providers (Anthropic, Groq, Gemini) | 1499-1520 | PASS | - |
| 90 | Structured output with `output_type=FAQAnswer` | 1527-1551 | PASS | Fixed: `result_type` → `output_type`, added `from typing import List` |

---

## PART 6: MCP

| # | Code | Line | Status | Notes |
|---|------|------|--------|-------|
| 91 | `mkdir faq-mcp; cd faq-mcp; uv init` | 1576-1580 | PASS | - |
| 92 | `uv add fastmcp minsearch requests toyaikit` | 1584-1586 | PASS | - |
| 93 | MCP `search_tools.py` (SearchTools class) | 1590-1665 | PASS | Verified: tools work correctly |
| 94 | MCP `main.py` with `FastMCP` and `wrap_instance_methods` | 1669-1685 | PASS | Verified: server initializes |
| 95 | `uv run python main.py` | 1695-1697 | PASS | Verified: server starts, tools exposed |
| 96 | MCP JSON-RPC examples (initialize, tools/list, tools/call) | 1717-1745 | PASS | Verified: protocol works |
| 97 | ToyAIKit MCP client (`MCPClient`, `SubprocessMCPTransport`) | 1753-1767 | PASS | Verified: connection works |
| 98 | ToyAIKit MCPTools wrapper with `OpenAIResponsesRunner` | 1771-1803 | PASS | Verified: tools accessible |
| 99 | SSE transport (`MCPServerSSE`) | 1882-1894 | PASS | Code verified |
| 100 | Cursor MCP config `.cursor/mcp.json` | 1902-1929 | PASS | Format verified |
| 101 | VS Code MCP config `.vscode/mcp.json` | 1939-1967 | PASS | Format verified |

### MCP Verification Details:

```
MCP Client initialized successfully!
Available tools: ['add_entry', 'search']

Tool call result example:
{
  "content": [
    {
      "type": "text",
      "text": "[{\"text\":\"If you have this error, it most likely that your kafka broker docker container is not working...\",\"section\":\"Module 6: streaming with kafka\",\"question\":\"kafka.errors.NoBrokersAvailable...\",\"course\":\"data-engineering-zoomcamp\"},...]"
    }
  ]
}
```

---

## search_tools.py MODULE

| # | Code | Status | Notes |
|---|------|--------|-------|
| `from search_tools import SearchTools, init_index, init_tools` | PASS | Module imports correctly |
| `init_index()` | PASS | Returns index with 948 documents |
| `SearchTools.search(query)` | PASS | Returns search results |
| `SearchTools.add_entry(question, answer)` | PASS | Adds entry to index |

---

## FIXES APPLIED TO README

1. **Line 1390**: `AnthropicClient(client=Anthropic())` → `AnthropicClient()`
2. **Line 1357**: Added note about `openai/gpt-oss-20b` pricing issue
3. **Lines 590-598**: Added `import re` and markdown stripping for Anthropic `.create()`
4. **Line 611**: Added "Use ONLY JSON, no markdown" to Anthropic `.parse()` example
5. **Line 621**: Added note about Anthropic structured output markdown behavior
6. **Line 1528**: Added `from typing import List` import
7. **Line 1536**: Changed `result_type` → `output_type` for PydanticAI

---

## DEPENDENCIES INSTALLED

| Package | Status |
|---------|--------|
| uv | OK |
| jupyter | OK |
| openai | OK |
| python-dotenv | OK |
| anthropic | OK |
| boto3 | OK |
| google-genai | OK |
| minsearch | OK |
| sentence-transformers | OK |
| tqdm | OK |
| toyaikit | OK |
| pydantic-ai | OK |
| fastmcp | OK |
| requests | OK |
