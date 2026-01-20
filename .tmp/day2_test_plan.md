# Day 2 Test Plan

## Overview
Testing all code blocks from Day 2 README to ensure they are self-contained and working.

## Test Strategy
1. Each Part gets its own test script
2. Scripts are standalone and can run independently
3. Mock external dependencies (OpenAI API) where needed
4. Test actual functionality (file operations, agent creation)

---

## Part 1: Django Template Setup

### Code Blocks to Test
1. **Environment Setup** - dirdotenv/python-dotenv installation
2. **Create Template** - mkdir, uv init, uv add django
3. **Create Makefile** - make file with install/migrate/run
4. **Create base.html** - template structure
5. **Create home view** - views.py and urls.py

### Test Cases
- [ ] Can create django_template directory structure
- [ ] Can run `uv init` and `uv add django`
- [ ] Can run `make migrate` successfully
- [ ] Can run `make run` (background process)
- [ ] Home page is accessible at http://localhost:8000

### Dependencies
- uv installed
- django_template directory created fresh

### Script
`test_part1_template.py`

---

## Part 2: AgentTools Class

### Code Blocks to Test
1. **Class Stub** - Basic class structure
2. **Full Implementation** - Complete agent_tools.py
3. **Testing Tools** - Initialize, see_file_tree, read_file, search_in_files
4. **Copy Template Function** - start() function

### Test Cases
- [ ] AgentTools can be imported
- [ ] AgentTools.__init__ stores project_path
- [ ] read_file() returns file contents
- [ ] write_file() creates files and directories
- [ ] execute_bash_command() runs simple commands
- [ ] execute_bash_command() blocks 'runserver'
- [ ] see_file_tree() returns sorted file list
- [ ] search_in_files() finds patterns
- [ ] start() function copies template correctly

### Dependencies
- django_template exists
- agent_tools.py exists

### Script
`test_part2_tools.py`

---

## Part 3: Single Agent with ToyAIKit

### Code Blocks to Test
1. **Simple Prompt** - DEVELOPER_PROMPT definition
2. **Import Libraries** - All imports
3. **Initialize Tools** - AgentTools and Tools object
4. **Create LLM Client** - OpenAIClient with model
5. **Create Runner** - OpenAIResponsesRunner setup
6. **Run Agent** - runner.run() and save result

### Test Cases
- [ ] All imports work (toyaikit, openai, etc.)
- [ ] AgentTools can be initialized with project_path
- [ ] Tools object can add agent_tools
- [ ] OpenAIClient can be created (mock API key)
- [ ] IPythonChatInterface can be created
- [ ] OpenAIResponsesRunner can be instantiated
- [ ] Result can be saved to file

### Dependencies
- OPENAI_API_KEY in environment
- toyaikit installed
- project exists

### Script
`test_part3_single_agent.py`

---

## Part 4: Multi-Agent with PydanticAI

### Code Blocks to Test
1. **Import and Initialize** - pydantic_ai imports
2. **Researcher Agent** - Create with instructions
3. **Writer Agent** - Create with instructions
4. **Coordinator Agent** - Create with other agents as tools
5. **Better Version** - Update instructions
6. **Reviewer Agent** - Create reviewer
7. **Run Multi-Agent** - PydanticAIRunner

### Test Cases
- [ ] pydantic_ai imports work
- [ ] Agent() can be created with string model
- [ ] Agent() accepts tools (functions)
- [ ] Agent() accepts other agents as tools
- [ ] Instructions variables work
- [ ] PydanticAIRunner can be created
- [ ] await runner.run() works (async)
- [ ] Result can be saved

### Dependencies
- OPENAI_API_KEY in environment
- pydantic-ai installed
- toyaikit installed
- project exists

### Script
`test_part4_multi_agent.py`

---

## Part 5: Monitoring & Guardrails

### Code Blocks to Test
1. **Logfire Setup** - pip install, configure
2. **Instrument PydanticAI** - instrument_pydantic_ai()
3. **Input Validation** - UserRequest model with validators
4. **Output Sanitization** - sanitize_code_output()
5. **Cost Limits** - RunLimits configuration
6. **Content Moderation** - moderate_content()
7. **Complete Agent** - Full monitored agent

### Test Cases
- [ ] logfire can be installed
- [ ] logfire.configure() works (with or without API key)
- [ ] instrument_pydantic_ai() runs without error
- [ ] UserRequest model validates input
- [ ] UserRequest rejects dangerous commands
- [ ] sanitize_code_output() catches dangerous imports
- [ ] RunLimits can be configured
- [ ] moderate_content() function structure is correct
- [ ] Complete agent code runs

### Dependencies
- logfire installed
- OPENAI_API_KEY in environment
- pydantic-ai installed

### Script
`test_part5_monitoring.py`

---

## Known Issues to Fix

### 1. Missing Imports
- `@validator` deprecated in Pydantic v2, use `field_validator`
- Need to add `from pydantic import field_validator`

### 2. Variable Dependencies
- Part 3: `project_name` must be defined before use
- Part 5: `user_input`, `task` variables undefined
- Part 5: `DEVELOPER_PROMPT` needs to be defined

### 3. Async Context
- Part 4: Need `asyncio.run()` for testing async functions
- Part 5: Agent.run() is async

### 4. Mock Requirements
- OpenAI API calls need mocking for unit tests
- Use `unittest.mock` for API calls

---

## Test Execution Order

1. Part 1 - Create template (must run first)
2. Part 2 - Test tools (depends on template)
3. Part 3 - Single agent (depends on tools + template)
4. Part 4 - Multi-agent (depends on tools + template)
5. Part 5 - Monitoring (can run independently)

---

## Success Criteria

- All scripts run without syntax errors
- All imports resolve correctly
- AgentTools methods work as expected
- Agents can be instantiated
- Results can be saved to files
- Code is self-contained (no undefined variables)
