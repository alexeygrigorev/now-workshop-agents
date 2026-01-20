from pydantic_ai import Agent
from pydantic import BaseModel
import asyncio

class J(BaseModel):
    j: str

a = Agent('openai:gpt-4o-mini', output_type=J)
r = asyncio.run(a.run('tell me a joke'))
print(type(r.output))
print(r.output)
print(hasattr(r.output, 'j'))
