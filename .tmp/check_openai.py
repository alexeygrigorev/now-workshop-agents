from openai import OpenAI
from pydantic import BaseModel

class J(BaseModel):
    s: str

c = OpenAI()
r = c.responses.parse(model='gpt-4o-mini', input='tell me a joke', text_format=J)
print([x for x in dir(r) if 'output' in x.lower() or 'parsed' in x.lower()])
