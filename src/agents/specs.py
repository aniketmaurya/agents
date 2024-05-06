from typing import List, Optional
from pydantic import BaseModel


class FunctionInfo(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: str
    function: FunctionInfo


class Message(BaseModel):
    role: str
    content: Optional[str]
    tool_calls: List[ToolCall]


class Choice(BaseModel):
    index: int
    logprobs: Optional[dict]
    message: Message
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletion(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Usage
