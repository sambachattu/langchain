from langchain_core.tools import StructuredTool
from pydantic import BaseModel,Field

class MultiplyInput(BaseModel):
    a:int = Field(required=True,description="The first number to add")
    b:int = Field(required=True,description="The second number to add")

def multiply_func(a:int,b:int)->int:
    return a*b

multiply_tool = StructuredTool.from_function(
    func=multiply_func,
    name="multiply",
    description="Multiply two numbers",
    args_schema=MultiplyInput
)

result = multiply_tool.invoke({"a":5,"b":5})

print(result)