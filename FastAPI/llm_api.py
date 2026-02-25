from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import asyncio
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Enable streaming
llm = ChatOpenAI(
    model="gpt-4o-mini",
    streaming=True
)

async def generate_response(prompt: str):
    async for chunk in llm.astream([HumanMessage(content=prompt)]):
        if chunk.content:
            yield chunk.content
            await asyncio.sleep(0)  # allow event loop switch

@app.get("/chat")
async def chat(prompt: str):
    return StreamingResponse(
        generate_response(prompt),
        media_type="text/plain"
    )