from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.chatbot import stream_chatbot
import uuid
import json
from typing import Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["x-session-id"]
)

class ChatMessage(BaseModel):
    user_input: str
    session_id: Optional[str] = None

@app.post("/chat/stream")
async def chat_stream_endpoint(request: Request, message: ChatMessage):
    try:
        session_id = message.session_id or str(uuid.uuid4())
        
        
        accept_header = request.headers.get("accept", "")
        is_sse = "text/event-stream" in accept_header
        
        if is_sse:
           
            def event_generator():
                for chunk in stream_chatbot(message.user_input, session_id):
                    yield f"data: {json.dumps({'message': chunk})}\n\n"
            
            headers = {
                "x-session-id": session_id,
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
            
            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers=headers
            )
        else:
           
            full_response = "".join([chunk for chunk in stream_chatbot(message.user_input, session_id)])
            return JSONResponse({
                "message": full_response,
                "session_id": session_id
            })
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))