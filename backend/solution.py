from __future__ import annotations

import logging

from backend.chat_schema import ChatMessage, ChatRequest, ChatResponse
from backend.helpers.agent import run_turn
from fastapi import HTTPException

log = logging.getLogger("chat")


def chat(request: ChatRequest) -> ChatResponse:
    """POST /chat: route the turn to a small tool set, then run the tool loop.

    See backend/helpers/ for the pieces: router.py (tool selection),
    agent.py (orchestration loop), executor.py (tool calls + log).
    """
    if request.messages[-1].role != "user":
        raise HTTPException(status_code=422, detail="The last message must come from the user.")

    try:
        reply, executor = run_turn(request.messages)
    except Exception as exc:
        # Model/API failure. Tool calls made before the failure still get logged.
        log.exception("chat turn failed")
        executor = getattr(exc, "executor", None)
        reply = f"Sorry, something went wrong while handling that request ({type(exc).__name__}). Please try again."
        return ChatResponse(
            messages=[*request.messages, ChatMessage(role="assistant", content=reply)],
            tool_calls=executor.log if executor else [],
        )

    return ChatResponse(
        messages=[*request.messages, ChatMessage(role="assistant", content=reply)],
        tool_calls=executor.log,
    )
