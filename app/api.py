import json
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import FileResponse
from app.config import ROOT, settings
from app.models import ArenaRequest, ArenaResponse, ChatRequest
from app.arena import execute

router = APIRouter()

@router.get('/')
def index():
    return FileResponse(ROOT / 'app/static/index.html')

@router.get('/health')
def health():
    return {
        'status': 'ok',
        'implementation': 'Inventory & Purchasing Sentinel Agent',
        'domain': 'Inbox Triage & Inventory Management'
    }

@router.get('/arena/manifest')
def manifest():
    return json.loads((ROOT / 'arena_manifest.json').read_text())

@router.get('/models')
def models():
    return {'models': ['gemini-2.5-flash', 'openai/gpt-oss-20b']}

@router.post('/arena/run', response_model=ArenaResponse)
async def arena_run(payload: ArenaRequest):
    model_to_use = payload.model or settings.model_name or 'gemini-2.5-flash'
    if model_to_use not in models()['models']:
        raise HTTPException(400, f"Model '{model_to_use}' is not supported. Supported models: {models()['models']}")
    return await execute(payload, model=model_to_use)

@router.post('/chat', response_model=ArenaResponse)
async def chat(payload: ChatRequest, request: Request):
    if payload.model not in models()['models']:
        raise HTTPException(400, f"Model '{payload.model}' is not enabled")
    busy = request.app.state.busy
    if payload.session_id in busy:
        raise HTTPException(409, 'This chat is already running')
    busy.add(payload.session_id)
    memory = request.app.state.memory
    try:
        session_state = memory.process_turn(payload.session_id, payload.task)
        result = await execute(
            payload,
            history=memory.get(payload.session_id),
            model=payload.model,
            session_state=session_state
        )
        memory.update_after_run(payload.session_id, payload.task, result.final_response, result.status)
        return result
    finally:
        busy.discard(payload.session_id)

@router.delete('/chat/{session_id}')
def reset(session_id: str, request: Request):
    if session_id in request.app.state.busy:
        raise HTTPException(409, 'Chat is running')
    request.app.state.memory.clear(session_id)
    return {'status': 'cleared'}
