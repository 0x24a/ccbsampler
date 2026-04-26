from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from schemas import ResampleRequest, ResampleResponse

router = APIRouter()


@router.post("/resample", response_model=ResampleResponse)
async def resample(req: ResampleRequest, request: Request) -> ResampleResponse:
    renderer = request.app.state.renderer
    gpu_queue = request.app.state.gpu_queue

    try:
        payload = await renderer.prepare(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    result: ResampleResponse = await gpu_queue.submit(payload)
    return result
