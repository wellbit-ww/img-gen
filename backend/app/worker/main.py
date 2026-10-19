from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
from collections import defaultdict
from typing import Any

from arq import create_pool
from arq.connections import RedisSettings
from arq.worker import Worker

from app.config import Settings, get_settings
from app.logging_setup import configure_logging
from app.services.redis_sync import set_worker_heartbeat
from app.services.workers_config import load_workers_config
from app.worker.tasks import (
    process_batch,
    process_batch_line,
    process_edit,
    process_generate,
    process_lora_train,
    process_remove_bg,
    process_upscale,
)

TASK_FUNCTIONS = [
    process_generate,
    process_edit,
    process_upscale,
    process_remove_bg,
    process_batch,
    process_batch_line,
    process_lora_train,
]

ORCHESTRATOR_QUEUES = frozenset({"batch_parent", "lora"})


async def _heartbeat(ctx: dict, busy: bool) -> None:
    settings = get_settings()
    worker_id = str(ctx.get("worker_id", "unknown"))
    set_worker_heartbeat(
        settings.redis_url,
        worker_id,
        {
            "worker_id": worker_id,
            "gpu": ctx.get("gpu"),
            "model": ctx.get("model_key"),
            "queues": ctx.get("queues", []),
            "busy": busy,
        },
    )


async def _run_gpu_workers_async(
    settings: Settings,
    gpu: int,
    model: str,
    queues: list[str],
    memory: str,
    worker_flags: dict[str, bool],
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["WORKER_MODEL"] = model
    os.environ["WORKER_MEMORY"] = memory
    os.environ["WORKER_TF32"] = "1" if worker_flags.get("tf32", True) else "0"
    os.environ["WORKER_VAE_TILING"] = "1" if worker_flags.get("vae_tiling", True) else "0"
    os.environ["WORKER_TORCH_COMPILE"] = "1" if worker_flags.get("torch_compile", False) else "0"

    worker_id = f"gpu{gpu}-{model}"
    shared_ctx: dict[str, Any] = {
        "worker_id": worker_id,
        "gpu": gpu,
        "model_key": model,
        "queues": queues,
        "engines": {},
        "active_jobs": 0,
    }

    heartbeat_task: asyncio.Task | None = None

    async def heartbeat_loop(ctx: dict) -> None:
        while True:
            busy = int(ctx.get("active_jobs", 0)) > 0
            await _heartbeat(ctx, busy)
            await asyncio.sleep(10)

    async def startup(ctx: dict) -> None:
        for key, value in shared_ctx.items():
            ctx[key] = value
        nonlocal heartbeat_task
        heartbeat_task = asyncio.create_task(heartbeat_loop(ctx))
        await _heartbeat(ctx, False)

    async def _on_job_start(ctx: dict) -> None:
        ctx["active_jobs"] = int(ctx.get("active_jobs", 0)) + 1
        await _heartbeat(ctx, True)

    async def _on_job_end(ctx: dict) -> None:
        ctx["active_jobs"] = max(0, int(ctx.get("active_jobs", 0)) - 1)
        await _heartbeat(ctx, int(ctx.get("active_jobs", 0)) > 0)

    redis = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    unique_queues = sorted(set(queues))
    orchestrator = [q for q in unique_queues if q in ORCHESTRATOR_QUEUES]
    gpu_queues = [q for q in unique_queues if q not in ORCHESTRATOR_QUEUES]
    tasks = []
    for queue_name in orchestrator:
        worker = Worker(
            functions=TASK_FUNCTIONS,
            redis_settings=RedisSettings.from_dsn(settings.redis_url),
            redis_pool=redis,
            queue_name=queue_name,
            max_jobs=8,
            on_startup=startup,
            on_job_start=_on_job_start,
            on_job_end=_on_job_end,
            ctx=shared_ctx,
        )
        tasks.append(asyncio.create_task(worker.main()))
    for queue_name in gpu_queues:
        worker = Worker(
            functions=TASK_FUNCTIONS,
            redis_settings=RedisSettings.from_dsn(settings.redis_url),
            redis_pool=redis,
            queue_name=queue_name,
            max_jobs=1,
            on_startup=startup,
            on_job_start=_on_job_start,
            on_job_end=_on_job_end,
            ctx=shared_ctx,
        )
        tasks.append(asyncio.create_task(worker.main()))
    await asyncio.gather(*tasks)


def _run_gpu_process(
    settings: Settings,
    gpu: int,
    model: str,
    queues: list[str],
    memory: str,
    worker_flags: dict[str, bool],
) -> None:
    from app.db.session import init_db

    configure_logging()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    init_db(settings.db_path)
    asyncio.run(
        _run_gpu_workers_async(settings, gpu, model, queues, memory, worker_flags)
    )


def run_supervisor(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    worker_cfgs = load_workers_config(settings)

    grouped: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "model": "mock",
            "queues": [],
            "memory": "text_encoder_offload",
            "flags": {"tf32": True, "vae_tiling": True, "torch_compile": False},
        }
    )
    for cfg in worker_cfgs:
        gpu = int(cfg.get("gpu", 0))
        grouped[gpu]["model"] = str(cfg.get("model", "mock"))
        grouped[gpu]["queues"].extend(cfg.get("queues", ["quality"]))
        grouped[gpu]["memory"] = str(cfg.get("memory", "text_encoder_offload"))
        for flag in ("tf32", "vae_tiling", "torch_compile"):
            if flag in cfg:
                grouped[gpu]["flags"][flag] = bool(cfg[flag])

    jobs = [
        (gpu, data["model"], data["queues"], data["memory"], data["flags"])
        for gpu, data in grouped.items()
    ]

    if len(jobs) == 1:
        gpu, model, queues, memory, flags = jobs[0]
        _run_gpu_process(settings, gpu, model, queues, memory, flags)
        return

    mp.set_start_method("spawn", force=True)
    processes: list[mp.Process] = []
    for gpu, model, queues, memory, flags in jobs:
        p = mp.Process(
            target=_run_gpu_process,
            args=(settings, gpu, model, queues, memory, flags),
            daemon=False,
            name=f"worker-gpu{gpu}",
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
