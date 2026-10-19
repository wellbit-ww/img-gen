from __future__ import annotations

import multiprocessing as mp
import os

from arq import Worker
from arq.connections import RedisSettings

from app.config import Settings, get_settings
from app.services.workers_config import load_workers_config
from app.worker.tasks import (
    process_batch,
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
    process_lora_train,
]


def _run_single_worker(settings: Settings, gpu: int, model: str, queue_name: str) -> None:
    from app.db.session import init_db

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    init_db(settings.db_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["WORKER_MODEL"] = model
    worker = Worker(
        functions=TASK_FUNCTIONS,
        redis_settings=RedisSettings.from_dsn(settings.redis_url),
        queue_name=queue_name,
        max_jobs=1,
    )
    worker.run()


def run_supervisor(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    worker_cfgs = load_workers_config(settings)
    jobs: list[tuple[int, str, str]] = []
    for cfg in worker_cfgs:
        gpu = int(cfg.get("gpu", 0))
        model = str(cfg.get("model", "mock"))
        for queue in cfg.get("queues", ["quality"]):
            jobs.append((gpu, model, queue))

    if len(jobs) == 1:
        gpu, model, queue = jobs[0]
        _run_single_worker(settings, gpu, model, queue)
        return

    mp.set_start_method("spawn", force=True)
    processes: list[mp.Process] = []
    for gpu, model, queue in jobs:
        p = mp.Process(
            target=_run_single_worker,
            args=(settings, gpu, model, queue),
            daemon=False,
            name=f"worker-gpu{gpu}-{queue}",
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
