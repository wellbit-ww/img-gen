from __future__ import annotations

import io
import zipfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from sqlmodel import Session, col, select

from app.api.deps import get_app_settings
from app.config import Settings
from app.core.schemas import GalleryResponse, ImageMeta, OverlayApplyRequest
from app.db.models import ImageRecord
from app.db.session import get_session
from app.services.storage import apply_and_save_overlay

router = APIRouter(tags=["gallery"])


def _to_meta(img: ImageRecord) -> ImageMeta:
    return ImageMeta(
        id=img.id,
        job_id=img.job_id,
        path=img.path,
        url=f"/api/images/{img.id}",
        prompt=img.prompt,
        preset_id=img.preset_id,
        seed=img.seed,
        width=img.width,
        height=img.height,
        favorite=img.favorite,
        has_overlay=img.has_overlay,
        created_at=img.created_at.isoformat(),
        params=img.params,
    )


@router.get("/gallery", response_model=GalleryResponse)
def gallery(
    search: str | None = None,
    preset: str | None = None,
    favorite: bool | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(24, ge=1, le=100),
    session: Session = Depends(get_session),
) -> GalleryResponse:
    stmt = select(ImageRecord).order_by(col(ImageRecord.created_at).desc())
    if search:
        stmt = stmt.where(col(ImageRecord.prompt).contains(search))
    if preset:
        stmt = stmt.where(ImageRecord.preset_id == preset)
    if favorite is not None:
        stmt = stmt.where(ImageRecord.favorite == favorite)
    all_items = session.exec(stmt).all()
    total = len(all_items)
    start = (page - 1) * page_size
    items = all_items[start : start + page_size]
    return GalleryResponse(
        items=[_to_meta(i) for i in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/images/{image_id}")
def get_image_file(image_id: str, session: Session = Depends(get_session)) -> FileResponse:
    img = session.get(ImageRecord, image_id)
    if not img:
        raise HTTPException(404, "Image not found")
    path = Path(img.path)
    if not path.exists():
        raise HTTPException(404, "File missing")
    return FileResponse(path)


@router.get("/images/{image_id}/meta", response_model=ImageMeta)
def get_image_meta(image_id: str, session: Session = Depends(get_session)) -> ImageMeta:
    img = session.get(ImageRecord, image_id)
    if not img:
        raise HTTPException(404, "Image not found")
    return _to_meta(img)


@router.post("/images/{image_id}/favorite")
def toggle_favorite(image_id: str, session: Session = Depends(get_session)) -> dict[str, bool]:
    img = session.get(ImageRecord, image_id)
    if not img:
        raise HTTPException(404, "Image not found")
    img.favorite = not img.favorite
    session.add(img)
    session.commit()
    return {"favorite": img.favorite}


@router.delete("/images/{image_id}")
def delete_image(image_id: str, session: Session = Depends(get_session)) -> dict[str, str]:
    img = session.get(ImageRecord, image_id)
    if not img:
        raise HTTPException(404, "Image not found")
    path = Path(img.path)
    if path.exists():
        path.unlink(missing_ok=True)
    session.delete(img)
    session.commit()
    return {"status": "deleted"}


@router.post("/images/{image_id}/overlay", response_model=ImageMeta)
def apply_overlay(
    image_id: str,
    body: OverlayApplyRequest,
    settings: Settings = Depends(get_app_settings),
    session: Session = Depends(get_session),
) -> ImageMeta:
    img = session.get(ImageRecord, image_id)
    if not img:
        raise HTTPException(404, "Image not found")
    fonts_dir = settings.fonts_dir
    if not fonts_dir.is_absolute():
        fonts_dir = (Path.cwd() / fonts_dir).resolve()
    source = Path(img.source_path or img.path)
    out_path = apply_and_save_overlay(source, body.overlay, fonts_dir)
    img.path = str(out_path)
    img.has_overlay = True
    session.add(img)
    session.commit()
    return _to_meta(img)


@router.get("/jobs/{job_id}/download.zip")
def download_job_zip(job_id: str, session: Session = Depends(get_session)) -> StreamingResponse:
    images = session.exec(select(ImageRecord).where(ImageRecord.job_id == job_id)).all()
    if not images:
        raise HTTPException(404, "No images for job")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for img in images:
            p = Path(img.path)
            if p.exists():
                zf.write(p, arcname=p.name)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{job_id}.zip"'},
    )
