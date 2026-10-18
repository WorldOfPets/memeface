from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import base64
from datetime import datetime
from pathlib import Path

app = FastAPI()

OUT_DIR = Path("data/webcam_test")
OUT_DIR.mkdir(exist_ok=True)

TEMPLATES_DIR = Path(__file__).parent / "templates"


class SnapRequest(BaseModel):
    image: str


def decode_data_url(value: str) -> bytes:
    if "," not in value:
        raise Exception("Ожидается data URL")
    _, payload = value.split(",", 1)
    return base64.b64decode(payload)


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = TEMPLATES_DIR / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post("/snap")
def snap(req: SnapRequest):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    full_path = OUT_DIR / f"{stamp}.jpg"
    full_path.write_bytes(decode_data_url(req.image))
    return {"ok": True, "file": full_path.name}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)