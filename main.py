import base64, json, os
import httpx, psycopg
from psycopg.rows import dict_row
from fastapi import Depends, FastAPI, File, Header, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
DATABASE_URL = os.environ["DATABASE_URL"]
APP_KEY = os.getenv("APP_KEY", "")  # clave de acceso compartida; si está vacía, no se pide
UNIDADES = "unidad, kg, g, litro, ml, caja, paquete, botella, lata, bolsa, atado, docena"

PROMPT = f"""Identificá el producto de la foto para un inventario de cocina/comercio en Paraguay.
Respondé SOLO un JSON con estas claves:
- "producto": nombre en español con presentación si se ve (ej: "Gaseosa cola 2 L"). Vacío si no podés identificarlo.
- "marca": marca del producto, vacío si es un producto fresco.
- "tipo": "envasado" o "fresco".
- "unidad": una de: {UNIDADES}.
- "codigo": dígitos del código de barras si se ven claramente, si no vacío."""


def conn():
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


def auth(x_app_key: str = Header(default="")):
    if APP_KEY and x_app_key != APP_KEY:
        raise HTTPException(401, "Clave incorrecta")


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def iniciar():
    with conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS catalogo(
            codigo TEXT PRIMARY KEY, producto TEXT NOT NULL,
            marca TEXT DEFAULT '', unidad TEXT DEFAULT 'unidad')""")
        c.execute("""CREATE TABLE IF NOT EXISTS inventario(
            id SERIAL PRIMARY KEY, fecha TIMESTAMPTZ NOT NULL DEFAULT now(),
            producto TEXT NOT NULL, marca TEXT DEFAULT '', cantidad NUMERIC NOT NULL,
            unidad TEXT NOT NULL, codigo TEXT DEFAULT '')""")


class Item(BaseModel):
    producto: str
    marca: str = ""
    cantidad: float
    unidad: str
    codigo: str = ""


@app.get("/")
def salud():
    return {"ok": True}


@app.get("/inventario", dependencies=[Depends(auth)])
def listar():
    with conn() as c:
        return c.execute("""SELECT id,
            to_char(fecha AT TIME ZONE 'America/Asuncion','YYYY-MM-DD HH24:MI') AS fecha,
            producto, marca, cantidad::float AS cantidad, unidad, codigo
            FROM inventario ORDER BY id""").fetchall()


@app.post("/inventario", dependencies=[Depends(auth)])
def agregar(it: Item):
    with conn() as c:
        c.execute("INSERT INTO inventario(producto,marca,cantidad,unidad,codigo) VALUES(%s,%s,%s,%s,%s)",
                  (it.producto, it.marca, it.cantidad, it.unidad, it.codigo))
        if it.codigo:
            c.execute("""INSERT INTO catalogo(codigo,producto,marca,unidad) VALUES(%s,%s,%s,%s)
                ON CONFLICT (codigo) DO UPDATE SET producto=EXCLUDED.producto,
                marca=EXCLUDED.marca, unidad=EXCLUDED.unidad""",
                      (it.codigo, it.producto, it.marca, it.unidad))
    return {"ok": True}


@app.delete("/inventario/{id}", dependencies=[Depends(auth)])
def borrar(id: int):
    with conn() as c:
        c.execute("DELETE FROM inventario WHERE id=%s", (id,))
    return Response(status_code=204)


@app.delete("/inventario", dependencies=[Depends(auth)])
def vaciar():
    with conn() as c:
        c.execute("DELETE FROM inventario")
    return Response(status_code=204)


@app.get("/catalogo/{codigo}", dependencies=[Depends(auth)])
def catalogo(codigo: str):
    with conn() as c:
        r = c.execute("SELECT producto,marca,unidad FROM catalogo WHERE codigo=%s", (codigo,)).fetchone()
    if not r:
        raise HTTPException(404, "No está en el catálogo")
    return r


@app.post("/reconocer", dependencies=[Depends(auth)])
async def reconocer(file: UploadFile = File(...)):
    img = await file.read()
    if len(img) > 8_000_000:
        raise HTTPException(413, "Imagen demasiado grande")
    body = {
        "contents": [{"parts": [
            {"text": PROMPT},
            {"inline_data": {"mime_type": file.content_type or "image/jpeg",
                             "data": base64.b64encode(img).decode()}},
        ]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.1},
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(url, json=body, headers={"x-goog-api-key": GEMINI_API_KEY})
    if r.status_code != 200:
        raise HTTPException(502, f"Gemini respondió {r.status_code}")
    try:
        texto = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(texto)
    except Exception:
        raise HTTPException(502, "Respuesta inesperada de Gemini")
