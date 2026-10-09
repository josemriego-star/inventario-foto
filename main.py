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


BASES_ABIERTAS = ["world.openfoodfacts.org", "world.openbeautyfacts.org", "world.openproductsfacts.org"]


async def buscar_web(codigo: str):
    """Bases abiertas: alimentos, cosmética/limpieza y otros productos."""
    for host in BASES_ABIERTAS:
        try:
            async with httpx.AsyncClient(timeout=8) as c:
                r = await c.get(f"https://{host}/api/v2/product/{codigo}.json",
                                params={"fields": "product_name,product_name_es,brands,quantity"},
                                headers={"User-Agent": "InventarioFoto/1.0"})
            p = r.json().get("product") if r.status_code == 200 else None
            nombre = ((p or {}).get("product_name_es") or (p or {}).get("product_name") or "").strip()
            if not nombre:
                continue
            cant = (p.get("quantity") or "").strip()
            marca = (p.get("brands") or "").split(",")[0].strip()
            return {"producto": f"{nombre} {cant}".strip(), "marca": marca, "unidad": "unidad", "origen": "web"}
        except Exception:
            continue
    return None


async def buscar_ia(codigo: str):
    """Gemini con búsqueda de Google: encuentra el producto por su código de barras."""
    prompt = (f"Buscá en internet el producto con código de barras {codigo} "
              "(si es posible, el que se vende en Paraguay o la región). "
              'Respondé SOLO un JSON, sin texto extra ni markdown: {"producto":"nombre con presentación (ej: Gaseosa cola 2 L)",'
              f'"marca":"","unidad":"una de: {UNIDADES}"}}. '
              'Si no encontrás el producto con certeza, respondé {"producto":""}. No inventes.')
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "tools": [{"google_search": {}}],
            "generationConfig": {"temperature": 0}}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    try:
        async with httpx.AsyncClient(timeout=45) as c:
            r = await c.post(url, json=body, headers={"x-goog-api-key": GEMINI_API_KEY})
        if r.status_code != 200:
            return None
        partes = r.json()["candidates"][0]["content"]["parts"]
        texto = "".join(p.get("text", "") for p in partes)
        d = json.loads(texto[texto.index("{"):texto.rindex("}") + 1])
        if not str(d.get("producto", "")).strip():
            return None
        return {"producto": d["producto"].strip(), "marca": str(d.get("marca", "")).strip(),
                "unidad": d.get("unidad") or "unidad", "origen": "ia"}
    except Exception:
        return None


@app.get("/catalogo/{codigo}", dependencies=[Depends(auth)])
async def catalogo(codigo: str, solo: int = 0):
    with conn() as c:
        r = c.execute("SELECT producto,marca,unidad FROM catalogo WHERE codigo=%s", (codigo,)).fetchone()
    if r:
        return {**r, "origen": "catalogo"}
    if not solo and codigo.isdigit() and 8 <= len(codigo) <= 14:
        res = await buscar_web(codigo) or await buscar_ia(codigo)
        if res:
            return res
    raise HTTPException(404, "No encontrado")


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
