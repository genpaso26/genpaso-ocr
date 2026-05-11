"""
GenPaso OCR v2 — Ingesta masiva con árbol genealógico
Procesa múltiples registros, gestiona DB maestra y extrae ancestros.
"""

import base64
import json
import os
import uuid
from io import BytesIO
from pathlib import Path

import anthropic
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ── Constantes ─────────────────────────────────────────────────────────────────

MASTER_DB_PATH = Path(__file__).parent / "GenPaso_Master_DB.csv"

COLUMNAS_MASTER = [
    "horse_id", "registration_number", "Horse_Chip", "name", "gender",
    "birth_date", "Gait", "color", "height", "country_origin",
    "issuing_association_id", "sire_id", "dam_id",
]

PROMPT_EXTRACCION = """Analiza este registro de caballo. Identifica al ejemplar principal y todos sus ancestros mencionados.
Devuelve EXCLUSIVAMENTE un objeto JSON con esta estructura exacta:

{
  "main": {
    "registration_number": "<número de registro oficial, o null>",
    "Horse_Chip": "<número de microchip, o null>",
    "name": "<nombre completo del caballo, respetando mayúsculas>",
    "gender": "<Semental | Yegua | Castrado>",
    "birth_date": "<YYYY-MM-DD, o null>",
    "Gait": "<modalidad: Paso Fino, Paso Colombiano, Trote, etc., o null>",
    "color": "<capa del caballo, o null>",
    "height": "<alzada como texto, o null>",
    "country_origin": "<código ISO 2 letras, o null>",
    "issuing_association_id": "<asociación o entidad que emite el registro, o null>"
  },
  "ancestors": [
    {
      "relationship": "<sire | dam | paternal_grandsire | paternal_granddam | maternal_grandsire | maternal_granddam>",
      "name": "<nombre completo. Conserva títulos: FC, CH, etc.>",
      "registration_number": "<número de registro, o null>",
      "gender": "<Semental | Yegua | Castrado, o null>",
      "Gait": "<modalidad, o null>",
      "association": "<asociación, o null>"
    }
  ]
}

Reglas críticas:
- "gender" solo acepta: Semental, Yegua, Castrado.
- Los nombres de padre y madre son vitales para la red genética de GenPaso. Inclúyelos completos con títulos (FC = Fuera de Concurso, CH = Champion).
- Es crítico identificar el Gait (Modalidad) y la Asociación de cada ejemplar.
- Si un ancestro no aparece en el documento, no lo incluyas en la lista.
- Solo devuelve el JSON, sin texto adicional ni bloques de código.
"""

# ── DB Maestra ──────────────────────────────────────────────────────────────────

def cargar_master_db() -> pd.DataFrame:
    if MASTER_DB_PATH.exists():
        df = pd.read_csv(MASTER_DB_PATH, dtype=str)
        for col in COLUMNAS_MASTER:
            if col not in df.columns:
                df[col] = None
        return df[COLUMNAS_MASTER]
    return pd.DataFrame(columns=COLUMNAS_MASTER)


def guardar_master_db(df: pd.DataFrame):
    df.to_csv(MASTER_DB_PATH, index=False)


def buscar_caballo(db: pd.DataFrame, registration_number=None, chip=None, nombre=None) -> str | None:
    """Busca un caballo en la DB. Retorna horse_id si existe, None si no."""
    if registration_number and registration_number != "None":
        mask = db["registration_number"].str.strip().str.upper() == str(registration_number).strip().upper()
        if mask.any():
            return db.loc[mask, "horse_id"].iloc[0]
    if chip and chip != "None":
        mask = db["Horse_Chip"].str.strip() == str(chip).strip()
        if mask.any():
            return db.loc[mask, "horse_id"].iloc[0]
    if nombre and nombre != "None":
        mask = db["name"].str.strip().str.upper() == str(nombre).strip().upper()
        if mask.any():
            return db.loc[mask, "horse_id"].iloc[0]
    return None


def insertar_o_vincular(db: pd.DataFrame, datos: dict) -> tuple[pd.DataFrame, str, bool]:
    """
    Busca el caballo en la DB. Si no existe lo inserta.
    Retorna (db_actualizada, horse_id, es_nuevo).
    """
    horse_id = buscar_caballo(
        db,
        registration_number=datos.get("registration_number"),
        chip=datos.get("Horse_Chip"),
        nombre=datos.get("name"),
    )
    if horse_id:
        return db, horse_id, False

    horse_id = str(uuid.uuid4())
    fila = {col: datos.get(col) for col in COLUMNAS_MASTER}
    fila["horse_id"] = horse_id
    nueva_fila = pd.DataFrame([fila])
    db = pd.concat([db, nueva_fila], ignore_index=True)
    return db, horse_id, True


# ── API ─────────────────────────────────────────────────────────────────────────

def obtener_cliente() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        st.error("No se encontró ANTHROPIC_API_KEY. Revisa tu archivo .env o el panel lateral.")
        st.stop()
    return anthropic.Anthropic(api_key=api_key)


def construir_contenido(archivo_bytes: bytes, media_type: str) -> list:
    datos_b64 = base64.standard_b64encode(archivo_bytes).decode("utf-8")
    if media_type == "application/pdf":
        bloque = {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": datos_b64}}
    else:
        bloque = {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": datos_b64}}
    return [bloque, {"type": "text", "text": PROMPT_EXTRACCION}]


def llamar_api(archivo_bytes: bytes, media_type: str) -> dict:
    cliente = obtener_cliente()
    respuesta = cliente.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{"role": "user", "content": construir_contenido(archivo_bytes, media_type)}],
    )
    texto = respuesta.content[0].text.strip()
    if texto.startswith("```"):
        lineas = texto.splitlines()
        texto = "\n".join(lineas[1:-1] if lineas[-1].strip() == "```" else lineas[1:])
    return json.loads(texto)


# ── Procesamiento de un archivo ─────────────────────────────────────────────────

def procesar_archivo(archivo_bytes: bytes, media_type: str, db: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Procesa un archivo: extrae datos del caballo principal y sus ancestros.
    Retorna la DB actualizada y un resumen del procesamiento.
    """
    resultado = llamar_api(archivo_bytes, media_type)
    main = resultado.get("main", {})
    ancestors = resultado.get("ancestors", [])

    resumen = {"nombre": main.get("name", "Desconocido"), "nuevos": 0, "existentes": 0, "ancestros": []}

    # Procesar ancestros primero para obtener sire_id y dam_id
    sire_id = None
    dam_id = None

    for anc in ancestors:
        datos_anc = {
            "name": anc.get("name"),
            "registration_number": anc.get("registration_number"),
            "Horse_Chip": None,
            "gender": anc.get("gender"),
            "Gait": anc.get("Gait"),
            "issuing_association_id": anc.get("association"),
        }
        db, anc_id, es_nuevo = insertar_o_vincular(db, datos_anc)

        rel = anc.get("relationship", "")
        if rel == "sire":
            sire_id = anc_id
        elif rel == "dam":
            dam_id = anc_id

        resumen["ancestros"].append({
            "nombre": anc.get("name"),
            "relacion": rel,
            "es_nuevo": es_nuevo,
            "horse_id": anc_id,
        })
        if es_nuevo:
            resumen["nuevos"] += 1
        else:
            resumen["existentes"] += 1

    # Procesar caballo principal
    datos_main = {
        "registration_number": main.get("registration_number"),
        "Horse_Chip": main.get("Horse_Chip"),
        "name": main.get("name"),
        "gender": main.get("gender"),
        "birth_date": main.get("birth_date"),
        "Gait": main.get("Gait"),
        "color": main.get("color"),
        "height": main.get("height"),
        "country_origin": main.get("country_origin"),
        "issuing_association_id": main.get("issuing_association_id"),
        "sire_id": sire_id,
        "dam_id": dam_id,
    }
    db, main_id, es_nuevo = insertar_o_vincular(db, datos_main)
    resumen["horse_id"] = main_id
    resumen["es_nuevo"] = es_nuevo
    if es_nuevo:
        resumen["nuevos"] += 1
    else:
        resumen["existentes"] += 1

    return db, resumen


# ── Exportación ─────────────────────────────────────────────────────────────────

def df_a_excel(df: pd.DataFrame, sheet_name: str = "GenPaso") -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return buffer.getvalue()


# ── UI ──────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="GenPaso OCR v2", page_icon="🐴", layout="wide")

st.title("🐴 GenPaso — Ingesta Masiva de Registros Equinos")
st.caption("Procesa múltiples registros, detecta ancestros y construye la base de datos genética de GenPaso.")

# Estado de sesión
for key in ["resumenes", "db_sesion", "db_master_snapshot"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuración")
    api_key_input = st.text_input("API Key de Anthropic", type="password", placeholder="sk-ant-...")
    if api_key_input:
        os.environ["ANTHROPIC_API_KEY"] = api_key_input

    st.divider()
    db_actual = cargar_master_db()
    st.metric("Registros en DB Maestra", len(db_actual))

    if MASTER_DB_PATH.exists():
        st.download_button(
            "⬇️ Descargar DB Maestra (.csv)",
            data=MASTER_DB_PATH.read_bytes(),
            file_name="GenPaso_Master_DB.csv",
            mime="text/csv",
            use_container_width=True,
        )
        if st.button("🗑️ Resetear DB Maestra", use_container_width=True, type="secondary"):
            MASTER_DB_PATH.unlink()
            st.rerun()

    st.divider()
    st.markdown("**Campos extraídos**")
    st.markdown("`horse_id` · `registration_number` · `Horse_Chip` · `name` · `gender` · `birth_date` · `Gait` · `color` · `height` · `country_origin` · `issuing_association_id` · `sire_id` · `dam_id`")

# ── Carga de archivos ────────────────────────────────────────────────────────────
st.subheader("1. Cargar Documentos")
archivos = st.file_uploader(
    "Arrastra o selecciona uno o varios registros",
    type=["jpg", "jpeg", "png", "pdf"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if archivos:
    cols = st.columns(min(len(archivos), 4))
    for i, arch in enumerate(archivos):
        with cols[i % 4]:
            if arch.type.startswith("image/"):
                st.image(arch, caption=arch.name, use_container_width=True)
            else:
                st.info(f"📄 {arch.name}")

procesar = st.button(
    f"🔍 Procesar {len(archivos)} Registro(s)" if archivos else "🔍 Procesar Registros",
    disabled=not archivos,
    type="primary",
    use_container_width=True,
)

# ── Procesamiento en lote ────────────────────────────────────────────────────────
if procesar and archivos:
    db = cargar_master_db()
    resumenes = []
    registros_sesion = []

    barra = st.progress(0, text="Iniciando procesamiento...")
    contenedor_log = st.container()

    for i, archivo in enumerate(archivos):
        barra.progress((i) / len(archivos), text=f"Procesando {archivo.name} ({i+1}/{len(archivos)})...")
        try:
            archivo_bytes = archivo.read()
            db, resumen = procesar_archivo(archivo_bytes, archivo.type, db)
            resumen["archivo"] = archivo.name
            resumen["error"] = None
            resumenes.append(resumen)

            fila_sesion = db[db["horse_id"] == resumen["horse_id"]].copy()
            registros_sesion.append(fila_sesion)

            with contenedor_log:
                estado = "NUEVO" if resumen["es_nuevo"] else "EXISTENTE"
                st.success(f"✅ **{resumen['nombre']}** — {estado} | {len(resumen['ancestros'])} ancestros detectados")

        except json.JSONDecodeError:
            resumenes.append({"archivo": archivo.name, "error": "JSON inválido — imagen poco legible"})
            with contenedor_log:
                st.warning(f"⚠️ {archivo.name}: imagen poco legible, intenta con mejor resolución.")
        except anthropic.BadRequestError:
            resumenes.append({"archivo": archivo.name, "error": "Archivo no procesable por la API"})
            with contenedor_log:
                st.warning(f"⚠️ {archivo.name}: formato no compatible con la API.")
        except Exception as e:
            resumenes.append({"archivo": archivo.name, "error": str(e)})
            with contenedor_log:
                st.error(f"❌ {archivo.name}: {e}")

    barra.progress(1.0, text="Procesamiento completo.")
    guardar_master_db(db)

    st.session_state.resumenes = resumenes
    st.session_state.db_sesion = pd.concat(registros_sesion, ignore_index=True) if registros_sesion else pd.DataFrame(columns=COLUMNAS_MASTER)
    st.session_state.db_master_snapshot = db

# ── Resultados ───────────────────────────────────────────────────────────────────
if st.session_state.resumenes:
    st.divider()
    st.subheader("2. Resumen de Procesamiento")

    total_archivos = len(st.session_state.resumenes)
    exitosos = [r for r in st.session_state.resumenes if not r.get("error")]
    total_nuevos = sum(r.get("nuevos", 0) for r in exitosos)
    total_existentes = sum(r.get("existentes", 0) for r in exitosos)
    total_ancestros = sum(len(r.get("ancestros", [])) for r in exitosos)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Archivos procesados", f"{len(exitosos)}/{total_archivos}")
    c2.metric("Perfiles nuevos creados", total_nuevos)
    c3.metric("Perfiles existentes vinculados", total_existentes)
    c4.metric("Ancestros identificados", total_ancestros)

    st.info(f"Se procesaron **{total_archivos}** archivos. Se crearon **{total_nuevos}** nuevos perfiles y se identificaron **{total_existentes}** ancestros/perfiles existentes.")

    # Detalle de ancestros
    with st.expander("Ver detalle de ancestros detectados"):
        for r in exitosos:
            if r.get("ancestros"):
                st.markdown(f"**{r['nombre']}**")
                for anc in r["ancestros"]:
                    icono = "🆕" if anc["es_nuevo"] else "🔗"
                    st.markdown(f"  {icono} `{anc['relacion']}` — {anc['nombre']} (ID: `{anc['horse_id'][:8]}...`)")

    st.divider()
    st.subheader("3. Datos de Esta Sesión")
    df_sesion = st.session_state.db_sesion
    if not df_sesion.empty:
        df_editado = st.data_editor(df_sesion, use_container_width=True, num_rows="fixed", key="editor_sesion")

        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button(
                "⬇️ Descargar sesión actual (.xlsx)",
                data=df_a_excel(df_editado, "Sesion_Actual"),
                file_name="sesion_actual.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with col_b:
            if st.session_state.db_master_snapshot is not None:
                st.download_button(
                    "⬇️ Descargar DB Maestra completa (.xlsx)",
                    data=df_a_excel(st.session_state.db_master_snapshot, "GenPaso_Master_DB"),
                    file_name="GenPaso_Master_DB.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
