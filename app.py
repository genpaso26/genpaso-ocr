"""
GenPaso OCR — Extractor inteligente de registros equinos
Usa Claude Sonnet para analizar imágenes/PDFs y devolver datos estructurados.
"""

import base64
import json
import os
import uuid
from io import BytesIO

import anthropic
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Configuración ──────────────────────────────────────────────────────────────

CAMPOS_GENPASO = [
    "horse_id",
    "registration_number",
    "Horse_Chip",
    "name",
    "gender",
    "birth_date",
    "Gait",
    "color",
    "height",
    "sire_name",
    "dam_name",
    "country_origin",
    "issuing_association_id",
]

PROMPT_EXTRACCION = """Analiza este registro de caballo y extrae la información solicitada.
Devuelve EXCLUSIVAMENTE un objeto JSON válido con exactamente estas claves:

{
  "registration_number": "<número de registro oficial del caballo, o null>",
  "Horse_Chip": "<número de microchip, o null>",
  "name": "<nombre completo del caballo, respetando mayúsculas>",
  "gender": "<uno de: Semental | Yegua | Castrado>",
  "birth_date": "<fecha de nacimiento en formato YYYY-MM-DD, o null si no está>",
  "Gait": "<modalidad o andadura del caballo, ej. Paso Fino, Trote, etc., o null>",
  "color": "<color o capa del caballo>",
  "height": "<alzada o altura en centímetros o manos, como texto, o null>",
  "sire_name": "<nombre completo del PADRE (sire). Conserva títulos como FC, CH, etc.>",
  "dam_name": "<nombre completo de la MADRE (dam). Conserva títulos como FC, CH, etc.>",
  "country_origin": "<código de país ISO de 2 letras (ej. CO, US, PR), o nombre del país>",
  "issuing_association_id": "<nombre o ID de la asociación o entidad que emite el registro, o null>"
}

Reglas críticas:
- El campo "gender" solo puede ser: Semental, Yegua o Castrado.
- Los nombres del padre (sire_name) y la madre (dam_name) son vitales para la red genética.
  Inclúyelos completos con sus títulos (FC = Fuera de Concurso, CH = Champion, etc.).
- Si un campo no aparece en el documento, usa null.
- No incluyas texto adicional. Solo el JSON.
"""

# ── Helpers ────────────────────────────────────────────────────────────────────

def obtener_cliente() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        st.error("❌ No se encontró ANTHROPIC_API_KEY. Revisa tu archivo .env")
        st.stop()
    return anthropic.Anthropic(api_key=api_key)


def construir_contenido(archivo_bytes: bytes, media_type: str) -> list:
    """Construye el bloque de contenido para la API según el tipo de archivo."""
    datos_b64 = base64.standard_b64encode(archivo_bytes).decode("utf-8")

    if media_type == "application/pdf":
        bloque_archivo = {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": datos_b64,
            },
        }
    else:
        bloque_archivo = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": datos_b64,
            },
        }

    return [bloque_archivo, {"type": "text", "text": PROMPT_EXTRACCION}]


def extraer_datos(archivo_bytes: bytes, media_type: str) -> dict:
    """Envía el archivo a Claude y retorna el dict con los datos extraídos."""
    cliente = obtener_cliente()
    contenido = construir_contenido(archivo_bytes, media_type)

    respuesta = cliente.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": contenido}],
    )

    texto = respuesta.content[0].text.strip()

    # Eliminar bloques de código markdown si Claude los añade
    if texto.startswith("```"):
        lineas = texto.splitlines()
        texto = "\n".join(lineas[1:-1]) if lineas[-1] == "```" else "\n".join(lineas[1:])

    datos = json.loads(texto)
    datos["horse_id"] = str(uuid.uuid4())
    return datos


def df_a_excel(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="GenPaso_Registro")
    return buffer.getvalue()


def fila_vacia() -> dict:
    return {campo: None for campo in CAMPOS_GENPASO}


# ── UI ─────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GenPaso OCR",
    page_icon="🐴",
    layout="wide",
)

st.title("🐴 GenPaso — Extractor de Registros Equinos")
st.caption("Carga una imagen o PDF de un registro de caballo para extraer sus datos automáticamente.")

# Estado de sesión
if "datos_extraidos" not in st.session_state:
    st.session_state.datos_extraidos = None
if "df_editable" not in st.session_state:
    st.session_state.df_editable = None

# ── Panel lateral ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuración")
    api_key_input = st.text_input(
        "API Key de Anthropic",
        type="password",
        placeholder="sk-ant-...",
        help="También puedes definirla en el archivo .env como ANTHROPIC_API_KEY",
    )
    if api_key_input:
        os.environ["ANTHROPIC_API_KEY"] = api_key_input

    st.divider()
    st.markdown("**Campos extraídos**")
    for c in CAMPOS_GENPASO:
        st.markdown(f"- `{c}`")

# ── Cuerpo principal ───────────────────────────────────────────────────────────
col_upload, col_preview = st.columns([1, 2], gap="large")

with col_upload:
    st.subheader("1. Cargar Documento")
    archivo = st.file_uploader(
        "Arrastra o selecciona un archivo",
        type=["jpg", "jpeg", "png", "pdf"],
        label_visibility="collapsed",
    )

    if archivo:
        if archivo.type.startswith("image/"):
            st.image(archivo, caption=archivo.name, use_container_width=True)
        else:
            st.info(f"📄 PDF cargado: **{archivo.name}**")

    procesar = st.button(
        "🔍 Procesar Registro",
        disabled=archivo is None,
        use_container_width=True,
        type="primary",
    )

# ── Lógica de extracción ───────────────────────────────────────────────────────
if procesar and archivo:
    with col_preview:
        with st.spinner("Analizando documento con Claude..."):
            try:
                archivo_bytes = archivo.read()
                datos = extraer_datos(archivo_bytes, archivo.type)
                st.session_state.datos_extraidos = datos
                # Asegurar todas las columnas en orden correcto
                fila = fila_vacia()
                fila.update(datos)
                st.session_state.df_editable = pd.DataFrame([fila], columns=CAMPOS_GENPASO)
                st.success("✅ Datos extraídos correctamente.")
            except json.JSONDecodeError:
                st.error(
                    "⚠️ El modelo no devolvió un JSON válido. "
                    "Intenta con una imagen más nítida o un PDF con texto seleccionable."
                )
            except anthropic.BadRequestError:
                st.error(
                    "⚠️ El documento no pudo ser procesado por la API. "
                    "Verifica que la imagen sea legible y esté en un formato compatible."
                )
            except Exception as e:
                st.error(f"❌ Error inesperado: {e}")

# ── Visualización y exportación ────────────────────────────────────────────────
with col_preview:
    if st.session_state.df_editable is not None:
        st.subheader("2. Revisar y Editar Datos")
        st.caption("Puedes editar cualquier celda antes de descargar.")

        df_editado = st.data_editor(
            st.session_state.df_editable,
            use_container_width=True,
            num_rows="fixed",
            key="editor_principal",
        )

        st.divider()
        st.subheader("3. Exportar")

        excel_bytes = df_a_excel(df_editado)
        nombre_caballo = (
            df_editado["name"].iloc[0] if df_editado["name"].iloc[0] else "registro"
        )
        nombre_archivo = f"genpaso_{nombre_caballo.replace(' ', '_')}.xlsx"

        st.download_button(
            label="⬇️ Descargar Excel (.xlsx)",
            data=excel_bytes,
            file_name=nombre_archivo,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

        with st.expander("Ver JSON extraído"):
            st.json(st.session_state.datos_extraidos)
    else:
        st.info("Los datos extraídos aparecerán aquí después de procesar un documento.")
