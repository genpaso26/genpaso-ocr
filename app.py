"""
GenPaso OCR v3 — Ingesta masiva con login, anti-duplicados y árbol genealógico
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

# ── Rutas y constantes ─────────────────────────────────────────────────────────

MASTER_DB_PATH = Path(__file__).parent / "GenPaso_Master_DB.csv"
LOGO_TYPO      = Path(__file__).parent / "images" / "LogoTypoTrans.png"
LOGO_ICON      = Path(__file__).parent / "images" / "LogoTrans1.png"

# Columnas exactas según horses.csv de GenPaso
COLUMNAS_MASTER = [
    "horse_id", "registration_number", "Horse_Chip",
    "name", "gender", "sire_id", "dam_id",
    "Gait", "color", "issuing_association_id",
]

PROMPT_EXTRACCION = """Analiza este registro de caballo. Identifica al ejemplar principal y todos sus ancestros mencionados.
Devuelve EXCLUSIVAMENTE un objeto JSON con esta estructura:

{
  "main": {
    "registration_number": "<número de registro oficial, o null>",
    "Horse_Chip": "<número de microchip, o null>",
    "name": "<nombre completo, respetando mayúsculas>",
    "gender": "<Semental | Yegua | Castrado>",
    "Gait": "<modalidad: Paso Fino, Paso Colombiano, Trote, etc., o null>",
    "color": "<capa del caballo, o null>",
    "issuing_association_id": "<asociación o entidad que emite el registro, o null>"
  },
  "ancestors": [
    {
      "relationship": "<sire | dam | paternal_grandsire | paternal_granddam | maternal_grandsire | maternal_granddam>",
      "name": "<nombre completo. Conserva títulos: FC, CH, etc.>",
      "registration_number": "<número de registro, o null>",
      "gender": "<Semental | Yegua | Castrado, o null>",
      "Gait": "<modalidad, o null>",
      "issuing_association_id": "<asociación, o null>"
    }
  ]
}

Reglas críticas:
- "gender" solo acepta: Semental, Yegua, Castrado.
- Conserva títulos en nombres de ancestros (FC = Fuera de Concurso, CH = Champion, etc.).
- Es crítico identificar el Gait (Modalidad) y la Asociación de cada ejemplar para la pureza de la DB de GenPaso.
- Si un ancestro no aparece en el documento, no lo incluyas.
- Solo devuelve el JSON, sin texto adicional ni bloques de código markdown.
"""

# ── Login ──────────────────────────────────────────────────────────────────────

def obtener_usuarios() -> dict:
    """Carga usuarios desde st.secrets o .env como fallback."""
    try:
        creds = st.secrets["credentials"]
        return {k: v for k, v in creds.items()}
    except Exception:
        user = os.getenv("LOGIN_USER", "")
        pwd  = os.getenv("LOGIN_PASS", "")
        return {user: pwd} if user else {}


def pantalla_login():
    col_l, col_c, col_r = st.columns([1, 1.5, 1])
    with col_c:
        if LOGO_TYPO.exists():
            st.image(str(LOGO_TYPO), use_container_width=True)
        st.markdown("### 🔐 Acceso Restringido")
        st.caption("Solo personal autorizado de GenPaso")
        with st.form("login_form"):
            usuario  = st.text_input("Usuario", placeholder="admin_genpaso")
            password = st.text_input("Contraseña", type="password")
            entrar   = st.form_submit_button("Ingresar", use_container_width=True, type="primary")

        if entrar:
            usuarios = obtener_usuarios()
            if usuario in usuarios and usuarios[usuario] == password:
                st.session_state["autenticado"] = True
                st.session_state["usuario_activo"] = usuario
                st.rerun()
            else:
                st.error("❌ Usuario o contraseña incorrectos.")


def verificar_autenticacion():
    if not st.session_state.get("autenticado", False):
        pantalla_login()
        st.stop()


# ── DB Maestra ─────────────────────────────────────────────────────────────────

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


def _es_vacio(valor) -> bool:
    return pd.isna(valor) or str(valor).strip() in ("", "None", "nan", "NaN")


def buscar_caballo(db: pd.DataFrame, registration_number=None, chip=None, nombre=None) -> str | None:
    """Retorna horse_id si el caballo ya existe en la DB."""
    if registration_number and not _es_vacio(registration_number):
        mask = db["registration_number"].str.strip().str.upper() == str(registration_number).strip().upper()
        if mask.any():
            return db.loc[mask, "horse_id"].iloc[0]
    if chip and not _es_vacio(chip):
        mask = db["Horse_Chip"].str.strip() == str(chip).strip()
        if mask.any():
            return db.loc[mask, "horse_id"].iloc[0]
    if nombre and not _es_vacio(nombre):
        mask = db["name"].str.strip().str.upper() == str(nombre).strip().upper()
        if mask.any():
            return db.loc[mask, "horse_id"].iloc[0]
    return None


def actualizar_campos_vacios(db: pd.DataFrame, horse_id: str, nuevos_datos: dict) -> pd.DataFrame:
    """Rellena campos vacíos de un registro existente con nueva información del OCR."""
    idx = db.index[db["horse_id"] == horse_id].tolist()
    if not idx:
        return db
    i = idx[0]
    for col, valor in nuevos_datos.items():
        if col in db.columns and col != "horse_id" and not _es_vacio(valor):
            if _es_vacio(db.at[i, col]):
                db.at[i, col] = valor
    return db


def insertar_o_actualizar(db: pd.DataFrame, datos: dict) -> tuple[pd.DataFrame, str, str]:
    """
    Busca el caballo. Si existe, actualiza campos vacíos (no duplica).
    Si no existe, crea fila nueva.
    Retorna (db, horse_id, accion) donde accion es 'nuevo' | 'actualizado' | 'vinculado'.
    """
    horse_id = buscar_caballo(
        db,
        registration_number=datos.get("registration_number"),
        chip=datos.get("Horse_Chip"),
        nombre=datos.get("name"),
    )
    if horse_id:
        db = actualizar_campos_vacios(db, horse_id, datos)
        return db, horse_id, "vinculado"

    horse_id = str(uuid.uuid4())
    fila = {col: datos.get(col) for col in COLUMNAS_MASTER}
    fila["horse_id"] = horse_id
    db = pd.concat([db, pd.DataFrame([fila])], ignore_index=True)
    return db, horse_id, "nuevo"


# ── API Claude ─────────────────────────────────────────────────────────────────

def obtener_api_key() -> str:
    try:
        return st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        return os.getenv("ANTHROPIC_API_KEY", "")


def obtener_cliente() -> anthropic.Anthropic:
    api_key = obtener_api_key()
    if not api_key:
        st.error("No se encontró ANTHROPIC_API_KEY. Configúrala en Secrets o .env")
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


# ── Procesamiento ──────────────────────────────────────────────────────────────

def procesar_archivo(archivo_bytes: bytes, media_type: str, db: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    resultado = llamar_api(archivo_bytes, media_type)
    main      = resultado.get("main", {})
    ancestors = resultado.get("ancestors", [])

    resumen = {
        "nombre": main.get("name", "Desconocido"),
        "nuevos": 0, "duplicados": 0, "ancestros_vinculados": 0,
        "ancestros": [],
    }

    sire_id = dam_id = None

    # 1. Procesar ancestros primero
    for anc in ancestors:
        datos_anc = {
            "name":                  anc.get("name"),
            "registration_number":   anc.get("registration_number"),
            "Horse_Chip":            None,
            "gender":                anc.get("gender"),
            "Gait":                  anc.get("Gait"),
            "issuing_association_id": anc.get("issuing_association_id"),
        }
        db, anc_id, accion = insertar_o_actualizar(db, datos_anc)

        rel = anc.get("relationship", "")
        if rel == "sire":
            sire_id = anc_id
        elif rel == "dam":
            dam_id = anc_id

        resumen["ancestros"].append({"nombre": anc.get("name"), "relacion": rel, "accion": accion, "horse_id": anc_id})
        if accion == "nuevo":
            resumen["nuevos"] += 1
        else:
            resumen["ancestros_vinculados"] += 1

    # 2. Procesar caballo principal
    datos_main = {
        "registration_number":   main.get("registration_number"),
        "Horse_Chip":            main.get("Horse_Chip"),
        "name":                  main.get("name"),
        "gender":                main.get("gender"),
        "Gait":                  main.get("Gait"),
        "color":                 main.get("color"),
        "issuing_association_id": main.get("issuing_association_id"),
        "sire_id":               sire_id,
        "dam_id":                dam_id,
    }
    db, main_id, accion = insertar_o_actualizar(db, datos_main)
    resumen["horse_id"] = main_id
    resumen["accion"]   = accion
    if accion == "nuevo":
        resumen["nuevos"] += 1
    else:
        resumen["duplicados"] += 1

    return db, resumen


# ── Exportación ────────────────────────────────────────────────────────────────

def df_a_excel(df: pd.DataFrame, sheet_name: str = "GenPaso") -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return buffer.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# APP PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="GenPaso OCR",
    page_icon=str(LOGO_ICON) if LOGO_ICON.exists() else "🐴",
    layout="wide",
)

# ── Verificar login ────────────────────────────────────────────────────────────
verificar_autenticacion()

# ── Header ────────────────────────────────────────────────────────────────────
col_logo, col_titulo = st.columns([1, 4], gap="medium")
with col_logo:
    if LOGO_TYPO.exists():
        st.image(str(LOGO_TYPO), use_container_width=True)
with col_titulo:
    usuario_activo = st.session_state.get("usuario_activo", "")
    st.markdown("## Ingesta Masiva de Registros Equinos")
    st.caption(f"Base de datos genética GenPaso · Usuario: **{usuario_activo}**")

st.divider()

# ── Estado de sesión ───────────────────────────────────────────────────────────
for key, val in [("resumenes", None), ("db_sesion", None), ("db_master_snapshot", None)]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    if LOGO_ICON.exists():
        st.markdown("<style>[data-testid='stSidebar'] img { image-rendering: high-quality; }</style>", unsafe_allow_html=True)
        st.image(str(LOGO_ICON), width=180)

    st.divider()

    # Estadísticas DB Maestra
    db_live = cargar_master_db()
    total_db   = len(db_live)
    sementales = len(db_live[db_live["gender"] == "Semental"]) if total_db else 0
    yeguas     = len(db_live[db_live["gender"] == "Yegua"])    if total_db else 0

    st.markdown("### 📊 DB Maestra")
    st.metric("Registros totales", total_db)
    c1, c2 = st.columns(2)
    c1.metric("Sementales", sementales)
    c2.metric("Yeguas", yeguas)

    if st.session_state.resumenes:
        exitosos = [r for r in st.session_state.resumenes if not r.get("error")]
        st.metric("Duplicados prevenidos", sum(r.get("duplicados", 0) for r in exitosos))
        st.metric("Ancestros vinculados",  sum(r.get("ancestros_vinculados", 0) for r in exitosos))

    st.divider()
    if MASTER_DB_PATH.exists():
        st.download_button(
            "⬇️ DB Maestra (.csv)",
            data=MASTER_DB_PATH.read_bytes(),
            file_name="GenPaso_Master_DB.csv",
            mime="text/csv",
            use_container_width=True,
        )
        if st.button("🗑️ Resetear DB Maestra", use_container_width=True, type="secondary"):
            MASTER_DB_PATH.unlink()
            st.rerun()

    st.divider()
    if st.button("🚪 Cerrar sesión", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# ── Carga de archivos ──────────────────────────────────────────────────────────
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

# ── Procesamiento en lote ──────────────────────────────────────────────────────
if procesar and archivos:
    db        = cargar_master_db()
    resumenes = []
    registros_sesion = []

    barra         = st.progress(0, text="Iniciando procesamiento...")
    log_container = st.container()

    for i, archivo in enumerate(archivos):
        barra.progress(i / len(archivos), text=f"Procesando {archivo.name} ({i+1}/{len(archivos)})...")
        try:
            db, resumen = procesar_archivo(archivo.read(), archivo.type, db)
            resumen["archivo"] = archivo.name
            resumen["error"]   = None
            resumenes.append(resumen)
            registros_sesion.append(db[db["horse_id"] == resumen["horse_id"]].copy())

            with log_container:
                accion_txt = "🆕 NUEVO" if resumen["accion"] == "nuevo" else "🔗 ACTUALIZADO"
                st.success(f"✅ **{resumen['nombre']}** — {accion_txt} | {len(resumen['ancestros'])} ancestros")

        except json.JSONDecodeError:
            resumenes.append({"archivo": archivo.name, "error": "JSON inválido"})
            with log_container:
                st.warning(f"⚠️ {archivo.name}: imagen poco legible.")
        except anthropic.BadRequestError:
            resumenes.append({"archivo": archivo.name, "error": "Archivo no compatible"})
            with log_container:
                st.warning(f"⚠️ {archivo.name}: formato no compatible con la API.")
        except Exception as e:
            resumenes.append({"archivo": archivo.name, "error": str(e)})
            with log_container:
                st.error(f"❌ {archivo.name}: {e}")

    barra.progress(1.0, text="✅ Procesamiento completo. DB Maestra guardada.")
    guardar_master_db(db)  # Guardado automático tras cada proceso exitoso

    st.session_state.resumenes          = resumenes
    st.session_state.db_sesion          = pd.concat(registros_sesion, ignore_index=True) if registros_sesion else pd.DataFrame(columns=COLUMNAS_MASTER)
    st.session_state.db_master_snapshot = db
    st.rerun()

# ── Resultados ─────────────────────────────────────────────────────────────────
if st.session_state.resumenes:
    st.divider()
    st.subheader("2. Resumen de Procesamiento")

    exitosos           = [r for r in st.session_state.resumenes if not r.get("error")]
    total_nuevos       = sum(r.get("nuevos", 0)               for r in exitosos)
    total_duplicados   = sum(r.get("duplicados", 0)            for r in exitosos)
    total_anc_vinc     = sum(r.get("ancestros_vinculados", 0)  for r in exitosos)
    total_ancestros    = sum(len(r.get("ancestros", []))        for r in exitosos)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros en DB Maestra",    len(cargar_master_db()))
    c2.metric("Perfiles nuevos creados",     total_nuevos)
    c3.metric("Duplicados prevenidos",       total_duplicados)
    c4.metric("Ancestros vinculados auto.",  total_anc_vinc)

    st.info(
        f"Se procesaron **{len(st.session_state.resumenes)}** archivos · "
        f"**{total_nuevos}** perfiles nuevos · "
        f"**{total_duplicados}** duplicados prevenidos · "
        f"**{total_ancestros}** ancestros detectados ({total_anc_vinc} ya existían en la DB)"
    )

    with st.expander("Ver detalle de ancestros"):
        for r in exitosos:
            if r.get("ancestros"):
                st.markdown(f"**{r['nombre']}**")
                for anc in r["ancestros"]:
                    icono = "🆕" if anc["accion"] == "nuevo" else "🔗"
                    st.markdown(f"  {icono} `{anc['relacion']}` — {anc['nombre']} · `{anc['horse_id'][:8]}...`")

    st.divider()
    st.subheader("3. Datos de Esta Sesión")
    df_sesion = st.session_state.db_sesion
    if not df_sesion.empty:
        df_editado = st.data_editor(df_sesion, use_container_width=True, num_rows="fixed", key="editor_sesion")

        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button(
                "⬇️ Sesión actual (.xlsx)",
                data=df_a_excel(df_editado, "Sesion_Actual"),
                file_name="sesion_actual.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with col_b:
            if st.session_state.db_master_snapshot is not None:
                st.download_button(
                    "⬇️ DB Maestra completa (.xlsx)",
                    data=df_a_excel(st.session_state.db_master_snapshot, "GenPaso_Master_DB"),
                    file_name="GenPaso_Master_DB.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
