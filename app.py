"""
GenPaso OCR v6 — Persistencia robusta con Excel + CSV de respaldo,
                  restauración manual y reset con confirmación.

NOTA TÉCNICA: En Streamlit Community Cloud el filesystem es efímero y puede
resetearse en reinicios o redeploys. Esta versión mitiga el problema con:
  1. Guardado inmediato en Excel + CSV después de cada archivo procesado.
  2. Restauración manual subiendo el Excel descargado previamente.
  3. Backups automáticos antes de resetear.
Para persistencia 100% permanente se recomienda integrar una DB externa:
  - Supabase / PostgreSQL
  - Google Drive API
  - AWS S3
  - Firebase Firestore
"""

import base64
import json
import os
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path

import anthropic
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ── Rutas ──────────────────────────────────────────────────────────────────────

BASE_DIR         = Path(__file__).parent
MASTER_XLSX_PATH = BASE_DIR / "GenPaso_Master_DB.xlsx"
MASTER_CSV_PATH  = BASE_DIR / "GenPaso_Master_DB.csv"
BACKUPS_DIR      = BASE_DIR / "backups"
LOGO_TYPO        = BASE_DIR / "images" / "LogoTypoTrans.png"
LOGO_ICON        = BASE_DIR / "images" / "LogoTrans1.png"

COLUMNAS_MASTER = [
    "horse_id", "registration_number", "Horse_Chip",
    "name", "gender", "sire_id", "dam_id",
    "Gait", "color", "issuing_association_id",
    "date_of_birth", "registration_date", "breeder",
    "owner", "place_of_birth", "markings",
]

CAMPOS_EDITABLES = [
    "registration_number", "Horse_Chip", "name", "gender",
    "sire_id", "dam_id", "Gait", "color", "issuing_association_id",
    "date_of_birth", "registration_date", "breeder",
    "owner", "place_of_birth", "markings",
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
    "issuing_association_id": "<asociación o entidad que emite el registro, o null>",
    "date_of_birth": "<fecha de nacimiento (DOB / Date of Birth / Foaling Date / Date Foaled / Nacimiento). Formato preferido MM-DD-YYYY. Si no es posible normalizar, conservar texto original. null si no aparece>",
    "registration_date": "<fecha de registro (Registration Date / Date Registered / Fecha de registro / Registered on). Mismo formato de fecha. null si no aparece>",
    "breeder": "<criador o criadero (Breeder / Bred by / Criador). Si hay varios, separar por coma. null si no aparece>",
    "owner": "<dueño o propietario actual (Owner / Current Owner / Propietario / Dueño). Si hay varios, separar por coma. null si no aparece>",
    "place_of_birth": "<lugar de nacimiento (POB / Place of Birth / Lugar de nacimiento / Born in / país o finca de nacimiento). null si no aparece>",
    "markings": "<marcas o señas distintivas (Markings / White markings / Señales / Señas particulares / Marcas / Descripción de marcas). Conservar descripción completa. null si no aparece>"
  },
  "ancestors": [
    {
      "relationship": "<sire | dam | paternal_grandsire | paternal_granddam | maternal_grandsire | maternal_granddam>",
      "name": "<nombre completo. Conserva títulos: FC, CH, etc.>",
      "registration_number": "<número de registro, o null>",
      "gender": "<Semental | Yegua | Castrado, o null>",
      "Gait": "<modalidad, o null>",
      "issuing_association_id": "<asociación, o null>",
      "date_of_birth": "<fecha de nacimiento del ancestro si aparece, o null>",
      "registration_date": "<fecha de registro del ancestro si aparece, o null>",
      "breeder": "<criador del ancestro si aparece, o null>",
      "owner": "<propietario del ancestro si aparece, o null>",
      "place_of_birth": "<lugar de nacimiento del ancestro si aparece, o null>",
      "markings": "<marcas del ancestro si aparece, o null>"
    }
  ]
}

Reglas críticas:
- "gender" solo acepta: Semental, Yegua, Castrado.
- Conserva títulos en nombres de ancestros (FC = Fuera de Concurso, CH = Champion, etc.).
- Es crítico identificar el Gait (Modalidad) y la Asociación de cada ejemplar.
- No inventes criador, dueño, lugar de nacimiento ni marcas. Si no aparece claramente, usa null.
- Si un ancestro no aparece en el documento, no lo incluyas en la lista.
- Solo devuelve el JSON, sin texto adicional ni bloques de código markdown.
"""

# ══════════════════════════════════════════════════════════════════════════════
# LOGIN
# ══════════════════════════════════════════════════════════════════════════════

def obtener_usuarios() -> dict:
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
                st.session_state["autenticado"]    = True
                st.session_state["usuario_activo"] = usuario
                st.rerun()
            else:
                st.error("❌ Usuario o contraseña incorrectos.")


def verificar_autenticacion():
    if not st.session_state.get("autenticado", False):
        pantalla_login()
        st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# DB MAESTRA — PERSISTENCIA
# ══════════════════════════════════════════════════════════════════════════════

def _es_vacio(valor) -> bool:
    return pd.isna(valor) or str(valor).strip() in ("", "None", "nan", "NaN")


def normalizar_master_db(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garantiza columnas correctas, orden, sin duplicados exactos y sin nan sueltos.
    """
    # Agregar columnas faltantes
    for col in COLUMNAS_MASTER:
        if col not in df.columns:
            df[col] = None

    # Mantener columnas extra que puedan existir y no estén en COLUMNAS_MASTER
    cols_extra = [c for c in df.columns if c not in COLUMNAS_MASTER]
    df = df[COLUMNAS_MASTER + cols_extra]

    # Convertir todo a string limpio, reemplazar nan/None por vacío
    df = df.astype(str)
    df = df.replace({"nan": "", "None": "", "NaN": "", "<NA>": ""})

    # Eliminar duplicados exactos
    df = df.drop_duplicates()

    return df.reset_index(drop=True)


def cargar_master_db() -> pd.DataFrame:
    """
    Orden de prioridad:
      1. GenPaso_Master_DB.xlsx  (fuente principal)
      2. GenPaso_Master_DB.csv   (migrar y generar xlsx)
      3. DataFrame vacío
    """
    if MASTER_XLSX_PATH.exists():
        try:
            df = pd.read_excel(MASTER_XLSX_PATH, dtype=str)
            return normalizar_master_db(df)
        except Exception:
            pass  # Si el Excel está corrupto, intentar CSV

    if MASTER_CSV_PATH.exists():
        try:
            df = pd.read_csv(MASTER_CSV_PATH, dtype=str)
            df = normalizar_master_db(df)
            guardar_master_db(df)  # Migrar a Excel
            return df
        except Exception:
            pass

    return pd.DataFrame(columns=COLUMNAS_MASTER)


def guardar_master_db(df: pd.DataFrame):
    """Guarda en Excel (fuente principal) y CSV (respaldo)."""
    df = normalizar_master_db(df)
    df.to_excel(MASTER_XLSX_PATH, index=False, engine="openpyxl")
    df.to_csv(MASTER_CSV_PATH, index=False)


def backup_master_db() -> Path | None:
    """Crea backup con timestamp antes de operaciones destructivas."""
    if not MASTER_XLSX_PATH.exists():
        return None
    BACKUPS_DIR.mkdir(exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = BACKUPS_DIR / f"GenPaso_Master_DB_backup_{ts}.xlsx"
    import shutil
    shutil.copy2(MASTER_XLSX_PATH, dest)
    return dest


def hay_db_persistente() -> bool:
    return MASTER_XLSX_PATH.exists() or MASTER_CSV_PATH.exists()


# ── Correcciones manuales ──────────────────────────────────────────────────────

def guardar_correcciones_en_master(df_editado: pd.DataFrame) -> tuple[int, list]:
    db           = cargar_master_db()
    actualizados = 0
    advertencias = []

    for _, fila in df_editado.iterrows():
        horse_id = fila.get("horse_id")
        if _es_vacio(horse_id):
            advertencias.append("Fila sin horse_id ignorada.")
            continue
        idx = db.index[db["horse_id"] == str(horse_id)].tolist()
        if not idx:
            advertencias.append(f"horse_id {str(horse_id)[:8]}... no encontrado — no se creó duplicado.")
            continue
        i = idx[0]
        for col in CAMPOS_EDITABLES:
            if col in df_editado.columns:
                db.at[i, col] = fila[col]
        actualizados += 1

    guardar_master_db(db)
    st.session_state.db_master_snapshot = db
    return actualizados, advertencias


# ── Anti-duplicados ────────────────────────────────────────────────────────────

def buscar_caballo(db: pd.DataFrame, registration_number=None, chip=None, nombre=None) -> dict:
    if registration_number and not _es_vacio(registration_number):
        mask = (
            db["registration_number"].fillna("").str.strip().str.upper()
            == str(registration_number).strip().upper()
        )
        if mask.any():
            return {"tipo": "fuerte", "horse_id": db.loc[mask, "horse_id"].iloc[0],
                    "motivo": "registration_number", "coincidencias": []}

    if chip and not _es_vacio(chip):
        mask = db["Horse_Chip"].fillna("").str.strip() == str(chip).strip()
        if mask.any():
            return {"tipo": "fuerte", "horse_id": db.loc[mask, "horse_id"].iloc[0],
                    "motivo": "Horse_Chip", "coincidencias": []}

    if nombre and not _es_vacio(nombre):
        mask = (
            db["name"].fillna("").str.strip().str.upper()
            == str(nombre).strip().upper()
        )
        if mask.any():
            return {"tipo": "probable", "horse_id": db.loc[mask, "horse_id"].iloc[0],
                    "motivo": "name", "coincidencias": db[mask][COLUMNAS_MASTER].to_dict("records")}

    return {"tipo": "no_encontrado", "horse_id": None, "motivo": None, "coincidencias": []}


def actualizar_campos_vacios(db: pd.DataFrame, horse_id: str, nuevos_datos: dict) -> pd.DataFrame:
    idx = db.index[db["horse_id"] == horse_id].tolist()
    if not idx:
        return db
    i = idx[0]
    for col, valor in nuevos_datos.items():
        if col in db.columns and col != "horse_id" and not _es_vacio(valor):
            if _es_vacio(db.at[i, col]):
                db.at[i, col] = valor
    return db


def actualizar_parentesco_si_vacio(
    db: pd.DataFrame, horse_id: str, sire_id=None, dam_id=None
) -> pd.DataFrame:
    idx = db.index[db["horse_id"] == horse_id].tolist()
    if not idx:
        return db
    i = idx[0]
    if sire_id and _es_vacio(db.at[i, "sire_id"]):
        db.at[i, "sire_id"] = sire_id
    if dam_id and _es_vacio(db.at[i, "dam_id"]):
        db.at[i, "dam_id"] = dam_id
    return db


def insertar_o_actualizar(
    db: pd.DataFrame, datos: dict, duplicados_probables: list | None = None
) -> tuple[pd.DataFrame, str, str]:
    resultado = buscar_caballo(
        db,
        registration_number=datos.get("registration_number"),
        chip=datos.get("Horse_Chip"),
        nombre=datos.get("name"),
    )

    if resultado["tipo"] == "fuerte":
        db = actualizar_campos_vacios(db, resultado["horse_id"], datos)
        return db, resultado["horse_id"], "vinculado"

    horse_id = str(uuid.uuid4())
    fila = {col: datos.get(col) for col in COLUMNAS_MASTER}
    fila["horse_id"] = horse_id
    db = pd.concat([db, pd.DataFrame([fila])], ignore_index=True)

    if resultado["tipo"] == "probable" and duplicados_probables is not None:
        duplicados_probables.append({
            "nombre":             datos.get("name"),
            "horse_id_nuevo":     horse_id,
            "coincidencia_id":    resultado["horse_id"],
            "coincidencia_datos": resultado["coincidencias"][0] if resultado["coincidencias"] else {},
            "motivo":             "Coincidencia solo por nombre",
        })
        return db, horse_id, "probable"

    return db, horse_id, "nuevo"


# ══════════════════════════════════════════════════════════════════════════════
# API CLAUDE
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# PROCESAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

def procesar_archivo(
    archivo_bytes: bytes, media_type: str, db: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    resultado = llamar_api(archivo_bytes, media_type)
    main      = resultado.get("main", {})
    ancestors = resultado.get("ancestors", [])

    resumen = {
        "nombre": main.get("name", "Desconocido"),
        "nuevos": 0, "duplicados": 0, "ancestros_vinculados": 0,
        "duplicados_probables": 0,
        "ancestros": [],
        "duplicados_probables_lista": [],
    }

    ids_por_relacion: dict[str, str] = {}

    for anc in ancestors:
        datos_anc = {
            "name":                   anc.get("name"),
            "registration_number":    anc.get("registration_number"),
            "Horse_Chip":             None,
            "gender":                 anc.get("gender"),
            "Gait":                   anc.get("Gait"),
            "issuing_association_id": anc.get("issuing_association_id"),
            "date_of_birth":          anc.get("date_of_birth"),
            "registration_date":      anc.get("registration_date"),
            "breeder":                anc.get("breeder"),
            "owner":                  anc.get("owner"),
            "place_of_birth":         anc.get("place_of_birth"),
            "markings":               anc.get("markings"),
        }
        db, anc_id, accion = insertar_o_actualizar(db, datos_anc, resumen["duplicados_probables_lista"])
        rel = anc.get("relationship", "")
        ids_por_relacion[rel] = anc_id
        resumen["ancestros"].append({"nombre": anc.get("name"), "relacion": rel, "accion": accion, "horse_id": anc_id})
        if accion == "nuevo":
            resumen["nuevos"] += 1
        elif accion == "probable":
            resumen["nuevos"] += 1
            resumen["duplicados_probables"] += 1
        else:
            resumen["ancestros_vinculados"] += 1

    if "sire" in ids_por_relacion:
        db = actualizar_parentesco_si_vacio(
            db, ids_por_relacion["sire"],
            sire_id=ids_por_relacion.get("paternal_grandsire"),
            dam_id=ids_por_relacion.get("paternal_granddam"),
        )
    if "dam" in ids_por_relacion:
        db = actualizar_parentesco_si_vacio(
            db, ids_por_relacion["dam"],
            sire_id=ids_por_relacion.get("maternal_grandsire"),
            dam_id=ids_por_relacion.get("maternal_granddam"),
        )

    datos_main = {
        "registration_number":    main.get("registration_number"),
        "Horse_Chip":             main.get("Horse_Chip"),
        "name":                   main.get("name"),
        "gender":                 main.get("gender"),
        "Gait":                   main.get("Gait"),
        "color":                  main.get("color"),
        "issuing_association_id": main.get("issuing_association_id"),
        "sire_id":                ids_por_relacion.get("sire"),
        "dam_id":                 ids_por_relacion.get("dam"),
        "date_of_birth":          main.get("date_of_birth"),
        "registration_date":      main.get("registration_date"),
        "breeder":                main.get("breeder"),
        "owner":                  main.get("owner"),
        "place_of_birth":         main.get("place_of_birth"),
        "markings":               main.get("markings"),
    }
    db, main_id, accion = insertar_o_actualizar(db, datos_main, resumen["duplicados_probables_lista"])
    resumen["horse_id"] = main_id
    resumen["accion"]   = accion
    if accion == "nuevo":
        resumen["nuevos"] += 1
    elif accion == "probable":
        resumen["nuevos"] += 1
        resumen["duplicados_probables"] += 1
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

verificar_autenticacion()

# ── Header ─────────────────────────────────────────────────────────────────────
col_logo, col_titulo = st.columns([1, 4], gap="medium")
with col_logo:
    if LOGO_TYPO.exists():
        st.image(str(LOGO_TYPO), use_container_width=True)
with col_titulo:
    st.markdown("## Ingesta Masiva de Registros Equinos")
    st.caption(f"Base de datos genética GenPaso · Usuario: **{st.session_state.get('usuario_activo', '')}**")

st.divider()

# ── Estado de sesión ───────────────────────────────────────────────────────────
for key, val in [("resumenes", None), ("db_sesion", None), ("db_master_snapshot", None)]:
    if key not in st.session_state:
        st.session_state[key] = val

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    if LOGO_ICON.exists():
        st.markdown("<style>[data-testid='stSidebar'] img { image-rendering: high-quality; }</style>", unsafe_allow_html=True)
        st.image(str(LOGO_ICON), width=180)

    st.divider()

    # ── REQ 7: Advertencia si no hay DB persistente ────────────────────────────
    if not hay_db_persistente():
        st.warning(
            "No se encontró DB Maestra persistente. Se iniciará una DB en blanco. "
            "Si tienes un respaldo Excel, súbelo en **Restaurar DB Maestra** ↓"
        )

    # ── Estadísticas DB Maestra ────────────────────────────────────────────────
    db_live    = cargar_master_db()
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
        st.metric("Duplicados prevenidos", sum(r.get("duplicados", 0)           for r in exitosos))
        st.metric("Ancestros vinculados",  sum(r.get("ancestros_vinculados", 0) for r in exitosos))
        st.metric("⚠️ Duplicados probables", sum(r.get("duplicados_probables", 0) for r in exitosos))

    st.divider()

    # ── REQ 6: Descargas de DB Maestra ────────────────────────────────────────
    st.markdown("### ⬇️ Descargar DB Maestra")
    if MASTER_XLSX_PATH.exists():
        st.download_button(
            "📊 Descargar Excel (.xlsx)",
            data=MASTER_XLSX_PATH.read_bytes(),
            file_name="GenPaso_Master_DB.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    if MASTER_CSV_PATH.exists():
        st.download_button(
            "📄 Descargar CSV (.csv)",
            data=MASTER_CSV_PATH.read_bytes(),
            file_name="GenPaso_Master_DB.csv",
            mime="text/csv",
            use_container_width=True,
        )
    if not MASTER_XLSX_PATH.exists() and not MASTER_CSV_PATH.exists():
        st.caption("Aún no hay DB guardada.")

    st.divider()

    # ── REQ 5: Restaurar DB desde archivo ─────────────────────────────────────
    st.markdown("### 📂 Restaurar DB Maestra")
    st.caption("Sube el Excel o CSV descargado previamente para restaurar la DB.")
    archivo_restaurar = st.file_uploader(
        "Subir DB Maestra",
        type=["xlsx", "csv"],
        key="uploader_restaurar",
        label_visibility="collapsed",
    )
    if archivo_restaurar:
        if st.button("✅ Confirmar restauración", use_container_width=True, type="primary"):
            try:
                if archivo_restaurar.name.endswith(".xlsx"):
                    df_rest = pd.read_excel(archivo_restaurar, dtype=str)
                else:
                    df_rest = pd.read_csv(archivo_restaurar, dtype=str)

                if "horse_id" not in df_rest.columns or "name" not in df_rest.columns:
                    st.error("El archivo no tiene las columnas mínimas requeridas (horse_id, name).")
                else:
                    guardar_master_db(df_rest)
                    st.success(f"✅ DB restaurada con {len(df_rest)} registros.")
                    st.rerun()
            except Exception as e:
                st.error(f"Error al restaurar: {e}")

    st.divider()

    # ── REQ 8: Reset con confirmación y backup automático ─────────────────────
    st.markdown("### ⚠️ Resetear DB Maestra")
    confirmar_reset = st.checkbox("Confirmo que deseo borrar la DB Maestra actual")
    if st.button("⚠️ Resetear DB Maestra", use_container_width=True, type="secondary", disabled=not confirmar_reset):
        backup = backup_master_db()
        if MASTER_XLSX_PATH.exists():
            MASTER_XLSX_PATH.unlink()
        if MASTER_CSV_PATH.exists():
            MASTER_CSV_PATH.unlink()
        if backup:
            st.success(f"DB reseteada. Backup guardado en: `{backup.name}`")
        else:
            st.success("DB reseteada. No había archivo previo.")
        st.rerun()

    st.divider()
    if st.button("🚪 Cerrar sesión", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# CUERPO PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

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

# ── REQ 4: Procesamiento con guardado inmediato por archivo ───────────────────
if procesar and archivos:
    db               = cargar_master_db()  # Siempre carga desde archivo persistente
    resumenes        = []
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

            # Guardado inmediato tras cada archivo exitoso
            guardar_master_db(db)

            with log_container:
                if resumen["accion"] == "nuevo":
                    icono = "🆕 NUEVO"
                elif resumen["accion"] == "probable":
                    icono = "⚠️ NUEVO (duplicado probable)"
                else:
                    icono = "🔗 ACTUALIZADO"
                st.success(
                    f"✅ **{resumen['nombre']}** — {icono} | "
                    f"{len(resumen['ancestros'])} ancestros | "
                    f"DB Maestra: {len(db)} registros guardados"
                )

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

    barra.progress(1.0, text=f"✅ Procesamiento completo — DB Maestra guardada con {len(db)} registros.")

    st.session_state.resumenes          = resumenes
    st.session_state.db_sesion          = pd.concat(registros_sesion, ignore_index=True) if registros_sesion else pd.DataFrame(columns=COLUMNAS_MASTER)
    st.session_state.db_master_snapshot = db
    st.rerun()

# ── Resultados ─────────────────────────────────────────────────────────────────
if st.session_state.resumenes:
    st.divider()
    st.subheader("2. Resumen de Procesamiento")

    exitosos         = [r for r in st.session_state.resumenes if not r.get("error")]
    total_nuevos     = sum(r.get("nuevos", 0)                for r in exitosos)
    total_duplicados = sum(r.get("duplicados", 0)            for r in exitosos)
    total_anc_vinc   = sum(r.get("ancestros_vinculados", 0)  for r in exitosos)
    total_ancestros  = sum(len(r.get("ancestros", []))        for r in exitosos)
    total_prob       = sum(r.get("duplicados_probables", 0)  for r in exitosos)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Registros en DB Maestra",   len(cargar_master_db()))
    c2.metric("Perfiles nuevos creados",    total_nuevos)
    c3.metric("Duplicados prevenidos",      total_duplicados)
    c4.metric("Ancestros vinculados auto.", total_anc_vinc)
    c5.metric("⚠️ Duplicados probables",    total_prob)

    st.info(
        f"Se procesaron **{len(st.session_state.resumenes)}** archivos · "
        f"**{total_nuevos}** perfiles nuevos · "
        f"**{total_duplicados}** duplicados prevenidos · "
        f"**{total_ancestros}** ancestros detectados ({total_anc_vinc} ya existían) · "
        f"**{total_prob}** coincidencias por nombre pendientes de revisión"
    )

    with st.expander("Ver detalle de ancestros y vínculos genealógicos"):
        for r in exitosos:
            if r.get("ancestros"):
                st.markdown(f"**{r['nombre']}** (`{r.get('horse_id','')[:8]}...`)")
                for anc in r["ancestros"]:
                    icono = "🆕" if anc["accion"] == "nuevo" else ("⚠️" if anc["accion"] == "probable" else "🔗")
                    st.markdown(f"  {icono} `{anc['relacion']}` — {anc['nombre']} · `{anc['horse_id'][:8]}...`")

    # ── Revisión de duplicados probables ──────────────────────────────────────
    todos_probables = []
    for r in exitosos:
        for dp in r.get("duplicados_probables_lista", []):
            dp["archivo"] = r.get("archivo", "")
            todos_probables.append(dp)

    if todos_probables:
        st.divider()
        st.subheader("⚠️ Revisión de Duplicados Probables")
        st.warning(
            f"Se detectaron **{len(todos_probables)}** registro(s) que coinciden solo por **nombre** "
            "con entradas existentes. Fueron insertados como nuevos. Revisa antes de continuar."
        )
        for dp in todos_probables:
            with st.expander(f"🐴 {dp['nombre']} — {dp['archivo']}"):
                col_n, col_e = st.columns(2)
                with col_n:
                    st.markdown("**Registro NUEVO insertado**")
                    st.code(dp["horse_id_nuevo"])
                with col_e:
                    st.markdown("**Registro EXISTENTE en DB**")
                    datos_ex = dp.get("coincidencia_datos", {})
                    st.code(dp.get("coincidencia_id", ""))
                    for campo in ["registration_number", "Horse_Chip", "gender", "Gait",
                                  "issuing_association_id", "date_of_birth", "registration_date",
                                  "breeder", "owner", "place_of_birth", "markings"]:
                        v = datos_ex.get(campo, "")
                        st.caption(f"`{campo}`: {v if not _es_vacio(v) else '—'}")
                st.info(
                    "Si son el mismo animal: edita el campo `registration_number` o `Horse_Chip` "
                    "en la tabla de sesión y guarda en DB Maestra. La próxima carga lo reconocerá automáticamente."
                )

    # ── Editor de sesión con botón guardar ────────────────────────────────────
    st.divider()
    st.subheader("3. Datos de Esta Sesión")
    df_sesion = st.session_state.db_sesion
    if not df_sesion.empty:
        df_editado = st.data_editor(
            df_sesion,
            use_container_width=True,
            num_rows="fixed",
            key="editor_sesion",
            column_config={"horse_id": st.column_config.TextColumn("horse_id", disabled=True)},
        )

        if st.button("💾 Guardar correcciones en DB Maestra", type="primary", use_container_width=True):
            actualizados, advertencias = guardar_correcciones_en_master(df_editado)
            if actualizados:
                st.success(f"✅ {actualizados} registro(s) actualizados en DB Maestra.")
            for adv in advertencias:
                st.warning(f"⚠️ {adv}")
            if not actualizados and not advertencias:
                st.info("No hubo cambios que guardar.")

        st.divider()
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
            db_completa = cargar_master_db()
            st.download_button(
                "⬇️ DB Maestra completa (.xlsx)",
                data=df_a_excel(db_completa, "GenPaso_Master_DB"),
                file_name="GenPaso_Master_DB.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
