import streamlit as st
import os
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Literal, List, Dict, TypedDict, Optional
from dotenv import load_dotenv

# Importar Groq y Gemini
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import START, END, StateGraph

# (Código para compatibilidad de create_stuff_documents_chain)
try:
    from langchain.chains.combine_documents import create_stuff_documents_chain
except Exception:
    def create_stuff_documents_chain(llm, prompt):
        class SimpleDocumentChain:
            def __init__(self, llm, prompt):
                self.llm = llm
                self.prompt = prompt

            def invoke(self, inputs):
                input_text = inputs.get("input", "")
                context_docs = inputs.get("context", []) or []
                context = "\n\n".join(getattr(d, "page_content", str(d)) for d in context_docs)
                system = "Eres el asistente de Inteligencia Artificial de Mercado Central 24h. Responde utilizando ÚNICAMENTE la información del contexto. Si no lo sabes, responde 'No lo sé'."
                human = f"Contexto: {context}.\nPregunta: {input_text}"
                try:
                    resp = self.llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
                    return resp.content if hasattr(resp, "content") else str(resp)
                except Exception:
                    return ""
        return SimpleDocumentChain(llm, prompt)

# ==========================================
# 1. CONFIGURACIÓN INICIAL DE STREAMLIT
# ==========================================
st.set_page_config(page_title="Agente Corporativo - Mercado Central 24h", page_icon="🛒", layout="wide")

# --- NUEVA INTERFAZ: BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1008/1008014.png", width=80) # Icono de carrito
    st.title("🏢 Mercado Central 24h")
    st.markdown("""
    **Supermercado moderno de operación continua (24/7)** que integra la experiencia de tienda física con servicios de delivery y app propia. 
    Su enfoque principal es la eficiencia operativa en la gestión de stock y una fuerte política de atención al cliente.
    """)
    
    st.divider()
    
    st.subheader("🧠 Sobre este Asistente")
    st.markdown("""
    Este chat utiliza **RAG (Generación Aumentada por Recuperación)**. Es una IA que lee en tiempo real la documentación interna de la empresa para brindarte respuestas precisas y automatizadas, mejorando drásticamente la eficiencia en el acceso a la información.
    """)
    
    st.divider()
    
    st.subheader("📄 Base de Conocimiento")
    st.markdown("Puedes descargar y consultar los documentos en los que se basa esta IA:")
    
    # Generar botones de descarga dinámicos si los archivos existen
    archivos = [
        "Política de Atención al Cliente y Devoluciones — Mercado Central 24h.pdf",
        "Preguntas Frecuentes (FAQ) — Mercado Central 24h.pdf",
        "Reglamento Interno y Procedimientos Operativos — Mercado Central 24h.pdf",
        "Manual de Proveedores y Política de Compras — Mercado Central 24h.pdf",
        "inventario_de_supermercado_latam.xlsx"
    ]
    
    for archivo in archivos:
        ruta_archivo = Path("documentos") / archivo
        if ruta_archivo.exists():
            with open(ruta_archivo, "rb") as f:
                st.download_button(
                    label=f"⬇️ {archivo[:25]}...", 
                    data=f, 
                    file_name=archivo,
                    help=f"Descargar {archivo}"
                )
        else:
            st.caption(f"📄 {archivo} (No disponible)")

    st.divider()
    
    st.subheader("💡 Ejemplos de Preguntas")
    st.info("""
    - ¿A cuántos días de vacaciones tengo derecho si llevo 3 años?
    - ¿Cuál es el proceso para devolver un producto lácteo?
    - ¿Cuál es el plazo de pago para proveedores Categoría A?
    - Necesito pedir una excepción médica para el uniforme.
    """)

# --- INTERFAZ PRINCIPAL ---
st.title("🛒 Asistente IA - Mercado Central 24h")

# Cargar API Keys desde el archivo .env
load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GEMINI_API_KEY or not GROQ_API_KEY:
    st.error("⚠️ Faltan las claves GEMINI_API_KEY o GROQ_API_KEY en tu archivo .env.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ==========================================
# 2. INICIALIZAR MODELOS
# ==========================================
@st.cache_resource
def load_models():
    # --- CORRECCIÓN: Usamos el nuevo modelo Llama 3.1 de Groq ---
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    # Mantenemos los embeddings de Google para leer tu base de datos FAISS
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    return llm, embeddings

llm, modelo_embeddings = load_models()

# ==========================================
# 3. CARGAR BASE VECTORIAL YA CONSTRUIDA
# ==========================================
@st.cache_resource
def load_vector_store():
    ruta_indice = "indice_faiss_mercado"
    if not os.path.exists(ruta_indice):
        return None
    return FAISS.load_local(ruta_indice, modelo_embeddings, allow_dangerous_deserialization=True)

vectorstore = load_vector_store()
if vectorstore is None:
    st.warning("⚠️ No se encontró la base de conocimiento. Por favor, ejecuta primero el script `python construir_indice.py` en tu terminal.")
    st.stop()

retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.3, "k": 4})

# ==========================================
# 4. FUNCIONES DEL AGENTE (TRIAJE Y RAG)
# ==========================================
PROMPT_TRIAJE = """
Eres un especialista en triaje del Service Desk de Mercado Central 24h.
Dado el mensaje del usuario, devuelve SÓLO un JSON con:
{
 "decision": "AUTO_RESOLVER" | "PEDIR_INFO" | "ABRIR_TICKET",
 "urgencia": "BAJA" | "MEDIANA" | "ALTA",
 "campos_faltantes": ["..."]
}
Reglas:
- AUTO_RESOLVER: Preguntas claras sobre políticas, vacaciones, devoluciones, inventario, etc.
- PEDIR_INFO: Mensajes vagos (Ej.: "Necesito ayuda", "Tengo una duda").
- ABRIR_TICKET: Quejas graves, excepciones, permisos especiales o incidentes.
"""

class TriajeOut(BaseModel):
    decision: Literal["AUTO_RESOLVER", "PEDIR_INFO", "ABRIR_TICKET"]
    urgencia: Literal["BAJA", "MEDIANA", "ALTA"]
    campos_faltantes: List[str] = Field(default_factory=list)

chain_de_triaje = llm.with_structured_output(TriajeOut)

def realizar_triaje(mensaje: str) -> dict:
    salida = chain_de_triaje.invoke([SystemMessage(content=PROMPT_TRIAJE), HumanMessage(content=mensaje)])
    return salida.model_dump()

prompt_rag = ChatPromptTemplate([
    ("system", "Eres el asistente de Inteligencia Artificial de Mercado Central 24h. Responde utilizando ÚNICAMENTE la información del contexto. Si no lo sabes, responde 'No lo sé'."),
    ("human", "Contexto: {context}.\nPregunta: {input}")
])
document_chain = create_stuff_documents_chain(llm, prompt_rag)

def buscar_respuesta_rag(pregunta: str) -> dict:
    docs_relacionados = retriever.invoke(pregunta)
    if not docs_relacionados:
        return {"respuesta": "No lo sé.", "citaciones": [], "documentos_encontrados": False}

    answer = document_chain.invoke({"input": pregunta, "context": docs_relacionados})
    if answer.strip().rstrip(".!?") == 'No lo sé':
        return {"respuesta": "No lo sé.", "citaciones": [], "documentos_encontrados": False}

    return {"respuesta": answer, "citaciones": docs_relacionados, "documentos_encontrados": True}

# ==========================================
# 5. GRAFO DEL AGENTE (LANGGRAPH)
# ==========================================
class AgentState(TypedDict, total=False):
    pregunta: str
    triaje: dict
    respuesta: Optional[str]
    citaciones: Optional[list]
    rag_exito: bool
    accion_final: str

def nodo_triaje(state: AgentState):
    return {"triaje": realizar_triaje(state["pregunta"])}

def nodo_auto_resolver(state: AgentState):
    res_rag = buscar_respuesta_rag(state["pregunta"])
    update = {"respuesta": res_rag["respuesta"], "citaciones": res_rag["citaciones"], "rag_exito": res_rag["documentos_encontrados"]}
    if res_rag["documentos_encontrados"]:
        update["accion_final"] = "AUTO_RESOLVER"
    return update

def nodo_pedir_info(state: AgentState):
    return {"respuesta": "Para poder ayudarte mejor, ¿podrías darme un poco más de detalles sobre tu consulta?", "citaciones": [], "accion_final": "PEDIR_INFO"}

def nodo_abrir_ticket(state: AgentState):
    tri = state["triaje"]
    return {"respuesta": f"He registrado tu solicitud como un **Ticket de Soporte** (Urgencia: {tri['urgencia']}). Un supervisor se pondrá en contacto pronto.", "citaciones": [], "accion_final": "ABRIR_TICKET"}

def arista_decision_triaje(state: AgentState) -> str:
    dec = state["triaje"]["decision"]
    if dec == "AUTO_RESOLVER": return "rag"
    elif dec == "PEDIR_INFO": return "info"
    else: return "ticket"

def arista_decision_rag(state: AgentState) -> str:
    if state["rag_exito"]: return "ok"
    palabras_clave = ["aprobación", "excepción", "permiso", "ticket", "problema"]
    if any(k in state["pregunta"].lower() for k in palabras_clave): return "ticket"
    return "info"

workflow = StateGraph(AgentState)
workflow.add_node("triaje", nodo_triaje)
workflow.add_node("auto_resolver", nodo_auto_resolver)
workflow.add_node("pedir_info", nodo_pedir_info)
workflow.add_node("abrir_ticket", nodo_abrir_ticket)

workflow.add_edge(START, "triaje")
workflow.add_conditional_edges("triaje", arista_decision_triaje, {"rag": "auto_resolver", "info": "pedir_info", "ticket": "abrir_ticket"})
workflow.add_conditional_edges("auto_resolver", arista_decision_rag, {"info": "pedir_info", "ticket": "abrir_ticket", "ok": END})
workflow.add_edge("pedir_info", END)
workflow.add_edge("abrir_ticket", END)

agente = workflow.compile()

# ==========================================
# 6. INTERFAZ DE USUARIO (STREAMLIT CHAT)
# ==========================================
st.markdown("¡Hola! Hazme cualquier consulta sobre el reglamento, atención al cliente, compras o recursos humanos y buscaré la respuesta en nuestra base de datos institucional.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Escribe tu pregunta aquí... (Ej. ¿A cuántos días de vacaciones tengo derecho?)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Buscando en la base de conocimiento..."):
            resultado = agente.invoke({"pregunta": prompt})
            respuesta_texto = resultado["respuesta"]
            
            if resultado.get("citaciones"):
                respuesta_texto += "\n\n**Fuentes consultadas:**\n"
                fuentes_unicas = list(set([doc.metadata.get("file_path", "Documento") for doc in resultado["citaciones"]]))
                for fuente in fuentes_unicas:
                    nombre_archivo = Path(fuente).name
                    respuesta_texto += f"- *{nombre_archivo}*\n"

            st.markdown(respuesta_texto)
    st.session_state.messages.append({"role": "assistant", "content": respuesta_texto})