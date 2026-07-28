import os
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

print("=== INICIANDO CONSTRUCTOR DE BASE DE CONOCIMIENTO ===")

# 1. Cargar API KEY
load_dotenv()
try:
    import toml
    secrets = toml.load(".streamlit/secrets.toml")
    API_KEY = secrets.get("GEMINI_API_KEY")
except Exception:
    API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    print("❌ ERROR: No se encontró la GEMINI_API_KEY en .streamlit/secrets.toml ni en el entorno.")
    exit()

os.environ["GOOGLE_API_KEY"] = API_KEY
modelo_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# 2. Cargar Documentos
docs = []
ruta_documentos = Path("./documentos")
if not ruta_documentos.exists():
    print("❌ ERROR: No existe la carpeta 'documentos'")
    exit()

print("Cargando PDFs y Excels...")
for documento in ruta_documentos.glob("*.pdf"):
    try:
        loader = PyMuPDFLoader(str(documento))
        docs.extend(loader.load())
        print(f"✔️ Cargado: {documento.name}")
    except Exception as e:
        print(f"❌ Error en {documento.name}: {e}")
        
for documento in ruta_documentos.glob("*.xlsx"):
    try:
        df = pd.read_excel(documento)
        text_content = df.to_string(index=False)
        from langchain_core.documents import Document
        docs.append(Document(page_content=text_content, metadata={"file_path": str(documento)}))
        print(f"✔️ Cargado: {documento.name}")
    except Exception as e:
        print(f"❌ Error en {documento.name}: {e}")

# 3. Fragmentar (TAMAÑO MUCHO MÁS GRANDE para evitar el límite de la API)
splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
chunks = splitter.split_documents(docs)
total_chunks = len(chunks)
print(f"\nTotal de fragmentos a procesar: {total_chunks}")

# 4. Procesar y guardar en lotes
vectorstore = None
tamaño_lote = 5 

for i in range(0, total_chunks, tamaño_lote):
    lote = chunks[i : i + tamaño_lote]
    print(f"Procesando lote {i} a {min(i+tamaño_lote, total_chunks)} de {total_chunks}...")
    
    intentos = 0
    while intentos < 3:
        try:
            if vectorstore is None:
                vectorstore = FAISS.from_documents(lote, modelo_embeddings)
            else:
                vectorstore.add_documents(lote)
            break # Salió bien, rompemos el bucle while
        except Exception as e:
            intentos += 1
            print(f"⚠️ Google pidió pausa. Reintentando en 30 segundos... (Intento {intentos}/3)")
            time.sleep(30)
            
    time.sleep(5) # Pausa segura entre lotes exitosos

# 5. Guardar
ruta_indice = "indice_faiss_mercado"
if vectorstore is not None:
    vectorstore.save_local(ruta_indice)
    print(f"\n✅ ¡ÉXITO! Base de conocimiento guardada en la carpeta '{ruta_indice}'.")
    print("Ya puedes ejecutar 'streamlit run app.py'")