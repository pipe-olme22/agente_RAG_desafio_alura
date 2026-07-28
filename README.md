# Agente Corporativo - Mercado Central 24h

## Descripción general

Este proyecto implementa un agente de asistencia corporativa para Mercado Central 24h usando una interfaz de chat en Streamlit. El agente responde preguntas sobre políticas internas, atención al cliente, devoluciones, procedimientos operativos, gestión de proveedores y consultas de inventario, basándose en una base de conocimiento construida a partir de documentos internos.

La solución combina:
- un modelo de lenguaje Gemini a través de `langchain-google-genai`,
- un índice vectorial FAISS para recuperación de información,
- y un flujo de decisión con `langgraph` para clasificar consultas en triage, respuesta automática o apertura de ticket.

## Arquitectura de la solución

1. `app.py`
   - Carga la clave de API de Gemini desde `.env` o `.streamlit/secrets.toml`.
   - Inicializa el modelo de chat y el modelo de embeddings de Google Gemini.
   - Carga el índice FAISS local (`indice_faiss_mercado/`) que contiene los vectores de los documentos.
   - Utiliza un grafo de estados (`StateGraph`) para decidir si la consulta se resuelve con RAG, se pide más información o se marca como ticket.
   - Usa un prompt de triage estructurado para clasificar la intención y urgencia.
   - Recupera documentos relevantes del índice y genera la respuesta con contexto.

2. `construir_indice.py`
   - Convierte los documentos PDF y XLSX de la carpeta `documentos/` en texto.
   - Fragmenta el texto en trozos grandes adecuados para embeddings.
   - Genera embeddings con Gemini y construye un índice FAISS.
   - Guarda el índice en `indice_faiss_mercado/`.

3. `documentos/`
   - Contiene los archivos fuente que alimentan la base de conocimiento del agente.
   - Ejemplos incluidos: políticas internas, FAQ, manuales de proveedores y un Excel de inventario.

4. `indice_faiss_mercado/`
   - Carpeta donde se almacena el índice vectorial local que utiliza el agente para búsquedas semánticas.

## Tecnologías y herramientas utilizadas

- Python 3.14
- Streamlit
- LangChain
- LangGraph
- FAISS
- Google Gemini Developer API
- python-dotenv
- PyMuPDF
- pandas
- openpyxl
- langchain-google-genai
- langchain-community

## Instrucciones para ejecutar el proyecto

1. Clona el repositorio y entra en la carpeta del proyecto.

2. Crea y activa un entorno virtual:

```powershell
python -m venv venv
.\venv\Scripts\activate
```

3. Instala las dependencias:

```powershell
pip install -r requirements.txt
```

4. Añade tu clave de API de Gemini en un archivo `.env` en la raíz del proyecto o en `.streamlit/secrets.toml`.

Ejemplo de `.env`:

```env
GEMINI_API_KEY="tu_api_key_aqui"
GEMINI_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
```

Ejemplo de `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "tu_api_key_aqui"
```

5. Si no dispones del índice FAISS, ejecútalo una vez para construirlo:

```powershell
python construir_indice.py
```

6. Inicia la aplicación Streamlit:

```powershell
streamlit run app.py
```

7. Abre el navegador en `http://localhost:8501`.

## Ejemplos de preguntas que el agente puede responder

- ¿Cuál es la política de devoluciones de Mercado Central 24h?
- ¿Cómo se registra un nuevo proveedor?
- ¿Qué debo hacer si un cliente solicita una excepción en el horario de entrega?
- ¿Cuál es el procedimiento interno para abrir un ticket de soporte?
- ¿Cuáles son los criterios de urgencia para las incidencias?
- ¿Cómo se gestiona el inventario de productos frescos?

## Ejemplos de respuestas generadas por el agente

- "Según la política de atención al cliente, las devoluciones se pueden tramitar dentro de los 30 días siguientes a la compra y deben conservar el comprobante de pago."
- "Para registrar un proveedor se requiere su documentación fiscal, datos de contacto y la aprobación del área de compras según el Manual de Proveedores."
- "El sistema clasifica como alta urgencia las incidencias que afectan la seguridad, el acceso a zonas restringidas o el proceso de facturación."
- "Si la consulta es ambigua, se solicita información adicional al usuario antes de continuar con el soporte."

## Notas

- No incluyas tu clave real de API en el repositorio.
- Si el modelo configurado no está disponible para tu cuenta, cambia `GEMINI_MODEL` en `.env` a un modelo compatible como `gemini-2.5-flash`, `gemini-2.5-pro` o `gemini-2.0-flash`.
- Asegúrate de ejecutar `python construir_indice.py` cada vez que actualices los documentos de `documentos/`.
