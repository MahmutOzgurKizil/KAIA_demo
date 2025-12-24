from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware
import os
import httpx
import chromadb
from chromadb.config import Settings
from typing import List, Optional
import asyncio
from collections import defaultdict
import logging
import secrets

# Logging Setup 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("course-rag")

# Configuration 
PERSONAS = {
    "tutor": "You are a patient course tutor. Explain step-by-step, use simple language, and reference course materials when relevant.",
    "examiner": "You are a strict examiner. Provide concise answers and ask clarifying follow-ups aligned with course objectives.",
    "coach": "You are a motivational study coach. Encourage, summarize key points, and suggest next actions."
}

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# Data Models 
class ChatRequest(BaseModel):
    model: str = "llama3.2"
    prompt: str
    stream: bool = False
    persona: Optional[str] = None
    session_id: Optional[str] = None
    use_rag: bool = False
    rag_top_k: int = 4
    course_id: Optional[int] = None

class HistoryItem(BaseModel):
    role: str
    content: str

# App Setup 
app = FastAPI(title="Course RAG API")

# Add session middleware
# IMPORTANT: The session key should be a composite of session_id and course_id
# to ensure chat histories are isolated per course.
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SESSION_SECRET", secrets.token_hex(32)))

# Setup templates
templates = Jinja2Templates(directory="templates")

if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

session_histories: defaultdict[str, List[HistoryItem]] = defaultdict(list)

# ChromaDB Setup 
chroma_client = chromadb.Client(Settings(anonymized_telemetry=False, is_persistent=True, persist_directory="./chroma_db"))

# Helper function to get course-specific collection
def get_course_collection(course_id: int):
    collection_name = f"course-{course_id}-rag"
    return chroma_client.get_or_create_collection(name=collection_name)

# Course objectives storage (simple file-based storage)
import json
OBJECTIVES_FILE = "./course_objectives.json"

def load_objectives():
    if os.path.exists(OBJECTIVES_FILE):
        with open(OBJECTIVES_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_objectives(objectives):
    with open(OBJECTIVES_FILE, 'w') as f:
        json.dump(objectives, f, indent=2)

def get_course_objectives(course_id: int):
    objectives = load_objectives()
    return objectives.get(str(course_id), "")

def set_course_objectives(course_id: int, text: str):
    objectives = load_objectives()
    objectives[str(course_id)] = text
    save_objectives(objectives)

# Sample Courses Data
COURSES = [
    {
        "id": 1,
        "name": "Introduction to Machine Learning",
        "code": "CS 4301",
        "instructor": "Dr. Sarah Johnson",
        "students": 156,
        "progress": 75,
        "color": "bg-[#003366]",
        "modules": 12,
        "assignments": 8,
    },
    {
        "id": 2,
        "name": "Deep Learning & Neural Networks",
        "code": "CS 4302",
        "instructor": "Prof. Michael Chen",
        "students": 142,
        "progress": 60,
        "color": "bg-[#FF6B35]",
        "modules": 10,
        "assignments": 6,
    },
    {
        "id": 3,
        "name": "Natural Language Processing",
        "code": "CS 4303",
        "instructor": "Dr. Emily Rodriguez",
        "students": 128,
        "progress": 45,
        "color": "bg-[#005588]",
        "modules": 14,
        "assignments": 10,
    },
    {
        "id": 4,
        "name": "Computer Vision",
        "code": "CS 4304",
        "instructor": "Prof. David Kim",
        "students": 134,
        "progress": 80,
        "color": "bg-[#FF8C42]",
        "modules": 11,
        "assignments": 7,
    },
    {
        "id": 5,
        "name": "Reinforcement Learning",
        "code": "CS 4305",
        "instructor": "Dr. Amanda Lee",
        "students": 98,
        "progress": 30,
        "color": "bg-[#004477]",
        "modules": 13,
        "assignments": 9,
    },
    {
        "id": 6,
        "name": "AI Ethics & Society",
        "code": "CS 4306",
        "instructor": "Prof. James Wilson",
        "students": 201,
        "progress": 90,
        "color": "bg-[#FF9F55]",
        "modules": 8,
        "assignments": 5,
    },
]

# Helper Functions 
async def embed_texts(texts: List[str]) -> List[List[float]]:
    # Low concurrency to prevent Ollama from hanging
    concurrency = int(os.getenv("EMBED_CONCURRENCY", "2"))
    sem = asyncio.Semaphore(concurrency)
    
    async def embed_one(client, text, idx):
        async with sem:
            try:
                r = await client.post(
                    f"{OLLAMA_HOST}/api/embeddings",
                    json={"model": EMBED_MODEL, "prompt": text[:1500]},
                    timeout=120.0
                )
                r.raise_for_status()
                return r.json().get("embedding", [])
            except Exception as e:
                logger.error(f"Embed error chunk {idx}: {e}")
                return []

    async with httpx.AsyncClient(timeout=120.0) as client:
        tasks = [embed_one(client, t, i) for i, t in enumerate(texts)]
        return await asyncio.gather(*tasks)

# Web Endpoints

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    if request.session.get("user"):
        return RedirectResponse(url="/dashboard", status_code=302)
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(request: Request, email: str = Form(...), password: str = Form(...), user_type: str = Form(...)):
    # Simple authentication (you should add proper authentication in production)
    request.session["user"] = {"email": email, "type": user_type}
    request.session["session_id"] = secrets.token_hex(8)
    return RedirectResponse(url="/dashboard", status_code=302)

@app.post("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/", status_code=302)

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    if not request.session.get("user"):
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse("dashboard.html", {"request": request, "active_tab": "dashboard"})

@app.get("/courses", response_class=HTMLResponse)
async def courses(request: Request):
    if not request.session.get("user"):
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse("courses.html", {"request": request, "active_tab": "courses", "courses": COURSES})

@app.get("/course/{course_id}/chatbot", response_class=HTMLResponse)
async def course_chatbot(request: Request, course_id: int):
    if not request.session.get("user"):
        return RedirectResponse(url="/", status_code=302)
    
    course = next((c for c in COURSES if c["id"] == course_id), None)
    if not course:
        return RedirectResponse(url="/courses", status_code=302)
    
    session_id = request.session.get("session_id", "user-1")
    user_type = request.session.get("user", {}).get("type", "student")
    
    # Get documents for this course
    collection = get_course_collection(course_id)
    doc_count = collection.count()
    
    # Get course objectives
    objectives = get_course_objectives(course_id)
    
    return templates.TemplateResponse("chatbot.html", {
        "request": request,
        "course": course,
        "session_id": session_id,
        "user_type": user_type,
        "doc_count": doc_count,
        "course_id": course_id,
        "objectives": objectives
    })

@app.get("/calendar", response_class=HTMLResponse)
async def calendar(request: Request):
    if not request.session.get("user"):
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse("dashboard.html", {"request": request, "active_tab": "calendar"})

@app.get("/assignments", response_class=HTMLResponse)
async def assignments(request: Request):
    if not request.session.get("user"):
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse("dashboard.html", {"request": request, "active_tab": "assignments"})

@app.get("/grades", response_class=HTMLResponse)
async def grades(request: Request):
    if not request.session.get("user"):
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse("dashboard.html", {"request": request, "active_tab": "grades"})

@app.get("/course/{course_id}/documents", response_class=HTMLResponse)
async def course_documents(request: Request, course_id: int):
    if not request.session.get("user"):
        return RedirectResponse(url="/", status_code=302)
    
    course = next((c for c in COURSES if c["id"] == course_id), None)
    if not course:
        return RedirectResponse(url="/courses", status_code=302)
    
    user_type = request.session.get("user", {}).get("type", "student")
    
    # Get documents metadata
    collection = get_course_collection(course_id)
    try:
        results = collection.get()
        documents_meta = []
        seen_sources = set()
        
        for metadata in results.get("metadatas", []):
            source = metadata.get("source", "Unknown")
            if source not in seen_sources:
                seen_sources.add(source)
                documents_meta.append({
                    "name": source,
                    "chunks": sum(1 for m in results.get("metadatas", []) if m.get("source") == source)
                })
    except:
        documents_meta = []
    
    return templates.TemplateResponse("documents.html", {
        "request": request,
        "course": course,
        "user_type": user_type,
        "documents": documents_meta
    })

@app.get("/course/{course_id}/materials")
async def get_course_materials(course_id: int):
    """API endpoint to get course materials as JSON"""
    try:
        collection = get_course_collection(course_id)
        results = collection.get()
        documents_meta = []
        seen_sources = set()
        
        for metadata in results.get("metadatas", []):
            source = metadata.get("source", "Unknown")
            if source not in seen_sources:
                seen_sources.add(source)
                documents_meta.append({
                    "name": source,
                    "chunks": sum(1 for m in results.get("metadatas", []) if m.get("source") == source)
                })
        
        return {"documents": documents_meta}
    except:
        return {"documents": []}

@app.get("/course/{course_id}/document/{document_name}")
async def get_document_content(course_id: int, document_name: str):
    """Get the full content of a document"""
    try:
        collection = get_course_collection(course_id)
        results = collection.get()
        
        # Collect all chunks for this document
        chunks = []
        for i, metadata in enumerate(results.get("metadatas", [])):
            if metadata.get("source") == document_name:
                chunks.append({
                    "chunk_id": metadata.get("chunk", 0),
                    "content": results.get("documents", [])[i]
                })
        
        # Sort by chunk ID
        chunks.sort(key=lambda x: x["chunk_id"])
        full_content = "\n".join([c["content"] for c in chunks])
        
        return {"name": document_name, "content": full_content}
    except Exception as e:
        return {"error": str(e)}

@app.get("/course/{course_id}/objectives")
async def get_objectives(course_id: int):
    """Get course objectives"""
    return {"objectives": get_course_objectives(course_id)}

@app.post("/course/{course_id}/objectives")
async def update_objectives(request: Request, course_id: int, objectives: str = Form(...)):
    """Update course objectives (instructor only)"""
    user_type = request.session.get("user", {}).get("type", "student")
    if user_type != "instructor":
        return {"error": "Only instructors can update objectives"}
    
    set_course_objectives(course_id, objectives)
    return {"success": True, "objectives": objectives}

@app.get("/demo")
async def demo():
    return FileResponse("static/index.html")

# API Endpoints

@app.get("/health")
async def health():
    return {"status": "ok", "ollama": OLLAMA_HOST}

@app.post("/chat")
async def chat(req: ChatRequest):
    async with httpx.AsyncClient(timeout=180.0) as client:
        rag_contexts = []
        rag_sources = []
        
        # 1. RAG Retrieval (course-specific)
        if req.use_rag:
            logger.info("Embedding prompt for RAG...")
            query_vecs = await embed_texts([req.prompt])
            if query_vecs and query_vecs[0]:
                # Use course_id from request
                if req.course_id:
                    collection = get_course_collection(req.course_id)
                else:
                    collection = chroma_client.get_or_create_collection(name="course-rag")
                
                try:
                    results = collection.query(query_embeddings=query_vecs, n_results=req.rag_top_k)
                    rag_contexts = results.get("documents", [[]])[0]
                    rag_sources = results.get("metadatas", [[]])[0]
                except Exception as e:
                    logger.error(f"RAG query error: {e}")
                    rag_contexts = []
                    rag_sources = []

        # 2. Construct Prompt
        context_block = "\n\n".join([f"[Source {i}] {c}" for i, c in enumerate(rag_contexts, start=1)]) if rag_contexts else ""
        
        # Get course objectives if course_id is provided
        objectives_block = ""
        if req.course_id:
            objectives = get_course_objectives(req.course_id)
            if objectives:
                objectives_block = f"\n\nCourse Objectives:\n{objectives}"
        
        messages = []
        
        sys_msg = PERSONAS.get(req.persona or "", "")
        if sys_msg or context_block or objectives_block:
            content = sys_msg
            if objectives_block:
                content += objectives_block
            if context_block:
                content += f"\n\nCourse Materials:\n{context_block}\nCite sources as [Source #]."
            messages.append({"role": "system", "content": content})

        if req.session_id and req.course_id:
            history_key = f"{req.session_id}-{req.course_id}"
            for h in session_histories.get(history_key, [])[-10:]:
                messages.append({"role": h.role, "content": h.content})

        messages.append({"role": "user", "content": req.prompt})

        # 3. Chat Generation
        try:
            resp = await client.post(f"{OLLAMA_HOST}/api/chat", json={
                "model": req.model,
                "messages": messages,
                "stream": False,
                "options": {"num_ctx": 4096}
            })
            resp.raise_for_status()
            answer = resp.json().get("message", {}).get("content", "")
        except Exception as e:
            return {"response": f"Error: {str(e)}", "sources": []}

        # 4. Save History (with course-specific key)
        if req.session_id and req.course_id:
            history_key = f"{req.session_id}-{req.course_id}"
            session_histories[history_key].append(HistoryItem(role="user", content=req.prompt))
            session_histories[history_key].append(HistoryItem(role="assistant", content=answer))

        return {"response": answer, "sources": rag_sources}

@app.get("/history/{course_id}/{session_id}")
async def get_history(course_id: int, session_id: str):
    """Gets chat history for a specific course and session."""
    history_key = f"{session_id}-{course_id}"
    return {"history": session_histories.get(history_key, [])}

@app.post("/course/{course_id}/ingest")
async def ingest_course_documents(course_id: int, files: List[UploadFile] = File(...)):
    try:
        docs, metadatas, ids = [], [], []
        
        for f in files:
            raw = await f.read()
            content = raw.decode("utf-8", errors="ignore")
            chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
            
            for idx, ch in enumerate(chunks):
                if len(ch.strip()) > 10:
                    docs.append(ch)
                    metadatas.append({"source": f.filename, "chunk": idx})
                    ids.append(f"{course_id}-{f.filename}-{idx}")

        if not docs:
            return {"ingested": 0}

        logger.info(f"Ingesting {len(docs)} chunks for course {course_id}...")
        vectors = await embed_texts(docs)
        
        # Filter failed embeddings
        valid = [(d, m, i, v) for d, m, i, v in zip(docs, metadatas, ids, vectors) if v]
        
        if valid:
            collection = get_course_collection(course_id)
            collection.add(
                documents=[x[0] for x in valid],
                metadatas=[x[1] for x in valid],
                ids=[x[2] for x in valid],
                embeddings=[x[3] for x in valid]
            )

        return {"ingested": len(valid), "skipped": len(docs) - len(valid)}
    except Exception as e:
        return {"error": str(e)}

@app.post("/course/{course_id}/rag/clear")
async def rag_clear_course(course_id: int):
    """Clears the RAG database for a specific course."""
    try:
        collection_name = f"course-{course_id}-rag"
        chroma_client.delete_collection(name=collection_name)
        # Recreate it empty
        chroma_client.get_or_create_collection(name=collection_name)
        return {"status": "cleared"}
    except Exception as e:
        # It might fail if the collection doesn't exist, which is fine.
        logger.warning(f"Could not clear collection for course {course_id}: {e}")
        return {"status": "error", "detail": str(e)}
