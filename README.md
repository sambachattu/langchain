# 🦜🔗 LangChain Learning Project

A hands-on project for learning LangChain fundamentals — covering document loading, text splitting, vector stores, retrievers, runnables, chains, and FastAPI integration using OpenAI models.

---

## 📁 Project Structure

```
langchain-main/
├── chat/
│   └── simpleChat.py
├── chains/
│   ├── simpleChain.py
│   ├── sequencialChain.py
│   └── parallelChain.py
├── document_loader/
│   ├── text_loader.py
│   ├── pdf_loader.py
│   ├── directory_loader.py
│   └── textFile.txt
├── text_splitters/
│   ├── length_based.py
│   ├── structure_based.py
│   ├── semantic_based.py
│   └── code_splitter.py
├── vector_stores/
│   └── chroma_vector_store/
│       └── chroma_db.py
├── retrievers/
│   ├── vector_store_retriever.py
│   ├── multi_query_retriever.py
│   ├── contextual_compression_retriever.py
│   ├── max_marginal_retriever.py
│   └── wikipedia_retriever.py
├── runnables/
│   ├── runnable_sequence.py
│   ├── runnable_parallel.py
│   ├── runnable_passthrough.py
│   ├── runnable_lambda.py
│   └── runnable_branch.py
├── FastAPI/
│   └── llm_api.py
└── requirements.txt
```

---

## ⚙️ Setup & Installation

### 1. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\Activate.ps1     # Windows PowerShell
source venv/bin/activate       # Mac/Linux
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Create a `.env` file in the root directory:
```
OPENAI_API_KEY=your_openai_api_key_here
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `langchain` | Core LangChain framework |
| `langchain-core` | Base interfaces and runnables |
| `langchain-community` | Document loaders, retrievers |
| `langchain-openai` | OpenAI LLM and embeddings |
| `langchain-text-splitters` | Text splitting utilities |
| `langchain-experimental` | SemanticChunker |
| `langchain-chroma` | Chroma vector store integration |
| `chromadb` | Chroma database |
| `python-dotenv` | Load `.env` variables |
| `pypdf` | PDF parsing |
| `fastapi` | REST API framework |

---

## 🗂️ Module Notes

---

### 💬 chat/simpleChat.py

> **Concept:** Basic LLM invocation — the simplest way to call an OpenAI model.

```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI()
response = llm.invoke('Explain RAG')
print(response.content)
```

**Key Points:**
- `ChatOpenAI()` uses `gpt-3.5-turbo` by default
- `.invoke()` sends a single message and returns an `AIMessage` object
- Access the text with `.content`

---

### 📄 document_loader/

> **Concept:** Document Loaders read files and convert them into LangChain `Document` objects containing `page_content` and `metadata`.

#### text_loader.py
- Uses `TextLoader` to load a `.txt` file
- Passes the loaded content into a chain that summarizes the poem using a `PromptTemplate`
- Shows how loaders integrate with chains via the `|` pipe operator

#### pdf_loader.py
- Uses `PyPDFLoader` to load a PDF file page by page
- Each page becomes a separate `Document` object
- Prints `metadata` (source, page number) and `page_content`

#### directory_loader.py
- Uses `DirectoryLoader` to load **all PDF files** in the `books/` folder at once
- `glob='*.pdf'` filters only PDF files
- `loader_cls=PyPDFLoader` tells it how to load each file

**Document Object Structure:**
```python
Document(
    page_content="text from the file...",
    metadata={"source": "file.pdf", "page": 0}
)
```

---

### ✂️ text_splitters/

> **Concept:** Text splitters break large documents into smaller chunks for embedding and retrieval. Chunk size and overlap are key parameters.

#### length_based.py — `CharacterTextSplitter`
- Splits text by a fixed number of characters
- `chunk_size=100`, `chunk_overlap=5`
- Loaded a PDF and split it into character-based chunks
- **Use when:** You need simple, fast splitting without caring about sentence/paragraph boundaries

#### structure_based.py — `RecursiveCharacterTextSplitter`
- Tries to split by `\n\n`, then `\n`, then spaces, then characters
- Respects paragraph and sentence structure better than `CharacterTextSplitter`
- `chunk_size=200`, `chunk_overlap=0`
- **Use when:** You want smarter splitting that preserves meaning ✅ (Most recommended)

#### semantic_based.py — `SemanticChunker`
- Uses **OpenAI embeddings** to split by meaning, not by character count
- `breakpoint_threshold_type="standard_deviation"` controls sensitivity
- Groups semantically related sentences into the same chunk
- **Use when:** Chunk quality matters (e.g., RAG pipelines)
- **Note:** Requires an embedding model — costs API calls

#### code_splitter.py — `RecursiveCharacterTextSplitter.from_language()`
- Splits code intelligently using Python-aware parsing
- `Language.PYTHON` ensures it splits at class/function boundaries
- **Use when:** Splitting source code files for code search or analysis

**Comparison Table:**

| Splitter | Splits By | Best For |
|---|---|---|
| `CharacterTextSplitter` | Character count | Simple/fast splitting |
| `RecursiveCharacterTextSplitter` | Structure (paragraphs → sentences → chars) | General text ✅ |
| `SemanticChunker` | Meaning (embeddings) | High-quality RAG |
| `from_language(PYTHON)` | Code structure | Source code |

---

### 🗄️ vector_stores/chroma_vector_store/chroma_db.py

> **Concept:** A vector store stores document embeddings and enables similarity search. Chroma is a local, file-based vector database.

**What this file does:**
- Creates 5 `Document` objects about IPL cricket players with team metadata
- Embeds them using `OpenAIEmbeddings` and stores in a local Chroma DB (`chroma_db/` folder)
- Performs `similarity_search_with_score()` filtered by team name

**Key Operations shown (some commented out):**
```python
# Add documents
vector_store.add_documents(docs)

# Similarity search
vector_store.similarity_search(query='Who is a bowler', k=1)

# Filtered search by metadata
vector_store.similarity_search_with_score(query='', filter={'team': 'Chennai Super Kings'})

# Update a document
vector_store.update_document(document_id='...', document=updated_doc)
```

**Key Point:** `persist_directory='chroma_db'` saves the vector store to disk so it persists between runs.

---

### 🔍 retrievers/

> **Concept:** Retrievers fetch relevant documents from a vector store based on a query. Different retriever types offer different strategies for improving result quality.

#### vector_store_retriever.py
- The most basic retriever — wraps a Chroma vector store
- `as_retriever(search_kwargs={'k': 2})` returns top 2 most similar documents
- **Use when:** Simple semantic search is sufficient

#### multi_query_retriever.py *(incomplete — imports only)*
- Uses an LLM to generate **multiple variations** of the query
- Each variation searches the vector store independently
- Results are combined and deduplicated
- **Use when:** Queries are ambiguous or a single search might miss relevant docs

#### contextual_compression_retriever.py
- **Two-stage retrieval:** first fetch docs, then use LLM to compress them
- `LLMChainExtractor` strips out irrelevant sentences from each retrieved document
- Returns only the parts of documents that are relevant to the query
- Example: searching "What is photosynthesis?" in docs mixed with unrelated content → returns only photosynthesis sentences
- **Use when:** Documents contain mixed topics and you need precision

#### max_marginal_retriever.py — MMR (Maximal Marginal Relevance)
- Balances **relevance** and **diversity** in results
- `lambda_mult=1` → pure relevance (0 = pure diversity, 0.5 = balanced)
- Prevents returning 3 nearly identical documents
- Uses FAISS vector store (in-memory, faster than Chroma)
- **Use when:** You want varied results that cover different angles

#### wikipedia_retriever.py
- Fetches live Wikipedia articles as documents
- `top_k_results=2` returns 2 articles
- No embedding model needed — searches Wikipedia directly
- **Use when:** You need factual/encyclopedic knowledge without building your own vector store

**Retriever Comparison:**

| Retriever | Strategy | Best For |
|---|---|---|
| Vector Store | Top-k similarity | Basic search |
| MultiQuery | Multiple query variations | Ambiguous queries |
| Contextual Compression | Fetch + compress with LLM | Mixed-topic documents |
| MMR | Relevant + diverse | Avoiding redundancy |
| Wikipedia | Live Wikipedia search | General knowledge |

---

### 🔗 chains/

> **Concept:** Chains connect prompts, models, and parsers using the `|` pipe operator (LCEL — LangChain Expression Language).

#### simpleChain.py
- Basic `prompt | model | parser` chain
- `PromptTemplate` formats the input with variables
- `StrOutputParser` converts `AIMessage` to plain string
- `chain.get_graph().print_ascii()` prints a visual diagram of the chain

```
PromptTemplate → ChatOpenAI → StrOutputParser
```

#### sequencialChain.py
- **Two-step chain:** first generate a report, then summarize it
- Output of step 1 automatically becomes input of step 2
- `chain = prompt1 | model | parser | prompt2 | model | parser`
- **Use when:** You need multi-step processing where each step builds on the previous

```
prompt1 → model → parser → prompt2 → model → parser
```

#### parallelChain.py
- **Three-step chain:** split into parallel branches, then merge
- `RunnableParallel` runs "notes" and "quiz" generation simultaneously
- Results are merged by a third prompt into a single document
- Input text is about embedding models and cosine similarity
- **Use when:** You want to process the same input in multiple ways and combine results

```
text → [notes branch + quiz branch] → merge prompt → final output
```

---

### ⚙️ runnables/

> **Concept:** Runnables are the building blocks of LCEL. Every component (prompts, models, parsers, functions) implements the Runnable interface with `.invoke()`, `.batch()`, `.stream()`.

#### runnable_sequence.py
- Chains components in order using `RunnableSequence(step1, step2, ...)`
- Generates a joke about a topic, then explains the joke
- Equivalent to using `|` pipe operator

#### runnable_parallel.py
- `RunnableParallel` runs two chains at the same time
- Generates a **tweet** and a **LinkedIn post** about the same topic simultaneously
- Returns a dict: `{'tweet': '...', 'linkedin': '...'}`

#### runnable_passthrough.py
- `RunnablePassthrough()` passes input through unchanged
- Used here to keep the original joke while also generating an explanation
- Result: `{'joke': 'original joke text', 'explanation': 'explanation of the joke'}`
- Also calls `.get_graph().print_ascii()` to visualize the chain

#### runnable_lambda.py
- `RunnableLambda` wraps any plain Python function as a Runnable
- Defines a `word_count(text)` function and plugs it into a parallel chain
- Result: `{'joke': 'joke text', 'word_count': 42}`
- **Use when:** You need custom logic (formatting, counting, filtering) inside a chain

#### runnable_branch.py
- `RunnableBranch` adds **conditional logic** to a chain
- Generates a detailed report on a topic
- If the report is **> 500 words** → summarizes it
- If the report is **≤ 500 words** → passes it through unchanged
- **Use when:** You need if/else logic depending on intermediate output

**Runnables Summary:**

| Runnable | Purpose |
|---|---|
| `RunnableSequence` | Execute steps in order |
| `RunnableParallel` | Execute steps simultaneously |
| `RunnablePassthrough` | Pass input unchanged |
| `RunnableLambda` | Wrap a Python function |
| `RunnableBranch` | Conditional routing (if/else) |

---

### 🚀 FastAPI/llm_api.py

> **Concept:** Expose the LLM as a REST API with streaming support using FastAPI.

**What it does:**
- Creates a FastAPI app with a `/chat` endpoint
- Accepts a `prompt` query parameter
- Returns a **streaming response** — tokens appear word by word instead of waiting for the full response
- Uses `gpt-4o-mini` with `streaming=True`

**How to run:**
```bash
pip install fastapi uvicorn
uvicorn FastAPI.llm_api:app --reload
```

**How to call:**
```
GET http://localhost:8000/chat?prompt=Explain LangChain
```

**Key Concepts:**
- `astream()` — async generator that yields chunks as they arrive
- `StreamingResponse` — FastAPI streams chunks to the client as they come
- `asyncio.sleep(0)` — yields control back to the event loop between chunks

---

## 🧠 Key Concepts Summary

| Concept | What it does |
|---|---|
| **Document Loaders** | Read files → `Document(page_content, metadata)` |
| **Text Splitters** | Break large docs into smaller chunks for embedding |
| **Embeddings** | Convert text → vectors (numbers) capturing meaning |
| **Vector Store** | Database that stores and searches embeddings |
| **Retriever** | Fetches relevant docs from a vector store |
| **Chain (LCEL)** | `prompt \| model \| parser` pipeline |
| **Runnable** | Any component with `.invoke()` / `.stream()` / `.batch()` |
| **PromptTemplate** | Reusable prompt with `{variable}` placeholders |
| **StrOutputParser** | Converts `AIMessage` → plain `str` |

---

## 🔑 Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Required for all OpenAI model and embedding calls |

---

## 📝 Notes

- All scripts use `load_dotenv()` to load the `.env` file automatically
- `langchain-classic` package is required for `ContextualCompressionRetriever` in LangChain v1.x
- FAISS is used as an in-memory vector store in some retrievers (faster, no persistence)
- Chroma persists to disk via `persist_directory` — reuse without re-embedding
- The `chain.get_graph().print_ascii()` method is great for visualizing chain structure during learning
