# VidIntel — AI Video Intelligence & RAG Assistant

> **VidIntel** is an AI-powered video intelligence system that converts long-form video/audio content into structured, searchable knowledge.

The system can process a **YouTube video or local audio/video file**, convert it into audio chunks, transcribe the content using **Whisper or Sarvam AI**, generate an intelligent meeting/video summary, extract **action items, key decisions, and unresolved questions**, and finally create a **Retrieval-Augmented Generation (RAG)** pipeline for context-aware question answering.

The project is built using **Python, PyTorch, OpenAI Whisper, LangChain, Groq, Hugging Face Sentence Transformers, ChromaDB, FFmpeg, Pydub, and yt-dlp**.

---

# Project Overview

Traditional video files contain a large amount of information, but searching and understanding that information manually can be difficult.

VidIntel transforms unstructured video/audio into structured and queryable information.

```text
YouTube URL / Local Video / Audio
                │
                ▼
        Audio Extraction
                │
                ▼
       WAV Conversion
                │
                ▼
        Audio Chunking
                │
                ▼
        Speech-to-Text
       ┌────────┴────────┐
       │                 │
    English           Hinglish
       │                 │
       ▼                 ▼
    Whisper           Sarvam AI
       │                 │
       └────────┬────────┘
                ▼
           Transcript
                │
        ┌───────┼────────┐
        │       │        │
        ▼       ▼        ▼
      Title   Summary  Extraction
                        │
                 ┌──────┼──────┐
                 ▼      ▼      ▼
              Actions Decisions Questions
                        │
                        ▼
                 Vector Embeddings
                        │
                        ▼
                    ChromaDB
                        │
                        ▼
                     Retriever
                        │
                        ▼
                  Relevant Context
                        │
                        ▼
                    Groq LLM
                        │
                        ▼
                  Context-Aware QA
```

---

# Main Objectives

VidIntel is designed to solve several problems associated with long-form video and meeting content:

- Convert video/audio into text
- Handle long audio files through chunking
- Support English transcription
- Support Hinglish-to-English transcription
- Generate professional summaries
- Generate a meaningful title
- Extract action items
- Extract responsible owners and deadlines when mentioned
- Extract key decisions
- Identify unresolved questions
- Store transcript information as vector embeddings
- Perform semantic search over the transcript
- Answer questions using RAG
- Reduce hallucination by restricting answers to transcript context
- Provide a reusable backend pipeline that can be connected to a UI

---

# Key AI Concepts Used

This project demonstrates several important AI Engineering concepts:

```text
Speech-to-Text
        ↓
Large Language Models
        ↓
Prompt Engineering
        ↓
Text Chunking
        ↓
Embeddings
        ↓
Vector Database
        ↓
Semantic Retrieval
        ↓
Retrieval-Augmented Generation
        ↓
AI Question Answering
```

The project is therefore more than a simple chatbot.

It combines:

```text
Computer/Audio Processing
+
Speech Recognition
+
LLM Applications
+
Information Extraction
+
Vector Search
+
RAG
```

---

# Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| AI/ML Framework | PyTorch |
| Speech-to-Text | OpenAI Whisper |
| Hinglish Translation + STT | Sarvam AI |
| LLM | Groq + `openai/gpt-oss-20b` |
| LLM Framework | LangChain |
| Prompt Pipeline | LangChain LCEL |
| Embeddings | `all-MiniLM-L6-v2` |
| Embedding Framework | Sentence Transformers / Hugging Face |
| Vector Database | ChromaDB |
| Audio Processing | Pydub |
| Video/Audio Download | yt-dlp |
| Media Conversion | FFmpeg |
| Environment Management | python-dotenv |
| UI | Streamlit |
| PDF Generation Dependency | ReportLab |
| Data Utilities | Pandas |
| GPU Acceleration | CUDA / PyTorch |

> **Note:** The Streamlit UI implementation in `app.py` is intentionally not documented here. This README focuses on the AI/backend architecture.

---

# 📁 Project Structure

```text
VidIntel-main/
│
├── core/
│   ├── extractor.py
│   ├── rag_engine.py
│   ├── summerized.py
│   ├── transcriber.py
│   └── vector_store.py
│
├── utils/
│   └── audio_processor.py
│
├── main.py
├── test.py
├── requirements.txt
├── .gitignore
├── README.md
│
├── downloads/          # Generated at runtime
├── vector_db/          # Generated ChromaDB storage
└── data/               # Optional local data
```

---

# Component Architecture

The project is divided into several logical components.

```text
                    main.py
                       │
                       ▼
              run_pipeline()
                       │
       ┌───────────────┼────────────────┐
       │               │                │
       ▼               ▼                ▼
 audio_processor   transcriber      LLM Modules
       │               │                │
       │               │        ┌───────┼────────┐
       │               │        ▼       ▼        ▼
       │               │     summary  extractor  title
       │               │
       │               ▼
       │          transcript
       │               │
       └───────────────┼─────────────────────┐
                       │                     │
                       ▼                     ▼
                 vector_store          rag_engine
                       │                     │
                       ▼                     ▼
                    Chroma                Groq
                       │                     │
                       └─────────┬───────────┘
                                 ▼
                            Question Answer
```

---

# 📂 Module 1 — `utils/audio_processor.py`

This module is responsible for converting the original input into manageable WAV audio chunks.

It handles:

```text
YouTube URL
     OR
Local Audio/Video
     ↓
WAV Audio
     ↓
10-minute Chunks
```

---

# YouTube Audio Download

The function:

```python
def download_yt_audio(url: str) -> str:
```

uses:

```python
yt_dlp
```

to download the best available audio stream.

The configuration selects:

```python
"format": "bestaudio[ext=m4a]/bestaudio/best"
```

This tells yt-dlp to prefer an audio-only stream.

The downloaded media is then processed through FFmpeg:

```python
"postprocessors": [
    {
        "key": "FFmpegExtractAudio",
        "preferredcodec": "wav",
        "preferredquality": "192",
    }
]
```

The result is a WAV file.

---

# Why yt-dlp?

`yt-dlp` is used to retrieve audio from supported online video platforms.

The important advantage for this project is that VidIntel does not need to process the complete video frames.

Instead:

```text
Video
  ↓
Audio
  ↓
Speech Recognition
```

This significantly simplifies the AI pipeline because the primary information source is speech.

---

# Browser Cookie Support

The code optionally supports browser cookies:

```python
browser = os.getenv("YTDLP_BROWSER", "").strip().lower()
```

If configured, it uses:

```python
cookiesfrombrowser
```

This can be useful when accessing content where browser authentication/cookies are required and permitted.

For public videos, cookies are normally not required.

---

# Local File Processing

The function:

```python
def convert_to_wav(input_path: str) -> str:
```

uses Pydub:

```python
audio = AudioSegment.from_file(input_path)
```

Then converts the audio to:

```text
Mono
16 kHz
WAV
```

using:

```python
audio = audio.set_channels(1).set_frame_rate(16000)
```

---

# Why Mono Audio?

The speech recognition pipeline generally does not require stereo information.

For speech transcription:

```text
Stereo
  ↓
Mono
```

reduces unnecessary audio data while preserving the speech signal.

---

# Why 16 kHz?

The audio is converted to:

```text
16,000 Hz
```

sampling rate.

This is a common speech-processing representation and provides enough temporal resolution for speech recognition while keeping the audio manageable.

---

# Audio Chunking

Long audio is divided into chunks using:

```python
def chunk_audio(
    wav_path: str,
    chunk_minutes: int = 10
) -> list:
```

The default chunk size is:

```text
10 minutes
```

Therefore, a 50-minute recording becomes approximately:

```text
Chunk 1 → 0–10 min
Chunk 2 → 10–20 min
Chunk 3 → 20–30 min
Chunk 4 → 30–40 min
Chunk 5 → 40–50 min
```

---

# Why Chunk Long Audio?

Processing a very long audio file in one operation can create several problems:

- High memory consumption
- Longer processing time
- Model context limitations
- API limits
- Increased failure probability
- Difficult recovery when a request fails

Chunking provides:

```text
Long Audio
    ↓
Small Independent Segments
    ↓
Process Each Segment
    ↓
Combine Results
```

This makes the pipeline more robust.

---

# 📂 `process_input()`

The main function is:

```python
def process_input(source: str) -> list:
```

It determines whether the input is:

```text
HTTP/HTTPS URL
```

or:

```text
Local File
```

If the input is a URL:

```python
download_yt_audio(source)
```

is called.

Otherwise:

```python
convert_to_wav(source)
```

is called.

Finally:

```python
chunk_audio(wav_path)
```

creates the audio chunks.

---

# Module 2 — `core/transcriber.py`

This module handles speech-to-text.

VidIntel supports two transcription paths:

```text
English
   ↓
OpenAI Whisper

Hinglish
   ↓
Sarvam AI
   ↓
English Transcript
```

---

# Whisper

The project uses:

```python
import whisper
```

The default model is:

```python
WHISPER_MODEL = os.getenv(
    "WHISPER_MODEL",
    "small"
)
```

Therefore, if no environment variable is provided:

```text
Whisper Small
```

is used.

---

# ⚙️ Whisper Model Loading

The project uses lazy loading:

```python
_model = None
```

and:

```python
def load_model():
```

The model is loaded only when required.

```python
if _model is None:
    _model = whisper.load_model(WHISPER_MODEL)
```

This is useful because loading Whisper can be relatively expensive.

Once loaded, the same model instance can be reused for multiple chunks.

---

# Whisper Transcription

Each audio chunk is processed using:

```python
result = model.transcribe(
    chunk_path,
    task="transcribe"
)
```

The text is extracted using:

```python
result["text"]
```

---

# Language Routing

The central function is:

```python
def transcribe_chunk(
    chunk_path: str,
    language: str = "english"
) -> str:
```

The logic is:

```text
language == "hinglish"
        ↓
Sarvam AI

otherwise
        ↓
Whisper
```

This gives the system two transcription modes.

---

# 🇬🇧 English Mode

When:

```text
language = "english"
```

the system uses:

```text
Whisper
```

The model performs local speech recognition.

Advantages:

- Local inference
- No external STT request for English
- Can use GPU through the underlying Whisper/PyTorch stack
- Good general-purpose speech recognition

---

# 🇮🇳 Hinglish Mode

When:

```text
language = "hinglish"
```

the system uses:

```text
Sarvam AI
```

through:

```text
https://api.sarvam.ai/speech-to-text-translate
```

The project expects Sarvam to provide an English transcript.

This is particularly useful when the spoken content contains Indian-language/Hinglish speech and the desired output is English.

---

# Sarvam API Audio Limitation

The code intentionally splits Sarvam requests into:

```python
SARVAM_PIECE_SECONDS = 25
```

seconds.

The implementation comments explain that the synchronous endpoint has a 30-second audio limit.

Therefore:

```text
10-minute chunk
       ↓
25-second pieces
       ↓
Sarvam API
       ↓
Piece transcripts
       ↓
Concatenate
```

The 25-second size provides a safety margin below the 30-second limit.

---

# Sarvam Processing Flow

For every 10-minute chunk:

```text
10-minute WAV
      ↓
Split into 25-second pieces
      ↓
Piece 1 → Sarvam
Piece 2 → Sarvam
Piece 3 → Sarvam
...
      ↓
Combine transcripts
      ↓
Return complete chunk transcript
```

Temporary files are deleted after processing:

```python
if os.path.exists(piece_path):
    os.remove(piece_path)
```

This prevents unnecessary accumulation of temporary audio files.

---

# 📝 Full Transcription

The function:

```python
def transcribe_all(
    chunks: list,
    language: str = "english"
) -> str:
```

iterates over every audio chunk:

```python
for i, chunk in enumerate(chunks):
```

Each chunk is transcribed.

Then:

```python
full_transcript += text + " "
```

combines all results.

Finally:

```python
return full_transcript.strip()
```

returns one complete transcript.

---

# 📚 Module 3 — `core/summerized.py`

This module performs:

```text
Transcript
    ↓
Chunking
    ↓
Partial Summaries
    ↓
Combined Summary
```

It also generates a professional title.

---

# Summary Chunking

The summarization splitter uses:

```python
RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=200
)
```

Therefore:

```text
Chunk Size    = 3000 characters
Overlap       = 200 characters
```

---

# Why Use Chunking for Summarization?

Large transcripts may exceed the context limits or become expensive to process as one prompt.

Instead of:

```text
Entire Transcript → LLM
```

VidIntel uses:

```text
Transcript
   ↓
3000-character chunks
   ↓
Summary of each chunk
   ↓
Combine summaries
   ↓
Final summary
```

This is essentially a **map-reduce style summarization pipeline**.

---

# Map Phase

The project creates:

```python
map_prompt = ChatPromptTemplate.from_messages(...)
```

with the instruction:

```text
Summarize this portion of a meeting transcript concisely.
```

Each transcript chunk is sent through the LLM:

```python
map_chain = map_prompt | llm | StrOutputParser()
```

The result is a list of partial summaries.

Conceptually:

```text
Transcript Chunk 1 → Summary 1
Transcript Chunk 2 → Summary 2
Transcript Chunk 3 → Summary 3
...
```

---

# Reduce Phase

All partial summaries are combined:

```python
combined = "\n\n".join(chunk_summerizes)
```

Then a second LLM call combines them into one final professional summary.

The prompt requests:

```text
one final professional meeting summary
in bullet point format
```

Therefore:

```text
Partial Summary 1
Partial Summary 2
Partial Summary 3
       ↓
Combined LLM Prompt
       ↓
Final Summary
```

---

# Automatic Title Generation

VidIntel also generates a short title:

```python
def generate_title(transcript: str) -> str:
```

The title prompt requests:

```text
short professional meeting title
max 8 words
only return the title
```

Only the first:

```text
2000 characters
```

of the transcript are used:

```python
transcript[:2000]
```

This reduces unnecessary input to the title-generation model.

---

# Module 4 — `core/extractor.py`

This module extracts structured information from the transcript.

It currently extracts three categories:

```text
Action Items
Key Decisions
Open Questions
```

---

# Common LLM Chain

The helper function:

```python
def build_chain(system_prompt: str):
```

creates a reusable LangChain pipeline.

The structure is:

```text
Input Transcript
       ↓
RunnablePassthrough
       ↓
RunnableLambda
       ↓
ChatPromptTemplate
       ↓
Groq LLM
       ↓
StrOutputParser
       ↓
Text Output
```

---

# ✅ Action Item Extraction

The function:

```python
extract_action_item(transcript)
```

asks the LLM to extract:

```text
Task Description
Owner
Deadline
```

The output is requested as a numbered list.

If no action items exist, the model is instructed to return:

```text
No action items found.
```

---

# 📌 Example Action Item Structure

The intended output format is approximately:

```text
1. Task: Prepare the project report
   Owner: Rahul
   Deadline: Friday

2. Task: Review the model performance
   Owner: Priya
   Deadline: Not specified
```

The actual result depends on the transcript and model output.

---

# 🔑 Key Decision Extraction

The function:

```python
extract_key_decisions(transcript)
```

asks the model to identify important decisions made during the meeting.

Example conceptual output:

```text
1. The team decided to use PostgreSQL.
2. The model will be deployed using Docker.
3. The project deadline was moved to next Friday.
```

If no decisions are identified:

```text
No key decisions found.
```

---

# Open Question Extraction

The function:

```python
extract_questions(transcript)
```

identifies:

```text
Unresolved Questions
Topics Requiring Follow-up
```

If none are found:

```text
No open questions found.
```

---

# Why Information Extraction Is Useful

Instead of forcing the user to read the entire transcript, VidIntel converts unstructured speech into structured knowledge:

```text
Meeting Transcript
       │
       ├── Summary
       │
       ├── Action Items
       │
       ├── Key Decisions
       │
       └── Open Questions
```

This is one of the major practical applications of LLMs in productivity systems.

---

# Module 5 — `core/vector_store.py`

This module creates the semantic search layer.

The technology stack is:

```text
Transcript
   ↓
Text Chunks
   ↓
Hugging Face Embeddings
   ↓
Vector Representations
   ↓
ChromaDB
```

---

# Embedding Model

The project uses:

```python
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
```

This model converts text into numerical vectors.

For example:

```text
"The project deadline is Friday."
```

is transformed into a vector representation.

Conceptually:

```text
Text
 ↓
Embedding Model
 ↓
[0.12, -0.04, 0.81, ...]
```

The exact vector values are generated by the model.

---

# Why Embeddings?

Keyword search only looks for matching words.

For example:

```text
Question:
"When is the project due?"
```

The transcript might say:

```text
"The team needs to finish everything by Friday."
```

There may be no exact phrase:

```text
"project due"
```

Semantic embeddings allow the system to identify that these texts have similar meaning.

---

# Vector Store Configuration

The vector database is configured as:

```python
CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"
```

Therefore, ChromaDB data is persisted in:

```text
vector_db/
```

The repository's `.gitignore` excludes this directory.

---

# RAG Chunking

The vector store uses a smaller chunk size:

```python
RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
```

Configuration:

```text
Chunk Size    = 500 characters
Overlap       = 50 characters
```

This is different from summarization.

### Summarization:

```text
3000 characters
200 overlap
```

### RAG retrieval:

```text
500 characters
50 overlap
```

The smaller RAG chunks provide more precise retrieval.

---

# Why Chunk Overlap?

Suppose a sentence is divided near a chunk boundary.

Without overlap:

```text
Chunk 1:
"The team discussed the deployment strategy and"

Chunk 2:
"decided to use Docker."
```

Important context can become separated.

With overlap:

```text
Chunk 1:
"...deployment strategy and decided to use Docker."

Chunk 2:
"decided to use Docker..."
```

The overlap preserves contextual continuity.

---

# 📄 LangChain Documents

Each chunk is converted into a:

```python
Document
```

with metadata:

```python
metadata={
    "chunk_index": i
}
```

This allows each vectorized chunk to retain its original chunk index.

---

# Building the Vector Store

The main function:

```python
build_vector_store(transcript)
```

performs:

```text
Transcript
    ↓
Split into 500-character chunks
    ↓
Create LangChain Documents
    ↓
Generate embeddings
    ↓
Store vectors in ChromaDB
```

---

# Persistent Vector Database

Chroma is configured with:

```python
persist_directory=CHROMA_DIR
```

Therefore, the vector store can persist on disk.

```text
vector_db/
```

This allows the database to be loaded later rather than necessarily rebuilding it from scratch.

---

# Retriever

The retriever is created using:

```python
def get_retriever(
    vector_store: Chroma,
    k: int = 4
):
```

with:

```python
search_type="similarity"
```

and:

```python
search_kwargs={"k": 4}
```

Therefore, for each user question, the system retrieves the:

```text
Top 4 semantically similar chunks
```

---

# Why Top-K Retrieval?

Suppose the transcript contains hundreds of chunks.

Sending all chunks to the LLM for every question would be inefficient.

Instead:

```text
1000 Transcript Chunks
        ↓
Semantic Search
        ↓
Top 4 Relevant Chunks
        ↓
LLM
```

This reduces the amount of irrelevant context provided to the model.

---

# 🤖 Module 6 — `core/rag_engine.py`

This is the core question-answering component.

RAG stands for:

> **Retrieval-Augmented Generation**

Instead of asking the LLM to answer from its general knowledge, VidIntel first retrieves relevant information from the transcript.

---

# RAG Architecture

```text
User Question
      │
      ▼
Question Embedding
      │
      ▼
Vector Similarity Search
      │
      ▼
Top 4 Transcript Chunks
      │
      ▼
Context Construction
      │
      ▼
Prompt
      │
      ▼
Groq LLM
      │
      ▼
Answer
```

---

# Building the RAG Chain

The function:

```python
build_rag_chain(transcript)
```

first creates a vector store:

```python
vector_store = build_vector_store(transcript)
```

Then:

```python
retriever = get_retriever(
    vector_store,
    k=4
)
```

The LLM is initialized:

```python
llm = get_llm()
```

---

# LLM Used

The project currently initializes:

```python
init_chat_model(
    model="groq:openai/gpt-oss-20b"
)
```

Therefore, the LLM is accessed through:

```text
Groq
```

using the:

```text
openai/gpt-oss-20b
```

model identifier configured in the source code.

> Model availability and provider configuration can change over time, so the exact model identifier should be verified if the project is run in the future.

---

# Hallucination Control

One of the most important parts of the RAG prompt is:

```text
Answer the user's question based ONLY on the meeting transcript context provided below.
```

The model is also instructed:

```text
If the answer is not found in the context, say:
"I could not find this information in the meeting transcript."
```

This is an important AI Engineering practice.

The system attempts to prevent the LLM from answering using unrelated external knowledge.

---

# Why RAG Instead of Normal LLM Chat?

A normal LLM may answer:

```text
Question
   ↓
LLM's General Knowledge
   ↓
Answer
```

VidIntel uses:

```text
Question
   ↓
Transcript Retrieval
   ↓
Relevant Context
   ↓
LLM
   ↓
Answer
```

This makes the answer grounded in the processed video.

---

# LCEL RAG Pipeline

The project uses LangChain Expression Language (LCEL).

The core pipeline is:

```python
rag_chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

Conceptually:

```text
                 User Question
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
      Retriever                Passthrough
          │                         │
          ▼                         ▼
    Relevant Docs                Question
          │                         │
          ▼                         │
     format_docs                   │
          │                         │
          └──────────┬──────────────┘
                     ▼
                   Prompt
                     │
                     ▼
                    LLM
                     │
                     ▼
              String Output
```

---

# 📄 Document Formatting

The helper function:

```python
def format_docs(docs):
    return "\n\n".join(
        [doc.page_content for doc in docs]
    )
```

combines the retrieved documents into one context string.

For example:

```text
Retrieved Chunk 1

Retrieved Chunk 2

Retrieved Chunk 3

Retrieved Chunk 4
```

becomes the context provided to the LLM.

---

# Asking Questions

The function:

```python
def ask_question(
    rag_chain,
    question: str
) -> str:
```

invokes the RAG chain:

```python
answer = rag_chain.invoke(question)
```

The final answer is returned as text.

---

# Complete RAG Example

Suppose the transcript contains:

```text
The team decided to deploy the application using Docker.
The deployment will happen next Friday.
```

User asks:

```text
When will the deployment happen?
```

The system performs:

```text
Question
   ↓
Embedding / Similarity Search
   ↓
Relevant Transcript Chunk
   ↓
Context:
"The deployment will happen next Friday."
   ↓
LLM
   ↓
"Next Friday."
```

The LLM does not need to independently know anything about the project.

The transcript is the knowledge source.

---

# Module 7 — `main.py`

`main.py` is the main backend orchestration layer.

The most important function is:

```python
def run_pipeline(
    source: str,
    language: str = "english"
) -> dict:
```

---

# Complete Pipeline

The function executes:

```python
chunks = process_input(source)
```

Then:

```python
transcript = transcribe_all(
    chunks,
    language=language
)
```

Then:

```python
title = generate_title(transcript)
```

Then:

```python
summary = summerizes(transcript)
```

Then:

```python
action_item = extract_action_item(transcript)
decisions = extract_key_decisions(transcript)
questions = extract_questions(transcript)
```

Finally:

```python
rag_chain = build_rag_chain(transcript)
```

---

# Pipeline Output

The function returns:

```python
{
    "title": title,
    "transcript": transcript,
    "summary": summary,
    "action_items": action_item,
    "key_decisions": decisions,
    "open_questions": questions,
    "rag_chain": rag_chain,
}
```

Therefore, one pipeline execution produces:

```text
Title
Transcript
Summary
Action Items
Key Decisions
Open Questions
RAG Chain
```

---

# End-to-End System

The entire backend can be understood as:

```text
                 INPUT
                   │
                   ▼
       YouTube URL / Local File
                   │
                   ▼
          Audio Processor
                   │
                   ▼
             WAV Conversion
                   │
                   ▼
            10-Minute Chunks
                   │
                   ▼
             Transcription
             ┌─────┴─────┐
             ▼           ▼
          Whisper     Sarvam
          English    Hinglish
             │           │
             └─────┬─────┘
                   ▼
               Transcript
                   │
        ┌──────────┼───────────┐
        │          │           │
        ▼          ▼           ▼
      Title      Summary    Extraction
                              │
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
                 Actions   Decisions Questions
                              │
                              ▼
                         RAG Pipeline
                              │
                              ▼
                         Text Chunks
                              │
                              ▼
                         Embeddings
                              │
                              ▼
                           ChromaDB
                              │
                              ▼
                          Retriever
                              │
                              ▼
                       Relevant Context
                              │
                              ▼
                           Groq LLM
                              │
                              ▼
                         Final Answer
```

---

# Chunking Strategy

The project intentionally uses different chunking strategies for different tasks.

| Task | Chunk Size | Overlap |
|---|---:|---:|
| Audio processing | 10 minutes | None |
| Sarvam API processing | 25 seconds | None |
| Summarization | 3000 characters | 200 |
| RAG Vector Store | 500 characters | 50 |

This is an important architectural decision.

Different tasks have different requirements.

---

# Why Different Chunk Sizes?

## Audio

Audio is divided into large chunks:

```text
10 minutes
```

because processing each small audio segment individually would create excessive overhead.

---

## Sarvam

Sarvam pieces are:

```text
25 seconds
```

because the synchronous API has a shorter audio-duration constraint.

---

## Summarization

Summary chunks are:

```text
3000 characters
```

because the model needs enough context to understand a meaningful portion of the transcript.

---

## RAG

RAG chunks are:

```text
500 characters
```

because retrieval benefits from smaller, focused pieces of information.

---

# Prompt Engineering

VidIntel uses different prompts for different tasks.

This is an important design pattern.

Instead of one generic prompt, the system creates specialized tasks:

```text
Title Generation Prompt
        ↓
Summary Prompt
        ↓
Action Item Prompt
        ↓
Decision Extraction Prompt
        ↓
Question Extraction Prompt
        ↓
RAG Question Answering Prompt
```

Each prompt has a specific objective.

---

# 🔐 Grounded Question Answering

The RAG system explicitly instructs the model to use:

```text
ONLY the meeting transcript context
```

and to state when information cannot be found.

This provides a basic grounding strategy.

However, this is **not a guarantee of zero hallucinations**.

The final response is still generated by an LLM, so output quality depends on:

- Retrieved context
- Transcript quality
- Chunking
- Embedding quality
- Prompt quality
- LLM behavior

---

# 📈 Accuracy

Unlike a supervised classification project, VidIntel does not currently calculate a single numerical:

```text
Accuracy = XX%
```

There is no labeled evaluation dataset included in the provided implementation.

Therefore, it would be technically incorrect to claim something like:

```text
95% Accuracy
98% Accuracy
99% Accuracy
```

without a dedicated evaluation dataset.

---

# Transcription Quality

Speech-to-text quality depends on several factors:

```text
Audio Quality
+
Speaker Accent
+
Background Noise
+
Speaking Speed
+
Language Mixing
+
Microphone Quality
+
Model Selection
```

For English, the project uses:

```text
Whisper Small
```

by default.

For Hinglish:

```text
Sarvam AI
```

is used.

---

# Better Transcription Evaluation

A future version could evaluate transcription using:

### Word Error Rate (WER)

```text
WER =
(Substitutions + Deletions + Insertions)
---------------------------------------
               Number of Reference Words
```

A lower WER generally indicates better transcription.

For example:

```text
Reference Transcript
        vs
Generated Transcript
        ↓
       WER
```

A labeled collection of real audio recordings and reference transcripts would be required for meaningful measurement.

---

# RAG Evaluation

RAG quality should also be evaluated separately.

Useful metrics include:

```text
Retrieval Precision
Retrieval Recall
Context Relevance
Answer Relevance
Faithfulness
Groundedness
```

A proper RAG benchmark would contain:

```text
Transcript
Question
Expected Answer
Relevant Chunk
```

Then the retrieval and generated answer could be evaluated systematically.

---

# Example Evaluation Dataset

A future evaluation dataset could look like:

```text
Transcript:
"The project deadline is October 10."

Question:
"When is the project deadline?"

Expected Answer:
"October 10."

Retrieved Context:
"The project deadline is October 10."

Generated Answer:
"October 10."
```

This allows automated evaluation of:

```text
Retrieval Quality
+
Answer Quality
```

---

# ⚠️ Current Limitations

## 1. No Ground-Truth Accuracy Dataset

The project does not include a manually labeled benchmark.

Therefore:

```text
No official numerical AI accuracy is reported.
```

---

## 2. No Speaker Diarization

The Sarvam request explicitly sets:

```python
"with_diarization": "false"
```

Therefore, the current system does not identify individual speakers.

The transcript is essentially treated as one continuous stream of text.

A future version could support:

```text
Speaker 1:
...

Speaker 2:
...
```

---

## 3. No Timestamps in Final Transcript

The current transcription pipeline combines text:

```python
full_transcript += text + " "
```

Therefore, detailed timestamps are not preserved in the final transcript returned by the pipeline.

A future implementation could retain:

```text
[00:01:24] Speaker 1: ...
[00:01:39] Speaker 2: ...
```

This would make the system much more useful for video navigation.

---

## 4. No Advanced RAG Reranking

The current retriever uses:

```text
Similarity Search
```

with:

```text
k = 4
```

There is no dedicated reranker.

A future architecture could use:

```text
Vector Search
      ↓
Top 10–20 Chunks
      ↓
Reranker
      ↓
Top 4 Relevant Chunks
      ↓
LLM
```

This can improve retrieval quality for difficult questions.

---

## 5. Vector Database Management

The current code builds the vector store using:

```python
Chroma.from_documents(...)
```

with a persistent directory.

Repeatedly building the same collection from the same transcript can potentially result in duplicate documents depending on how the persistent collection is managed.

A production implementation should use:

- Unique document IDs
- Transcript/session IDs
- Collection management
- Deduplication
- Metadata filtering
- Explicit update/delete operations

---

## 6. Single Collection Design

The vector store currently uses:

```python
COLLECTION_NAME = "meeting_transcript"
```

A production application processing many independent videos should separate documents by:

```text
video_id
meeting_id
session_id
user_id
```

For example:

```text
video_001
video_002
video_003
```

This prevents unrelated videos from being mixed together.

---

# 🔒 Security Considerations

API keys should never be hard-coded.

The project uses:

```python
load_dotenv()
```

and environment variables.

Sensitive files are excluded through `.gitignore`:

```text
.env
.env.*
```

Never commit:

```text
GROQ_API_KEY
SARVAM_API_KEY
```

to GitHub.

---

# Environment Variables

The project uses several configurable environment variables.

Recommended `.env` structure:

```env
GROQ_API_KEY=your_groq_api_key

SARVAM_API_KEY=your_sarvam_api_key

WHISPER_MODEL=small

SARVAM_STT_MODEL=saaras:v2.5

YTDLP_BROWSER=
```

---

# Environment Variable Explanation

## `GROQ_API_KEY`

Used by the Groq/LangChain LLM integration.

---

## `SARVAM_API_KEY`

Required when:

```text
language = hinglish
```

because the Hinglish transcription path uses Sarvam's API.

---

## `WHISPER_MODEL`

Controls the local Whisper model.

Default:

```text
small
```

Depending on the installed Whisper implementation, other Whisper model sizes can be selected.

Larger models generally trade:

```text
More compute
+
More memory
+
Slower inference
```

for potentially better transcription quality.

---

## `SARVAM_STT_MODEL`

Default:

```text
saaras:v2.5
```

This controls the Sarvam model used by the transcription API.

---

## `YTDLP_BROWSER`

Optional browser configuration for yt-dlp cookie loading.

Example conceptually:

```env
YTDLP_BROWSER=chrome
```

Only configure this when necessary and supported by your local environment.

---

# Installation

## 1. Clone the Repository

```bash
git clone <your-github-repository-url>
```

Then:

```bash
cd VidIntel-main
```

---

# 2. Create a Virtual Environment

Recommended:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

---

# 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

The project recommends:

```text
Python 3.12
```

as indicated in the provided `requirements.txt`.

---

# 4. Install FFmpeg

FFmpeg is required for audio/video conversion.

The project uses:

```text
ffmpeg-python
```

and:

```text
Pydub
```

but a system-level FFmpeg installation is still required for the underlying media processing workflow.

Verify installation:

```bash
ffmpeg -version
```

If FFmpeg is correctly installed, the command should display the installed FFmpeg version.

---

# 5. Node.js Requirement

The yt-dlp configuration includes:

```python
"js_runtimes": {
    "node": {}
}
```

Therefore, Node.js may be required by the configured yt-dlp workflow.

Verify:

```bash
node --version
```

---

# 6. Configure `.env`

Create:

```text
.env
```

in the project root.

Example:

```env
GROQ_API_KEY=your_key_here
SARVAM_API_KEY=your_key_here

WHISPER_MODEL=small
SARVAM_STT_MODEL=saaras:v2.5

YTDLP_BROWSER=
```

Do not commit this file to GitHub.

---

# Running the Backend Pipeline

The main backend entry point is:

```text
main.py
```

Run:

```bash
python main.py
```

The program asks:

```text
Enter YouTube URL or local file path:
```

Then:

```text
Language (english/hinglish):
```

Example:

```text
Enter YouTube URL or local file path:
https://www.youtube.com/watch?v=example

Language (english/hinglish):
english
```

---

# Testing the Pipeline

The project also contains:

```text
test.py
```

This file demonstrates the backend pipeline using a YouTube URL.

It performs:

```text
Download
   ↓
Transcription
   ↓
Transcript Printing
   ↓
Title Generation
   ↓
Summary
   ↓
Action Items
   ↓
Key Decisions
   ↓
Open Questions
```

Run:

```bash
python test.py
```

---

# Requirements

The project contains a comprehensive `requirements.txt`.

Major dependencies include:

```text
streamlit
python-dotenv

langchain
langchain-core
langchain-community
langchain-text-splitters

openai
groq
langchain-groq

google-genai
langchain-google-genai

anthropic
langchain-anthropic

openai-whisper

torch
torchaudio

ffmpeg-python
pydub

yt-dlp

chromadb
langchain-chroma

sentence-transformers
transformers
huggingface-hub
langchain-huggingface

tiktoken
pydantic
reportlab
requests
pandas
tqdm
```

---

# GPU Support

Whisper is built on PyTorch and can take advantage of GPU acceleration when the environment supports it.

Check CUDA:

```python
import torch

print(torch.cuda.is_available())
```

If the result is:

```text
True
```

PyTorch can use the available CUDA device for supported operations.

---

# Performance Considerations

VidIntel contains several computationally expensive operations.

The main cost centers are:

```text
1. Whisper transcription
2. LLM summarization
3. LLM information extraction
4. Embedding generation
5. RAG LLM inference
```

---

# API Cost Considerations

The project uses a combination of local and external AI processing.

### English Transcription

```text
Whisper
↓
Local inference
```

This does not require an external transcription API.

However, it requires local compute resources.

---

### Hinglish Transcription

```text
Sarvam API
↓
External API request
```

This depends on the provider's current pricing, quota, and account configuration.

---

### LLM Processing

The project uses:

```text
Groq
```

for:

```text
Title Generation
Summary
Action Items
Key Decisions
Open Questions
RAG Answers
```

Each of these operations can result in LLM requests.

Therefore, a single video can produce multiple LLM calls.

---

# Approximate LLM Call Pattern

For one transcript, the current backend can perform approximately:

```text
1 × Title Generation
1 × Summary per 3000-character transcript chunk
1 × Final Summary
1 × Action Item Extraction
1 × Decision Extraction
1 × Question Extraction
```

Then:

```text
RAG question
    ↓
1 LLM call per user question
```

The exact number of calls depends on transcript length and user interaction.

---

# AI Architecture Summary

VidIntel is effectively a multi-stage AI system:

```text
Stage 1
------
Media Processing

Stage 2
------
Speech Recognition

Stage 3
------
LLM Information Understanding

Stage 4
------
Embedding Generation

Stage 5
------
Vector Storage

Stage 6
------
Semantic Retrieval

Stage 7
------
Retrieval-Augmented Generation
```

---

# Production Architecture

A more production-oriented architecture could look like:

```text
                         User
                          │
                          ▼
                   Video / Audio
                          │
                          ▼
                ┌─────────────────┐
                │ Media Processor │
                └────────┬────────┘
                         │
                         ▼
                    Audio Chunks
                         │
                         ▼
                 ┌───────────────┐
                 │ Speech-to-Text│
                 └───────┬───────┘
                         │
                         ▼
                    Transcript
                         │
             ┌───────────┼───────────┐
             │           │           │
             ▼           ▼           ▼
          Summary     Extraction    Title
             │           │           │
             └───────────┼───────────┘
                         │
                         ▼
                    Text Chunks
                         │
                         ▼
                    Embeddings
                         │
                         ▼
                    Vector DB
                         │
                         ▼
                      Search
                         │
                         ▼
                  Relevant Context
                         │
                         ▼
                       LLM
                         │
                         ▼
                     Answer
```

---

# Future Improvements

## 1. Speaker Diarization

Add:

```text
Speaker Identification
```

so transcripts become:

```text
Speaker 1:
We should deploy on Friday.

Speaker 2:
I agree. I'll prepare the Docker image.
```

---

# 2. Timestamp-Aware Transcript

Preserve:

```text
Start Time
End Time
Text
Speaker
```

Example:

```text
[00:12:32 - 00:12:45]
Speaker 2:
We will deploy this Friday.
```

This could enable direct video navigation.

---

# 3. Better RAG Retrieval

Current:

```text
Similarity Search
k = 4
```

Future:

```text
Hybrid Search
+
Vector Search
+
Keyword Search
+
Reranker
```

---

# 4. Metadata Filtering

Store metadata such as:

```text
video_id
chunk_index
timestamp
speaker
source
language
```

Then retrieval can be filtered more intelligently.

---

# 5. Multi-Video Knowledge Base

Currently, the vector collection is:

```text
meeting_transcript
```

A future implementation could support:

```text
Multiple Videos
       ↓
Shared Knowledge Base
       ↓
Cross-Video Search
```

For example:

```text
"Compare the decisions made in Meeting 1 and Meeting 3."
```

---

# 6. Structured JSON Outputs

Instead of returning plain text for extracted information, the system could return structured JSON.

For example:

```json
{
  "action_items": [
    {
      "task": "Prepare project report",
      "owner": "Rahul",
      "deadline": "Friday"
    }
  ]
}
```

This would make downstream application development easier.

---

# 7. Evaluation Framework

A production version should include a benchmark containing:

```text
Ground Truth Transcript
Ground Truth Summary
Ground Truth Action Items
Ground Truth Decisions
Ground Truth Questions
Ground Truth RAG Answers
```

Then evaluate:

```text
WER
ROUGE
BERTScore
Retrieval Recall
Retrieval Precision
Answer Faithfulness
Answer Relevance
```

---

# 8. Background Job Processing

For long videos, a production application could use:

```text
FastAPI
+
Celery / Background Workers
+
Redis
```

instead of performing the complete pipeline synchronously.

Architecture:

```text
User
 ↓
API
 ↓
Job Queue
 ↓
Worker
 ↓
Transcription
 ↓
LLM Processing
 ↓
Vector DB
 ↓
Result
```

---

# 9. Database for Metadata

A production system could use:

```text
PostgreSQL
```

for metadata:

```text
video_id
title
user_id
created_at
duration
language
transcript_status
processing_status
```

while ChromaDB or another vector database stores semantic embeddings.

---

# 🧪 Example End-to-End Usage

Suppose the input is:

```text
https://www.youtube.com/watch?v=example
```

The system performs:

```text
1. Detect YouTube URL
          ↓
2. Download audio
          ↓
3. Convert to WAV
          ↓
4. Split into 10-minute chunks
          ↓
5. Transcribe each chunk
          ↓
6. Combine transcript
          ↓
7. Generate title
          ↓
8. Generate summary
          ↓
9. Extract action items
          ↓
10. Extract decisions
          ↓
11. Extract open questions
          ↓
12. Create embeddings
          ↓
13. Store embeddings in ChromaDB
          ↓
14. Create retriever
          ↓
15. Wait for user questions
          ↓
16. Retrieve relevant transcript chunks
          ↓
17. Send context + question to LLM
          ↓
18. Generate grounded answer
```

---

# 📊 Project Capabilities

| Capability | Implemented |
|---|:---:|
| YouTube Input | ✅ |
| Local Audio Input | ✅ |
| Local Video Input | ✅ |
| WAV Conversion | ✅ |
| Audio Chunking | ✅ |
| English Transcription | ✅ |
| Hinglish Transcription | ✅ |
| Whisper Integration | ✅ |
| Sarvam Integration | ✅ |
| Automatic Title | ✅ |
| Meeting Summary | ✅ |
| Action Item Extraction | ✅ |
| Owner Extraction | ✅ |
| Deadline Extraction | ✅ |
| Key Decision Extraction | ✅ |
| Open Question Extraction | ✅ |
| Text Embeddings | ✅ |
| ChromaDB | ✅ |
| Semantic Search | ✅ |
| RAG | ✅ |
| Context-Grounded Q&A | ✅ |
| Speaker Diarization | ❌ |
| Timestamp-Aware Q&A | ❌ |
| Automated Accuracy Benchmark | ❌ |
| RAG Reranking | ❌ |

---

# 🧠 Key AI Engineering Lessons

This project demonstrates how multiple AI components can be combined into a practical application.

The most important architectural lesson is:

```text
Don't use an LLM for everything.
```

Instead, use specialized components.

```text
Audio
 ↓
Speech Model

Text
 ↓
LLM

Semantic Search
 ↓
Embedding Model

Knowledge Retrieval
 ↓
Vector Database

Final Reasoning
 ↓
LLM
```

Each component solves a different problem.

---

# Why RAG Is Important

An LLM by itself is not a database for your private video.

RAG creates a bridge:

```text
Private / User Data
       ↓
Vector Database
       ↓
Relevant Context
       ↓
LLM
```

This makes it possible to build applications where users can ask questions about:

```text
Meetings
Lectures
Interviews
Podcasts
Presentations
Tutorials
Research Discussions
Recorded Classes
Business Calls
```

without manually searching through the complete transcript.

---

# Project Highlights

### Multi-Engine Speech Recognition

```text
English  → Whisper
Hinglish → Sarvam AI
```

---

### LLM-Powered Understanding

The system generates:

```text
Title
Summary
Action Items
Key Decisions
Open Questions
```

---

### Semantic Search

Instead of simple keyword matching:

```text
Question
   ↓
Embedding
   ↓
Semantic Similarity
   ↓
Relevant Transcript
```

---

### RAG-Based Question Answering

```text
Question
   ↓
Retrieve Relevant Context
   ↓
LLM
   ↓
Grounded Answer
```

---

### Persistent Vector Store

Transcript embeddings are stored using:

```text
ChromaDB
```

allowing the semantic retrieval layer to persist on disk.

---

# Important Technical Clarification

This project should not be described as a model that was:

```text
trained from scratch
```

or:

```text
fine-tuned
```

The project primarily uses existing pretrained models and APIs:

```text
Whisper
Hugging Face Embedding Model
Groq-hosted LLM
Sarvam AI
```

The engineering work is primarily in:

```text
System Architecture
+
Pipeline Design
+
Prompt Engineering
+
RAG
+
Data Processing
+
Model Integration
+
Retrieval
+
Application Logic
```

This is a legitimate and important category of modern **AI Engineering**.

---

# What Makes This an AI Engineering Project?

The project combines multiple AI systems into a single end-to-end pipeline.

```text
                 AI Engineering
                       │
       ┌───────────────┼────────────────┐
       │               │                │
       ▼               ▼                ▼
 Speech AI          LLM Apps          RAG
       │               │                │
       ▼               ▼                ▼
  Whisper/Sarvam   Groq/LangChain    ChromaDB
       │               │                │
       └───────────────┼────────────────┘
                       ▼
                 VidIntel System
```

The project demonstrates practical skills in:

- LLM integration
- LangChain
- LCEL
- Prompt engineering
- RAG architecture
- Embeddings
- Vector databases
- Semantic retrieval
- Speech recognition
- Audio preprocessing
- API integration
- GPU inference
- AI pipeline orchestration

---

# Important Code-Level Notes

## `summerized.py`

The filename contains:

```text
summerized.py
```

and the function is:

```python
summerizes()
```

The conventional spelling would be:

```text
summarized.py
summarizes()
```

This does not prevent the current implementation from working, but renaming it in a future cleanup would improve code quality and maintainability.

---

## Unused Imports

Some files contain imports that are not currently required by the implementation.

For example, `rag_engine.py` imports:

```python
RecursiveCharacterTextSplitter
```

but the actual RAG chunking is performed inside:

```text
vector_store.py
```

Similarly, some imports can be cleaned up as part of future refactoring.

---

# Recommended Code Quality Improvements

A production-quality version could introduce:

```text
Type-safe configuration
Logging framework
Exception handling
Unit tests
Integration tests
Structured output schemas
Centralized configuration
Unique document IDs
Session-based vector collections
Caching
Retry mechanisms
Async API processing
Monitoring
Evaluation
```

---

# Testing Strategy

A stronger testing setup could contain:

```text
tests/
│
├── test_audio_processor.py
├── test_transcriber.py
├── test_summary.py
├── test_extractor.py
├── test_vector_store.py
└── test_rag.py
```

Testing should cover:

### Audio Processing

```text
URL detection
File conversion
Chunk creation
```

### Transcription

```text
English routing
Hinglish routing
API failure handling
```

### Extraction

```text
Action item extraction
Decision extraction
Question extraction
```

### RAG

```text
Document creation
Embedding generation
Retrieval
Context construction
Answer generation
```

---

# Error Handling Improvements

The current code handles some errors, such as failed Sarvam requests.

For example:

```python
if not response.ok:
    response.raise_for_status()
```

A production system should additionally handle:

```text
Network failures
Timeouts
Invalid URLs
Unsupported media
Corrupted audio
Missing API keys
API rate limits
LLM failures
Embedding failures
Disk failures
Out-of-memory errors
```

---

# Future Version Architecture

A mature version of VidIntel could evolve toward:

```text
                    VidIntel
                       │
        ┌──────────────┼───────────────┐
        │              │               │
        ▼              ▼               ▼
   Video Engine    Speech Engine    Document Engine
        │              │               │
        ▼              ▼               ▼
   yt-dlp/FFmpeg  Whisper/Sarvam   Text Processor
                       │               │
                       └───────┬───────┘
                               ▼
                         LLM Processing
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
           Summary         Extraction          RAG
              │                │                │
              │                │          ┌─────┴─────┐
              │                │          ▼           ▼
              │                │      Embeddings   Retriever
              │                │          │           │
              │                │          ▼           │
              │                │       Chroma         │
              │                │          │           │
              └────────────────┴──────────┼───────────┘
                                          ▼
                                       Groq LLM
                                          │
                                          ▼
                                    AI Response
```

---

# 📌 Conclusion

**VidIntel** is an end-to-end AI Video Intelligence and RAG system designed to transform unstructured video/audio content into searchable and actionable knowledge.

The system combines:

```text
YouTube / Local Media
        ↓
Audio Processing
        ↓
Whisper / Sarvam Speech Recognition
        ↓
Transcript
        ↓
LLM Understanding
        ↓
Summary + Title + Action Items
        ↓
Key Decisions + Open Questions
        ↓
Embedding Generation
        ↓
ChromaDB
        ↓
Semantic Retrieval
        ↓
RAG
        ↓
Groq LLM
        ↓
Context-Aware Answers
```

The project demonstrates practical implementation of modern AI application architecture rather than relying on a single AI model.

---

# 🏆 Project Summary

| Category | Details |
|---|---|
| Project Name | **VidIntel** |
| Domain | AI / Generative AI / RAG |
| Primary Task | Video & Meeting Intelligence |
| Programming Language | Python |
| Speech Recognition | OpenAI Whisper |
| Hinglish STT | Sarvam AI |
| LLM | Groq — `openai/gpt-oss-20b` |
| LLM Framework | LangChain |
| Embedding Model | `all-MiniLM-L6-v2` |
| Vector Database | ChromaDB |
| Audio Processing | Pydub + FFmpeg |
| Video Download | yt-dlp |
| GPU Framework | PyTorch / CUDA |
| Input | YouTube URL / Local Audio / Local Video |
| Output | Transcript + Summary + Insights + RAG Q&A |
| Speaker Diarization | Not currently implemented |
| Numerical AI Accuracy | Not currently benchmarked |
| RAG | ✅ |
| Semantic Search | ✅ |
| Model Training | ❌ |
| Model Fine-Tuning | ❌ |

---

# ⭐ Project Status

```text
✅ YouTube Audio Processing
✅ Local Media Processing
✅ Audio Conversion
✅ Long Audio Chunking
✅ Whisper Transcription
✅ Hinglish → English Transcription
✅ Sarvam Integration
✅ Automatic Title Generation
✅ LLM Summarization
✅ Action Item Extraction
✅ Key Decision Extraction
✅ Open Question Extraction
✅ Hugging Face Embeddings
✅ ChromaDB Vector Store
✅ Semantic Retrieval
✅ LangChain LCEL
✅ RAG Pipeline
✅ Context-Grounded Q&A
✅ Persistent Vector Database
🚧 Speaker Diarization
🚧 Timestamp-Aware Retrieval
🚧 RAG Reranking
🚧 Automated Evaluation
🚧 Multi-Video Knowledge Base
```

---

# Skills Demonstrated

This project demonstrates practical experience with:

```text
Python
PyTorch
Whisper
Speech-to-Text
LLMs
Groq
LangChain
LCEL
Prompt Engineering
RAG
Embeddings
Hugging Face
ChromaDB
Vector Search
Semantic Search
Audio Processing
FFmpeg
yt-dlp
API Integration
GPU Acceleration
AI Pipeline Design
Information Extraction
```

---

# One-Line Project Description

> **VidIntel is an AI-powered video intelligence and RAG assistant that transforms YouTube or local media into transcripts, summaries, actionable insights, and context-grounded conversational knowledge.**

---


# 📜 License

```text
MIT License

Copyright (c) 2026 Niraj Singh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

# Acknowledgements

This project uses open-source and API-based technologies including:

```text
PyTorch
OpenAI Whisper
LangChain
Hugging Face
ChromaDB
yt-dlp
Pydub
FFmpeg
Groq
Sarvam AI
```

Each component contributes a specialized capability to the overall AI pipeline.

Dev: Niraj Singh
