# RAG-Based Profile Match

Match job descriptions against resumes using semantic search (LangChain + Chroma + OpenAI embeddings).

## Dataset

Put your resume PDFs in the **`Resume/`** folder (or pass a different path when running).

## How to start the application

### 1. Install dependencies

```bash
cd /Users/garvit/Rag_Based_Profile_Match
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

Copy `.env.example` to `.env`, add your key, or:

```bash
export OPENAI_API_KEY="your-key-here"
```

### 3. Index resumes (Part A)

Loads PDFs from `Resume/`, chunks them, embeds with OpenAI, and stores in Chroma. Run once (or whenever you add/change resumes):

```bash
python resume_rag.py
```

To use a different folder:

```bash
python resume_rag.py /path/to/your/resumes
```

### 4. Run job matching (Part B)

Pass a job description as the first argument; prints JSON with top matches:

```bash
python job_matcher.py "Senior Python developer. 5+ years experience. Machine Learning required."
```

With no argument, a sample job description is used.

## Output

`job_matcher.py` prints JSON in this shape:

- `job_description`: the input string  
- `top_matches`: list of `{ candidate_name, resume_path, match_score, matched_skills, relevant_excerpts, reasoning }`

## Summary

| Step | Command |
|------|--------|
| Install | `pip install -r requirements.txt` |
| Index resumes | `python resume_rag.py` (uses `Resume/` by default) |
| Match jobs | `python job_matcher.py "Your job description here"` |
