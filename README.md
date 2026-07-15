# RAG Advisor Project:

### Description:
This project is ...

### Features:
- Data ingestion using URLs for HTML, PDF, and Plain Text
- RAG system capabilities and vector similarity search for stored chunks
- Selection of advisors that will take in query and use RAG to make informed response
- Use locally run LLM agent to synthesize information and complete specified tasks

### Installation
- python -m venv .venv
- source .venv/bin/activate
- pip install -r requirements.txt
- deactivate

### Tech Stack
- Programming language: Python, SQL
- Database: PostgreSQL
- Data Tools: pandas, NumPy, YAML, sentence-transformers, BeautifulSoup, pypdf
- APIs/Services: psycopg, Ollama
- Testing: pytest
- Version Control: Git, GitHub


## Database Overview
```text  
└── Project schema  
├── advisor_documents  
│ ├── Purpose: Maps advisors to relevant documents  
│ └── Columns:  
│ ├── advisor_id  
│ ├── document_id  
│ ├── weight  
│ └── relevance_note  
│  
├── advisors  
│ ├── Purpose: Stores basic advisor information  
│ └── Columns:  
│ ├── advisor_id  
│ ├── name  
│ ├── description  
│ └── config  
│  
├── chunks  
│ ├── Purpose: Stores text chunks extracted from documents  
│ └── Columns:  
│ ├── chunk_id  
│ ├── document_id  
│ ├── chunk_index  
│ ├── chunk_text  
│ ├── token_count  
│ ├── embedding  
│ └── embedding_model  
│  
├── etl history  
│ ├── Purpose: Document ETL history for data insertion
│ └── Columns:  
│ ├── job_type  
│ ├── run_status  
│ ├── num_entries  
│ ├── urls  
│ ├── start_time  
│ ├── end_time  
│ ├── error_message  
│ └── log_file  
│  
├── documents  
│ ├── Purpose: Catalogues source documents
│ └── Columns:  
│ ├── document_id  
│ ├── url  
│ ├── source_type  
│ └── content_hash  