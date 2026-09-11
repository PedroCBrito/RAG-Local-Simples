# RAG Local com LangChain, FAISS e Ollama

Pipeline de *Retrieval-Augmented Generation* (RAG) 100% local, projetado para consulta semântica e geração de respostas baseadas exclusivamente em documentos armazenados no diretório `docs_consulta/`. A solução prioriza privacidade de dados, ausência de custos com APIs externas e baixa latência de inferência em hardware convencional.

---

## 1. Stacks e Tecnologias

| Componente | Tecnologia | Finalidade Técnica |
| :--- | :--- | :--- |
| **Linguagem** | Python 3.12+ | Ambiente base de execução |
| **Orquestração RAG** | LangChain (`langchain`, `langchain-community`, `langchain-ollama`) | Gestão de cadeias de recuperação, injeção de contexto e templates de prompt |
| **Vector Store** | FAISS (`faiss-cpu`) | Indexação vetorial em memória e persistência local para busca por similaridade |
| **Embeddings** | Ollama (`nomic-embed-text`) ou Sentence-Transformers | Vetorização semântica de chunks textuais |
| **LLM Local** | Ollama (`llama3.2:1b`, `llama3.2:3b` ou `phi3:mini`) | Inferência e geração de respostas com baixa demanda computacional |
| **Document Loaders** | PyPDF / TextLoader | Extração e leitura de arquivos `.pdf`, `.txt` |

---

## 2. Arquitetura da Solução

```
[ docs_consulta/ ] (PDF / TXT)
        │
        ▼
[ Document Loader & Text Splitter ]
        │
        ▼
[ Embeddings Model ] (Local via Ollama / HuggingFace)
        │
        ▼
[ FAISS Vector Store ] (Persistência local em disco)
        │
        ▼  <─── Pergunta do Usuário
[ Similarity Search (Top-K) ]
        │
        ▼
[ Contexto + Prompt Template ] ──► [ LLM Local Ollama ] ──► [ Resposta com Fontes ]
```

---

## 3. Estrutura do Projeto

```
.
├── .gitignore
├── README.md
├── requirements.txt
├── docs_consulta/        # Diretório de entrada dos documentos para consulta
├── faiss_index/          # Armazenamento persistido dos índices vetoriais
├── src/
│   ├── config.py         # Parâmetros gerais (modelos, chunk size, overlap)
│   ├── ingest.py         # Módulo de carga, fragmentação e indexação vetorial
│   └── rag_chain.py      # Definição do retriever e da cadeia de inferência
└── main.py               # Ponto de entrada para execução e interface CLI
```

---

## 4. Critérios de Aceite

1. **Privacidade e Operação Offline:** Todo o processamento (extração, embedding, indexação e inferência) deve ocorrer exclusivamente de forma local, sem chamadas a APIs pagas ou provedores em nuvem.
2. **Ingestão Dinâmica:** O sistema deve ler e processar automaticamente os arquivos contidos em `docs_consulta/`.
3. **Persistência do Índice:** O índice gerado pelo FAISS deve ser salvo em disco local e reutilizado nas consultas, evitando reprocessamento desnecessário dos documentos.
4. **Fidelidade e Mitigação de Alucinações:** As respostas devem ser estritamente fundamentadas no contexto recuperado da base documental, com declaração explícita de ausência de informação quando o contexto for insuficiente.
5. **Eficiência Computacional:** Uso de modelos leves (1B a 3B parâmetros) para viabilizar execução fluida em CPU ou GPUs com memória limitada.
6. **Rastreabilidade:** Identificação das fontes (nome do arquivo e trecho) utilizadas na composição da resposta.
