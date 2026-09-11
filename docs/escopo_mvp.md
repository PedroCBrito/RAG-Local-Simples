# Escopo do MVP - RAG Local

Documento de especificação do Produto Mínimo Viável (MVP) para o pipeline de *Retrieval-Augmented Generation* (RAG) com execução 100% local.

---

## 1. Objetivo

Implementar uma solução de RAG local e offline capaz de ingerir documentos proprietários, indexá-los em formato vetorial e responder a consultas em linguagem natural via linha de comando (CLI), garantindo privacidade dos dados, fidelidade ao contexto e custo zero de infraestrutura de nuvem/APIs.

---

## 2. Requisitos Funcionais (RF)

| ID | Requisito | Descrição |
| :--- | :--- | :--- |
| **RF-01** | Ingestão Documental | Carregar arquivos nos formatos `.pdf` e `.txt` a partir do diretório local `docs_consulta/`. |
| **RF-02** | Fragmentação (*Chunking*) | Dividir o conteúdo dos documentos em chunks textuais configuráveis (tamanho e sobreposição/overlap). |
| **RF-03** | Geração de Embeddings | Converter os chunks de texto em vetores numéricos utilizando um modelo local (ex.: `nomic-embed-text` ou Sentence-Transformers). |
| **RF-04** | Indexação e Persistência | Criar e salvar o índice vetorial em disco local via FAISS (`faiss_index/`), permitindo reuso sem reprocessamento completo. |
| **RF-05** | Recuperação Semântica (*Retriever*) | Buscar os $k$ chunks mais relevantes para a pergunta do usuário com base em similaridade vetorial (distância euclidiana/cosseno). |
| **RF-06** | Geração de Resposta Aumentada | Montar o prompt com contexto recuperado + instrução de sistema e submeter ao LLM local (via Ollama). |
| **RF-07** | Rastreabilidade de Fontes | Retornar na resposta os metadados dos chunks recuperados (nome do arquivo de origem e página/trecho). |
| **RF-08** | Mitigação de Alucinação | Instruir o modelo a responder explicitamente que não possui informações suficientes quando o contexto recuperado não cobrir a dúvida. |
| **RF-09** | Interface CLI | Disponibilizar entrada de perguntas e exibição de respostas estruturadas via terminal. |

---

## 3. Requisitos Não Funcionais (RNF)

| ID | Requisito | Critério Técnico |
| :--- | :--- | :--- |
| **RNF-01** | Privacidade e *Air-Gap* | Execução 100% *on-premises* sem envio de dados/telemetria para serviços externos. |
| **RNF-02** | Custo Operacional | Ausência de custos com tokens ou APIs de terceiros (ex.: OpenAI, Anthropic). |
| **RNF-03** | Consumo de Recursos | Operar em hardware convencional (CPU x86_64 ou GPU integrada/dedicada básica) utilizando modelos de 1B a 3B parâmetros. |
| **RNF-04** | Latência de Resposta | Tempo de recuperação vetorial < 200 ms e geração da resposta condicionada à vazão de tokens do hardware local. |
| **RNF-05** | Modularidade | Código estruturado em módulos independentes (`config`, `ingest`, `rag_chain`, `main`). |
| **RNF-06** | Compatibilidade de Ambiente | Execução suportada em ambientes Linux, WSL2 e Windows com Python 3.12+. |

---

## 4. Fora do Escopo (Out of Scope)

* Interface gráfica de usuário (GUI / Web UI como Streamlit, Gradio ou Next.js).
* Autenticação, autorização de usuários e controle de acesso baseado em papéis (RBAC).
* Mecanismos avançados de re-ranking (Cross-Encoders/Cohere Re-ranker) ou busca híbrida (BM25 + Dense).
* Agentes autônomos, roteamento semântico complexo, chamadas de ferramentas (*function calling*) ou memória de conversação persistente multissessão.
* OCR para extração de texto de imagens ou digitalizações não pesquisáveis.
* Conectores de dados para repositórios externos (SharePoint, Google Drive, bancos de dados SQL/NoSQL).
* Pipeline de deploy em produção (orquestração via Kubernetes, Docker Swarm ou APIs distribuídas via FastAPI/vLLM).
