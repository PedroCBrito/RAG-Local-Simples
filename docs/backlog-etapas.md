# Backlog de Desenvolvimento do MVP (Por Etapas)

Registro das etapas concluídas do pipeline RAG local, com mapeamento de Requisitos Funcionais (RF), Requisitos Técnicos (RT) e critérios de aceite validados.

**Status geral:** MVP implementado e validado localmente.

---

## Etapa 1: Ingestão, Embeddings e Indexação Vetorial (FAISS)

**Objetivo:** Processar os documentos de entrada, gerar representações vetoriais e persistir o índice FAISS em disco para consultas subsequentes.

### Tarefas e Requisitos:
- [x] **[RF-01 / RT-01] Configuração de Ambiente e Leitura de Documentos**
  - Configurar dependências no `requirements.txt` (`langchain`, `langchain-community`, `faiss-cpu`, `pypdf`, etc.).
  - Implementar módulo de carga capaz de varrer `docs_consulta/` e carregar arquivos `.pdf` e `.txt`.
  - *Critério de Aceite:* O loader lê todos os arquivos suportados na pasta sem lançar exceções para formatos válidos.

- [x] **[RF-02 / RT-02] Segmentação de Texto (*Chunking*)**
  - Implementar `RecursiveCharacterTextSplitter` com parâmetros customizáveis via `src/config.py` (ex.: `chunk_size=1000`, `chunk_overlap=150`).
  - Preservar metadados (`source`, `page`) em cada chunk gerado.
  - *Critério de Aceite:* Documentos extensos são divididos em trechos menores mantendo integridade e metadados de origem.

- [x] **[RF-03 / RT-03] Provedor de Embeddings Local**
  - Configurar `OllamaEmbeddings(model="nomic-embed-text")` no serviço local.
  - *Critério de Aceite:* Conversão de lotes de texto em vetores sem dependência de conexões de rede externa.

- [x] **[RF-04 / RT-03] Criação e Persistência do Índice FAISS**
  - Implementar script `src/ingest.py` para construir o índice vetorial e salvar os binários (`index.faiss` e `index.pkl`) no diretório `faiss_index/`.
  - Criar rotina de recarregamento do índice a partir do disco.
  - *Critério de Aceite:* `python src/ingest.py` gera os arquivos e `python src/ingest.py --load-only` recarrega o índice sem reprocessar os documentos.

---

## Etapa 2: Integração com LLM e Cadeia RAG

**Objetivo:** Configurar o modelo de linguagem local, integrar o *retriever* e construir o pipeline de injeção de contexto com mitigação de alucinação.

### Tarefas e Requisitos:
- [x] **[RT-04] Conexão com LLM Local (Ollama)**
  - Configurar `ChatOllama` com `llama3.2:3b` no endpoint local.
  - Utilizar `temperature=0.0` para reduzir variação nas respostas.
  - *Critério de Aceite:* O sistema comunica-se com a instância local do Ollama e executa inferência sem erros.

- [x] **[RF-05 / RT-05] Configuração do *Retriever***
  - Expor o índice FAISS como retriever (`as_retriever(search_kwargs={"k": 4})`).
  - *Critério de Aceite:* Para uma query textual, o retriever retorna os $k$ trechos com maior proximidade semântica.

- [x] **[RF-06 / RF-08 / RT-05] Template de Prompt e Cadeia RAG (LCEL)**
  - Construir prompt em `prompts/` ou `src/rag_chain.py` que instrua estritamente o uso do contexto fornecido.
  - Incluir o fallback: *"Não foi possível encontrar a resposta no contexto fornecido."*
  - Montar a cadeia via LCEL (`retriever | prompt | llm | StrOutputParser`).
  - *Critério de Aceite:* A LLM responde à pergunta baseando-se estritamente no contexto recuperado e recusa responder perguntas fora da base documental.

- [x] **[RF-07 / RT-05] Extração e Rastreabilidade de Fontes**
  - Estruturar a cadeia para retornar tanto a resposta sintetizada quanto os documentos fonte (`source_documents`).
  - *Critério de Aceite:* O retorno contém os nomes dos arquivos e números de páginas/trechos consultados.

---

## Etapa 3: Interface CLI e Experiência do Usuário

**Objetivo:** Fornecer um ponto de entrada interativo via terminal (`main.py`) para execução de perguntas, visualização de fontes e tratamento de exceções.

### Tarefas e Requisitos:
- [x] **[RF-09 / RT-06] Loop Interativo no Terminal**
  - Implementar `main.py` com loop interativo (`input("Pergunta: ")`), comando de saída (`exit`/`sair`) e verificação prévia do índice `faiss_index/`.
  - Reconstruir automaticamente o índice com `ingest_documents()` sempre que o `main.py` for iniciado.
  - Alertar o usuário caso o índice ainda não tenha sido gerado, sugerindo a execução do ingest.
  - *Critério de Aceite:* O usuário executa `python main.py` e consegue fazer múltiplas perguntas sucessivas no mesmo terminal.

- [x] **[RF-07 / RT-06] Formatação de Saída Estruturada**
  - Exibir a resposta gerada com destaque e listar abaixo as fontes utilizadas no formato:
    ```
    [Resposta]: <Texto gerado pela LLM>
    
    [Fontes Consultadas]:
    - doc1.pdf (Página 2)
    - notas.txt (Trecho 1)
    ```
  - *Critério de Aceite:* Resposta e fontes são apresentadas de forma legível e organizada no terminal.

- [x] **[RT-06] Tratamento de Exceções e Resiliência**
  - Tratar falhas comuns: serviço do Ollama offline/inacessível, diretório `docs_consulta/` vazio, índice corrompido ou ausente.
  - *Critério de Aceite:* Mensagens de erro amigáveis e orientativas são exibidas no console em caso de falha, sem crash com traceback não tratado.

---

## Evidências de validação

- Embeddings `nomic-embed-text` com 768 dimensões.
- Persistência e reload dos arquivos `index.faiss` e `index.pkl`.
- Recuperação Top-K com preservação dos metadados.
- Respostas fundamentadas, fallback para contexto insuficiente e retorno das fontes.
- Consulta direta e múltiplas consultas na mesma sessão do CLI.
- Tratamento de Ollama offline, timeout, modelo ausente e índice ausente/corrompido.
