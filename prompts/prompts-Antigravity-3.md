# Etapa 1 - Ingestão, Embeddings e Indexação Vetorial (FAISS)

## Prompt 1 - `Criar arquivo main.py`
```
Estou criando um projeto de Python em RAG rodando local.

Preciso criar o arquivo main.py que será a raiz do meu projeto. Ele será o ponto de entrada para execução e interface CLI

Gerar documento main.py
```

## Prompt 2 - `Criar Módulo de leitura em docs_consulta`
```
Estou criando um projeto de Python em RAG rodando local.

Preciso implementar um módulo de carga capaz de varrer a pasta docs_consultas para fazer a vetorização e embedding dos documentos.

Deve ser capaz de aceitas arquivos em formatos .pdf e .txt

Criar módulo inicial de varredura.
```

## Prompt 3 - `Criar Módulo de chunk para documentos`
```
Estou criando um projeto de Python RAG local, já tenho a extração das informações dos documentos.

Preciso implementar um módulo de RecursiveCharacterTextSplitter com parâmetros customizáveis via src/config.py (ex.: `chunk_size=1000`, `chunk_overlap=150`)

Deve ser preservar os metadados em cada chunk gerado

Criar módulo de chunks
```

## Prompt 4 - `Implementar Embedding Model`
```
Estou criando um projeto de Python RAG local, preciso transformar meus documentos parcionados pelo splitter.py

Configurar modelo de embedding local (OllamaEmbeddings) para fazer o embeddign apartir dos dados parcionados pelo loader e splitter. 

Os embeddings devem ser corretamente armazenados e feitos usando modelos LOCAIS.

Apenas instale e configure o modelo embedding
```

## Prompt 5 - `Implementar Indexação e Ingestão FAISS`
```
Estou criando um projeto de Python RAG local, os embeddings estão sendo feitos pelo OllamaEmbeddings

Agora Implemente um script src/ingest.py para construir o índice vetorial e salvar os binários no diretório /faiss_index/.

Deve criar rotina de recarregamento do índice a partir do disco e a execução do script src/ingest.py gera os arquivos na pasta faiss_index e permite reload em memória sem reprocessar os documentos originais.

Gere o arquivo /src/ingest.py com o conteudo do script
```

## Prompt 6 - `Revisão Etapa 1`
```
A Etapa 1 de ingestão, chunking, embeddings e indexação vetorial com FAISS foi implementada.

Revise todos os módulos criados nesta etapa (loader, splitter, embeddings, ingest e persistência em faiss_index/).

Execute testes de ponta a ponta na ingestão e verifique a integridade dos arquivos gerados.
```

## Prompt 7 - `Commit Etapa 1`
```
Concluí a Etapa 1 com ingestão de documentos, fragmentação de texto, geração de embeddings e persistência do índice FAISS.

Gere uma mensagem de commit seguindo o padrão conventional commits.

Faça o commit das alterações da Etapa 1.
```

---

# Etapa 2 - Integração com LLM e Cadeia RAG

## Prompt 1 - `Baixar e Conectar LLM`
```
Estou criando um RAG local em Python, a criação de embeddings está sendo feita e armazenada no /faiss_index/

Agora deve configurar cliente Ollama apontando para modelo local llama3.2:3b. Deve ser um modelo preciso, portanto parametrizar temperatura baixa.

Baixar e configurar temperatura da LLM local.
```

## Prompt 2 - `Configuração Retriever`
```
Estou criando um RAG Python local, temos o modelo local instalado e os embeddings sendo feitos.

Agora deve ser feita a configuração do Retriever.

Deve ser possivel encontrar o índice FAISS como retriever. O retorno esperado deve ser uma query textual, retornar os $k$ trechos com maior proximidade semântica.

Configure corretamente, se preciso crie testes para validação.
```

## Prompt 3 - `Template do Prompt`
```
Estou criando um RAG Python local, tem o modelo já retornando os $k$ trechos com maior proximidade semântica.

Deve ser criado um prompt que instrua estritamente o uso do contexto fornecido.
Não deve responder à pergunta baseando-se fora do contexto recuperado, para dados desconhecidos adicione uma mensagem de resposta não encontrada.

Montar a cadeia via LCEL (retriever | prompt | llm | StrOutputParser).

Crie um e adicione esse script em /src/rag_chain.py
```

## Prompt 4 - `Adicionar Rastreabilidade`
```
Estou criando um RAG Python local, já está criada a base para a recuperação e conexão com a LLM.

Garanta que seja retornado tanto a resposta sintetizada quanto os documentos fontes.
O retorno deve conter os nomes dos arquivos e números de páginas/trechos consultados.

Crie um script que garanta que essa regra seja seguida.
```

## Prompt 5 - `Revisão Etapa 2`
```
A Etapa 2 de conexão com LLM local via Ollama, configuração do Retriever, templates de prompt e cadeia RAG com rastreabilidade foi implementada.

Revise a cadeia RAG completa, a injeção de contexto e a mitigação de respostas fora do escopo.

Valide o fluxo de perguntas e respostas garantindo o retorno das fontes consultadas.
```

## Prompt 6 - `Commit Etapa 2`
```
Concluí a Etapa 2 com integração do Ollama, montagem da cadeia RAG via LCEL, controle de alucinação e rastreabilidade de fontes.

Gere uma mensagem de commit seguindo o padrão conventional commits.

Faça o commit das alterações da Etapa 2.
```

---

# Etapa 3 - Interface CLI e Experiência do Usuário

## Prompt 1 - `Implementar Interação`
```
O sistema de RAG Local está implementado. O usuário deve utilizar o cli para utilizar o RAG

Implemente em `main.py` um loop interativo (`input("Pergunta: ")`), comando de saída (`exit`/`sair`) e verificação prévia do índice `faiss_index/`.
Deve alertar o usuário caso o índice ainda não tenha sido gerado, sugerindo a execução do ingest.

O script deve possibilitar o usuário a fazer mais de uma consulta por execução.

Atualize o main.py.
```

## Prompt 2 - `Personalizar saída`
```
Com o RAG criado e o usuario já conseguindo interagir com o cli python.

A saída para cada interação de ser a resposta gerada e listar abaixo as fontes utilizadas no formato:

Exemplo de saída esperada:
    [Resposta]: <Texto gerado pela LLM>
    
    [Fontes Consultadas]:
    - doc1.pdf (Página 2)
    - notas.txt (Trecho 1)
    
Respostas e fontes devem ser apresentadas de forma legível e organizada no terminal.

Gerar script para garantir que as saídas serão como esperado.
```

## Prompt 3 - `Tratar exceções`
```
O RAG em Python está com o escopo fechado. Porem não trata exceções.

Adicione para tratar falhas comuns, como: serviço do Ollama offline/inacessível, diretório `docs_consulta/` vazio, índice corrompido ou ausente.
Deve conter mensagens de erro amigáveis e exibidas no console em caso de falha, verifique os erros mais comuns e trate todos.

Crie script para gerenciar exceções de consulta.
```

## Prompt 4 - `Revisão Etapa 3`
```
A Etapa 3 de interface CLI interativa, formatação de saídas e tratamento de exceções e erros de execução foi implementada.

Revise a usabilidade do terminal, validações de entrada, comportamento com serviços offline e legibilidade das respostas e fontes.

Valide a experiência completa do usuário via CLI.
```

## Prompt 5 - `Commit Etapa 3`
```
Concluí a Etapa 3 com interface CLI interativa em main.py, formatação estruturada de saídas e tratamento robusto de exceções.

Gere uma mensagem de commit seguindo o padrão conventional commits.

Faça o commit das alterações da Etapa 3.
```

---

# Etapa 4 - Revisão e Finalização

## Prompt 1 - `Revise todo o projeto`
```
Projeto de RAG local em Python, utilizando LLM ollama local.

Revise todo o projeto, as integrações entre as partes. Garanta que tudo está rodando como esperado, erros/exceções tratadas.

Todo o projeto deve ser considerado integro 
```

## Prompt 2 - `Revise a documentação`
```
O projeto de um RAG local em Python está implementado.

Revise a documentação, readme, requirements e a pasta /docs/ para garantir que está descrevendo corretamente o projeto, e que não houve desatualização durante a implementação.

Revise e atualize a documentação do projeto.
```

## Prompt 3 - `Criar testes unitários`
```
O projeto de um RAG local em Python está implementado.

Crie testes unitários para validar o funcionamento das principais funcionalidades do projeto.

Testes criados e executados com sucesso, validando o comportamento esperado do sistema.
```

## Prompt 4 - `Commit Final`
```
Projeto RAG Local concluído, revisado e documentado com sucesso.

Gere uma mensagem de commit final seguindo o padrão conventional commits.

Faça o commit de encerramento da implementação e documentação.
```
