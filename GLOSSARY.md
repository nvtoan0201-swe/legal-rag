# Legal RAG

Question answering over Vietnamese legal documents, where every answer cites the articles it rests on.

## Corpus

**Document** (văn bản):
One law or decree in the corpus, identified by a short id such as `blld2019`.
_Avoid_: file, source

**Article** (Điều):
A numbered article of a Document; the unit an answer cites.
_Avoid_: section, provision

**Clause** (khoản):
A numbered clause inside an Article.
_Avoid_: paragraph, point (điểm is a smaller unit)

**Chunk**:
The piece of a Document that is embedded and searched on its own; normally one Clause, or a whole Article when it has no Clauses.
_Avoid_: passage, segment

**Corpus index**:
The searchable store of every Chunk in the corpus, which also returns a whole Article (its Clauses in order) by id.
_Avoid_: retriever, vector store, collection

## Answering

**Graph expansion**:
Adding Articles that the retrieved Articles reference (or are referenced by) to the context of an answer.
_Avoid_: graph search, neighbour lookup
