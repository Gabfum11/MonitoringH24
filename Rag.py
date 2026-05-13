"""
Modulo RAG (Retrieval Augmented Generation).

Indicizza i riepiloghi orari in ChromaDB con embedding BGE-M3
e permette la ricerca semantica per rispondere a domande storiche.
"""

import json
from pathlib import Path
from datetime import date, timedelta

import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction
from FlagEmbedding import BGEM3FlagModel


class BGEM3EmbeddingFunction(EmbeddingFunction):
    """Wrapper BGE-M3 compatibile con ChromaDB."""

    def __init__(self):
        self._model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    def __call__(self, input):
        result = self._model.encode(input, batch_size=12, max_length=512)
        return result["dense_vecs"].tolist()


class RagIndex:
    def __init__(self, persist_dir="rag_index"):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._ef = BGEM3EmbeddingFunction()
        self._collection = self._client.get_or_create_collection(
            name="riepiloghi_orari",
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"}
        )

    # =========================================
    # INDICIZZAZIONE
    # =========================================
    def index_summary(self, summary_date: str, hour: int, hour_label: str, summary: str):
        """Indicizza un singolo riepilogo orario."""
        doc_id = f"{summary_date}_{hour:02d}"
        self._collection.upsert(
            ids=[doc_id],
            documents=[summary],
            metadatas=[{"date": summary_date, "hour": hour, "hour_label": hour_label}]
        )

    def index_existing_data(self, data_dir: str = "diari"):
        """Indicizza tutti i riepiloghi orari esistenti nell'archivio."""
        root = Path(data_dir)
        json_files = list(root.rglob("data.json"))
        print(f"[RAG] Indicizzazione di {len(json_files)} file...")

        indexed = 0
        for path in json_files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for s in data.get("hourly_summaries", []):
                    self.index_summary(
                        summary_date=data["date"],
                        hour=s["hour"],
                        hour_label=s["hour_label"],
                        summary=s["summary"]
                    )
                    indexed += 1
            except Exception as e:
                print(f"[RAG] Errore su {path}: {e}")

        print(f"[RAG] Indicizzati {indexed} riepiloghi orari.")

    # =========================================
    # RICERCA
    # =========================================
    def search(self, query: str, n_results: int = 5, days_back: int = None) -> str:
        """Cerca i riepiloghi più rilevanti per la query.

        Args:
            query: domanda del familiare
            n_results: numero di chunk da recuperare
            days_back: se impostato, filtra agli ultimi N giorni
        """
        total = self._collection.count()
        if total == 0:
            return ""

        where = None
        if days_back is not None:
            cutoff = (date.today() - timedelta(days=days_back)).isoformat()
            where = {"date": {"$gte": cutoff}}

        results = self._collection.query(
            query_texts=[query],
            n_results=min(n_results, total),
            where=where,
            include=["documents", "metadatas"]
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]

        if not docs:
            return ""

        lines = ["RIEPILOGHI RILEVANTI DALL'ARCHIVIO:"]
        for doc, meta in zip(docs, metas):
            lines.append(f"[{meta['date']} {meta['hour_label']}] {doc}")

        return "\n".join(lines)

    def count(self) -> int:
        return self._collection.count()
