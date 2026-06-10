"""
Modulo RAG (Retrieval Augmented Generation).

Indicizza i riepiloghi orari in ChromaDB con embedding BGE-M3
e permette la ricerca semantica per rispondere a domande storiche.
"""

import json
import time
from pathlib import Path
from datetime import date, timedelta

import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction
from FlagEmbedding import BGEM3FlagModel


class BGEM3EmbeddingFunction(EmbeddingFunction):
    """Wrapper BGE-M3 compatibile con ChromaDB."""

    def __init__(self):
        self._model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True) #fp16 serve per risparmiare memoria, non influisce sulla qualità degli embedding

    def __call__(self, input):
        result = self._model.encode(input, batch_size=12, max_length=512) #significa che tronca a 512 token, ma i nostri riepiloghi orari sono molto più corti, quindi va bene
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

        # Accumulatore latenze ricerca (resettato a inizio giornata)
        self._latency_samples = {
            "embedding": [],
            "retrieval": [],
            "build_context": [],
            "total": [],
        }

    # =========================================
    # METRICHE DI LATENZA
    # =========================================
    def _record_latency(self, embedding, retrieval, build, total):
        self._latency_samples["embedding"].append(embedding)
        self._latency_samples["retrieval"].append(retrieval)
        self._latency_samples["build_context"].append(build)
        self._latency_samples["total"].append(total)

    def get_latency_summary(self):
        summary = {}
        for phase, samples in self._latency_samples.items():
            if not samples:
                continue
            summary[phase] = {
                "n": len(samples),
                "mean": round(sum(samples) / len(samples), 3),
                "min": round(min(samples), 3),
                "max": round(max(samples), 3),
            }
        return summary

    def reset_latency(self):
        for v in self._latency_samples.values():
            v.clear()

    # =========================================
    # INDICIZZAZIONE
    # =========================================
    def index_summary(self, summary_date: str, hour: int, hour_label: str, summary: str):
        """Indicizza un singolo riepilogo orario."""
        doc_id = f"{summary_date}_{hour:02d}"
        self._collection.upsert(
            ids=[doc_id],
            documents=[summary],
            metadatas=[{"date": summary_date, "hour": hour, "hour_label": hour_label, "type": "orario"}]
        )

    def index_diary(self, summary_date: str, content: str):
        """Indicizza un diario giornaliero."""
        doc_id = f"diario_{summary_date}"
        self._collection.upsert(
            ids=[doc_id],
            documents=[content],
            metadatas=[{"date": summary_date, "type": "diario"}]
        )

    def index_weekly(self, start_date: str, end_date: str, content: str):
        """Indicizza un report settimanale."""
        doc_id = f"settimanale_{start_date}_{end_date}"
        self._collection.upsert(
            ids=[doc_id],
            documents=[content],
            metadatas=[{"date": end_date, "start_date": start_date, "type": "settimanale"}]
        )

    def index_monthly(self, year: int, month: int, content: str):
        """Indicizza un report mensile."""
        doc_id = f"mensile_{year}-{month:02d}"
        self._collection.upsert(
            ids=[doc_id],
            documents=[content],
            metadatas=[{"date": f"{year}-{month:02d}-01", "type": "mensile"}]
        )

    def index_existing_data(self, data_dir: str = "diari"):
        """Indicizza riepiloghi orari, diari giornalieri e report settimanali esistenti."""
        root = Path(data_dir)

        # Riepiloghi orari
        json_files = list(root.rglob("data.json"))
        print(f"[RAG] Indicizzazione di {len(json_files)} file orari...")
        indexed_orari = 0
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
                    indexed_orari += 1
            except Exception as e:
                print(f"[RAG] Errore su {path}: {e}")
        print(f"[RAG] Indicizzati {indexed_orari} riepiloghi orari.")

        # Diari giornalieri
        diary_files = list(root.rglob("diario.txt"))
        print(f"[RAG] Indicizzazione di {len(diary_files)} diari giornalieri...")
        indexed_diari = 0
        for path in diary_files:
            try:
                summary_date = path.parent.name
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                body = content.split("\n\n", 1)
                body = body[1] if len(body) > 1 else content
                self.index_diary(summary_date, body.strip())
                indexed_diari += 1
            except Exception as e:
                print(f"[RAG] Errore su {path}: {e}")
        print(f"[RAG] Indicizzati {indexed_diari} diari giornalieri.")

        # Report settimanali
        weekly_files = list(root.rglob("settimanale_*.txt"))
        print(f"[RAG] Indicizzazione di {len(weekly_files)} report settimanali...")
        indexed_weekly = 0
        for path in weekly_files:
            try:
                parts = path.stem.replace("settimanale_", "").split("_")
                start_date, end_date = parts[0], parts[1]
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                body = content.split("\n\n", 1)
                body = body[1] if len(body) > 1 else content
                self.index_weekly(start_date, end_date, body.strip())
                indexed_weekly += 1
            except Exception as e:
                print(f"[RAG] Errore su {path}: {e}")
        print(f"[RAG] Indicizzati {indexed_weekly} report settimanali.")

        # Report mensili
        monthly_files = list(root.rglob("mensile_*.txt"))
        print(f"[RAG] Indicizzazione di {len(monthly_files)} report mensili...")
        indexed_monthly = 0
        for path in monthly_files:
            try:
                parts = path.stem.replace("mensile_", "").split("-")
                year, month = int(parts[0]), int(parts[1])
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                body = content.split("\n\n", 1)
                body = body[1] if len(body) > 1 else content
                self.index_monthly(year, month, body.strip())
                indexed_monthly += 1
            except Exception as e:
                print(f"[RAG] Errore su {path}: {e}")
        print(f"[RAG] Indicizzati {indexed_monthly} report mensili.")

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

        # Fase 1: embedding della query
        t0 = time.perf_counter()
        query_embedding = self._ef([query])
        t1 = time.perf_counter()

        # Fase 2: retrieval da ChromaDB (passa l'embedding già calcolato
        # per evitare il doppio calcolo che farebbe query_texts)
        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=min(n_results, total),
            where=where,
            include=["documents", "metadatas"]
        )
        t2 = time.perf_counter()

        docs = results["documents"][0]
        metas = results["metadatas"][0]

        if not docs:
            self._record_latency(t1 - t0, t2 - t1, 0.0, t2 - t0)
            return ""

        # Fase 3: costruzione del contesto testuale
        lines = ["RIEPILOGHI RILEVANTI DALL'ARCHIVIO:"]
        for doc, meta in zip(docs, metas):
            label = meta.get("hour_label") or meta.get("type", "")
            lines.append(f"[{meta['date']} {label}] {doc}")
        context = "\n".join(lines)
        t3 = time.perf_counter()

        self._record_latency(t1 - t0, t2 - t1, t3 - t2, t3 - t0)
        return context

    def count(self) -> int:
        return self._collection.count()
