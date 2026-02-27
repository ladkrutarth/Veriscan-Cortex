import os
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from typing import List, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / ".chroma_db_local"

class RAGEngineLocal:
    """
    RAG Engine using local embeddings (sentence-transformers) and ChromaDB.
    No external API calls are made during indexing or retrieval.
    """
    
    def __init__(self, db_path: str = str(DB_PATH)):
        self.db_path = db_path
        self._client = chromadb.PersistentClient(path=self.db_path)
        
        # Use a high-performance local embedding function
        self._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        self._collection = self._client.get_or_create_collection(
            name="veriscan_intel_local",
            embedding_function=self._embedding_function
        )
        print(f"✅ Local RAG Engine initialized with {self._collection.count()} documents.")

    def index_data(self, force: bool = False):
        """Load and index transaction and complaint data."""
        if self._collection.count() > 0 and not force:
            return

        print("⚡ Indexing local knowledge base...")
        
        # 1. Index Transactions (as demonstration of local data)
        txn_path = PROJECT_ROOT / "dataset" / "csv_data" / "processed_fraud_train.csv"
        if txn_path.exists():
            print(f"Indexing transactions from {txn_path}...")
            df_txn = pd.read_csv(txn_path).head(300)
            documents, metadatas, ids = [], [], []
            for i, row in df_txn.iterrows():
                text = f"Transaction: {row.get('merchant', 'Unknown')} in {row.get('category', 'Unknown')}. Amount: ${row.get('amt', 0)}. Location: {row.get('city', 'Unknown')}, {row.get('state', 'Unknown')}."
                documents.append(text)
                metadatas.append({"type": "transaction", "user_id": str(row.get("user_id", "Unknown"))})
                ids.append(f"txn_{i}")
            self._collection.add(documents=documents, metadatas=metadatas, ids=ids)

        # 2. Index CFPB Complaints (the actual knowledge base)
        cfpb_path = PROJECT_ROOT / "dataset" / "csv_data" / "cfpb_credit_card.csv"
        if cfpb_path.exists():
            print(f"Indexing CFPB complaints from {cfpb_path}...")
            # Load a larger chunk for better coverage
            df_cfpb = pd.read_csv(cfpb_path, nrows=1000)
            documents, metadatas, ids = [], [], []
            for i, row in df_cfpb.iterrows():
                issue = row.get('Issue', 'Financial Issue')
                complaint = row.get('Consumer complaint narrative', '')
                if pd.isna(complaint) or complaint == '':
                    complaint = f"Consumer reported an issue regarding {issue} with {row.get('Company', 'a financial institution')}."
                
                text = f"CFPB Complaint: {issue}. Narrative: {complaint}"
                documents.append(text)
                metadatas.append({"type": "complaint", "company": str(row.get("Company", "Unknown")), "state": str(row.get("State", "Unknown"))})
                ids.append(f"cfpb_{i}")
            
            # Batch add
            batch_size = 100
            for j in range(0, len(documents), batch_size):
                self._collection.add(
                    documents=documents[j:j+batch_size],
                    metadatas=metadatas[j:j+batch_size],
                    ids=ids[j:j+batch_size]
                )
            
        print(f"✅ Indexed {self._collection.count()} items locally.")

    def query(self, text: str, n_results: int = 5) -> List[Dict]:
        """Perform semantic search."""
        results = self._collection.query(
            query_texts=[text],
            n_results=n_results
        )
        
        parsed = []
        if results['documents']:
            for i in range(len(results['documents'][0])):
                parsed.append({
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i],
                    "confidence": round(1.0 / (1.0 + results['distances'][0][i]), 3)
                })
        return parsed

    def get_context_for_query(self, query_text: str) -> str:
        """Helper to get a flat string of context for the LLM."""
        results = self.query(query_text)
        return "\n".join([r["text"] for r in results])

if __name__ == "__main__":
    engine = RAGEngineLocal()
    engine.index_data(force=True)
    
    test_q = "Are there any high value transactions in travel?"
    print(f"\nQuery: {test_q}")
    results = engine.query(test_q)
    for r in results:
        print(f"[{r['confidence']}] {r['text']}")
