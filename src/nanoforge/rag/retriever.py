"""Blockwise bounded top-k retrieval without materializing the score corpus."""
from __future__ import annotations
import numpy as np

class Retriever:
    def __init__(self,store,embedder):
        if store.meta.dim != embedder.dim:
            raise ValueError("index encoder dimension mismatch; rebuild with the selected encoder")
        self.store,self.embedder = store,embedder
    def close(self):
        self.store.close()
    def __enter__(self):
        return self
    def __exit__(self, *_args):
        self.close()
    def query(self,text,top_k=3):
        if getattr(self.store, "_closed", False):
            raise RuntimeError("RAG index is closed")
        if isinstance(top_k,bool) or not isinstance(top_k,int) or not 1 <= top_k <= 1000:
            raise ValueError("top_k must be in [1,1000]")
        if not isinstance(text,str):
            raise ValueError("query must be text")
        vectors = self.store.vectors
        if not len(vectors) or not text.strip():
            return []
        q = np.asarray(self.embedder.embed_batch([text]),dtype=np.float32)
        if q.shape != (1,self.store.meta.dim) or not np.isfinite(q).all():
            raise ValueError("invalid query embedding")
        q = q[0]/max(float(np.linalg.norm(q[0])),1e-9)
        winners = []
        for start in range(0,len(vectors),4096):
            scores = vectors[start:start+4096]@q
            k = min(top_k,len(scores))
            indices = np.argpartition(-scores,k-1)[:k]
            winners.extend((float(scores[i]),start+int(i)) for i in indices)
            winners = sorted(winners,key=lambda item:(-item[0],item[1]))[:top_k]
        return [{**self.store.docs[index],"score":score} for score,index in winners]
