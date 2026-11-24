"""Streaming chunks, bounded IDF fitting and immutable index generations."""
from __future__ import annotations
from itertools import islice
from pathlib import Path
from nanoforge.artifacts import fingerprint_files
from nanoforge.rag.chunker import chunk_text
from nanoforge.rag.embeddings import TfIdfEmbedder, embedding_identity, get_embedder
from nanoforge.rag.store import NumpyVectorStore

def _iter_chunks(input_paths,text_key,text_columns,max_chunk_chars,overlap_chars):
    from nanoforge.data.formats import iter_dataset_records
    for record in iter_dataset_records(input_paths,text_key=text_key,text_columns=tuple(text_columns) if text_columns else None):
        for chunk in chunk_text(record.text,max_chunk_chars,overlap_chars):
            yield chunk,record.source,dict(record.metadata)

def build_index(input_paths,out_dir,backend="tfidf",max_chunk_chars=512,overlap_chars=64,text_key="text",
                text_columns=None,batch_size=64,max_docs=None,idf_fit_limit=100000,progress_callback=None,
                engine=None,st_model="all-MiniLM-L6-v2",tfidf_vocab=16384):
    if idf_fit_limit < 1:
        raise ValueError("idf_fit_limit must be positive")
    input_paths = [Path(p) for p in input_paths]
    embedder = get_embedder(backend,engine=engine,st_model=st_model,tfidf_vocab=tfidf_vocab)
    effective = "tfidf" if isinstance(embedder,TfIdfEmbedder) else backend
    if isinstance(embedder,TfIdfEmbedder):
        limit = min(idf_fit_limit,max_docs) if max_docs is not None else idf_fit_limit
        embedder.fit(item[0] for item in islice(_iter_chunks(input_paths,text_key,text_columns,max_chunk_chars,overlap_chars),limit))
        state = embedder.to_dict()
    else:
        state = embedding_identity(embedder)
    return NumpyVectorStore.build_and_save(out_dir,embedder,
        _iter_chunks(input_paths,text_key,text_columns,max_chunk_chars,overlap_chars),effective,state,
        batch_size,max_docs,progress_callback,encoder_identity=embedding_identity(embedder),
        provenance={"sources_fingerprint":fingerprint_files(input_paths),"max_chunk_chars":max_chunk_chars,
                    "overlap_chars":overlap_chars,"idf_fit_limit":idf_fit_limit})
