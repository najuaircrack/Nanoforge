"""Immutable-generation, bounded-memory vector index with legacy readers."""
from __future__ import annotations
import datetime
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
import numpy as np
from nanoforge.artifacts import atomic_write_json, fingerprint_file

@dataclass
class IndexMeta:
    backend: str
    dim: int
    num_docs: int
    embedder_state: dict = field(default_factory=dict)
    created_at: str = ""
    schema_version: int = 2
    files: dict = field(default_factory=dict)
    fingerprints: dict = field(default_factory=dict)
    encoder_identity: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    def to_dict(self):
        return asdict(self)
    @classmethod
    def from_dict(cls, raw):
        version = raw.get("schema_version",1)
        if isinstance(version,bool) or not isinstance(version,int) or version not in {1,2}:
            raise ValueError("unsupported index schema")
        for key in ("dim","num_docs"):
            value = raw.get(key)
            if isinstance(value,bool) or not isinstance(value,int) or value < (1 if key == "dim" else 0):
                raise ValueError("invalid index dimensions")
        values = {key:raw[key] for key in cls.__dataclass_fields__ if key in raw}
        values["schema_version"] = version
        return cls(**values)

class JsonlDocuments:
    """Indexed lazy JSONL metadata; text is read only for selected hits."""
    def __init__(self,path,offsets,max_doc_bytes=1048576):
        self.path,self.offsets,self.max_doc_bytes = Path(path),offsets,max_doc_bytes
    def __len__(self):
        return len(self.offsets)
    def __getitem__(self,index):
        if isinstance(index,slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if not 0 <= index < len(self):
            raise IndexError(index)
        with self.path.open("rb") as handle:
            handle.seek(int(self.offsets[index]))
            line = handle.readline(self.max_doc_bytes+1)
        if len(line)>self.max_doc_bytes:
            raise ValueError("index document byte limit exceeded")
        return json.loads(line)

class NumpyVectorStore:
    def __init__(self,vectors,docs,meta):
        if vectors.ndim != 2 or vectors.shape != (meta.num_docs,meta.dim) or len(docs) != meta.num_docs:
            raise ValueError("index integrity failure: dimensions/counts")
        self.vectors,self.docs,self.meta = vectors,docs,meta
        self._closed = False
    def close(self):
        if getattr(self, "_closed", True):
            return
        for array in (self.vectors, getattr(self.docs, "offsets", None)):
            mapping = getattr(array, "_mmap", None)
            if mapping is not None:
                mapping.close()
        self._closed = True
    def __enter__(self):
        return self
    def __exit__(self, *_args):
        self.close()
    def __del__(self):
        self.close()
    @classmethod
    def load(cls,index_dir,*,verify=True):
        base = Path(index_dir).resolve()
        meta_path = base/"index_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing index file: {meta_path}")
        meta = IndexMeta.from_dict(json.loads(meta_path.read_text(encoding="utf-8")))
        if meta.schema_version == 1:
            vectors = np.load(base/"vectors.npy",mmap_mode="r",allow_pickle=False)
            docs = json.loads((base/"metadata.json").read_text(encoding="utf-8"))
            if vectors.dtype != np.float32:
                raise ValueError("index dtype must be float32")
            return cls(vectors,docs,meta)
        expected = {"vectors","docs","offsets"}
        if set(meta.files) != expected or set(meta.fingerprints) != expected:
            raise ValueError("index integrity failure: generation files")
        paths = {}
        for name,relative in meta.files.items():
            relative = Path(relative)
            path = (base/relative).resolve()
            if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(base) or not path.is_file():
                raise ValueError("index integrity failure: invalid generation path")
            if verify and fingerprint_file(path) != meta.fingerprints[name]:
                raise ValueError("index integrity failure: fingerprint")
            paths[name] = path
        if paths["vectors"].stat().st_size != meta.num_docs*meta.dim*4 or paths["offsets"].stat().st_size != meta.num_docs*8:
            raise ValueError("index integrity failure: size")
        if meta.num_docs:
            vectors = np.memmap(paths["vectors"],mode="r",dtype=np.float32,shape=(meta.num_docs,meta.dim))
            offsets = np.memmap(paths["offsets"],mode="r",dtype=np.int64,shape=(meta.num_docs,))
            if int(offsets[0]) != 0 or np.any(offsets[1:] <= offsets[:-1]) or int(offsets[-1]) >= paths["docs"].stat().st_size:
                raise ValueError("index integrity failure: offsets")
        else:
            vectors,offsets = np.empty((0,meta.dim),dtype=np.float32),np.empty((0,),dtype=np.int64)
        docs = JsonlDocuments(paths["docs"],offsets)
        return cls(vectors,docs,meta)
    @classmethod
    def build_and_save(cls,index_dir,embedder,chunks,backend,embedder_state=None,
                       batch_size=64,max_docs=None,progress_callback=None,encoder_identity=None,provenance=None):
        if isinstance(batch_size,bool) or not isinstance(batch_size,int) or not 1 <= batch_size <= 65536:
            raise ValueError("batch_size must be in [1,65536]")
        if max_docs is not None and (isinstance(max_docs,bool) or not isinstance(max_docs,int) or max_docs < 0):
            raise ValueError("max_docs must be nonnegative")
        dim = int(embedder.dim)
        if not 1 <= dim <= 65536:
            raise ValueError("embedding dimension limit exceeded")
        base = Path(index_dir).resolve()
        generation = base/".generations"/uuid.uuid4().hex
        generation.mkdir(parents=True)
        paths = {"vectors":generation/"vectors.f32","docs":generation/"docs.jsonl","offsets":generation/"offsets.i64"}
        count = 0
        buffer = []
        with paths["vectors"].open("wb") as vecfile, paths["docs"].open("wb") as docfile, paths["offsets"].open("wb") as offsetfile:
            def flush():
                nonlocal count
                if not buffer:
                    return
                array = np.asarray(embedder.embed_batch([item[0] for item in buffer]),dtype=np.float32)
                if array.shape != (len(buffer),dim) or not np.isfinite(array).all():
                    raise ValueError("invalid embedding batch shape/values")
                norms = np.linalg.norm(array,axis=1,keepdims=True)
                array = array/np.maximum(norms,1e-9)
                vecfile.write(array.tobytes())
                for text,source,metadata in buffer:
                    raw = (json.dumps({"text_chunk":text,"source":source,"metadata":metadata},ensure_ascii=False,allow_nan=False)+"\n").encode("utf-8")
                    if len(raw)>1048576:
                        raise ValueError("index document byte limit exceeded")
                    offsetfile.write(np.asarray([docfile.tell()],dtype=np.int64).tobytes())
                    docfile.write(raw)
                    count += 1
                buffer.clear()
                if progress_callback:
                    progress_callback(count)
            for text,source,metadata in chunks:
                if max_docs is not None and count+len(buffer) >= max_docs:
                    break
                if not isinstance(text,str) or not isinstance(source,str) or not isinstance(metadata,dict):
                    raise ValueError("invalid index document")
                buffer.append((text,source,metadata))
                if len(buffer)>=batch_size:
                    flush()
            flush()
            for handle in (vecfile,docfile,offsetfile):
                handle.flush()
                os.fsync(handle.fileno())
        state = embedder_state or (embedder.to_dict() if hasattr(embedder,"to_dict") else {})
        meta = IndexMeta(backend,dim,count,state,datetime.datetime.now(datetime.timezone.utc).isoformat(),
                         files={name:path.relative_to(base).as_posix() for name,path in paths.items()},
                         fingerprints={name:fingerprint_file(path) for name,path in paths.items()},
                         encoder_identity=encoder_identity or {},provenance=provenance or {})
        # This is the only mutable pointer; readers see a whole old/new generation.
        atomic_write_json(base/"index_meta.json",meta.to_dict())
        return cls.load(base)
