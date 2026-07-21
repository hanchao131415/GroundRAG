import json
from types import MethodType

from langchain_core.documents import Document

from rag_modules.index_construction import IndexConstructionModule


def _module(tmp_path):
    module = IndexConstructionModule.__new__(IndexConstructionModule)
    module.index_save_path = str(tmp_path)
    module.vectorstore = None
    module.build_calls = []

    def load_index(self):
        return self.vectorstore

    def build_vector_index(self, chunks):
        self.build_calls.append(list(chunks))
        self.vectorstore = object()
        return self.vectorstore

    module.load_index = MethodType(load_index, module)
    module.build_vector_index = MethodType(build_vector_index, module)
    return module


def _doc(text, source="book.pdf", chunk_index=0):
    return Document(
        page_content=text,
        metadata={"source": source, "chunk_index": chunk_index},
    )


def test_manifest_preserves_duplicate_source_and_chunk_index(tmp_path):
    module = _module(tmp_path)
    chunks = [_doc("page one"), _doc("page two")]

    _, changed = module.build_incremental(chunks)

    manifest = json.loads((tmp_path / "chunk_hashes.json").read_text(encoding="utf-8"))
    assert changed is True
    assert len(manifest) == len(chunks)


def test_equivalent_second_run_skips_rebuild(tmp_path):
    module = _module(tmp_path)
    first = [_doc("alpha", chunk_index=0), _doc("beta", chunk_index=1)]
    equivalent = [_doc("alpha", chunk_index=0), _doc("beta", chunk_index=1)]

    _, first_changed = module.build_incremental(first)
    _, second_changed = module.build_incremental(equivalent)

    assert first_changed is True
    assert second_changed is False
    assert len(module.build_calls) == 1


def test_deletion_only_change_rebuilds_index(tmp_path):
    module = _module(tmp_path)
    chunks = [_doc("alpha", chunk_index=0), _doc("beta", chunk_index=1)]
    module.build_incremental(chunks)

    _, changed = module.build_incremental(chunks[:1])

    assert changed is True
    assert len(module.build_calls) == 2
    assert module.build_calls[-1] == chunks[:1]


def test_deleting_final_chunk_clears_persisted_index(tmp_path):
    module = _module(tmp_path)
    module.build_incremental([_doc("alpha")])
    (tmp_path / "index.faiss").write_bytes(b"stale")
    (tmp_path / "index.pkl").write_bytes(b"stale")

    vectorstore, changed = module.build_incremental([])

    assert changed is True
    assert vectorstore is None
    assert not (tmp_path / "index.faiss").exists()
    assert not (tmp_path / "index.pkl").exists()
    assert len(module.build_calls) == 1


def test_line_ending_and_trailing_space_changes_do_not_rebuild(tmp_path):
    module = _module(tmp_path)
    module.build_incremental([_doc("alpha  \r\nbeta")])

    _, changed = module.build_incremental([_doc("alpha\nbeta")])

    assert changed is False
    assert len(module.build_calls) == 1
