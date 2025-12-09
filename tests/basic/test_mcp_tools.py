from hoi4_rag.mcp_server import create_server


class FakeSearchResult(dict):
    pass


class FakeSearcher:
    def __init__(self):
        self.calls = []

    def search(self, query, top_k=5, filters=None, use_colbert_rerank=False):
        self.calls.append((query, top_k, filters, use_colbert_rerank))
        return [FakeSearchResult(chunk_id="id", text="t", score=1.0, metadata={})]

    @property
    def lancedb_store(self):
        class Dummy:
            def get_chunk_by_id(self, doc_id):
                return type(
                    "Obj",
                    (),
                    {
                        "id": doc_id,
                        "text": "txt",
                        "file_path": "p",
                        "domain": "d",
                        "concept": "c",
                        "version": "v",
                        "section_heading": "h",
                        "section_anchor": "a",
                        "parent_headings": [],
                    },
                )

            def _require_table(self):
                import pandas as pd

                data = [{"id": "id", "text": "txt", "section_heading": "h", "section_anchor": "a"}]
                return type(
                    "T",
                    (),
                    {"to_pandas": lambda self, filter=None, columns=None: pd.DataFrame(data)},
                )

        return Dummy()


class FakeReranker:
    def rerank(self, query, candidates, top_k):
        return candidates[:top_k]


def test_mcp_tools_search_and_section():
    server = create_server(settings=None, searcher=FakeSearcher(), reranker=FakeReranker())

    # search tool
    res = server.tools[0]("q", None, 3)  # type: ignore[index]
    assert res and res[0]["chunk_id"] == "id"

    # get_document_section
    section_tool = [t for t in server.tools if t.__name__ == "get_document_section"][0]
    section = section_tool("id", False)
    assert section["id"] == "id"
