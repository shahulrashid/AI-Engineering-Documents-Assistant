import fitz

from core.ingestion import build_chunks, extract_pages, split_text


def make_pdf(pages: list[str]) -> bytes:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def test_extract_pages_keeps_page_numbers():
    pdf = make_pdf(["Hydraulic pump maintenance", "Torque spec for bolt A7"])
    pages = extract_pages(pdf)
    assert [p for p, _ in pages] == [1, 2]
    assert "Torque" in pages[1][1]


def test_split_text_respects_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = split_text(text, chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    # Overlap: the last word of one chunk should reappear at the start of the next.
    assert chunks[0].split()[-1] in chunks[1]


def test_split_text_rejects_bad_overlap():
    try:
        split_text("abc", chunk_size=10, chunk_overlap=10)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_build_chunks_preserves_page_and_doc_metadata():
    pages = [(1, "A" * 50), (2, "B" * 50)]
    chunks = build_chunks("manual.pdf", pages, chunk_size=100, chunk_overlap=10)
    assert {c.page for c in chunks} == {1, 2}
    assert all(c.doc_name == "manual.pdf" for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
