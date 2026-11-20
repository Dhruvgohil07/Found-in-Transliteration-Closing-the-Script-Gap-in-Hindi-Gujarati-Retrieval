"""Found in Transliteration: side-by-side search demo (Phase 11).

Launch from the repo root (see app/README.md for the prebuilt artefacts and the
IndicXlit service):

    streamlit run app/app.py

Type a query in Latin script (or native script); four methods search the same
collection and their top passages are shown side by side. All search logic is
in src/demo/pipeline.py and every setting comes from configs/demo.yaml, which
points at the experiment configs behind the reported results. This file only
lays out the page.
"""

import html
import sys
import urllib.request
from pathlib import Path

import streamlit as st

# Streamlit puts app/ on sys.path, not the repo root; src/ must be importable.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.demo.pipeline import Pipeline, SearchResult, highlight
from src.demo.store import PassageStore
from src.demo.xlit_client import XlitUnavailable
from src.utils.config import load_config, resolve_path

CONFIG = "configs/demo.yaml"
SNIPPET_CHARS = 400   # passage text shown per result (the full text is one click away)


@st.cache_resource(show_spinner=False)
def demo_config() -> dict:
    return load_config(CONFIG)


@st.cache_resource(show_spinner="Loading indexes, models and embeddings (first search only)...")
def pipeline(lang: str) -> Pipeline:
    cfg = demo_config()
    return Pipeline(lang, cfg["languages"][lang], cfg["xlit_service"])


@st.cache_resource(show_spinner=False)
def store(lang: str) -> PassageStore:
    return PassageStore(resolve_path(demo_config()["languages"][lang]["store"]))


def service_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=1) as response:
            return response.status == 200
    except OSError:
        return False


def marked(text: str, terms: set[str], rules: list[str]) -> str:
    """HTML for ``text`` with the query's terms highlighted (text is escaped first)."""
    return "".join(f"<mark>{html.escape(piece)}</mark>" if hit else html.escape(piece)
                   for piece, hit in highlight(text, terms, rules))


def show_column(result: SearchResult, col, passages: dict, top: int) -> None:
    relevant = set(result.judged["relevant"]) if result.judged else set()
    st.markdown(f"**{col.title}**")
    st.caption(f"{len(col.ranking):,} passages ranked · {col.seconds:.2f} s")
    if relevant:
        ranks = [r for r, (docno, _) in enumerate(col.ranking, start=1) if docno in relevant]
        st.caption(f"first relevant passage at rank **{ranks[0]}**" if ranks else "no relevant passage in the top 1000")
    if not col.ranking:
        st.info("No passage matches any query term.")
    for rank, (docno, score) in enumerate(col.ranking[:top], start=1):
        doc = passages.get(docno, {"title": "", "text": "(passage not in the store)"})
        tick = " ✅" if docno in relevant else ""
        text = doc["text"]
        short = text if len(text) <= SNIPPET_CHARS else text[:SNIPPET_CHARS] + " …"
        title = f"<b>{html.escape(doc['title'])}</b> · " if doc["title"] else ""
        st.markdown(
            f"<div style='font-size:0.9em;margin-bottom:0.8em'>"
            f"<span style='color:gray'>{rank}. {html.escape(docno)} · {score:.3f}</span>{tick}<br>"
            f"{title}{marked(short, col.terms, col.rules)}</div>",
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(page_title="Found in Transliteration", layout="wide")
    cfg = demo_config()
    langs = cfg["languages"]

    with st.sidebar:
        # A query typed for one language means nothing in the other: clear it on a switch.
        lang = st.radio("Language", list(langs), format_func=lambda code: langs[code]["label"],
                        on_change=lambda: st.session_state.update(query=""))
        top = st.slider("Passages per method", 3, 20, cfg["show_top"])
        if service_ok(cfg["xlit_service"]):
            st.success("IndicXlit service: running")
        else:
            st.warning("IndicXlit service: not reachable. Words from the experiments still work; "
                       "new words need the service (app/README.md).")
        st.caption("Settings of every column come from configs/demo.yaml, which points at the "
                   "experiment configs behind the reported results.")

    st.title("Found in Transliteration")
    st.write("Type a query the way people do, in Latin script (*kisan andolan news*), and see how each "
             f"method searches the {langs[lang]['label']} collection, which is written in native script.")

    st.write("Examples:")
    examples = langs[lang]["examples"]
    buttons = st.columns(len(examples))
    for button, example in zip(buttons, examples):
        if button.button(example["kind"], key=f"{lang}-{example['kind']}", help=example["query"],
                         width="stretch"):
            st.session_state["query"] = example["query"]

    query = st.text_input("Query", key="query", placeholder="e.g. kisan andolan news")
    if not query.strip():
        return

    try:
        result = pipeline(lang).search(query)
    except XlitUnavailable as e:
        st.error(str(e))
        return

    if result.native:
        st.write("Query in native script: no back-transliteration needed.")
    else:
        live = f" (live from IndicXlit: {', '.join(result.live_words)})" if result.live_words else ""
        st.write(f"Top-1 back-transliteration: **{result.bt1}**{live}")
    if result.judged:
        st.write(f"This is judged {result.judged['split']} query `{result.judged['qid']}`: "
                 "relevant passages are marked ✅.")

    shown = {docno for col in result.columns for docno, _ in col.ranking[:top]}
    passages = store(lang).get(sorted(shown))
    for column, col in zip(st.columns(len(result.columns)), result.columns):
        with column:
            show_column(result, col, passages, top)


main()
