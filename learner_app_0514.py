# learner_app_compact_0515.py
# ──────────────────────────
# Streamlit app: compact view – moves verbose text output into an expander
# Run with:  streamlit run learner_app_compact_0515.py

import streamlit as st
import pandas as pd
import json
from collections import defaultdict
from clustering_utils_0514 import readable_label, make_full_graph
import streamlit.components.v1 as components

st.set_page_config(layout="wide")
st.title("🧬 Learner Error → Replacement → Collocation Concept Groups")

# ──────────────────── Data helpers ────────────────────
@st.cache_data
def load_json(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)

@st.cache_data
def load_csv(path: str):
    return pd.read_csv(path)

DATA = load_json("data/merged_output_0515.json")
CSV  = load_csv("data/flat_report_all.csv")

# ───────────────────── Main UI ────────────────────────
err = st.text_input("Search for an error word:"  ).strip()

if err:
    # 0. basic validity check
    if err not in DATA or not DATA[err].get("replacements"):
        st.warning("No data found for this error word.")
        st.stop()

    # 1. JSON-level stats
    total_count = sum(r["count"] for r in DATA[err]["replacements"].values())
    st.header(f"🔹 Error: '{err}' — corrected {total_count} times")

    # 2. CSV slice & ordering
    df = CSV[CSV["錯誤字"] == err].copy()
    if df.empty:
        st.info("No detailed CSV data available for this word.")
        st.stop()

    sort_cols = [
        "此錯誤總次次數",
        "此錯誤-正確總錯誤次數",
        "此錯誤正確累積此搭配詞分類唯一次數",
        "此錯誤-正確搭配總錯誤次數",
    ]
    df = df.sort_values(sort_cols, ascending=[False]*4)

    df_with_colloc = df[df["此搭配詞"].notna() & (df["此搭配詞"] != "")]
    df_no_colloc   = df[df["此搭配詞"].isna()  | (df["此搭配詞"] == "")]

    # -----------------------------------------------------------------------
    # 3. EVERYTHING from here down to the graph lives inside an expander ✔︎
    # -----------------------------------------------------------------------
    with st.expander("📊 Correction details", expanded=False):
        # 3-a. detailed replacement-collocation summary
        st.subheader("1. Error → Replacement → Collocation")
        summary_data = defaultdict(lambda: {"count": 0, "collocations": {}})

        for rep, grp in df_with_colloc.groupby("正確搭配"):
            rep_count = grp["此錯誤-正確總錯誤次數"].iloc[0]
            summary_data[rep]["count"] = rep_count

            for concept, sg in grp.groupby("此搭配詞分類"):
                collocs = sg["此搭配詞"].unique().tolist()
                # keep the largest collocate list we meet for each (rep, concept)
                if len(collocs) > len(summary_data[rep]["collocations"].get(concept, [])):
                    summary_data[rep]["collocations"][concept] = collocs

            concept_strings = [
                f"- **{readable_label(c)}**: {', '.join(v)}"
                for c, v in summary_data[rep]["collocations"].items()
            ]
            st.markdown(f"* → **{rep}** ({rep_count}×)  " + " | ".join(concept_strings))

        if not df_no_colloc.empty:
            st.markdown(
                "##### Replacements with **no** collocations\n"
                + ", ".join(sorted(df_no_colloc["正確搭配"].unique()))
            )

        # 3-b. concept-centric overview
        st.subheader(f"2. Overall Replacement Groups for '{err}'")
        for concept, subdf in df_with_colloc.groupby("此搭配詞分類"):
            label = readable_label(concept)
            repls   = ", ".join(sorted(subdf["正確搭配"].unique()))
            collocs = ", ".join(sorted(subdf["此搭配詞"].unique()))
            st.markdown(f"- **{label}**  \n  "
                        f"Replacements: {repls}  \n  "
                        f"Collocations: {collocs}")

        # 3-c. optional raw table
        if st.checkbox("Show raw table ↔", value=False, key="show_table"):
            st.dataframe(df_with_colloc, use_container_width=True)

        # 3-d. “top confusable” quick view
        top_reps = (
            df_with_colloc
            .groupby("正確搭配")["此錯誤-正確總錯誤次數"]
            .max()
            .sort_values(ascending=False)
        )
        st.markdown(
            "##### Top confusable replacements\n"
            + ", ".join(top_reps.head(10).index)
        )

    # -----------------------------------------------------------------------
    # 4. Build graph-friendly data from `summary_data`
    # -----------------------------------------------------------------------
    graph_data = {
        err: {
            "replacements": {},
            "clusters": {},
        }
    }
    for rep, info in summary_data.items():
        graph_data[err]["replacements"][rep] = {
            "count": info["count"],
            "collocations": {},
        }
        for concept, collocs in info["collocations"].items():
            graph_data[err]["clusters"].setdefault(concept, collocs)
            graph_data[err]["replacements"][rep]["collocations"][concept] = {
                c: 1 for c in collocs
            }

    # 5. Show the PyVis graph (lazy-render on click)
    if st.button("🗺️ Show interactive graph"):
        st.info("Building graph – please wait…")
        net, _ = make_full_graph(graph_data, height="750px", width="100%")

        path = "pyvis_graph.html"
        net.write_html(path, notebook=False, open_browser=False)
        with open(path, "r", encoding="utf-8") as fh:
            components.html(fh.read(), height=750, scrolling=True)


with st.expander("🔍 Browse whole CSV", expanded=True):
    sort_by = st.multiselect(
        "Sort by columns:",
        options=list(CSV.columns),
        default=["此錯誤-正確總錯誤次數"],
        key="csv_sort",
    )
    st.dataframe(
        CSV.sort_values(sort_by, ascending=False),
        use_container_width=True,
    )
