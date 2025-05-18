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
st.title("🔮 WordGenie: Confusable Word Explorer")

# ──────────────────── Data helpers ────────────────────
@st.cache_data
def load_json(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)

@st.cache_data
def load_csv(path: str):
    return pd.read_csv(path)

DATA = load_json("data/merged_output_20250518_1257.json")
CSV  = load_csv("data/flat_report_all.csv")

# add a common error list in CSV[err] sorted from most to least common error
word = st.selectbox("Chose for the word in this dataset", options=CSV["error"].unique().tolist(), index=0)
# most common error word
# st.code(f"top ten error words {CSV["error"].value_counts().head(10).to_string()}")

# ───────────────────── Main UI ────────────────────────
err = st.text_input("Search for word you want to explore:", word, placeholder="enter the word..." ).strip().lower()
# label="this is for finding the word you can use instead of the worrd that are too general",
if err:
    # 0. basic validity check
    if err not in DATA or not DATA[err].get("replacements"):
        st.warning("No data found for this error word.")
        st.stop()

    # 1. JSON-level stats
    total_count = DATA[err]["total count"]
    st.header(f"🔹 Word you can use instead of {err} - corrected {total_count} times")

    # 2. CSV slice & ordering
    df = CSV[CSV["error"] == err].copy()
    if df.empty:
        st.info("No detailed CSV data available for this word.")
        st.stop()

    sort_cols = [
        "total num of error occur",
        "num of this error-correction pair occur",
        "num of this error-correction phrase occur",
        "unique num of error-correction pairs fall into this collocation category",
        "total num of error time this error-correction pairs fall into this collocation category",
        "category of collocation",
        "collocation",
        "correction",
        "error",
    ]
    df = df.sort_values(sort_cols, ascending=[False]*len(sort_cols))

    df_with_colloc = df[df["collocation"].notna() & (df["collocation"] != "")]
    df_no_colloc   = df[df["collocation"].isna()  | (df["collocation"] == "")]
    df_with_colloc = df_with_colloc.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    # st.write(df_with_colloc[sort_cols].head())
    # -----------------------------------------------------------------------
    # 3. EVERYTHING from here down to the graph lives inside axn expander ✔︎
    # -----------------------------------------------------------------------
    with st.expander("📊 Correction details", expanded=False):
        lines=[(f"The word {err} - corrected {total_count} times in this database. The → correction with collocations and its concept group are listed below:")]
        summary_data = defaultdict(lambda: {"count": 0, "collocations": {}})

        for rep, grp in df_with_colloc.groupby("correction"):
            rep_count = grp["num of this error-correction pair occur"].iloc[0]
            summary_data[rep]["count"] = rep_count
            for concept, sg in grp.groupby("category of collocation"):
                # print(concept, sg)
                collocs = sg["collocation"].unique().tolist()
                unique_val = sg["unique num of error-correction pairs fall into this collocation category"].iloc[0]
                summary_data[rep]["collocations"][concept] = {"collocs": collocs, "unique_count": unique_val}
        sorted_summary = sorted(
            summary_data.items(),
            key=lambda item: item[1]["count"],
            reverse=True
        )
        for rep, info in sorted_summary:
            # print(rep, info)
            rep_count = info["count"]
            concept_strings = []
            for concept, coll_info in sorted(info["collocations"].items(),
                                               key=lambda x: x[1]["unique_count"],
                                               reverse=True):
                # print(concept, coll_info)
                collocs = coll_info["collocs"]
                concept_name = readable_label(concept)
                unique_val = coll_info["unique_count"]
                if concept_name != "Concept 0":
                    concept_strings.append(f" {concept_name} ({unique_val}): ({', '.join(collocs)})")
            lines.append(f" → {rep} ({rep_count})  " + " ||| ".join(concept_strings))
        if not df_no_colloc.empty:
            lines.append(
                "##### Corrections with **no** collocations\n"
                + ", ".join(sorted(df_no_colloc["correction"].unique()))
            )
        st.code("\n".join(lines), language="markdown")

        st.subheader(f"2. Concept-Based Clusters for '{err}'")
        for concept, subdf in df_with_colloc.groupby("category of collocation"):
            label = readable_label(concept)
            repls = ", ".join(sorted(subdf["correction"].unique()))
            collocs = ", ".join(sorted(subdf["collocation"].unique()))
            st.markdown(f"- **{label}**  \n  Replacements: {repls}  \n  Collocations: {collocs}")
        # 3-c. optional raw table
        if st.checkbox("Show raw table ↔", value=False, key="show_table"):
            st.dataframe(df_with_colloc.reset_index(drop=True), use_container_width=True)
        # 3-d. “top confusable” quick view
        top_reps = (
            df_with_colloc
            .groupby("correction")["num of this error-correction phrase occur"]
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
    all_categories = list({concept for rep_info in summary_data.values() 
                               for concept in rep_info["collocations"].keys()})
    selected_categories = st.multiselect("category you want to show", options=all_categories, default=all_categories[0:5])

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
        for concept, coll_info in info["collocations"].items():
            # 如果該類別不在使用者選擇的列表中，就跳過，不加入圖形
            if concept not in selected_categories:
                continue
            graph_data[err]["clusters"].setdefault(concept, coll_info["collocs"])
            graph_data[err]["replacements"][rep]["collocations"][concept] = {
                c: 1 for c in coll_info["collocs"]
            }

    # 5. Show the PyVis graph (lazy-render on click)
    # st.info("Building graph – please wait…")
    net, _ = make_full_graph(graph_data, height="750px", width="100%")

    path = "pyvis_graph.html"
    net.write_html(path, notebook=False, open_browser=False)
    with open(path, "r", encoding="utf-8") as fh:
        components.html(fh.read(), height=750, scrolling=True)


with st.expander("🔍 Browse whole CSV", expanded=False):
    sort_by = st.multiselect(
        "Sort by columns:",
        options=list(CSV.columns),
        default=["total num of error occur"],
        key="csv_sort",
    )
    st.dataframe(
        CSV.sort_values(sort_by, ascending=False).reset_index(drop=True),
        use_container_width=True,
    )
