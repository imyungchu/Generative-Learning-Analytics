import streamlit as st
import pandas as pd
import json
import openai
import os
from collections import defaultdict
import streamlit.components.v1 as components

st.set_page_config(layout="wide")
st.title("🔮 WordGenie: Confusable Word Explorer")
# ──────────────────── Secrets & OpenAI setup ────────────────────
openai.api_key = st.secrets["openai"]["api_key"]  # or your env fallback…

import openai

def generate_micro_lesson(word, context_examples=None, debug=False):
    # 1. Base instruction
    intro = (
        "You are an advanced English instructor. Please do not say somthing like 'If you want, I can also provide exercises or further examples!' because you are a lesson generator, not a chatbot.\n"
        "You are a small part of the app called WordGenie which target on solving confusable words by clustering collocation to make concept group based on error, correction and collocation."
        f"Your task is to generate a micro-lesson for learners who struggle with the overly general adjective '{word}' Group the collocaiton and correction by yourself and try to make group as small as possible.\n"
        f"Generate a structured semantic summary table that which concept(clusters) with which group of adjective replacements for the overly general adjective '{word}',\n"
    )

    # 2. Optional context block
    if context_examples:
        # join with bullets or numbered list for clarity
        ctx_block = "Here are some learner errors and human corrections with collocation group:\n"
        ctx_block += "\n".join(f"- {ex}" for ex in context_examples)
    else:
        ctx_block = ""

    # 3. Lesson request
    lesson = (
        "\n\nThen produce a concise, slide-style micro-lesson:\n"
        "- A brief generalization of the concept\n"
        "- 2–3 replacement with collocation illustrating each cluster\n"
        "- Use headings or table formatting as if for a learner handout\n"
    )

    # build final prompt
    prompt = "\n\n".join(part for part in (intro, ctx_block, lesson) if part)

    # debug print
    if debug:
        print("===== PROMPT =====")
        print(prompt)
        print("==================")

    # call OpenAI
    resp = openai.ChatCompletion.create(
        model="gpt-4.1-mini-2025-04-14",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5000,
        temperature=0,
    )
    return resp.choices[0].message.content

# ──────────────────── Data helpers ────────────────────
@st.cache_data
def load_json(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)

@st.cache_data
def load_csv(path: str):
    return pd.read_csv(path)

DATA = load_json("data/merged_output_20250518_1257.json")
CSV  = load_csv("data/disambiguated_collocations.csv")

@st.cache_data
def get_error_options():
    options = []
    df = pd.read_csv("data/error_options.csv", encoding="utf-8")
    for opt in df.to_dict(orient="records"):
        options.append(f'{opt["error"]} ({opt["total num of error occur"]}, {opt["pos"]})')
    return options
option = get_error_options()
selected = st.selectbox("Chose for the word in this dataset", options=option, index=0)

err = selected.split(" (")[0]

num = len(option)
st.markdown(f"There are {num} number of confusable words in this dataset")

if err:
    # 1. JSON-level stats
    total_count = CSV[CSV["error"] == err]["total num of error occur"].unique()[0]

    
    st.header(f"🔹 Word you can use instead of {err} - corrected {total_count} times")

    # 2. CSV slice & ordering
def default_summary():
    return {"count": 0, "collocations": {}}

@st.cache_data
def get_summary_data(err, CSV):
    df = CSV[CSV["error"] == err].copy()
    if df.empty:
        return None, None
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
    df_with_colloc = df_with_colloc.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    
    summary_data = defaultdict(default_summary)
    for rep, grp in df_with_colloc.groupby("correction"):
        rep_count = grp["num of this error-correction pair occur"].iloc[0]
        summary_data[rep]["count"] = rep_count
        for concept, sg in grp.groupby("category of collocation"):
            collocs = sg["collocation"].unique().tolist()
            unique_val = sg["total num of error time this error-correction pairs fall into this collocation category"].iloc[0]
            summary_data[rep]["collocations"][concept] = {"collocs": collocs, "unique_count": unique_val}
    return df_with_colloc, summary_data
    
df_with_colloc, summary_data = get_summary_data(err, CSV)

if df_with_colloc is None:
    st.info("No detailed CSV data available for this word.")
    st.stop()
    # st.write(df_with_colloc[sort_cols].head())
    # -----------------------------------------------------------------------
    # 3. EVERYTHING from here down to the graph lives inside axn expander ✔︎
    # -----------------------------------------------------------------------

lines=[(f"The word {err} - corrected {total_count} times in this database. The → correction with collocations and its concept group are listed below:")]
summary_data = defaultdict(lambda: {"count": 0, "collocations": {}})

for rep, grp in df_with_colloc.groupby("correction"):
    rep_count = grp["num of this error-correction pair occur"].iloc[0]
    summary_data[rep]["count"] = rep_count
    for concept, sg in grp.groupby("category of collocation"):
        # print(concept, sg)
        collocs = sg["collocation"].unique().tolist()
        unique_val = sg["total num of error time this error-correction pairs fall into this collocation category"].iloc[0]
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
        concept_name = CSV[CSV["category of collocation"] == concept]["category of collocation"].unique()[0]
        unique_val = coll_info["unique_count"]
        if concept_name != "misc-ap-0":
            concept_strings.append(f" {concept_name} ({unique_val}): ({', '.join(collocs)})")
    lines.append(f" → {rep} ({rep_count})  " + " ||| ".join(concept_strings))

with st.expander("📊 Correction details", expanded=True):
    st.code("\n".join(lines), language="markdown")

# ──────────────────── Generate micro-lesson button ────────────────────
if st.button("Generate Micro-Lesson with OpenAI"):
    # Pass your lines as the context_examples
    lesson = generate_micro_lesson(err, context_examples=lines)
    st.write("#### Micro-Lesson")
    st.write(lesson)

with st.expander(f"2. Concept-Based Clusters for '{err} with replacements and collocations", expanded=False):
    category_summary = []
    for concept, subdf in df_with_colloc.groupby("category of collocation"):
        # Count the unique corrections in this category
        repls = ", ".join(sorted(subdf["correction"].unique()))
        collocs = ", ".join(sorted(subdf["collocation"].unique()))
        # merged_phrases = ", ".join([
        #     f"{subdf['correction'].iloc[i]} {subdf['collocation'].iloc[i]}" 
        #     for i in range(len(subdf))
        # ])
        merged_phrases = ", ".join([
            f"{DATA[err]['replacements'][subdf['correction'].iloc[i]]['collocations'][subdf['collocation'].iloc[i]]['phrase']}"
            for i in range(len(subdf))
        ])
        group_corr_count = 0  # initialize counter for this group
        for j in range(len(subdf["collocation"])):
            group_corr_count += subdf["num of this error-correction pair occur"].iloc[j]
        category_summary.append((concept, group_corr_count, repls, merged_phrases))
        # Sort categories by the number of unique corrections (descending)
        category_summary = sorted(category_summary, key=lambda x: x[1], reverse=True)
        
    # st.subheader(f"2. Concept-Based Clusters for '{err}' (sorted by number of corrections)")
    for concept, corr_count, repls, coll in category_summary:
        # Use the CSV label for display (if available)
        label_arr = CSV[CSV["category of collocation"] == concept]["category of collocation"].unique()
        if label_arr.size > 0:
            label = label_arr[0]
        else:
            label = concept
        if label != "misc-ap-0":
            st.markdown(
                f"- **{label}** ({corr_count} corrections):\n\n  Replacements: {repls}\n\n  Collocations: {coll}"
            )
    # # 3-c. optional raw table
    if st.checkbox("Show raw table ↔", value=False, key="show_table"):
        st.dataframe(df_with_colloc.reset_index(drop=True), use_container_width=True)

# -----------------------------------------------------------------------
# 4. Build graph-friendly data from `summary_data`
# -----------------------------------------------------------------------
all_categories = list({
    concept
    for rep_info in summary_data.values()
    for concept in rep_info["collocations"].keys()
    if concept != "misc-ap-0"
})
selected_categories = st.multiselect(
    "category you want to show",
    options=all_categories,
    default=all_categories[0:3]
)

# Improved graph data structure
graph_data = {
    err: {
        "replacements": {},
        "categories": {},
        "collocations": {},
        "edges": []
    }
}

for rep, info in summary_data.items():
    graph_data[err]["replacements"][rep] = {"count": info["count"]}
    for concept, coll_info in info["collocations"].items():
        if concept not in selected_categories:
            continue
        # Add category node if not exists
        if concept not in graph_data[err]["categories"]:
            graph_data[err]["categories"][concept] = {}
        # Link correction to category
        graph_data[err]["edges"].append(("rep", rep, "cat", concept))
        for c in coll_info["collocs"]:
            # Add collocation node if not exists
            if c not in graph_data[err]["collocations"]:
                graph_data[err]["collocations"][c] = {}
            # Link category to collocation
            graph_data[err]["edges"].append(("cat", concept, "colloc", c))

# Now build the graph using PyVis
import networkx as nx
from pyvis.network import Network

G = nx.Graph()
# Add nodes
for rep in graph_data[err]["replacements"]:
    G.add_node(f"rep:{rep}", label=rep, color="#812503", shape="box")
for cat in graph_data[err]["categories"]:
    G.add_node(f"cat:{cat}", label=cat, color="#1f77b4", shape="ellipse")
for colloc in graph_data[err]["collocations"]:
    G.add_node(f"colloc:{colloc}", label=colloc, color="#2ca02c", shape="dot")

# Add edges
for edge in graph_data[err]["edges"]:
    if edge[0] == "rep" and edge[2] == "cat":
        G.add_edge(f"rep:{edge[1]}", f"cat:{edge[3]}", color="#888")
    elif edge[0] == "cat" and edge[2] == "colloc":
        G.add_edge(f"cat:{edge[1]}", f"colloc:{edge[3]}", color="#aaa")

net = Network(height="750px", width="100%", bgcolor="#222", font_color="white")
net.from_nx(G)
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