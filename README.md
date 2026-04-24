# 🔮 WordGenie: Confusable Word Explorer

WordGenie is an interactive learning analytics application designed to help English language learners master "confusable words"—specifically adjectives that are often misused or overly general. By leveraging semantic clustering of collocations and LLM-powered pedagogical generation, WordGenie provides structured micro-lessons and interactive visualizations to clarify word usage.

## 🚀 Features

- **Confusable Word Explorer**: Search for common learner errors and explore the relationship between error adjectives, their corrections, and the nouns they collocate with.
- **Automated Micro-Lessons**: Generates structured, learner-friendly lessons using OpenAI's GPT-4, grouping corrections by semantic concepts.
- **Semantic Clustering**: Uses a hybrid approach for clustering collocates:
  - **Longman Lexicon**: Primary grouping based on the established Longman Lexicon categories.
  - **Vector Fallback**: Uses spaCy embeddings and Affinity Propagation/HDBSCAN for words missing from the lexicon.
- **Interactive Visualizations**: Explores collocation networks using interactive PyVis graphs.
- **Data-Driven Insights**: Powered by a large-scale dataset of learner errors, human corrections, and collocation data.

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **NLP & Embeddings**: [spaCy](https://spacy.io/)
- **Clustering & Machine Learning**: [scikit-learn](https://scikit-learn.org/) (Affinity Propagation), [HDBSCAN](https://hdbscan.readthedocs.io/)
- **LLM Integration**: [OpenAI API](https://openai.com/) (GPT-4)
- **Graph Visualization**: [PyVis](https://pyvis.readthedocs.io/), [NetworkX](https://networkx.org/)
- **Data Manipulation**: [pandas](https://pandas.pydata.org/), NumPy

## 📂 Project Structure

- `learner_app_0514.py`: The main Streamlit application script.
- `clustering_utils_0514.py`: Core logic for collocate clustering, Longman Lexicon integration, and graph generation.
- `data/`: Contains processed learner error datasets, correction pairs, and Longman Lexicon files.
- `notebooks/`: Jupyter notebooks (`0519.ipynb`, `ADJ_correction.ipynb`) for data exploration and model testing.
- `pyvis_graph.html`: Interactive graph output for visualization.

## 🚦 Getting Started

### Prerequisites

- Python 3.9+
- OpenAI API Key

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/Generative-Learning-Analytics.git
   cd Generative-Learning-Analytics
   ```

2. Install dependencies:

   ```bash
   pip install streamlit pandas openai spacy scikit-learn hdbscan networkx pyvis
   python -m spacy download en_core_web_lg
   ```

3. Set up your OpenAI API key in `.streamlit/secrets.toml` or as an environment variable:
   ```toml
   [openai]
   api_key = "your-api-key-here"
   ```

### Running the App

```bash
streamlit run learner_app_0514.py
```

## 🧠 Methodology

WordGenie implements a "Research-to-Lesson" pipeline:

1. **Error Extraction**: Identifies frequent adjective errors from learner corpora.
2. **Collocation Analysis**: Maps the nouns that co-occur with both the error and the correction.
3. **Semantic Mapping**: Groups these collocations using the Longman Lexicon or vector-based clustering to identify distinct usage "concepts."
4. **Pedagogical Generation**: Instructs GPT-4 to generate a structured table and micro-lesson based on these semantic clusters.
