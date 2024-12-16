import csv
import os
import random
from collections import Counter

import networkx as nx
import pandas as pd
import streamlit as st
from fuzzywuzzy import process
from streamlit_agraph import Config, Edge, Node, agraph

# Configurable Constants
MAX_EDGES_TO_SHOW = 200
GRAPH_HEIGHT = 800
GRAPH_WIDTH = 1200
HIGHLIGHTED_NODE_COLOR = "red"
HIGHLIGHTED_NODE_SIZE = 20
SEARCHED_NODE_COLOR = "#00FF00"  # Green for searched node
DEFAULT_NODE_COLOR = "blue"
DEFAULT_NODE_SIZE = 10
EDGE_THICKNESS = 6


# path_to_root_data = "/home/kbougatiotis/GIT/Prime_Adj/data/"

PREDEFINED_DATASETS = {
    "Toy": "./toy.csv",
    # "Simpathic": "/home/kbougatiotis/GIT/PAM_Biomedical/Simpathic/data/simpathic/simpathic-graph-triples.csv",
    # "YAGO3-10": os.path.join(path_to_root_data, "YAGO3-10", "train.txt"),
    # "YAGO3-10": os.path.join(path_to_root_data, "YAGO3-10-DR", "train_orig.txt"),
    # "codex-s": os.path.join(path_to_root_data, "codex-s", "train.txt"),
    # "WN18RR": os.path.join(path_to_root_data, "WN18RR", "train.txt"),
    # "FB15k-237": os.path.join(path_to_root_data, "FB15k-237", "train.txt"),
    # "NELL995": os.path.join(path_to_root_data, "NELL995", "train.txt"),
    # "DDB14": os.path.join(path_to_root_data, "DDB14", "train.txt"),
    # "lc-neo4j": os.path.join(path_to_root_data, "lc-neo4j", "train.txt"),
    # "ad-neo4j": os.path.join(path_to_root_data, "ad-neo4j", "train.txt"),
    # "DRKG": os.path.join(path_to_root_data, "DRKG", "train.tsv"),
    # "Hetionet": os.path.join(path_to_root_data, "Hetionet", "hetionet-v1.0-edges.tsv"),
    # "OpenBG500": os.path.join(path_to_root_data, "OpenBG500", "train.txt"),
    # "AD_NO_DB": os.path.join(path_to_root_data, "AD_NO_DB", "train_orig"),
}


@st.cache_data
# Function to load data
def load_data(file_path):
    with open(file_path, "r") as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.readline())
        print(dialect.delimiter)
        to_use = dialect.delimiter
    data = pd.read_csv(file_path, names=["head", "rel", "tail"], sep=to_use)
    return data


@st.cache_data
# Function to create and cache the graph
def create_graph(df, directed):
    # G = nx.DiGraph() if directed else nx.Graph()
    # for _, row in df.iterrows():
    #     G.add_edge(row["head"], row["tail"], rel=row["rel"])
    G = nx.from_pandas_edgelist(
        df, source="head", target="tail", edge_attr="rel", create_using=nx.DiGraph
    )
    return G


# Function to create induced subgraph
def get_induced_subgraph(G, selected_nodes, selected_relations, k):
    # Expand neighborhood
    induced_nodes = set()
    for node in selected_nodes:
        if node in G:
            induced_nodes.update(
                nx.single_source_shortest_path_length(G, node, cutoff=k).keys()
            )

    # Filter edges
    subgraph = G.subgraph(induced_nodes).copy()
    if selected_relations:
        edges_to_remove = [
            (u, v)
            for u, v, d in subgraph.edges(data=True)
            if d["rel"] not in selected_relations
        ]
        subgraph.remove_edges_from(edges_to_remove)

    # Subsample edges if too many
    if subgraph.number_of_edges() > MAX_EDGES_TO_SHOW:
        edges = list(subgraph.edges(data=True))
        sampled_edges = subsample_edges(edges, selected_nodes, MAX_EDGES_TO_SHOW)
        subgraph = nx.Graph()
        subgraph.add_edges_from(sampled_edges)

    return subgraph


# Function for edge subsampling
def subsample_edges(edges, selected_nodes, max_edges):
    # Prioritize edges connecting selected nodes
    connecting_edges = [
        e for e in edges if e[0] in selected_nodes or e[1] in selected_nodes
    ]
    sampled_edges = connecting_edges[:max_edges]

    # Add remaining edges to reach max_edges if needed
    if len(sampled_edges) < max_edges:
        additional_edges = edges[: max_edges - len(sampled_edges)]
        sampled_edges.extend(additional_edges)

    return sampled_edges


# Function to assign colors to relations
def assign_relation_colors(relations):
    random.seed(42)
    colors = ["#%06x" % random.randint(0, 0xFFFFFF) for _ in relations]
    return dict(zip(relations, colors))


# Function to convert graph to AGraph format
def convert_to_agraph(graph, selected_nodes, searched_node, relation_colors):
    nodes = []
    edges = []
    for node in graph.nodes():
        node_color = (
            SEARCHED_NODE_COLOR
            if node == searched_node
            else (
                HIGHLIGHTED_NODE_COLOR if node in selected_nodes else DEFAULT_NODE_COLOR
            )
        )
        node_size = (
            HIGHLIGHTED_NODE_SIZE
            if node in selected_nodes or node == searched_node
            else DEFAULT_NODE_SIZE
        )
        nodes.append(Node(id=node, label=node, color=node_color, size=node_size))
    for u, v, d in graph.edges(data=True):
        edges.append(
            Edge(
                source=u,
                target=v,
                label=d["rel"],
                color=relation_colors.get(d["rel"], "#000000"),
                thickness=EDGE_THICKNESS,
            )
        )
    return nodes, edges


# Streamlit UI
st.set_page_config(
    page_title="Focused Graph Visualization",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Report a bug": "mailto: bogas dot ko at gmail dot com",
        "About": "## Focused Graph Visualization",
    },
)
st.sidebar.header("Knowledge Graph Filters")
st.sidebar.write(
    """
## Visualize a KG focusing on specific nodes at a time.

Filter the wanted nodes and relations and see only the selected induced subgraph.

"""
)

# Dataset selection
dataset_choice = st.sidebar.selectbox(
    "Select a predefined dataset or upload your own:",
    ["Upload Your Own"] + list(PREDEFINED_DATASETS.keys()),
)


print(f"Loading the data\n")

if dataset_choice == "Upload Your Own":
    uploaded_file = st.sidebar.file_uploader("Upload your dataset (.csv)", type=["csv"])
    if uploaded_file:
        df = load_data(uploaded_file)
        directed = st.sidebar.checkbox("Consider Directionality", value=True)
        G = create_graph(df, directed)
    else:
        st.warning("Please upload a dataset to proceed.")
        st.stop()
else:
    print(f"Creating the graph\n")
    df = load_data(PREDEFINED_DATASETS[dataset_choice])
    # Directionality setting
    directed = st.sidebar.checkbox("Consider Directionality", value=True)
    G = create_graph(df, directed)


# Create and cache the graph


print(f"Counting rels\n")
# Relation frequency calculation
relation_counts = Counter(df["rel"])
sorted_relations = sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)
relation_colors = assign_relation_colors([r[0] for r in sorted_relations])


print("Filter\n")
# Filter Panel
all_nodes = set(df["head"]).union(df["tail"])
selected_nodes = st.sidebar.multiselect("Select Nodes", options=list(all_nodes))

# Relation filter with checkboxes
st.sidebar.subheader("Filter Relations")
selected_relations = []
for rel, count in sorted_relations:
    rel_checkbox = st.sidebar.checkbox(
        f"{rel} ({count} - {100*count/len(df):.2f} %)", value=True
    )
    if rel_checkbox:
        selected_relations.append(rel)

k_hops = st.sidebar.slider("Number of Hops", min_value=1, max_value=10, value=2)

# Generate Induced Subgraph
if selected_nodes:
    subgraph = get_induced_subgraph(G, selected_nodes, selected_relations, k_hops)

    # Node search feature
    searched_node = None
    fuzzy_matches = []
    node_search_query = st.text_input("Search for a node in the graph")
    if node_search_query:
        fuzzy_matches = [
            match
            for match, score in process.extract(
                node_search_query, list(subgraph.nodes()), limit=5
            )
            if score > 70
        ]
        st.write("Suggestions:", fuzzy_matches)

    if fuzzy_matches:
        searched_node = st.selectbox("Select a matching node:", fuzzy_matches)

    # Convert to AGraph format
    nodes, edges = convert_to_agraph(
        subgraph, selected_nodes, searched_node, relation_colors
    )

    # Graph Visualization
    config = Config(
        height=GRAPH_HEIGHT,
        width=GRAPH_WIDTH,
        directed=directed,
        nodeHighlightBehavior=True,
        collapsible=False,
        staticGraph=True,  # Fix nodes in place
    )

    col1, col2 = st.columns([4, 1])

    with col1:
        agraph(nodes=nodes, edges=edges, config=config)

    with col2:
        st.subheader("Relation Color Legend")
        for rel, color in relation_colors.items():
            st.markdown(
                f"<div style='background-color:{color};width:20px;height:20px;display:inline-block;'></div> {rel}",
                unsafe_allow_html=True,
            )

else:
    st.warning("Please select at least one node to display the graph.")
