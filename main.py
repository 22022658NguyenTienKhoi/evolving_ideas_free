import argparse
import copy
import json
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from openai import OpenAI
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

client = OpenAI()

def ideas_md_to_jsonl():
    """Convert ideas.md into ideas.jsonl with one idea per line."""
    with open("ideas.md", "r", encoding="utf-8") as fh:
        ideas_text = fh.read()

    ideas_list = []
    for idea_block in ideas_text.split("\n# "):
        idea_block = idea_block.strip()
        if not idea_block:
            continue
        title, *content_lines = idea_block.split("\n")
        ideas_list.append({
            "title": title.strip(),
            "content": "\n".join(content_lines).strip(),
        })

    with open("ideas.jsonl", "w", encoding="utf-8") as out:
        for idea in ideas_list:
            out.write(json.dumps(idea) + "\n")


def embed_text(text: str):
    """Return an OpenAI embedding for *text*."""
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-large",
    )
    return response.data[0].embedding


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Embed ideas and create 3-D visualisations.")

    # Pipeline toggles
    p.add_argument("--skip_mdjsonl", action="store_true", help="Skip .md → .jsonl conversion")
    p.add_argument("--skip_embeddings", action="store_true", help="Skip embedding generation")
    p.add_argument("--skip_dimred", action="store_true", help="Skip dimensionality-reduction")

    # Embedding granularity
    p.add_argument("--embedding_separate", action="store_true", help="Embed each sentence separately")

    # Dimensionality-reduction algorithm
    p.add_argument("--downsample_method", choices=["pca", "tsne"], default="pca", help="Dimensionality-reduction algorithm")
    p.add_argument("--tsne_perplexity", type=float, default=30.0, help="Perplexity for t-SNE")
    p.add_argument("--tsne_lr", type=float, default=200.0, help="Learning-rate for t-SNE")

    return p


def main():
    args = build_parser().parse_args()

    # Derive the skip flags for backward-compatibility ----------------------------------------------------------
    skip_mdjsonl = args.skip_mdjsonl or args.skip_embeddings or args.skip_dimred
    skip_embeddings = args.skip_embeddings or args.skip_dimred

    # -----------------------------------------------------------------------------------------------------------
    # 1. Markdown → JSONL                                                                                        
    # -----------------------------------------------------------------------------------------------------------
    if not skip_mdjsonl:
        ideas_md_to_jsonl()
        print("✅ Converted ideas.md → ideas.jsonl")

    with open("ideas.jsonl", "r", encoding="utf-8") as fh:
        ideas = [json.loads(line) for line in fh]
    print(f"📄 Loaded {len(ideas)} ideas")

    # -----------------------------------------------------------------------------------------------------------
    # 2. Text → Embeddings                                                                                       
    # -----------------------------------------------------------------------------------------------------------
    if not skip_embeddings:
        ideas_with_embeddings = copy.deepcopy(ideas)
        for idea in ideas_with_embeddings:
            idea["embedding_tgt"] = embed_text(f"{idea['title']}\n\n{idea['content']}")
            idea["embeddings"] = [embed_text(sentence) for sentence in idea["content"].split("\n") if sentence]
            print(f"🔗 Embedded: {idea['title']}")

        with open("ideas_embeddings.jsonl", "w", encoding="utf-8") as out:
            for idea in ideas_with_embeddings:
                out.write(json.dumps(idea) + "\n")
        print("✅ Saved embeddings → ideas_embeddings.jsonl")

    with open("ideas_embeddings.jsonl", "r", encoding="utf-8") as fh:
        ideas_with_embeddings = [json.loads(line) for line in fh]
    print(f"📄 Loaded {len(ideas_with_embeddings)} ideas with embeddings")

    # -----------------------------------------------------------------------------------------------------------
    # 3. Dimensionality-reduction (PCA or t-SNE → 3-D)                                                           
    # -----------------------------------------------------------------------------------------------------------
    if not args.skip_dimred:
        if not args.embedding_separate:
            embed_matrix = np.array([idea["embedding_tgt"] for idea in ideas_with_embeddings])
        else:
            nested = [idea["embeddings"] for idea in ideas_with_embeddings]
            embed_matrix = np.array([item for sublist in nested for item in sublist])
        print("🔢 Embed-matrix shape:", embed_matrix.shape)

        if args.downsample_method == "pca":
            reducer = PCA(n_components=3, svd_solver="full", random_state=42)
            print("🗜️  Using PCA")
        else:
            reducer = TSNE(
                n_components=3,
                perplexity=args.tsne_perplexity,
                learning_rate=args.tsne_lr,
                init="random",
                random_state=42,
            )
            print("🗜️  Using t-SNE")

        embed_3d = reducer.fit_transform(embed_matrix)

        with open("embeddings_3d.jsonl", "w", encoding="utf-8") as fh:
            for (x, y, z) in embed_3d:
                fh.write(json.dumps({"x": float(x), "y": float(y), "z": float(z)}) + "\n")
        print("✅ Saved 3-D embeddings → embeddings_3d.jsonl")

    # -----------------------------------------------------------------------------------------------------------
    # 4. Visualisation                                                                                           
    # -----------------------------------------------------------------------------------------------------------
    with open("embeddings_3d.jsonl", "r", encoding="utf-8") as fh:
        embeddings_3d = [json.loads(line) for line in fh]
    print(f"📊 Loaded {len(embeddings_3d)} 3-D points")

    # -----------------------------------------------------------------------------------------------
    # Interactive Plotly scatter with conditional labelling
    # -----------------------------------------------------------------------------------------------
    if not args.embedding_separate:
        # One point per idea → we can label directly with titles
        df = pd.DataFrame(embeddings_3d)
        df["title"] = [idea["title"] for idea in ideas_with_embeddings]
        fig = px.scatter_3d(
            df,
            x="x",
            y="y",
            z="z",
            text="title",          # visible labels
            hover_name="title",    # hover tool-tip
            opacity=0.85,
            color=df.index.astype(float),
        )
        fig.update_traces(textposition="top center", marker=dict(size=6))
    else:
        # Multiple points per idea → fall back to colour gradient only
        df = pd.DataFrame(embeddings_3d)
        fig = px.scatter_3d(
            df,
            x="x",
            y="y",
            z="z",
            opacity=0.85,
            color=df.index.astype(float),
        )
        fig.update_traces(marker=dict(size=6))

    fig.update_layout(
        title="3-D Embeddings of Ideas",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        showlegend=False,
    )

    html_out = "ideas_interactive.html"
    fig.write_html(html_out)
    print(f"✅ Saved interactive plot → {html_out}")

    # -----------------------------------------------------------------------------------------------
    # Matplotlib animation (unchanged)
    # -----------------------------------------------------------------------------------------------
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.08, ticks=[])
    cbar.set_label("recency", fontsize=22)

    def animate(frame):
        ax.clear()
        ax.grid()
        ax.set_title("3-D Embeddings of Ideas", fontsize=30)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        for i in range(frame + 1):
            p = embeddings_3d[i]
            colour_idx = i / len(embeddings_3d)
            ax.scatter(p["x"], p["y"], p["z"], color=plt.cm.viridis(colour_idx), s=100)
            if i > 0:
                prev = embeddings_3d[i - 1]
                ax.quiver(prev["x"], prev["y"], prev["z"],
                          p["x"] - prev["x"], p["y"] - prev["y"], p["z"] - prev["z"],
                          color=plt.cm.viridis(colour_idx), alpha=0.5, arrow_length_ratio=0.1)

        xs = [p["x"] for p in embeddings_3d]
        ys = [p["y"] for p in embeddings_3d]
        zs = [p["z"] for p in embeddings_3d]
        ax.set_xlim(min(xs) - 0.1, max(xs) + 0.1)
        ax.set_ylim(min(ys) - 0.1, max(ys) + 0.1)
        ax.set_zlim(min(zs) - 0.1, max(zs) + 0.1)

        ax.view_init(elev=20, azim=(frame % 360) * 1.2)

    anim = animation.FuncAnimation(fig, animate, frames=len(embeddings_3d), interval=100, repeat=False)
    video_out = "ideas_evolution.mp4"
    anim.save(video_out, writer="ffmpeg")
    plt.close()
    print(f"✅ Saved animation → {video_out}")


if __name__ == "__main__":
    main()
