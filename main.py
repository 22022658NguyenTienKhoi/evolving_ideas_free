import argparse
import copy
import json
from openai import OpenAI
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation
import plotly.express as px
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize



client = OpenAI()

def ideas_md_to_jsonl():
    # load data from ideas.md
    with open("ideas.md", "r") as file:
        ideas_text = file.read()

    # transform ideas into a list of dictionaries
    ideas_list = []
    for idea_text in ideas_text.split("\n# "):
        idea_text = idea_text.strip()
        if idea_text:
            idea_title = idea_text.split("\n")[0].strip()
            idea_content = "\n".join(idea_text.split("\n")[1:]).strip()
            ideas_list.append({
                "title": idea_title,
                "content": idea_content
            })

    # save the ideas list to a jsonl file
    with open("ideas.jsonl", "w") as jsonl_file:
        for idea in ideas_list:
            jsonl_file.write(json.dumps(idea) + "\n")


def embed_text(text):
    response = client.embeddings.create(
        input=text,
        # model="text-embedding-3-small"
        model="text-embedding-3-large",
    )
    return response.data[0].embedding


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_mdjsonl", action="store_true", help="Skip the conversion from markdown to JSONL.")
    parser.add_argument("--skip_embeddings", action="store_true", help="Skip the embedding process.")
    parser.add_argument("--skip_2d_embeddings", action="store_true", help="Skip the embedding process.")
    parser.add_argument("--embedding_separate", action="store_true", help="Use the 2D embeddings with each description as a separate embedding.")
    args = parser.parse_args()

    skip_mdjsonl = args.skip_mdjsonl or args.skip_embeddings or args.skip_2d_embeddings
    skip_embeddings = args.skip_embeddings or args.skip_2d_embeddings
    skip_2d_embeddings = args.skip_2d_embeddings

    # Convert ideas from markdown to jsonl
    if not skip_mdjsonl:
        ideas_md_to_jsonl()
        print("Ideas have been converted to ideas.jsonl")

    # Load the ideas from the jsonl file
    with open("ideas.jsonl", "r") as jsonl_file:
        ideas = [json.loads(line) for line in jsonl_file]
    print(f"Loaded {len(ideas)} ideas from ideas.jsonl")

    # Embed the ideas into a vector
    if not skip_embeddings:
        ideas_with_embeddings = copy.deepcopy(ideas)
        for idea in ideas_with_embeddings:
            idea['embedding_tgt'] = embed_text(idea['title'] + '\n\n' + idea["content"])
            idea['embeddings'] = [embed_text(xs) for xs in idea['content'].split('\n')]
            print(f"- embedding created: {idea['title']}")
        # save the embeddings
        with open("ideas_embeddings.jsonl", "w") as embeddings_file:
            for idea in ideas_with_embeddings:
                embeddings_file.write(json.dumps(idea) + "\n")
        print("Embeddings have been created and saved to ideas_embeddings.jsonl")


    # Load the embeddings from the jsonl file
    with open("ideas_embeddings.jsonl", "r") as embeddings_file:
        ideas_with_embeddings = [json.loads(line) for line in embeddings_file]
    print(f"Loaded {len(ideas_with_embeddings)} ideas with embeddings from ideas_embeddings.jsonl")

    # Downsample the embeddings to 3 dimensions
    if not skip_2d_embeddings:
        if not args.embedding_separate:
            embed_matrix = np.array([idea["embedding_tgt"] for idea in ideas_with_embeddings])
            print("Shape of embed_matrix:", embed_matrix.shape)
        else:
            embed_tmp = [idea["embeddings"] for idea in ideas_with_embeddings]
            embed_matrix = np.array([item for sublist in embed_tmp for item in sublist])
            print("Shape of embed_matrix:", embed_matrix.shape)
        pca = PCA(n_components=3, svd_solver="full", random_state=42)  # Changed to 3 components
        embed_3d = pca.fit_transform(embed_matrix)

        # Attach 3-D coords to each idea and write out
        out_path = "embeddings_3d.jsonl"
        with open(out_path, "w") as fh:
            for (x, y, z) in embed_3d:  # Added z coordinate
                idea_with_xyz = {
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),  # Added z coordinate
                }
                fh.write(json.dumps(idea_with_xyz) + "\n")

            print(f"3-D embeddings saved to {out_path}")

    # Load the 3D embeddings from the jsonl file
    with open("embeddings_3d.jsonl", "r") as embeddings_3d_file:
        embeddings_3d = [json.loads(line) for line in embeddings_3d_file]
    print(f"Loaded {len(embeddings_3d)} ideas with 3D embeddings from embeddings_3d.jsonl")



    # Create a 3D scatter plot using Plotly
    df = pd.DataFrame(embeddings_3d)
    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        color=df.index.astype(float),     # gives each point its own colour along a gradient
        opacity=0.85,
    )

    fig.update_traces(marker=dict(size=6))
    fig.update_layout(
        title="3-D Embeddings of Ideas",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
    )

    out_file = "ideas_interactive.html"
    fig.write_html(out_file)
    print(f"Interactive plot saved to {out_file}")



    # Create a 3D animation of the embeddings
    fig = plt.figure(figsize=(10, 10))
    ax  = fig.add_subplot(111, projection='3d')

    # Create a static colour-bar that maps 0‒1 → viridis
    norm = Normalize(vmin=0, vmax=1)
    sm   = ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm.set_array([])                          # no data needed
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.08, ticks=[])
    cbar.set_label("recency", fontsize=22)

    def animate(frame):
        ax.clear()
        ax.grid()
        ax.set_title("3D Embeddings of Ideas", fontsize=30)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        
        # Plot all points up to the current frame
        for i in range(frame + 1):
            idea = embeddings_3d[i]
            color_intensity = i / len(embeddings_3d)
            ax.scatter(
                idea["x"], idea["y"], idea["z"],
                color=plt.cm.viridis(color_intensity), s=100
            )
            
            # Arrow from previous to current point
            if i > 0:
                prev = embeddings_3d[i - 1]
                ax.quiver(
                    prev["x"], prev["y"], prev["z"],
                    idea["x"] - prev["x"],
                    idea["y"] - prev["y"],
                    idea["z"] - prev["z"],
                    color=plt.cm.viridis(color_intensity),
                    alpha=0.5, arrow_length_ratio=0.1,
                )

        # Consistent axis limits
        all_x = [p["x"] for p in embeddings_3d]
        all_y = [p["y"] for p in embeddings_3d]
        all_z = [p["z"] for p in embeddings_3d]
        ax.set_xlim(min(all_x) - 0.1, max(all_x) + 0.1)
        ax.set_ylim(min(all_y) - 0.1, max(all_y) + 0.1)
        ax.set_zlim(min(all_z) - 0.1, max(all_z) + 0.1)

        # Slow spin for visual flair
        ax.view_init(elev=20, azim=(frame % 360) * 1.2)

    # Build and save the animation
    anim = animation.FuncAnimation(
        fig, animate, frames=len(embeddings_3d),
        interval=100, repeat=False
    )
    anim.save("ideas_evolution.mp4", writer="ffmpeg")
    plt.close()
    print("Animation saved to ideas_evolution.mp4")
