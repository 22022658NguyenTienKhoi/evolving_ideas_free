# Ideas Evolution Visualizer

A script to see how my ideas evolve over time.

<div align="center">
<img src="https://github.com/jennyzzt/evolving_ideas/raw/main/ideas_evolution.gif" width="600" alt="Ideas Evolution Visualization">
</div>

## What it does
1.	Take ideas.md as input
2.	Embed ideas with a text embedding model
3.	Reduce the dimensionality to 3D
4.	Visualise into an animation

## Quick Start

Put all your ideas into one markdown file, with each idea title starting with '#'. For example:
```md
# Idea Title
any other idea description
```

```bash
# Install requirements
pip install -r requirements.txt

# Set API key
export OPENAI_API_KEY="sk-…"

# Run
python ideas_evolution.py
```
