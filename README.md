# SocialFin: An Interpretable Classifier based on Large scale Social Network Analysis

Mechanistic model interpretability is essential to understand AI decision making, ensuring safety, aligning with human values, improving model reliability and facilitating research. By revealing internal processes, it promotes transparency, mitigates risks, and fosters trust, ultimately leading to more effective and ethical AI systems in critical areas. In this study, we have explored social network data from BlueSky and built an easy-to-train, interpretable, simple classifier using Sparse Autoencoders features. We have used these posts to build a financial classifier that is easy to understand. Finally, we have visually explained important characteristics.

## Dataset 
A curated dataset based on Bluesky social network data is created. The dataset has random 3000 entries with an extra column called "answer" which provide "positive", "neutral" or "negative" label based on financial sentiment analysis.

# How to run

```
### Install uv 
curl -LsSf https://astral.sh/uv/install.sh | sh
### Create a virtual environment and actiavte
uv venv --python 3.11 && source .venv/bin/activate
## Install the dependencies
uv pip install -r requirements.txt
## Run the notebook using marimo
marimo edit notebook.py 
```



