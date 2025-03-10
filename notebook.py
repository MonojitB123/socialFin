import marimo

__generated_with = "0.11.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import goodfire
    import os

    GOODFIRE_API_KEY = os.getenv("GOODFIRE_API_KEY")
    client = goodfire.Client(api_key=GOODFIRE_API_KEY)


    # Import necessary libraries
    from datasets import load_dataset

    # Load a dataset from Hugging Face
    dataset = load_dataset('alpindale/two-million-bluesky-posts') 

    # Split the dataset into train, validation, and test sets
    # Here we assume the dataset has a 'train' split; adjust according to your dataset's structure.
    train_test_split = dataset['train'].train_test_split(test_size=0.2)  # 80% train, 20% test
    train_valid_split = train_test_split['train'].train_test_split(test_size=0.25)  # 60% train, 20% valid

    # Combine the splits into a dictionary
    final_dataset = {
        'train': train_valid_split['train'],
        'validation': train_valid_split['test'],
        'test': train_test_split['test']
    }

    # Display the splits
    print(f"Training set: {len(final_dataset['train'])} samples")
    print(f"Validation set: {len(final_dataset['validation'])} samples")
    print(f"Test set: {len(final_dataset['test'])} samples")
    return (
        GOODFIRE_API_KEY,
        client,
        dataset,
        final_dataset,
        goodfire,
        load_dataset,
        os,
        train_test_split,
        train_valid_split,
    )


@app.cell
def _(final_dataset, goodfire):
    from transformers import pipeline
    import torch
        # Instantiate a Goodfire variant model using "meta-llama/Meta-Llama-3.1-8B-Instruct"
    variant = goodfire.Variant("meta-llama/Meta-Llama-3.1-8B-Instruct")
    device = 0 if torch.cuda.is_available() else -1
    train_dataset = final_dataset["train"].shuffle(seed=42).select([i for i in list(range(3000))])

    # Load a sentiment analysis model
    model_path= "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    sentiment_analyzer = pipeline("sentiment-analysis", model=model_path,tokenizer=model_path, max_length=512, truncation=True, batch_size=8, device=device)

    sentiments = []
    for text in train_dataset["text"]:
        result = sentiment_analyzer(text)
          # Extract the sentiment label
        sentiment_label = result[0]['label']
        sentiments.append(sentiment_label)

    train_dataset = train_dataset.add_column("answer", sentiments)
    return (
        device,
        model_path,
        pipeline,
        result,
        sentiment_analyzer,
        sentiment_label,
        sentiments,
        text,
        torch,
        train_dataset,
        variant,
    )


@app.cell
def _(train_dataset):
    train_dataset.to_parquet('bsky_with_sentiment.parquet')
    return


@app.cell
def _(train_dataset):
    shuffled_df = train_dataset.shuffle(seed=42).to_pandas()
    positive_examples = shuffled_df[shuffled_df["answer"] == "positive"]
    negative_examples = shuffled_df[shuffled_df["answer"] == "negative"]
    neutral_examples = shuffled_df[shuffled_df["answer"] == "neutral"]

    FEATURE_COMPUTE_SIZE = 60
    CLASSIFIER_FULL_SET_SIZE = 150
    return (
        CLASSIFIER_FULL_SET_SIZE,
        FEATURE_COMPUTE_SIZE,
        negative_examples,
        neutral_examples,
        positive_examples,
        shuffled_df,
    )


@app.cell
def _(
    FEATURE_COMPUTE_SIZE,
    client,
    negative_examples,
    positive_examples,
    variant,
):
    positive_news_features, negative_news_features = client.features.contrast(
        dataset_1=[
            [
                {
                    "role": "user",
                    "content": f"Is the following good or bad news for investors? {text}",
                },
                {"role": "assistant", "content": "good"},
            ]
            for text in positive_examples[0:FEATURE_COMPUTE_SIZE]["text"].tolist()
        ],
        dataset_2=[
            [
                {
                    "role": "user",
                    "content": f"Is the following good or bad news for investors? {text}",
                },
                {"role": "assistant", "content": "bad"},
            ]
            for text in negative_examples[0:FEATURE_COMPUTE_SIZE]["text"].tolist()
        ],
        model=variant,
        top_k=100,
    )

    # Rerank the contrastive features for relevance to financial market sentiment
    positive_news_features = client.features.rerank(
        features=positive_news_features,
        query="bull market",
        model=variant,
        top_k=50
    )
    negative_news_features = client.features.rerank(
        features=negative_news_features,
        query="bear market",
        model=variant,
        top_k=50
    )
    features_to_look_at = positive_news_features | negative_news_features
    features_to_look_at
    return features_to_look_at, negative_news_features, positive_news_features


@app.cell
def _():
    from itertools import combinations


    class FeatureSearch:
        """A class for systematically searching through combinations of features to evaluate their predictive power."""

        def __init__(self, feature_group):
            self.feature_group = feature_group

        def grid(self, k_features_per_combo: int = 2):
            """Perform a grid search over all possible combinations of features.

            Args:
                k_features_per_combo (int): The number of features to include in each combination.

            Returns:
                list: All possible k-sized combinations of features from the feature group.
            """

            # Get all possible combinations of features
            return list(combinations(self.feature_group, k_features_per_combo))
    return FeatureSearch, combinations


@app.cell
def _(GOODFIRE_API_KEY, goodfire):
    async_client = goodfire.AsyncClient(api_key=GOODFIRE_API_KEY)
    return (async_client,)


@app.cell
async def _(
    CLASSIFIER_FULL_SET_SIZE,
    async_client,
    features_to_look_at,
    goodfire,
    negative_examples,
    neutral_examples,
    positive_examples,
    variant,
):
    import pandas as pd
    import asyncio
    from tqdm.asyncio import tqdm_asyncio

    MIN_SAMPLES_PER_CLASS = min(
        len(negative_examples),
        len(positive_examples),
        len(neutral_examples),
        CLASSIFIER_FULL_SET_SIZE,
    )

    async def _get_feature_acts_for_sample_class(
        sample_class: pd.DataFrame,
        features_to_use_for_classification: goodfire.FeatureGroup,
        k=100,
        batch_size=10
    ):
        if k < len(features_to_use_for_classification):
            raise ValueError(
                "k must be greater than the number of features to use for classification"
            )

        samples = []
        all_samples = sample_class[0:MIN_SAMPLES_PER_CLASS]

        # Process in batches
        for i in range(0, len(all_samples), batch_size):
            batch = all_samples[i:i + batch_size]
            tasks = []

            for idx, row in batch.iterrows():
                text = row["text"]
                tasks.append(
                    async_client.features.inspect(
                        [
                            {
                                "role": "user",
                                "content": f"is the following good or bad for investors? {text}",
                            }
                        ],
                        model=variant,
                        features=features_to_use_for_classification,
                    )
                )

            # Process this batch
            batch_results = await tqdm_asyncio.gather(*tasks)
            for context in batch_results:
                features = context.top(k=k)
                samples.append(features)

        return samples

    async def process_all_classes():
        print("Computing positive news features...")
        positive_class_features = await _get_feature_acts_for_sample_class(
            positive_examples, features_to_look_at, k=100
        )

        print("Computing negative news features...")
        negative_class_features = await _get_feature_acts_for_sample_class(
            negative_examples, features_to_look_at, k=100
        )

        print("Computing neutral news features...")
        neutral_class_features = await _get_feature_acts_for_sample_class(
            neutral_examples, features_to_look_at, k=100
        )

        return positive_class_features, negative_class_features, neutral_class_features

    # Run in Colab
    positive_class_features, negative_class_features, neutral_class_features = await process_all_classes()
    return (
        MIN_SAMPLES_PER_CLASS,
        asyncio,
        negative_class_features,
        neutral_class_features,
        pd,
        positive_class_features,
        process_all_classes,
        tqdm_asyncio,
    )


@app.cell
def _(
    FeatureSearch,
    features_to_look_at,
    negative_class_features,
    positive_class_features,
):
    from sklearn import tree
    from sklearn import svm
    from sklearn.model_selection import train_test_split as tts
    from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score
    import numpy as np
    import tqdm


    def train_tree(x, y, depth):
        train_x, test_x, train_y, test_y = tts(
            x, y, train_size=0.5, random_state=42
        )

        # Create a nice regularized tree
        model = tree.DecisionTreeClassifier(
            max_depth=depth, min_samples_leaf=len(train_x) // 10, random_state=42
        )

        model.fit(train_x, train_y)

        pred = model.predict(test_x)

        # Calculate the f1 score of the model
        score = f1_score(test_y, pred)

        return model, pred, score


    def find_best_combo(features, k_features_per_combo=2):
        combos = FeatureSearch(features).grid(k_features_per_combo=k_features_per_combo)
        best_combo = None
        best_model = None
        mean_act_negative = 0
        mean_act_positive = 0
        support_vector_distances = 0
        best_score = 0

        for combo in tqdm.tqdm(combos):
            # Create a linear regression model
            def _select_feature_acts(combo, row):
                output = []
                for index, feature in enumerate(combo):
                    for feature_act in row:
                        if feature_act.feature.uuid == feature.uuid:
                            output.append(feature_act.activation)
                            break

                return output

            x_negative = [
                _select_feature_acts(combo, row) for row in negative_class_features
            ]
            # x_neutral = [_select_feature_acts(combo, row) for row in neutral_class_features]
            x_positive = [
                _select_feature_acts(combo, row) for row in positive_class_features
            ]

            y_negative = [-1] * len(x_negative)
            # y_neutral = [0] * len(x_neutral)
            y_positive = [1] * len(x_positive)

            x = x_negative + x_positive
            y = y_negative + y_positive

            model, pred, score = train_tree(x, y, depth=len(combo))

            if score > best_score:
                best_score = score
                best_combo = combo
                best_model = model

        return best_combo, best_score, best_model


    best_combo_at_k = {}
    for i in range(3):
        best_combo, best_score, best_model = find_best_combo(
            features_to_look_at, k_features_per_combo=i + 1
        )
        print(i + 1, best_combo, best_score, best_model)
        best_combo_at_k[i + 1] = (best_combo, best_score, best_model)
    return (
        accuracy_score,
        balanced_accuracy_score,
        best_combo,
        best_combo_at_k,
        best_model,
        best_score,
        f1_score,
        find_best_combo,
        i,
        np,
        svm,
        tqdm,
        train_tree,
        tree,
        tts,
    )


@app.cell
def _(best_combo_at_k):
    for k in [1, 2, 3]:
        combo, score, model = best_combo_at_k[k]
        print(f"k={k} features: score={score}")
    return combo, k, model, score


@app.cell
def _(best_combo_at_k, client):
    # Inspect features of the best performing model

    best_individual_feature = best_combo_at_k[2][0][0]

    client.features.neighbors(best_individual_feature,model="meta-llama/Meta-Llama-3.1-8B-Instruct" )
    return (best_individual_feature,)


@app.cell
def _(best_combo_at_k):
    BEST_TREE_INDEX = 3
    best_features = best_combo_at_k[BEST_TREE_INDEX][0]
    best_tree = best_combo_at_k[BEST_TREE_INDEX][2]
    return BEST_TREE_INDEX, best_features, best_tree


@app.cell
def _(best_features, best_tree, tree):
    # Let's visualize the tree

    import graphviz


    dot_data = tree.export_graphviz(
        best_tree,
        out_file=None,
        feature_names=[feature.label for feature in best_features],
        class_names=["negative", "positive"],
        filled=True,
        rounded=True,
        special_characters=True,
    )
    graph = graphviz.Source(dot_data)

    graph.render("graphname", format="png")
    return dot_data, graph, graphviz


@app.cell
def _():
    import marimo as mo
    _src = (
        "graphname.png"
    )
    mo.image(src=_src, width="2080px", height="577px")
    return (mo,)


@app.cell
def _(pd):
    df = pd.read_parquet('bsky_with_sentiment.parquet')

    df
    return (df,)


@app.cell
def _(df):
    import sweetviz as sv
    report = sv.analyze(df)
    # save file in html format
    report.show_html('report.html')
    return report, sv


if __name__ == "__main__":
    app.run()
