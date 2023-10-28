from metaflow import (
    FlowSpec,
    step,
    Parameter,
    IncludeFile,
    card,
    current
)
from metaflow.cards import Table, Markdown, Artifact

def labeling_function(row):
    """
    Label a provided row based on the "rating" column value.
    
    Parameters:
    - row (pd.Series): A row from a DataFrame with a "rating" key.
    
    Returns:
    - int: 1 if rating is 4 or 5 (indicating a positive review), otherwise 0.
    """
    return 1 if row["rating"] in [4, 5] else 0

class BaselineNLPFlow(FlowSpec):
    split_size = Parameter("split-sz", default=0.2)
    seed = Parameter("random_seed",default=42)
    data = IncludeFile("data", default="../data/Womens Clothing E-Commerce Reviews.csv")

    @step
    def start(self):
        import pandas as pd
        import io
        from sklearn.model_selection import train_test_split

        # Load and preprocess the dataset
        df = pd.read_csv(io.StringIO(self.data))
        df.columns = ["_".join(name.lower().strip().split()) for name in df.columns]
        df["review_text"] = df["review_text"].astype("str")
        _has_review_df = df[df["review_text"] != "nan"]
        reviews = _has_review_df["review_text"]
        labels = _has_review_df.apply(labeling_function, axis=1)
        self.df = pd.DataFrame({"label": labels, **_has_review_df})

        # Split data for training and validation
        _df = pd.DataFrame({"review": reviews, "label": labels})
        self.traindf, self.valdf = train_test_split(_df, 
                                                     test_size=self.split_size,
                                                     random_state=self.seed)
        self.next(self.baseline)

    @step
    def baseline(self):
        from sklearn.dummy import DummyClassifier
        from sklearn.metrics import accuracy_score, roc_auc_score

        # Train and score the baseline model
        dummy_model = DummyClassifier(strategy="most_frequent")
        dummy_model.fit(self.traindf["review"],self.traindf["label"])
        self.probas = dummy_model.predict_proba(self.valdf["review"])[:,1]
        self.preds = dummy_model.predict(self.valdf["review"])
        self.base_acc = accuracy_score(self.valdf["label"],self.preds)
        self.base_rocauc = roc_auc_score(self.valdf["label"],self.probas)

        self.next(self.end)

    @card(type="corise")
    @step
    def end(self):
        pass

if __name__ == "__main__":
    BaselineNLPFlow()
