from metaflow import (
    FlowSpec,
    step,
    Parameter,
    IncludeFile,
    card,
    current
)
from metaflow.cards import Table, Markdown, Artifact
import numpy as np

def labeling_function(row):
    """
    Label a provided row based on the "rating" column value.
    
    Parameters:
    - row (pd.Series): A row from a DataFrame with a "rating" key.
    
    Returns:
    - int: 1 if rating is 4 or 5 (indicating a positive review), otherwise 0.
    """
    return 1 if row["rating"] in [4, 5] else 0

class ParallelFlow(FlowSpec):
    split_size = Parameter("split-sz", default=0.2)
    seed = Parameter("random_seed", default=42)
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
        self.next(self.baseline, self.make_grid)

    @step
    def baseline(self):
        from sklearn.dummy import DummyClassifier
        from sklearn.metrics import accuracy_score, roc_auc_score

        # Train and score the baseline model
        self.dummy_model = DummyClassifier(strategy="most_frequent")
        self.dummy_model.fit(self.traindf["review"], self.traindf["label"])
        self.probas = self.dummy_model.predict_proba(self.valdf["review"])[:, 1]
        self.preds = self.dummy_model.predict(self.valdf["review"])
        self.base_acc = accuracy_score(self.valdf["label"], self.preds)
        self.base_rocauc = roc_auc_score(self.valdf["label"], self.probas)

        self.next(self.final_join)

    @step
    def make_grid(self):
        from sklearn.model_selection import ParameterGrid

        param_values = {'tfidfvectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)]}
        self.grid_points = list(ParameterGrid(param_values))

        # evaluate each in cross product of ParameterGrid.
        self.next(self.tfidf_lr_text_model, 
                  foreach='grid_points')

    @step
    def tfidf_lr_text_model(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.pipeline import make_pipeline
        from sklearn.model_selection import cross_val_score

        # Get the current grid point
        current_grid_point = self.input

        # Tokenization using the ngram_range from the current grid point
        vectorizer = TfidfVectorizer(ngram_range=current_grid_point['tfidfvectorizer__ngram_range'])

        # Estimator
        estimator = LogisticRegression(max_iter=1_000)

        # Model
        self.lr_text_model = make_pipeline(vectorizer, estimator)

        # Score via 5-fold Cross-Validation
        self.model_scores = cross_val_score(self.lr_text_model,
                                            self.traindf["review"],
                                            self.traindf["label"],
                                            scoring="roc_auc",
                                            cv=5)

        self.lr_text_roc_auc = np.mean(self.model_scores)
        self.lr_text_model.fit(self.traindf["review"], self.traindf["label"])

        # Storing the model, parameters, oof data and score
        self.current_grid_point = current_grid_point
        self.current_model = self.lr_text_model
        self.current_score = self.lr_text_roc_auc
        self.oof_data = self.valdf


        self.next(self.tfidf_join)


    @step
    def tfidf_join(self, inputs):

        # Gather results from each tfidf_lr_text_model branch.
        self.lr_text_rocauc = np.max([inp.lr_text_roc_auc for inp in inputs])

        # Find the output with the highest score from the parallel branches
        best_input = max(inputs, key=lambda inp: inp.current_score)
        
        # Retrieve and store the best model, its parameters, and its score
        self.best_model = best_input.current_model
        self.best_parameters = best_input.current_grid_point
        
        self.oof_data = inputs.tfidf_lr_text_model.oof_data
        self.next(self.final_join)


    @step
    def final_join(self, inputs):
        self.base_rocauc = inputs.baseline.base_rocauc
        self.lr_text_rocauc = inputs.tfidf_join.lr_text_rocauc
        self.best_model = inputs.tfidf_join.best_model
        self.best_model_parameters = inputs.tfidf_join.best_parameters
        self.oof_data = inputs.tfidf_join.oof_data
        
        self.next(self.end)

    @card(type="corise")
    @step
    def end(self):
        from metaflow import Flow, current
        from sklearn.metrics import roc_auc_score

        # Is the model better than a baseline?
        self.beats_baseline = self.lr_text_rocauc > self.base_rocauc

        # Smoke testing
        _smoke_test_reviews = ["Hate it. Horrible shoes",
                               "Beautiful shoes. Loved it",
                              ]

        _smoke_test_preds = self.best_model.predict_proba(_smoke_test_reviews)[:,1]
        print(_smoke_test_preds) 
        negative_review = _smoke_test_preds[0] 
        positive_review = _smoke_test_preds[1] 
        self.passed_smoke_test = positive_review > negative_review

        if self.beats_baseline and self.passed_smoke_test:
            margin = self.lr_text_rocauc - self.base_rocauc
            print("Model passed the smoke test")
            print(f"The model beats the baseline by {margin:0.2f}")
            print(f"Model 5-fold CV ROC_AUC: {self.lr_text_rocauc:0.2f}")
            run = Flow(current.flow_name)[current.run_id]
            run.add_tag('Deployment_candidate')
        else:
            print("Review this model. Have not passed the minimal tests.")
        
        # Out of fold rocauc
        oof_preds = self.best_model.predict_proba(self.oof_data["review"])[:,1]
        self.oof_rocauc = roc_auc_score(self.oof_data["label"],oof_preds)
        print(f"Out of Fold Model ROC_AUC:{self.oof_rocauc:0.2f}") 

if __name__ == "__main__":
    ParallelFlow()
