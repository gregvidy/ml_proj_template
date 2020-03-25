import pandas as pd
from sklearn import model_selection

"""
- binary classification
- multi class classification
- multi label classification
- single column regression
- multi column regression
- holdout
"""

class CrossValidation:
    def __init__(self,
                 df,
                 target_cols,
                 shuffle=False,
                 problem_type="binary_classification",
                 num_folds=5,
                 random_state=123
                 ):
        self.dataframe = df
        self.target_cols = target_cols
        self.shuffle = shuffle
        self.problem_type = problem_type
        self.num_folds = num_folds
        self.random_state = random_state
        self.num_target = len(target_cols)
        
        if self.shuffle is True:
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)
        
        self.dataframe["kfold"] = -1

    def split(self):
        if self.problem_type in ("binary_classification", "multiclass_classification"):
            if self.num_target != 1:
                raise Exception("Invalid number of targets for this problem!")
            target = self.target_cols[0]
            unique_values = self.dataframe[target].nunique()
            if unique_values == 1:
                raise Exception("Only one unique value found!")
            elif unique_values > 1:
                kf = model_selection.StratifiedKFold(n_splits=self.num_folds,
                                                     shuffle=self.shuffle,
                                                     random_state=self.random_state)
                for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe)):
                    self.dataframe.loc[val_idx, "kfold"] = fold

if __name__ == "__main__":
    df = pd.read_csv("../input/train.csv")
    cv = CrossValidation(df, shuffle=True,
                         target_cols=["target"],
                         problem_type="binary_classification")
    df_split = cv.split()
    print(df_split.head())
    print(df_split["kfolds"].value_counts())

