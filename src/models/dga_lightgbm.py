"""DGA classifier -- LightGBM on lexical features."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score

from src.features.lexical import lexical_features

FEATURE_COLS = [
    "length", "entropy", "digit_ratio", "vowel_consonant_ratio",
    "max_consonant_run", "ngram_score", "dict_word_ratio", "hex_char_ratio",
]

def build_feature_frame(domains, labels, families=None):
    df = pd.DataFrame({"domain": domains, "label": labels})
    if families is not None:
        df["family"] = families
    feats = df["domain"].apply(lexical_features).apply(pd.Series)
    return pd.concat([df, feats], axis=1)

def train(df, train_idx):
    model = LGBMClassifier(n_estimators=200, class_weight="balanced", verbosity=-1)
    model.fit(df.loc[train_idx, FEATURE_COLS], df.loc[train_idx, "label"])
    return model

def evaluate(model, df, test_idx):
    X_test = df.loc[test_idx, FEATURE_COLS]
    y_test = df.loc[test_idx, "label"]
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, preds, digits=3, output_dict=True, zero_division=0)
    auc = roc_auc_score(y_test, probs) if len(set(y_test)) > 1 else float("nan")
    return report, auc

def held_out_family_eval(df, held_out_family, benign_sample_frac=0.3, random_state=1):
    """Honest generalization test: train on every family except one, test on that one."""
    train_idx = df[df["family"] != held_out_family].index
    benign_test_idx = df[df["family"] == "benign"].sample(
        frac=benign_sample_frac, random_state=random_state
    ).index
    dga_test_idx = df[df["family"] == held_out_family].index
    test_idx = benign_test_idx.union(dga_test_idx)

    model = train(df, train_idx)
    report, auc = evaluate(model, df, test_idx)
    return report, auc, model