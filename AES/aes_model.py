import numpy as np
import pandas as pd
import nltk
import re
import textstat
import joblib
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings("ignore")

nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)

_stop_words = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_model.pkl")


def clean_text(text: str) -> str:
    text  = str(text).lower()
    text  = re.sub(r"[^a-zA-Z ]", " ", text)
    words = text.split()
    words = [_lemmatizer.lemmatize(w) for w in words if w not in _stop_words]
    return " ".join(words)


def extract_features(text: str) -> dict:
    words     = text.split()
    sentences = [s for s in text.split('.') if s.strip()]
    return {
        "word_count":       len(words),
        "unique_ratio":     len(set(words)) / (len(words) + 1),
        "avg_word_len":     float(np.mean([len(w) for w in words])) if words else 0.0,
        "sentence_count":   len(sentences),
        "avg_sentence_len": float(np.mean([len(s.split()) for s in sentences])) if sentences else 0.0,
        "flesch_score":     textstat.flesch_reading_ease(text),
        "flesch_kincaid":   textstat.flesch_kincaid_grade(text),
        "difficult_words":  textstat.difficult_words(text),
        "syllable_count":   textstat.syllable_count(text),
    }


def load_and_filter(csv_path: str, prompt_name=None):
    df = pd.read_csv(csv_path)
    if "prompt_name" in df.columns:
        if prompt_name is None:
            prompt_name = df["prompt_name"].value_counts().index[0]
        df = df[df["prompt_name"] == prompt_name].copy()
    df = df[["full_text", "score"]].dropna().copy()
    df = df.reset_index(drop=True)
    return df, prompt_name


def get_prompt_names(csv_path: str) -> list:
    df = pd.read_csv(csv_path)
    if "prompt_name" in df.columns:
        return df["prompt_name"].value_counts().index.tolist()
    return ["default"]


def train(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["cleaned"] = df["full_text"].apply(clean_text)

    feat_df = df["full_text"].apply(extract_features).apply(pd.Series)

    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned"], df["score"], test_size=0.2, random_state=42
    )

    vectorizer    = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    feat_train    = csr_matrix(feat_df.loc[X_train.index].values)
    feat_test     = csr_matrix(feat_df.loc[X_test.index].values)
    X_train_final = hstack([X_train_tfidf, feat_train])
    X_test_final  = hstack([X_test_tfidf,  feat_test])

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_final, y_train, test_size=0.1, random_state=42
    )

    model = XGBRegressor(
        n_estimators=1000, learning_rate=0.01, max_depth=4,
        random_state=42, tree_method="hist",
        reg_alpha=0.1, reg_lambda=1.5, min_child_weight=3,
        gamma=0.1, subsample=0.8, colsample_bytree=0.8,
        colsample_bylevel=0.8, early_stopping_rounds=50, eval_metric="rmse"
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=True)

    y_pred    = model.predict(X_test_final)
    score_min = int(df["score"].min())
    score_max = int(df["score"].max())
    y_pred_r  = np.clip(np.round(y_pred), score_min, score_max).astype(int)

    metrics = {
        "mse":         float(mean_squared_error(y_test, y_pred)),
        "rmse":        float(mean_squared_error(y_test, y_pred) ** 0.5),
        "r2":          float(r2_score(y_test, y_pred)),
        "mae":         float(mean_absolute_error(y_test, y_pred)),
        "exact_match": float(np.mean(y_pred_r == y_test)),
        "within_one":  float(np.mean(np.abs(y_pred_r - y_test) <= 1)),
        "within_two":  float(np.mean(np.abs(y_pred_r - y_test) <= 2)),
        "best_iter":   int(model.best_iteration),
        "score_min":   score_min,
        "score_max":   score_max,
        "n_train":     len(X_train),
        "n_test":      len(X_test),
    }

    return {
        "model":      model,
        "vectorizer": vectorizer,
        "metrics":    metrics,
        "y_test":     y_test,
        "y_pred":     y_pred,
        "y_pred_r":   y_pred_r,
    }


def predict(text: str, model, vectorizer, score_min: int, score_max: int) -> dict:
    cleaned  = clean_text(text)
    tfidf    = vectorizer.transform([cleaned])
    feats    = extract_features(text)
    feat_mat = csr_matrix(np.array(list(feats.values())).reshape(1, -1))
    combined = hstack([tfidf, feat_mat])
    raw      = float(model.predict(combined)[0])
    rounded  = int(np.clip(round(raw), score_min, score_max))
    return {"raw": raw, "rounded": rounded, "feats": feats}


def save_model(results: dict, path: str = MODEL_PATH) -> None:
    joblib.dump({
        "model":      results["model"],
        "vectorizer": results["vectorizer"],
        "score_min":  results["metrics"]["score_min"],
        "score_max":  results["metrics"]["score_max"],
        "metrics":    results["metrics"],
    }, path)
    print(f" Model saved to: {path}")


def load_model(path: str = MODEL_PATH) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved model at '{path}'. Run aes_model.py once to train & save.")
    data = joblib.load(path)
    print(f"Model loaded from: {path}")
    return data


def model_exists(path: str = MODEL_PATH) -> bool:
    return os.path.exists(path)


def get_or_train_model(csv_path: str, prompt_name=None, force_retrain: bool = False) -> dict:
    if not force_retrain and model_exists():
        return load_model()
    print("No saved model found. Training now...")
    df, prompt = load_and_filter(csv_path, prompt_name)
    print(f"Training on prompt: {prompt} | Essays: {len(df)}")
    results = train(df)
    save_model(results)
    return {
        "model":      results["model"],
        "vectorizer": results["vectorizer"],
        "score_min":  results["metrics"]["score_min"],
        "score_max":  results["metrics"]["score_max"],
        "metrics":    results["metrics"],
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, traceback

    csv = sys.argv[1] if len(sys.argv) > 1 else "ASAP.csv"
    print(f">> Script location : {os.path.abspath(__file__)}")
    print(f">> Model will save : {MODEL_PATH}")

    if model_exists():
        print(f" Model already exists. Delete saved_model.pkl to retrain.")
        data = load_model()
        m    = data["metrics"]
    else:
        try:
            print(f"Loading data from : {csv}")
            df, prompt = load_and_filter(csv)
            print(f"Prompt  : {prompt}")
            print(f"Essays  : {len(df)}")
            print(f"Scores  : {df['score'].min()} – {df['score'].max()}")
            print("\nTraining... (this runs only once)\n")
            results = train(df)
            save_model(results)          # ← saves right here
            m = results["metrics"]
        except Exception as e:
            print(f"\n ERROR: {e}")
            traceback.print_exc()
            sys.exit(1)

    print(f"\n{'='*40}")
    print(f"R²           : {m['r2']:.4f}")
    print(f"RMSE         : {m['rmse']:.4f}")
    print(f"MAE          : {m['mae']:.4f}")
    print(f"Exact match  : {m['exact_match']*100:.2f}%")
    print(f"Within ±1    : {m['within_one']*100:.2f}%")
    print(f"Within ±2    : {m['within_two']*100:.2f}%")
    print(f"{'='*40}")