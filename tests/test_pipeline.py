import pandas as pd

from src import segment, train


def test_extract_max_numeric_handles_messy_text():
    assert segment.extract_max_numeric("100 - 140 Nm") == 140
    assert segment.extract_max_numeric("$1,100,000 ") == 1_100_000
    assert pd.isna(segment.extract_max_numeric(None))
    assert pd.isna(segment.extract_max_numeric("N/A"))


def test_training_data_is_clean():
    df = train.load_data()
    assert (df["fuel_type"] == "gas").all()
    assert df[train.FEATURES + [train.TARGET]].notna().all().all()


def test_model_beats_baseline():
    report = train.evaluate(train.load_data())
    best = report["results"][report["best_model"]]
    assert best["test_mae"] < report["baseline_mae"] / 2


def test_segments_have_readable_names():
    df = segment.load_data()
    labels, best_k, _ = segment.fit_segments(df)
    df["segment_id"] = labels
    names = segment.name_segments(df)
    assert len(names) == best_k
    assert len(set(names.values())) == best_k
