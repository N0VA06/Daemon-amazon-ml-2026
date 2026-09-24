"""PDF unit test: pred {A,B,C}, truth {A,C} → 0.714."""

from src.evaluate import f05_entity, f05_from_counts, macro_f05


def test_pdf_example():
    score = f05_entity({"A", "B", "C"}, {"A", "C"})
    assert abs(score - 0.7142857142857143) < 1e-12
    assert abs(score - 0.714) < 1e-3


def test_from_counts_matches_formula():
    # 1.25 * TP / (0.25 * |truth| + |pred|)
    assert abs(f05_from_counts(2, 3, 2) - 1.25 * 2 / (0.25 * 2 + 3)) < 1e-12


def test_singleton():
    assert f05_entity([], []) == 1.0
    assert f05_entity(["X"], []) == 0.0
    assert f05_entity([], ["X"]) == 0.0


def test_macro_includes_singletons():
    truth = {"s1": {"A"}, "s2": set()}
    pred = {"s1": {"A"}, "s2": []}
    m = macro_f05(pred, truth, ["s1", "s2"])
    assert abs(m["macro_f05"] - 1.0) < 1e-12
    pred_bad = {"s1": {"A"}, "s2": ["Z"]}
    m2 = macro_f05(pred_bad, truth, ["s1", "s2"])
    assert abs(m2["macro_f05"] - 0.5) < 1e-12
