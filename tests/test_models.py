from evalsq.models import accuracy, fit_models, make_models, train_test_split_by_year

CUTOFF = 2020


def test_split_sizes(featured_df):
    train, test = train_test_split_by_year(featured_df, cutoff=CUTOFF)
    assert len(train) and len(test)
    assert train.index.year.max() < CUTOFF <= test.index.year.min()


def test_fit_models_matches_zoo(featured_df):
    train, _ = train_test_split_by_year(featured_df, cutoff=CUTOFF)
    scaler, models = fit_models(train)
    assert set(models) == set(make_models())


def test_accuracy_in_range(featured_df):
    train, test = train_test_split_by_year(featured_df, cutoff=CUTOFF)
    scaler, models = fit_models(train)
    for m in models.values():
        assert 0.0 <= accuracy(m, scaler, test) <= 1.0
