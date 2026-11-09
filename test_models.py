import breast_cancer_tree as bc
import car_title_tree as cars


def test_breast_cancer_data_shape():
    X, y = bc.load_data()
    assert len(X) == 569 and X.shape[1] == 30
    assert set(y.unique()) == {0, 1}


def test_depth_curve_is_reasonable():
    X, y = bc.load_data()
    curve = bc.depth_curve(X, y, range(2, 5))
    assert (curve["mean"] > 0.85).all()


def test_days_left_parser():
    assert cars.days_left("10 days left") == 10
    assert cars.days_left("12 hours left") == 0.5
    assert cars.days_left("Listing Expired") == 0


def test_car_features_have_no_text_columns():
    X, y = cars.load_data()
    assert all(dtype.kind in "if" for dtype in X.dtypes)
    assert 0 < y.mean() < 0.2
