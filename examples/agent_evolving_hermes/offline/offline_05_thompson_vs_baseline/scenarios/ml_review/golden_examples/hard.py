# ══════════════════════════════════════════════════════════════════════════════
# HARD examples — data leakage bugs that LOOK correct to a generic reviewer.
#
# Why the baseline scores near ZERO on these:
#   - `scaler.fit_transform(X)` before split IS valid Python — no syntax error
#   - `KFold` on patient rows IS cross-validation — sounds like "best practice"
#   - SMOTE before split DOES fix class balance — appears to help
#   - All code runs without exception — no obvious red flags
#
# A generic "review code quality" skill produces "looks good, add comments"
# on these examples.  The LLM-as-judge gives ~0.05–0.15 for that answer.
# An evolved skill that knows data leakage patterns scores ~0.70–0.90.
#
# Technical keywords that ONLY appear in correct answers:
#   data_leakage, Pipeline, fit_transform, GroupKFold, TimeSeriesSplit,
#   SMOTE, imblearn.pipeline, TargetEncoder, SelectKBest, nested_cross_validation,
#   temporal_leakage, group_leakage, target_encoding_leakage
# ══════════════════════════════════════════════════════════════════════════════

example_01 = {
    "task_input": (
        "Review this preprocessing pipeline:\n\n"
        "scaler = StandardScaler()\n"
        "X_scaled = scaler.fit_transform(X)          # scale entire dataset\n"
        "X_train, X_test, y_train, y_test = \\\n"
        "    train_test_split(X_scaled, y, test_size=0.2, random_state=42)"
    ),
    "expected_behavior": (
        "Must identify critical data_leakage — `StandardScaler.fit_transform(X)` "
        "is called on the ENTIRE dataset BEFORE the train/test split. "
        "The scaler learns the mean and std of the test samples, allowing test-set "
        "statistics to influence preprocessing. This inflates performance estimates. "
        "Fix: split first, then fit the scaler only on training data: "
        "`X_train, X_test = train_test_split(X, y, test_size=0.2); "
        "scaler.fit(X_train); X_train_s = scaler.transform(X_train); "
        "X_test_s = scaler.transform(X_test)`. "
        "Better: use `Pipeline([('scaler', StandardScaler()), ('model', model)])` "
        "which makes fit/transform boundaries automatic and correct."
    ),
    "difficulty": "hard",
    "category": "data-leakage",
    "source": "golden",
}

example_02 = {
    "task_input": (
        "Review this time-series feature engineering:\n\n"
        "df['next_day_return'] = df['price'].shift(-1) / df['price'] - 1\n"
        "df['price_momentum'] = df['price'] - df['price'].shift(-3)\n"
        "X = df[['next_day_return', 'price_momentum', 'volume']].dropna()\n"
        "y = (df['price'].shift(-1) > df['price']).dropna()\n"
        "model.fit(X, y)"
    ),
    "expected_behavior": (
        "Must identify temporal_leakage — `shift(-1)` and `shift(-3)` shift values "
        "backward in time, incorporating tomorrow's and three days' future prices "
        "as input features. At prediction time these future values do not exist. "
        "The model learns from information it could never have in production, "
        "producing unrealistically good backtest metrics. "
        "Fix: only use features available at prediction time: `shift(1)` for "
        "yesterday's values, or `shift(0)` for same-day values. "
        "For evaluation use `TimeSeriesSplit` to prevent training on future folds."
    ),
    "difficulty": "hard",
    "category": "data-leakage",
    "source": "golden",
}

example_03 = {
    "task_input": (
        "Review this medical model cross-validation:\n\n"
        "# patient_df has multiple rows per patient (one per visit)\n"
        "X = patient_df.drop('readmitted', axis=1)\n"
        "y = patient_df['readmitted']\n"
        "scores = cross_val_score(\n"
        "    model, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=42)\n"
        ")"
    ),
    "expected_behavior": (
        "Must identify group_leakage — the dataset has multiple rows per patient, "
        "so standard `KFold` places visits from the same patient in both training "
        "and test folds. The model memorises patient-specific patterns (age, "
        "diagnosis codes) rather than learning generalisable features. "
        "The reported cross-validation score is optimistically biased. "
        "Fix: use `GroupKFold(n_splits=5)` with `groups=patient_df['patient_id']`: "
        "`cross_val_score(model, X, y, cv=GroupKFold(5), groups=patient_df['patient_id'])`. "
        "This guarantees each patient appears in exactly one fold."
    ),
    "difficulty": "hard",
    "category": "data-leakage",
    "source": "golden",
}

example_04 = {
    "task_input": (
        "Review this categorical feature engineering:\n\n"
        "df['cat_mean_target'] = (\n"
        "    df.groupby('category')['target'].transform('mean')\n"
        ")\n"
        "X = df.drop('target', axis=1)\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, df['target'])"
    ),
    "expected_behavior": (
        "Must identify target_encoding_leakage — `groupby('category')['target'].transform('mean')` "
        "computes per-category mean targets using the ENTIRE dataset including the "
        "test set before the split. Target information leaks into input features. "
        "Fix: compute encoding only from training data and apply to test: "
        "first split, then `means = X_train.groupby('category')['target'].mean()`, "
        "then `X_test['cat_mean_target'] = X_test['category'].map(means)`. "
        "For cross-validation use `TargetEncoder` from `sklearn.preprocessing` "
        "inside a `Pipeline` which handles per-fold fitting automatically."
    ),
    "difficulty": "hard",
    "category": "data-leakage",
    "source": "golden",
}

example_05 = {
    "task_input": (
        "Review this feature selection + cross-validation:\n\n"
        "selector = SelectKBest(f_classif, k=10)\n"
        "X_selected = selector.fit_transform(X, y)  # uses all data\n"
        "scores = cross_val_score(model, X_selected, y, cv=5)"
    ),
    "expected_behavior": (
        "Must identify feature_selection_leakage — `SelectKBest.fit_transform(X, y)` "
        "uses the complete dataset (including what will become test folds) to select "
        "features. The selected features are chosen with knowledge of all labels, "
        "inflating the subsequent cross-validation scores. "
        "Fix: include feature selection INSIDE the cross-validation Pipeline: "
        "`pipeline = Pipeline([('selector', SelectKBest(f_classif, k=10)), ('model', model)]); "
        "cross_val_score(pipeline, X, y, cv=5)`. "
        "Inside `Pipeline`, `SelectKBest.fit_transform` is called only on the "
        "training portion of each fold."
    ),
    "difficulty": "hard",
    "category": "data-leakage",
    "source": "golden",
}

example_06 = {
    "task_input": (
        "Review this time-series cross-validation:\n\n"
        "scores = cross_val_score(\n"
        "    model, X_timeseries, y_timeseries,\n"
        "    cv=KFold(n_splits=5, shuffle=True, random_state=42)\n"
        ")"
    ),
    "expected_behavior": (
        "Must flag wrong cross-validation strategy for time-series data. "
        "`KFold` with `shuffle=True` randomly mixes observations across folds, "
        "so a model can be trained on data from 2023 and tested on data from 2021 — "
        "training on the 'future' relative to the test set. "
        "This is temporal_leakage and produces unrealistically optimistic scores. "
        "Fix: use `TimeSeriesSplit(n_splits=5)` which creates expanding training "
        "windows where each test fold is always strictly after its training fold. "
        "Never use `shuffle=True` with time-ordered data."
    ),
    "difficulty": "hard",
    "category": "data-leakage",
    "source": "golden",
}

example_07 = {
    "task_input": (
        "Review this oversampling code:\n\n"
        "smote = SMOTE(random_state=42)\n"
        "X_res, y_res = smote.fit_resample(X, y)   # applied to full dataset\n"
        "X_train, X_test, y_train, y_test = \\\n"
        "    train_test_split(X_res, y_res, stratify=y_res, random_state=42)"
    ),
    "expected_behavior": (
        "Must identify SMOTE leakage — oversampling is applied to the entire dataset "
        "before the split. SMOTE generates synthetic minority samples by interpolating "
        "between real samples. Some synthetic training samples are interpolations of "
        "real test samples, contaminating the test set. "
        "The reported test score is optimistically biased. "
        "Fix: apply SMOTE only after splitting: "
        "`X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y); "
        "X_train_res, y_train_res = SMOTE(random_state=42).fit_resample(X_train, y_train)`. "
        "For cross-validation, use `imblearn.pipeline.Pipeline` (not sklearn's) "
        "so SMOTE is applied per fold."
    ),
    "difficulty": "hard",
    "category": "data-leakage",
    "source": "golden",
}

example_08 = {
    "task_input": (
        "Review this nested cross-validation implementation:\n\n"
        "outer_scores = []\n"
        "for train_idx, test_idx in KFold(n_splits=5).split(X):\n"
        "    X_tr, X_te = X[train_idx], X[test_idx]\n"
        "    y_tr, y_te = y[train_idx], y[test_idx]\n"
        "    grid = GridSearchCV(model, param_grid, cv=5)\n"
        "    grid.fit(X_tr, y_tr)\n"
        "    outer_scores.append(grid.best_score_)  # collecting inner CV score"
    ),
    "expected_behavior": (
        "Must identify the nested_cross_validation bias — `grid.best_score_` is the "
        "INNER cross-validation score on the training fold, not the outer test fold. "
        "The inner score is the optimised best result from hyperparameter search; "
        "using it as the outer performance estimate gives an upward-biased evaluation. "
        "Fix: collect the outer test score instead: "
        "`outer_scores.append(grid.score(X_te, y_te))`. "
        "The outer test fold was held out from the inner `GridSearchCV`, so "
        "`grid.score(X_te, y_te)` gives an unbiased estimate of generalisation."
    ),
    "difficulty": "hard",
    "category": "evaluation",
    "source": "golden",
}

GOLDEN_EXAMPLES_HARD = [
    example_01, example_02, example_03, example_04,
    example_05, example_06, example_07, example_08,
]
