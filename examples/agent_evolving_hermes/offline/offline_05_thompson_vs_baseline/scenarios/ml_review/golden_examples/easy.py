# ══════════════════════════════════════════════════════════════════════════════
# EASY examples — general pandas/sklearn issues a decent code reviewer catches.
# The baseline skill should score ~0.40–0.55 here; these provide the floor that
# prevents evolution from purely regressing on generic quality issues.
# ══════════════════════════════════════════════════════════════════════════════

example_01 = {
    "task_input": (
        "Review this training loop:\n\n"
        "for epoch in range(100):\n"
        "    df = pd.read_csv('training_data.csv')\n"
        "    X = df.drop('label', axis=1).values\n"
        "    y = df['label'].values\n"
        "    model.partial_fit(X, y)"
    ),
    "expected_behavior": (
        "Must flag the redundant I/O — `pd.read_csv()` is called every epoch, "
        "reading the same file from disk 100 times. This is a serious performance "
        "bug; CSV loading dominates the runtime. "
        "Fix: load the data once before the loop: "
        "`df = pd.read_csv('training_data.csv')` then use `X` and `y` in the loop. "
        "For large datasets that don't fit in memory, use a `DataLoader` or batch "
        "generator pattern instead of re-reading the full CSV."
    ),
    "difficulty": "easy",
    "category": "performance",
    "source": "golden",
}

example_02 = {
    "task_input": (
        "Review this split and training code:\n\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
        "model = RandomForestClassifier(n_estimators=100)\n"
        "model.fit(X_train, y_train)\n"
        "print(f'Score: {model.score(X_test, y_test)}')"
    ),
    "expected_behavior": (
        "Must flag the missing `random_state` parameter in both `train_test_split()` "
        "and `RandomForestClassifier`. Without a fixed `random_state`, every run "
        "produces a different train/test split and a different model, making results "
        "non-reproducible. "
        "Fix: `train_test_split(X, y, random_state=42)` and "
        "`RandomForestClassifier(n_estimators=100, random_state=42)`. "
        "All stochastic operations in an ML pipeline — splits, model initialization, "
        "data augmentation — must have a fixed `random_state` for reproducibility."
    ),
    "difficulty": "easy",
    "category": "reproducibility",
    "source": "golden",
}

example_03 = {
    "task_input": (
        "Review this row-wise computation:\n\n"
        "results = []\n"
        "for idx, row in df.iterrows():\n"
        "    score = row['feature_a'] * 0.3 + row['feature_b'] * 0.7\n"
        "    results.append(score)\n"
        "df['score'] = results"
    ),
    "expected_behavior": (
        "Must flag use of `.iterrows()` which is 100–1000x slower than vectorized "
        "pandas operations. Each call creates a Python object per row, bypassing "
        "numpy's C-level operations. "
        "Fix: replace the entire loop with a single vectorized expression: "
        "`df['score'] = df['feature_a'] * 0.3 + df['feature_b'] * 0.7`. "
        "For more complex per-row logic, prefer `.apply()` with a lambda, "
        "or better yet restructure to use `np.where` or `np.select` for "
        "conditional logic."
    ),
    "difficulty": "easy",
    "category": "performance",
    "source": "golden",
}

example_04 = {
    "task_input": (
        "Review this preprocessing step:\n\n"
        "X_train = df_train.drop('target', axis=1)\n"
        "y_train = df_train['target']\n"
        "model.fit(X_train, y_train)\n"
        "predictions = model.predict(X_test)"
    ),
    "expected_behavior": (
        "Must flag missing data validation before training. "
        "If `X_train` contains NaN or infinite values, `model.fit()` either raises "
        "a cryptic ValueError or silently produces wrong results depending on the "
        "estimator. "
        "Fix: check before fitting — `assert not X_train.isnull().any().any()` "
        "or use `SimpleImputer(strategy='median')` as a preprocessing step. "
        "Always add imputation and an outlier check (`np.isinf(X_train.values).any()`) "
        "to the pipeline before the estimator. "
        "The same check is needed for `X_test`."
    ),
    "difficulty": "easy",
    "category": "data-validation",
    "source": "golden",
}

GOLDEN_EXAMPLES_EASY = [example_01, example_02, example_03, example_04]
