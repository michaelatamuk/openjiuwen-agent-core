# ══════════════════════════════════════════════════════════════════════════════
# MEDIUM examples — ML-specific issues requiring sklearn/pandas knowledge.
# The baseline "generic code review" skill scores ~0.20–0.40 here because
# it won't naturally produce terms like fit_transform, stratify, GridSearchCV,
# StratifiedKFold, class_weight, or Pipeline.
# ══════════════════════════════════════════════════════════════════════════════

example_01 = {
    "task_input": (
        "Review this preprocessing code:\n\n"
        "scaler = StandardScaler()\n"
        "X_train_scaled = scaler.fit_transform(X_train)\n"
        "X_test_scaled  = scaler.fit_transform(X_test)  # ← note: fit_transform on test"
    ),
    "expected_behavior": (
        "Must identify `scaler.fit_transform(X_test)` as a bug. "
        "Calling `fit_transform` on the test set re-computes mean and std from test "
        "data, so each set is normalised to its own distribution. This breaks the "
        "invariant that test data must be processed identically to training data. "
        "Fix: `X_test_scaled = scaler.transform(X_test)` — apply the scaler that was "
        "already fitted on `X_train`. "
        "The correct pattern: `fit_transform` on train, `transform` only on test/val."
    ),
    "difficulty": "medium",
    "category": "preprocessing",
    "source": "golden",
}

example_02 = {
    "task_input": (
        "Review this classification evaluation:\n\n"
        "# Dataset: 950 samples class-0, 50 samples class-1\n"
        "model.fit(X_train, y_train)\n"
        "score = accuracy_score(y_test, model.predict(X_test))\n"
        "print(f'Model accuracy: {score:.1%}')"
    ),
    "expected_behavior": (
        "Must flag the misleading metric. With 95% class imbalance, a constant "
        "predictor that always returns class-0 achieves 95% accuracy while learning "
        "nothing. Accuracy is an unreliable metric for imbalanced datasets. "
        "Fix: use `f1_score(y_test, preds, average='weighted')` or "
        "`roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])`. "
        "Also add `class_weight='balanced'` to the classifier and `stratify=y` "
        "to `train_test_split` so the split preserves the class ratio."
    ),
    "difficulty": "medium",
    "category": "evaluation",
    "source": "golden",
}

example_03 = {
    "task_input": (
        "Review this train/test split:\n\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    X, y, test_size=0.2, random_state=42\n"
        ")"
    ),
    "expected_behavior": (
        "Must flag the missing `stratify=y` parameter for a classification task. "
        "Without stratification, a random 20% split can produce train and test sets "
        "with very different class distributions — particularly harmful for imbalanced "
        "data where the minority class might appear disproportionately in one split. "
        "Fix: `train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)`. "
        "Always use `stratify=y` for classification to ensure both splits have "
        "representative class proportions."
    ),
    "difficulty": "medium",
    "category": "data-splitting",
    "source": "golden",
}

example_04 = {
    "task_input": (
        "Review this hyperparameter search:\n\n"
        "best_k, best_score = None, 0\n"
        "for k in [3, 5, 10, 20]:\n"
        "    model = KNeighborsClassifier(n_neighbors=k)\n"
        "    model.fit(X_train, y_train)\n"
        "    score = model.score(X_test, y_test)\n"
        "    if score > best_score:\n"
        "        best_k, best_score = k, score\n"
        "print(f'Best k={best_k}, score={best_score}')"
    ),
    "expected_behavior": (
        "Must flag hyperparameter selection leakage — choosing `k` based on test set "
        "score makes the test set part of model selection. The reported `best_score` "
        "is now an optimistic, biased estimate of generalization because the test set "
        "was used to make a modelling decision. "
        "Fix: use `GridSearchCV(KNeighborsClassifier(), {'n_neighbors': [3, 5, 10, 20]}, cv=5)` "
        "on the training set only. Report `grid.best_params_` and reserve "
        "the test set for one final unbiased evaluation."
    ),
    "difficulty": "medium",
    "category": "model-selection",
    "source": "golden",
}

example_05 = {
    "task_input": (
        "Review this imputation code:\n\n"
        "imputer = SimpleImputer(strategy='mean')\n"
        "X_train = imputer.fit_transform(X_train)\n"
        "X_test  = imputer.fit_transform(X_test)   # ← fit_transform on test"
    ),
    "expected_behavior": (
        "Must flag `imputer.fit_transform(X_test)` — the imputer re-learns column "
        "means from the test set, causing two problems: "
        "(1) test data is normalised to its own means, not the training distribution; "
        "(2) test column means (computed from test samples) contaminate the imputer state. "
        "Fix: `X_test = imputer.transform(X_test)` using the imputer fitted on train. "
        "Better: wrap in a `Pipeline` so imputation is automatically confined to "
        "training folds during any cross-validation."
    ),
    "difficulty": "medium",
    "category": "preprocessing",
    "source": "golden",
}

example_06 = {
    "task_input": (
        "Review this model evaluation:\n\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\n"
        "model.fit(X_train, y_train)\n"
        "print(f'Test accuracy: {model.score(X_test, y_test):.3f}')"
    ),
    "expected_behavior": (
        "Must flag the high-variance evaluation. A single 80/20 split gives an "
        "estimate that heavily depends on which samples happen to fall in the test "
        "set — on small datasets the reported score can swing ±0.10 across runs "
        "even with `random_state` fixed. "
        "Fix: use k-fold cross-validation: "
        "`scores = cross_val_score(model, X, y, cv=StratifiedKFold(5), scoring='f1_weighted')` "
        "and report `f'{scores.mean():.3f} ± {scores.std():.3f}'`. "
        "Single splits are only appropriate for very large datasets (>100k samples)."
    ),
    "difficulty": "medium",
    "category": "evaluation",
    "source": "golden",
}

example_07 = {
    "task_input": (
        "Review this neural network training:\n\n"
        "model = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=500)\n"
        "model.fit(X_train, y_train)\n"
        "print(f'Train: {model.score(X_train, y_train):.3f}')\n"
        "print(f'Test:  {model.score(X_test, y_test):.3f}')"
    ),
    "expected_behavior": (
        "Must flag two issues: "
        "(1) Missing feature scaling — `MLPClassifier` uses gradient descent which "
        "is extremely sensitive to feature scale; unscaled features with different "
        "magnitudes cause slow convergence, vanishing/exploding gradients, and poor "
        "results. Fix: prepend `StandardScaler()` in a Pipeline. "
        "(2) Only reporting final scores — there is no early stopping or convergence "
        "check. With `max_iter=500` the model may overfit or not converge. "
        "Add `early_stopping=True, validation_fraction=0.1` to `MLPClassifier`."
    ),
    "difficulty": "medium",
    "category": "neural-networks",
    "source": "golden",
}

example_08 = {
    "task_input": (
        "Review this model comparison code:\n\n"
        "model.fit(X_train, y_train)\n"
        "train_score = model.score(X_train, y_train)\n"
        "print(f'Model performance: {train_score:.1%}')"
    ),
    "expected_behavior": (
        "Must flag reporting training score as the model's performance metric. "
        "Training score measures memorisation, not generalisation — a model that "
        "memorises all training examples scores 100% on train but fails on new data. "
        "Must also compute and report test score: "
        "`test_score = model.score(X_test, y_test)`. "
        "Compare both: a large gap between `train_score` and `test_score` (>0.10) "
        "indicates overfitting and should trigger regularisation, reduced complexity, "
        "or more training data."
    ),
    "difficulty": "medium",
    "category": "evaluation",
    "source": "golden",
}

GOLDEN_EXAMPLES_MEDIUM = [
    example_01, example_02, example_03, example_04,
    example_05, example_06, example_07, example_08,
]
