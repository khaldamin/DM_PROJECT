from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

def tune_hyperparameters(estimator, param_grid, X_train, y_train, method='random', n_iter=15, cv=3, scoring='f1', n_jobs=-1, random_state=42):
    """
    Executes GridSearchCV or RandomizedSearchCV hyperparameter tuning.
    """
    if method == 'grid':
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1
        )
    else:
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=1
        )

    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_
