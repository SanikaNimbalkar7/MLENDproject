import os
import dill
import sys
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException


def save_object(file_path, obj):
    """
    Save any Python object to disk using dill serialization.

    Args:
        file_path (str): Path where the object should be saved.
        obj: Python object to be saved.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(x_train, y_train, x_test, y_test, models, params):
    """
    Trains and evaluates multiple models using GridSearchCV.

    Args:
        x_train, y_train: Training features and target.
        x_test, y_test: Testing features and target.
        models (dict): Dictionary of models {name: model_instance}.
        params (dict): Dictionary of hyperparameters {name: param_grid}.

    Returns:
        dict: Model names mapped to their R² score on the test set.
    """
    try:
        report = {}

        for model_name, model in models.items():
            para = params.get(model_name, {})

            # Hyperparameter tuning
            gs = GridSearchCV(
                estimator=model,
                param_grid=para,
                cv=3,
                n_jobs=-1,
                scoring='r2',
                verbose=0
            )
            gs.fit(x_train, y_train)

            # Best tuned model
            best_model = gs.best_estimator_
            models[model_name] = best_model  # Update dict with tuned model

            # Predictions
            y_train_pred = best_model.predict(x_train)
            y_test_pred = best_model.predict(x_test)

            # Scores
            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            print(
                f"{model_name} | Best Params: {gs.best_params_} | "
                f"Train R²: {train_score:.4f} | Test R²: {test_score:.4f}"
            )

            report[model_name] = test_score

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Load any Python object from disk using dill deserialization.

    Args:
        file_path (str): Path of the saved object.

    Returns:
        Loaded Python object.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
