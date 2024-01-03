import logging
logger = logging.getLogger(__name__)

import numpy as np
from sklearn.metrics import mean_absolute_error
from itertools import combinations, combinations_with_replacement

class WeightedAverageEnsemble:
    """
    This class implements a Weighted Average Ensemble method to select models 
    based on minimizing the Mean Absolute Error (MAE) on a validation dataset 
    while adhering to a budget constraint.

    Attributes:
        models (dict): A dictionary containing various models for ensemble.
        X_val (array-like): Validation dataset features.
        y_val (array-like): Validation dataset labels.
        budget (int, default=5): Maximum number of models allowed in the ensemble.

    Methods:
        __init__: Initializes the class with models, validation features, labels, and a budget.
        calculate_mae: Calculates the MAE for a given set of selected models.
        ensemble: Performs model selection based on a weighted average ensemble approach
        considering a budget restriction.
    """
    def __init__(self, models, X_val, y_val, budget=5):
        self.models = models
        self.X_val = X_val
        self.y_val = y_val
        self.budget = budget

    def calculate_mae(self, selected_models):
        predictions = np.zeros_like(self.y_val)
        
        for model in selected_models:
            predictions += self.models[model].predict(self.X_val)
        
        ensemble_prediction = predictions / len(selected_models)
        mae = mean_absolute_error(self.y_val, ensemble_prediction)
        
        return mae

    def ensemble(self):
        best_model = min(self.models, key=lambda model: mean_absolute_error(self.y_val, self.models[model].predict(self.X_val)))
        selected_models = [best_model]
        best_mae = self.calculate_mae(selected_models)

        for _ in range(self.budget):
            remaining_models = set(self.models.keys()) - set(selected_models)
            candidate_models = list(combinations_with_replacement(remaining_models, 1))

            for candidate_model in candidate_models:
                candidate_models_list = list(candidate_model)
                current_models = selected_models + candidate_models_list
                current_mae = self.calculate_mae(current_models)

                if current_mae < best_mae:
                    best_mae = current_mae
                    selected_models = current_models
                    print("Selected models:", selected_models)
                    print("Current MAE:", best_mae)

                else:
                    print("Failed to find a better ensemble")
                    break

        print("Final selected models:", selected_models)
        print("Best MAE:", best_mae)
        
        return selected_models, best_mae

class GreedyAlgorithmEnsemble:
    """
    This class implements a Greedy Algorithm Ensemble method to select a combination 
    of models that minimizes the Mean Absolute Error (MAE) on a validation dataset.

    Attributes:
        models (dict): A dictionary containing various models for ensemble.
        X_val (array-like): Validation dataset features.
        y_val (array-like): Validation dataset labels.

    Methods:
        __init__: Initializes the class with models, validation features, and labels.
        calculate_mae: Calculates the MAE for a given set of selected models.
        ensemble: Performs a greedy algorithm to select models minimizing the MAE.
    """
    def __init__(self, models, X_val, y_val):
        self.models = models
        self.X_val = X_val
        self.y_val = y_val

    def calculate_mae(self, selected_models):
        predictions = np.zeros_like(self.y_val)
        
        for model in selected_models:
            predictions += self.models[model].predict(self.X_val)
        
        ensemble_prediction = predictions / len(selected_models)
        mae = mean_absolute_error(self.y_val, ensemble_prediction)
        
        return mae

    def ensemble(self):
        selected_models = []
        best_mae = float('inf')

        while True:
            remaining_models = set(self.models.keys()) - set(selected_models)
            if not remaining_models:
                break

            candidate_models = list(combinations(remaining_models, 1))

            for candidate_model in candidate_models:
                candidate_models_list = list(candidate_model)
                current_models = selected_models + candidate_models_list
                current_mae = self.calculate_mae(current_models)

                if current_mae < best_mae:
                    best_mae = current_mae
                    selected_models = current_models
                    print("Selected models:", selected_models)
                    print("Current MAE:", best_mae)

        print("Final selected models:", selected_models)
        print("Best MAE:", best_mae)
        
        return selected_models, best_mae

class ClassificationBasedEnsemble:
    """
    This class utilizes a classification model (classifier_model) to predict a class 
    (0 or 1) based on sequence data (seq), and then selects regression models 
    (regression_models) optimized separately for each predicted class to minimize 
    Mean Absolute Error (MAE) on a separate set of target values (Tm).

    Attributes:
        seq (array-like): Sequence data used for prediction.
        Tm (array-like): Target values.
        classifier_model: Model predicting classes based on sequence data.
        regression_models (dict): Regression models optimized for different classes.

    Methods:
        __init__:Initializes the class with sequence data, target values, 
                a classifier model, and regression models.
        calculate_mae: Calculates the MAE for a given set of selected regression models.
        greedy_optimization: Performs a greedy optimization to select the best combination 
                of regression models based on minimizing MAE.
        using_cls: Uses the classifier to predict classes and selects optimized regression 
                models separately for each predicted class, aiming to minimize MAE.
    """
    def __init__(self, seq, Tm, classifier_model, regression_models):
        self.seq = seq
        self.Tm = Tm
        self.classifier_model = classifier_model
        self.regression_models = regression_models

    def calculate_mae(self, models):
        predictions = np.zeros_like(self.Tm)
        
        for model_name, model in models.items():
            predictions += model.predict(self.seq)
        
        ensemble_prediction = predictions / len(models)
        mae = mean_absolute_error(self.Tm, ensemble_prediction)
        
        return mae

    def greedy_optimization(self, models):
        selected_models = []
        best_mae = float('inf')

        for _ in range(len(models)):
            remaining_models = set(models.keys()) - set(selected_models)

            for candidate_model in remaining_models:
                current_models = selected_models + [candidate_model]
                current_mae = self.calculate_mae({model: models[model] for model in current_models})

                if current_mae < best_mae:
                    best_mae = current_mae
                    selected_models = current_models
                    print("Selected models:", selected_models)
                    print("Current MAE:", best_mae)
    
        print("Final selected models:", selected_models)
        print("Best MAE:", best_mae)
        
        return selected_models, best_mae

    def ensemble(self):
        org_class = self.classifier_model.predict(self.seq)
        print('*** Greedy optimization for non thermophilic ***')
        models_non_thermo, best_mae_non_thermo = self.greedy_optimization({key: value for key, value in self.regression_models.items() if org_class == 0})
        print('*** Greedy optimization for thermophilic ***')
        models_thermo, best_mae_thermo= self.greedy_optimization({key: value for key, value in self.regression_models.items() if org_class == 1})

        print("Selected models for non thermophilic (class 0):", models_non_thermo)
        print("Best MAE non thermo:", best_mae_non_thermo)
        print("Selected models for thermophilic (class 1):", models_thermo)
        print("Best MAE thermo:", best_mae_thermo)
        
        return models_non_thermo, best_mae_non_thermo, models_thermo, best_mae_thermo
