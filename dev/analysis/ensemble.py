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
            print('remaining',remaining_models)
            print('remaining',len(remaining_models))
            print('candidates', candidate_models)

            for candidate_model in candidate_models:
                candidate_models_list = list(candidate_model)
                current_models = selected_models + candidate_models_list
                current_mae = self.calculate_mae(current_models)
                print('current mae', current_mae)

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
            #print('model_preds',self.models[model].predict(self.X_val))
        
        ensemble_prediction = predictions / len(selected_models)
        mae = mean_absolute_error(self.y_val, ensemble_prediction)
        
        return mae

    def ensemble(self):
        #selected_models = []
        #best_mae = float('inf')
        
        best_model = min(self.models, key=lambda model: mean_absolute_error(self.y_val, self.models[model].predict(self.X_val)))
        selected_models = [best_model]
        print('best selected model', best_model)
        best_mae = self.calculate_mae(selected_models)
        print('best model mae', best_model)
        
        model_keys = list(self.models.keys())
        
        remaining_models = sorted(set(self.models.keys()) - set(selected_models))
        prev_remaining_models = set()

        
        while remaining_models != prev_remaining_models:
            for i in range(3):
                remaining_models = sorted(set(self.models.keys()) - set(selected_models))
                print('remaining',remaining_models)
                print('remaining',len(remaining_models))
                
                prev_remaining_models = remaining_models.copy() 

                candidate_models = list(combinations(remaining_models, 1))
                print('candidates',candidate_models)

                for candidate_model in candidate_models:
                    candidate_models_list = list(candidate_model)
                    current_models = selected_models + candidate_models_list
                    print('current models',current_models)
                    current_mae = self.calculate_mae(current_models)
                    print('current mae', current_mae)

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
        self.classifier_model =classifier_model
        self.regression_models = regression_models

    def calculate_mae(self, models,Tm, seq):
        predictions = np.zeros_like(Tm)
        
        for  model in models:
            predictions += self.regression_models[model].predict(seq)
            #print('model_predict',model.predict(seq))
        #print(('mae predictions',predictions))
        #print('real label', Tm)
        
        ensemble_prediction = predictions / len(models)
        mae = mean_absolute_error(Tm, ensemble_prediction)
        
        return mae

    def greedy_optimization(self, models,Tm,seq):
        #selected_models = []
        #best_mae = float('inf')

        best_model = min(models, key=lambda model: mean_absolute_error(Tm, models[model].predict(seq)))
        selected_models = [best_model]
        print('best selected model', best_model)
        best_mae = self.calculate_mae(selected_models,Tm,seq)
        print('best model mae', best_model)
        
        remaining_models = sorted(set(models.keys()) - set(selected_models))
        prev_remaining_models = set()

        while remaining_models != prev_remaining_models:
            for i in range(3):
                remaining_models = sorted(set(models.keys()) - set(selected_models))
                print('remaining',remaining_models)
                print('remaining',len(remaining_models))
                
                prev_remaining_models = remaining_models.copy() 
            
                print('remaining models', remaining_models)
                print('remaining',len(remaining_models))
                for candidate_model in remaining_models:
                    current_models = selected_models + [candidate_model]
                    current_mae = self.calculate_mae({model: models[model] for model in current_models},Tm,seq)

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
        org_non_thermo_seq = [self.seq[i] for i in range(len(org_class)) if org_class[i] == 0]
        #print(len(org_non_thermo_seq))
        org_non_thermo_tm = [self.Tm[i] for i in range(len(org_class)) if org_class[i] == 0]
        #print(len(org_non_thermo_tm))
        org_thermo_seq = [self.seq[i] for i in range(len(org_class)) if org_class[i] == 1]
        #print(len(org_thermo_seq))
        org_thermo_tm = [self.Tm[i] for i in range(len(org_class)) if org_class[i] == 1]
        #print(len(org_thermo_tm))
        
        print('*** Greedy optimization for non-thermophilic ***')
        models_non_thermo, best_mae_non_thermo = self.greedy_optimization({key: value for key, value in self.regression_models.items()}, seq=org_non_thermo_seq, Tm=org_non_thermo_tm)
        print('*** Greedy optimization for thermophilic ***')
        models_thermo, best_mae_thermo= self.greedy_optimization({key: value for key, value in self.regression_models.items()}, seq=org_thermo_seq, Tm=org_thermo_tm)

        print("Selected models for non thermophilic (class 0):", models_non_thermo)
        print("Best MAE non thermo:", best_mae_non_thermo)
        print("Selected models for thermophilic (class 1):", models_thermo)
        print("Best MAE thermo:", best_mae_thermo)
        
        return models_non_thermo, best_mae_non_thermo, models_thermo, best_mae_thermo


from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_absolute_error
from itertools import combinations

class StackingEnsemble:
    def __init__(self, models, X_val, y_val):
        self.models = models
        self.X_val = X_val
        self.y_val = y_val
        self.best_ensemble = None
        self.best_mae = float('inf')

    def calculate_mae(self, selected_models):
        model_list = [self.models[model] for model in selected_models]
        print('model_list',model_list)
        stack = StackingRegressor(estimators=model_list, final_estimator=None, passthrough=True)
        stack.fit(self.X_val, self.y_val)
        predictions = stack.predict(self.X_val)
        print(predictions)
        mae = mean_absolute_error(self.y_val, predictions)
        return mae

    def ensemble(self): #find_best_ensemble
        #all_models = list(self.models.keys())
        all_models = list(self.models.items())
        print('all_models',all_models)
        for r in range(1, len(all_models) + 1):
            combinations_r = combinations(all_models, r)
            print(combinations_r)
            for ensemble in combinations_r:
                print('ensemble',ensemble)
                mae = self.calculate_mae(ensemble)
                if mae < self.best_mae:
                    self.best_mae = mae
                    self.best_ensemble = ensemble
                    print("Selected models:", self.best_ensemble)
                    print("Current MAE:", self.best_mae)
        print("Final selected models:", self.best_ensemble)
        print("Best MAE:", self.best_mae)
        return self.best_ensemble, self.best_mae

# Esempio di utilizzo:
# Supponiamo di avere un dizionario di modelli chiamato 'models' e i dati di validazione X_val, y_val
# ensemble = StackingEnsemble(models, X_val, y_val)
# best_ensemble, best_mae = ensemble.find_best_ensemble()
