from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import pandas as pd
import shap
import numpy as np
import mlflow
from mlflow.models.signature import infer_signature
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet


class TrainingPipeline(mlflow.pyfunc.PythonModel):
      
    def __init__(self, **kwds):
        self.verbose = True
        self.grid_params = {
        "trees": {'model__max_features': ['auto', 'sqrt'],
               'model__max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
               'model__min_samples_split': [2, 5, 10],
               'model__min_samples_leaf': [1, 2, 4]
        },

        "gradientboost": {
              'model__learning_rate': [.03, 0.05, .07], #so called `eta` value
              'model__max_depth': [5, 6, 7],
              'model__min_samples_split': [2, 5, 10],
              'model__min_samples_leaf': [1, 2, 4],
              'model__subsample': [0.7],
              'model__n_estimators': [500]},


        "elasticnet": {"model__alpha": [0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        "model__l1_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}

        }
        self.generic_models = {"gradientboost": GradientBoostingRegressor(n_estimators=500, max_depth=4, min_samples_split=5, learning_rate = 0.01,loss = "squared_error"),
        "trees": DecisionTreeRegressor(random_state=0), "elasticnet":ElasticNet()
        }
        self.template_models = {"gradientboost": GradientBoostingRegressor, "trees": DecisionTreeRegressor, "elasticnet":ElasticNet}
        super().__init__(**kwds)
        return


    def hyperparam_search(self,X, Y, train_param, is_scaled):
        model = self.template_models[train_param["hypotesis_model"]]()
        if "param_grid" in train_param["hyperparam"]:
            print("------- hypermarameter grid search using user param_grid dictionary")
            param_grid = train_param["hyperparam"]["param_grid"]
        else:
            print("------- hypermarameter grid search using default grid search dictionary")
            param_grid = self.grid_params[train_param["hypotesis_model"]]
        
        if is_scaled:
            print("------- hypermarameter grid search with cv 4 partitions and scaling the inputs")
            pipeline_ml = Pipeline([('scaler',  StandardScaler()), ('model', model)])
        else:
            print("------- hypermarameter grid search with cv 4 partitions")   
            pipeline_ml = Pipeline([('model', model)])
        grid = GridSearchCV(estimator = pipeline_ml, param_grid = param_grid, verbose=self.verbose, cv=4, n_jobs=6) 
        # Run the grid
        print("------- Running the grid search fit")
        grid.fit(X, Y)
        return grid.best_params_


    def _get_variable_importance(self, X, model):
        imp_var = model.feature_importances_
        imp_var_dict = dict(zip(X.columns, imp_var))
        imp_var_df = pd.DataFrame(list(imp_var_dict.items()), columns=['variable', 'importance_split'])
        imp_var_df.sort_values('importance_split', ascending=False, inplace=True)
        return imp_var_df


    def _feature_importance(self, X, model, importance_treshold = 0.01, min_features = 5):
        # Function to select important features
        imp_df=self._get_variable_importance(X, model)
        # Sort by Importance Split Ascending order
        imp_df.sort_values(by='importance_split',ascending=False,inplace=True)
        # Selecting the features based on the threshold
        features=imp_df[imp_df['importance_split'] >= importance_treshold]['variable'].tolist()
        if (len(features) < min_features):
            print('------- Since # features are less than', min_features, 'selecting the top',min_features,'features')
            features = imp_df["variable"].head(min_features).tolist()
        print("------- Features selecteds are: "+str(features))
        return features


    def _feature_selection(self,X,Y,train_param, is_scaled):
        if is_scaled:
            print("------- Standarizing data with mean 0 std of 1")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
        # Generic model for feature selection
        gen_model = self.generic_models[train_param["hypotesis_model"]]
        print("------- Training model for feature selection")
        model = gen_model.fit(X, Y)
        print("------- Compute feature importance")
        important_features = self._feature_importance(X, model)
        return important_features


    def fit(self, data, train_param):
        self.train_param = train_param
        is_scaled = train_param["scale"]
        # Split in train - test sets
        print("1. Splitting data in train - test sets: test size = " +str(train_param["test_size"]*100)+ " %")
        X_train, X_test, Y_train, Y_test = train_test_split(data[train_param["features"]], data[train_param["target"]], test_size=train_param["test_size"])
        Y_test = Y_test.values.ravel()
        Y_train = Y_train.values.ravel()
        self.X_test = X_test
        self.Y_test = Y_test
        self.X_train = X_train
        self.Y_train = Y_train
        print("-------------------- Finished train - test split stage")

        # Feature selection
        if train_param["feature_selection"]:
            print("2. Starting feature selection using a generic model")
            features_names = self._feature_selection(X_train, Y_train, train_param, is_scaled)
            X_train = X_train[features_names]
            X_test = X_test[features_names]
        else:
            print("2. No feature selection selected, using all columns defined in input param")
            features_names = X_train.columns
        self.features_names = features_names
        print("-------------------- Finished feature selection stage")

        # Hyperparameter and model search
        if len(train_param["hyperparam"]) == 0 :
            print("3. Grid search for hyperparameter tunning")
            param = self.hyperparam_search(X_train, Y_train, train_param, is_scaled)
            temp_param = {}
            for key in param:
                temp_param[key.replace("model__", "")] = param[key]
            param = temp_param
        else:
            print("3. Using hyperparameters given by user")
            param = train_param["hyperparam"]
        self.hyperparam = param
        print("------- the hypermarameter selected are: "+str(param))
        print("-------------------- Finished hyperparameter tunning stage")

        # Model training
        print("4. Training final model")
        if is_scaled:
            print("------- Scaling training data")
            scaler = StandardScaler()
            X = scaler.fit_transform(X_train)
            self.scaler = scaler
        X_train = pd.DataFrame(X, index=X_train.index, columns=X_train.columns)
        temp_model = self.template_models[train_param["hypotesis_model"]](**param)
        model = temp_model.fit(X_train, Y_train)
        print("-------------------- Finished model training")

        # Save model in object
        self.model = model
        print("5. Generation model signature")
        self.signature = infer_signature(X_train, self.predict(None, X_train))
        print("-------------------- Finished model signature stage")
        return


    def Model_metrics(self):
        '''Description: Metrics for the evaluation of the model
        Args:
            -df: Test dataframe
        '''
        print("Starting model evaluation")
        Y_train_pred = self.predict_proba(self.X_train)
        Y_test_pred = self.predict_proba(self.X_test)
        self.mae_score_train = mean_absolute_error(self.Y_train, Y_train_pred)
        print(f"mae in training is {self.mae_score_train}")
        self.mae_score_test = mean_absolute_error(self.Y_test, Y_test_pred)
        print(f"mae in testing is {self.mae_score_test}")
        self.mape_score_train = mean_absolute_percentage_error(self.Y_train, Y_train_pred)
        print(f"mape in training is {self.mape_score_train}")
        self.mape_score_test = mean_absolute_percentage_error(self.Y_test, Y_test_pred)
        print(f"mape in testing is {self.mape_score_test}")
        return


    def predict_proba(self, X_in):
        X_in = X_in[self.features_names]
        is_scaled = self.train_param["scale"]
        if is_scaled:
            X = self.scaler.transform(X_in)
            X_test = pd.DataFrame(X, index=X_in.index, columns=X_in.columns)
        else:
            X_test = X_in
        Y_pred = self.model.predict(X_test)
        return Y_pred
  

    def shap_explain_plot(self, X_in):
        X_in = X_in[self.features_names]
        is_scaled = self.train_param["scale"]
        if is_scaled:
            X = self.scaler.transform(X_in)
            X = pd.DataFrame(X, index=X_in.index, columns=X_in.columns)
        else:
            X = X_in     
        # shap plot values
        #try:
        explainer = shap.Explainer(self.model)
        shap_test = explainer(X)
        shap_values = shap_test.values
        #fig1 = plt.figure()
        #shap.summary_plot(shap_values, X, max_display=40, plot_size = (25,25))
        fig2 = plt.figure()
        shap.summary_plot(shap_values, X, plot_type="bar", plot_size = (25,25))
        #except:
        #    print("Can't apply shap explainer to the given model")
        #    fig1 = None
        #    fig2 = None
        return fig2
