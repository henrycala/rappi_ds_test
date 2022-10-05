import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
import lightgbm as lgb
import pandas as pd
import shap
import numpy as np
import pickle
import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature


class training_pipeline(mlflow.pyfunc.PythonModel):
  
  
  def _init_(self, verbose, **kwds):
    super()._init_(**kwds)
    self.verbose = True
    self.grid_params = {"lgbm" :{
    'model__learning_rate': [0.005, 0.01],
    'model__n_estimators': [8,16,24],
    'model__num_leaves': [6,8,12,16], # large num_leaves helps improve accuracy but might lead to over-fitting
    'model__boosting_type' : ['gbdt', 'dart'], # for better accuracy -> try dart
    'model__objective' : ['binary'],
    'model__max_bin':[255, 510], # large max_bin helps improve accuracy but might slow down training progress
    'model__random_state' : [500],
    'model__colsample_bytree' : [0.64, 0.65, 0.66],
    'model__subsample' : [0.7,0.75],
    'model__reg_alpha' : [1,1.2],
    'model__reg_lambda' : [1,1.2,1.4],
    'model__importance_type': ['split'],
    'model__max_depth': [-1,5,10,20],
   'model__min_child_samples': [5,10,15,20],
   'model__min_child_weight': [0.001],
   'model__min_split_gain': [0.0],
   'model__n_jobs': [6],
   'model__silent': ['warn'],
   'model__subsample_for_bin': [200000],
   'model__subsample_freq': [0],
   'model__scale_pos_weight': [2]
    }
    }
    self.generic_models = { "lgbm": lgb.LGBMClassifier(objective='binary', n_jobs=6, max_depth=-1, n_estimators=800, random_state = 10,scale_pos_weight=2)
    }
    self.template_models = {"lgbm": lgb.LGBMClassifier}
    return
  

  
  def _encode_cols(self, data, cols_list):        
    for col_name in cols_list:
      if col_name in data.columns:
        l_encoder = LabelEncoder()
        data[col_name] = l_encoder.fit_transform(data[col_name])
    return data
      
      
  def _preprocesing(self, data, preprocesing_param, train_param):
    # Select only the training and target columns in the dataframe
    data = data[train_param["cols_train"]+[train_param["target"]]]
    # Processing for encoded columns
    if preprocesing_param["encode"]["active"]:
      print("------- Encoding columns:"+str(preprocesing_param["encode"]["cols_encode"]))
      data = self._encode_cols(data, preprocesing_param["encode"]["cols_encode"])
    # Convert all columns to float
    print("------- Converting data to float")
    data = data.astype(float)
    return data

  
  def _get_variable_importance(self, X, model):
    imp_var = model.feature_importances_
    imp_var_dict = dict(zip(X.columns, imp_var))
    imp_var_df = pd.DataFrame(list(imp_var_dict.items()), columns=['variable', 'importance_split'])
    imp_var_df.sort_values('importance_split', ascending=False, inplace=True)
    return imp_var_df
  
  
  def _feature_importance(self, X, model, importance_treshold = 0.01, min_features = 25):
    # Function to select important features
    imp_df=self._get_variable_importance(X, model)
    # Sort by Importance Split Ascending order
    imp_df.sort_values(by='importance_split',ascending=False,inplace=True)
    # Selecting the features based on the threshold
    features=imp_df[imp_df['importance_split'] >= importance_treshold]['variable'].tolist()
    if (len(features) < min_features):
        print('------- Features with importance > ',cut_off_imp,' : ', len(features))
        print('------- Since # features are less than', min_features, 'selecting the top',min_features,'features')
        features = imp_df["variable"].head(min_num_features).tolist()
    print("------- Features selecteds are: "+str(features))
    return features
  
  
  def _feature_selection(self,X,Y,train_param, is_scaled):
    method_name = train_param["feature_selection"]["method"]
    # Thinking in adding more methods like cv
    if method_name not in ["generic"]:
      method_name = "generic"
    print("------- Feature selection using {} method".format(method_name))
    if  method_name == "generic":
      if is_scaled:
        print("------- Standarizing data with mean 0 std of 1".format(method_name))
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
      # Generic model for feature selection
      gen_model = self.generic_models[train_param["hypotesis_model"]]
      print("------- Training model for feature selection")
      model = gen_model.fit(X, Y, verbose = self.verbose, eval_metric=train_param["eval_metric"])
      print("------- Compute feature importance")
      important_features = self._feature_importance(X, model)
    return important_features

  
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
      pipeline_ml = Pipeline([('scaler',  StandardScaler()),
            ('model', model)])
      
    else:
      print("------- hypermarameter grid search with cv 4 partitions")   
      pipeline_ml = Pipeline([('model', model)])
    grid = GridSearchCV(estimator = pipeline_ml, param_grid = param_grid, verbose=self.verbose, cv=4, n_jobs=6) 
    # Run the grid
    print("------- Running the grid search fit")
    grid.fit(X, Y)
    return grid.best_params_
    
    
  def fit(self, data, preprocesing_param, train_param):
    self.preprocesing_param = preprocesing_param
    self.train_param = train_param
    is_scaled = preprocesing_param["scale"]["active"]
    if self.verbose:
      print("1. Starting preprocessing:")
    data = self._preprocesing(data, preprocesing_param, train_param)
    print("-------------------- Finished preprocessing stage")
    # Split in train - test sets
    print("2. Splitting data in train - test sets: test size = " +str(train_param["test_size"]*100)+ " %")
    X_train, X_test, Y_train, Y_test = train_test_split(data[train_param["cols_train"]], data[train_param["target"]], test_size=train_param["test_size"], stratify=data[train_param["target"]])
    self.X_test = X_test
    self.Y_test = Y_test
    self.X_train = X_train
    self.Y_train = Y_train
    print("-------------------- Finished train - test split stage")
    # Feature selection
    if train_param["feature_selection"]["active"]:
      print("3. Starting feature selection using a generic model")
      features_names = self._feature_selection(X_train, Y_train, train_param, is_scaled)
      X_train = X_train[features_names]
      X_test = X_test[features_names]
    else:
      print("3. No feature selection selected, using all columns defined in input param")
      features_names = X_train.columns
    self.features_names = features_names
    print("-------------------- Finished feature selection stage")
    # Hyperparameter and model search
    if train_param["hyperparam"]["search"]:
      print("4. Grid search for hyperparameter tunning")
      param = self.hyperparam_search(X_train, Y_train, train_param, is_scaled)
    else:
      print("4. Using hyperparameters given by user")
      param = train_param["hyperparam"]["param"]
    self.hyperparam = param
    print("------- the hypermarameter selected are: "+str(param))
    print("-------------------- Finished hyperparameter tunning stage")
    # Model training
    print("5. Training final model")
    if is_scaled:
      print("------- Scaling training data")
      scaler = StandardScaler()
      X = scaler.fit_transform(X_train)
      self.scaler = scaler
      X_train = pd.DataFrame(X, index=X_train.index, columns=X_train.columns)
    temp_model = self.template_models[train_param["hypotesis_model"]](**param)
    model = temp_model.fit(X_train, Y_train, verbose = self.verbose, eval_metric= train_param["eval_metric"])
    print("-------------------- Finished model training")
    # Save model in object
    self.model = model
    print("6. Generation model signature")
    self.signature = infer_signature(X_train, self.predict(None, X_train))
    print("-------------------- Finished model signature stage")
    return

  
  def predict_proba(self, X_in):
    is_scaled = self.preprocesing_param["scale"]["active"]
    if is_scaled:
      X = self.scaler.transform(X_in)
      X_test = pd.DataFrame(X, index=X_in.index, columns=X_in.columns)
    else:
      X_test = X_in
    Y_pred = self.model.predict_proba(X_test)[:,1]
    return Y_pred
  
  
  def shap_explain_plot(self, X_in):
    is_scaled = self.preprocesing_param["scale"]["active"]
    if is_scaled:
      X = self.scaler.transform(X_in)
      X = pd.DataFrame(X, index=X_in.index, columns=X_in.columns)
    else:
      X = X_in     
    # shap plot values
    try:
      shap_values = shap.TreeExplainer(self.model).shap_values(X)
      fig1 = plt.figure()
      shap.summary_plot(shap_values[1], X, max_display=40, plot_size = (25,25))
      fig2 = plt.figure()
      shap.summary_plot(shap_values[1], X, plot_type="bar", plot_size = (25,25))
    except:
      print("Can't apply shap.TreeExplainer to the given model")
    return fig1, fig2


  def roc_curve_plot(self,y_true, y_predicted):
      fpr, tpr, _ = roc_curve(y_true, y_predicted)
      auc = roc_auc_score(y_true, y_predicted)
      fig = plt.figure()
      plt.plot([0, 1], [0, 1], 'k--')
      plt.plot(fpr, tpr, label=f'Auc:{auc: .2f}')
      plt.xlabel('False positive rate')
      plt.ylabel('True positive rate')
      plt.title('ROC curve')
      plt.legend(loc='best')
      return fig
 

  def precision_recall_plot(self,y_true, y_scores, filename=None):
      precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
      fig = plt.figure()
      plt.plot(thresholds, precision[:-1]) 
      plt.plot(thresholds, recall[:-1]) 
      leg = plt.legend(('precision', 'recall'), frameon=True, loc='best') 
      leg.get_frame().set_edgecolor('k') 
      plt.xlabel('threshold') 
      plt.ylabel('%')
      return fig
 

  def lift_chart_plot(self,y_true, y_score, num_cuts=10):
    score_df = pd.DataFrame(y_score, index=y_true.index)
    score_df = pd.concat([y_true, score_df], axis = 1)  
    score_df.columns = ['y_true','y_score']
    score_df['rank']=score_df['y_score'].rank(method='first')
    score_df['deciles']=pd.qcut(score_df['rank'], num_cuts, labels=list(range(1,num_cuts + 1)))
    n_targets = score_df['y_true'].sum()
    n_size = score_df.shape[0]
    prevalance = n_targets/n_size
    lift_df = score_df.groupby("deciles").agg(Number_of_targets = pd.NamedAgg(column="y_true", aggfunc="sum"), number_of_samples = pd.NamedAgg(column="y_true", aggfunc="count"),
                                              threshold = pd.NamedAgg(column="y_score", aggfunc="min")).reset_index()
    lift_df.sort_values('deciles', ascending=False, inplace=True)
    lift_df['cumulative_capture_rate'] = lift_df['Number_of_targets'].cumsum()
    lift_df['total_targets'] =  n_targets
    lift_df['Lift Cumm'] = (lift_df['Number_of_targets'].cumsum()/lift_df['number_of_samples'].cumsum())/prevalance
    lift_df['cumulative_capture_rate_percent'] = lift_df['cumulative_capture_rate']/lift_df['total_targets']
    lift_df['lift'] = lift_df['Number_of_targets']/(lift_df['number_of_samples']*prevalance)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    width = 0.4
    lift_df.number_of_samples.plot(kind='bar', color='#72A2C0', ax=ax, width=width, position=1)
    lift_df.Number_of_targets.plot(kind='bar', color='#0444BF', ax=ax, width=width, position=0)
    ax2.plot(ax.get_xticks(), lift_df['lift'], linestyle = '-', marker='o', linewidth = 2.0, color = 'lime')
    ax.plot(ax.get_xticks(), lift_df['total_targets'], linestyle = '--', marker='o', linewidth = 2.0, color = 'dimgray')
    ax.plot(ax.get_xticks(), lift_df['cumulative_capture_rate'], linestyle = '-.', marker='o', linewidth = 2.0, color = 'crimson')
    cumulative_capture_rate_percent = lift_df['cumulative_capture_rate_percent'].tolist()
    cumulative_capture_rate = lift_df['cumulative_capture_rate'].tolist()
    for i,j in enumerate(cumulative_capture_rate_percent):
        ax.annotate(str(int(j*100)) + '%', xy=(i, cumulative_capture_rate[i]))
    ax.legend(loc = 'lower right', fontsize = 'xx-small')
    ax2.legend(loc = "best")
    ax.set_xlabel('Deciles')
    ax.set_ylabel('Size')
    ax2.set_ylabel('Lift')
    plt.title('Lift Chart')
    plt.show()
    return fig

  
  def confusion_matrix_plot(self, y_true, y_pred, treshold):
    y_pred_label = (y_pred > treshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_label)
    # Not normalized confusion matrix
    fig1 = plt.figure()
    plt.matshow(cm, plt.gcf().number)
    for i in range(len(cm)):
      for j in range(len(cm)):
        plt.annotate(cm[i,j],xy=(j,i),horizontalalignment='center',verticalalignment='center',size=15,color='white') 
    plt.title('Confusion matrix of the classifier')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # Normalized confusion matrix
    cm_norm = np.zeros_like(cm).astype(float)
    for i in range(len(cm)):
      for j in range(len(cm)):
        cm_norm[i,j] = round(cm[i,j]*100/np.sum(cm[i,:]),2)
    fig2 = plt.figure()
    plt.matshow(cm_norm, plt.gcf().number)
    for i in range(len(cm_norm)):
      for j in range(len(cm_norm)):
        plt.annotate(cm_norm[i,j],xy=(j,i),horizontalalignment='center',verticalalignment='center',size=15,color='white') 
    plt.title('Normalized confusion matrix of the classifier')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    return fig1, fig2
    
        
  def model_evaluation(self,X,Y, treshold = 0.5, save_mlflow = False):
    Y_pred = self.predict_proba(X)
    # Shap values plot
    fig1, fig2 = self.shap_explain_plot(X)
    # Roc curve plot
    fig3 = self.roc_curve_plot(Y, Y_pred)
    # precision and recall plot
    fig4 = self.precision_recall_plot(Y, Y_pred)
    # generating plots
    fig5 = self.lift_chart_plot(Y, Y_pred, num_cuts=10)
    # Confusion matrix
    fig6, fig7 = self.confusion_matrix_plot(Y, Y_pred, treshold)
    if save_mlflow:
      mlflow.log_figure(fig1, 'shap_summary_plot.png')
      mlflow.log_figure(fig2, 'shap_summary_barplot.png')
      mlflow.log_figure(fig3, 'roc_curve.png')
      mlflow.log_figure(fig4, 'precision_recall_treshold.png')
      mlflow.log_figure(fig5, 'lift_chart.png')
      mlflow.log_figure(fig6, 'confusion_matrix.png')
      mlflow.log_figure(fig7, 'normalized_confusion_matrix.png')
    return
    
    
  def model_scores(self, treshold = 0.5, savefig_in_mlflow = False):
    self.model_evaluation(self.X_test, self.Y_test, treshold, savefig_in_mlflow)
    return
  
  
  def predict(self, context, model_input):
    return self.predict_proba(model_input)
  
  
  def model_metrics(self):
    Y_train_pred = self.predict_proba(self.X_train)
    Y_test_pred = self.predict_proba(self.X_test)
    self.auc_score_train = roc_auc_score(self.Y_train, Y_train_pred)
    self.auc_score_test = roc_auc_score(self.Y_test, Y_test_pred)
    return

# COMMAND ----------

cols_selected = ['Others_margin_F', 'Engine_Mechanical_margin_F', 'DAYS_SINCE_LAST_SERVICE', 'DISTANCETOLASTSERVICE', 'MARGIN', 'Fuel_Injection_margin_F', 'ODOMETERREADING', 'INVOICE_LABOR_CUMULATIVE_F', 'HISTORICAL_INVOICE', 'DAYSTOLASTSCHEDULEDSERVICE', 'DAYSLEFTFORWARRANTY', 'INVOICE_PARTS_CUMULATIVE_F', 'INVOICE_AMOUNT_CUMULATIVE_F', 'SCHEDULEDINVOICERATIO', 'AGEATSERVICE', 'HISTORICAL_CUSTOMERAMOUNT', 'AVERAGE_DAYS_BETWEEN_SERVICES', 'AVERAGE_SERVICE_DURATION_F', 'CURRENT_VISIT_AVERAGE_DISCOUNT_F', 'CARMODEL', 'ACTUALTOEXPECTEDSCHEDULESERVICERATIO', 'CUSTOMER_AMOUNT_CUMULATIVE_F', 'Climate_control_margin_F', 'Vehicle_trim_margin_F', 'Braking_components_margin_F', 'Body_Electrical_margin_F', 'CUST_TYPE', 'REPAIR_COUNT_F', 'HISTORICAL_CHURN_COUNT', 'CITYFLAG', 'SECONDHANDFLAG', 'SCHEDULEDSERVICESCOUNT', 'OTHER_DEPARTMENT_VISITS_F', 'LIMAFLAG', 'EXTENDEDWARRANTYFLAG', 'warranty_flag']

col_target = "CHURN"

preprocesing_param = {"encode" : {"active": True, "cols_encode": ["CARMODEL"]},
                      "scale" : {"active": True}
                     }
training_param = {"cols_train" : cols_selected,
                  "target" : col_target,
                  "test_size": 0.2,
                  "hypotesis_model" : "lgbm",
                  "eval_metric" : 'auc',
                  "feature_selection" : {"active": False, "method": "generic"},
                  "hyperparam" : {"search": False, "param": {'boosting_type': 'gbdt', 'colsample_bytree': 0.64, 'learning_rate': 0.005, 'max_bin': 255, 'n_estimators': 8, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500}}
}
# Training object
churn_train = training_pipeline(verbose = True)
churn_train.fit(data, preprocesing_param, training_param)
churn_train.model_scores(treshold = 0.204)

# COMMAND ----------

# MAGIC %md Include the model in ML FLOW

# COMMAND ----------

# mlflow.start_run creates a new MLflow run to track the performance of this model. 
# Within the context, you call mlflow.log_param to keep track of the parameters used, and
# mlflow.log_metric to record metrics like accuracy.
with mlflow.start_run(run_name='lgbm_churn_pe_colteam_test'):
  # Training the model
  churn_train = training_pipeline(verbose = True)
  churn_train.fit(data, preprocesing_param, training_param)
  # Log model
  mlflow.pyfunc.log_model("PE_Churn_lgbm_colteam", python_model = churn_train, signature=churn_train.signature)
  # Log metrics
  churn_train.model_metrics()
  mlflow.log_metric('auc', churn_train.auc_score_test)
  # Log model parameters
  mlflow.log_params(churn_train.hyperparam)
  # Log figure artifacts
  churn_train.model_scores(treshold = 0.204, savefig_in_mlflow = True)
  
  # storing pickle model file
  outfile = open('lgbm_model_churn_pe_colteam.pkl','wb')
  pickle.dump({'training_params':training_param,'preprocesing_param':preprocesing_param,'model':churn_train},outfile)
  
  
  dict_model = {'training_params':training_param,'preprocesing_param':preprocesing_param,'model':churn_train}
  
  churn_model = dict_model['model']
  churn_model.predict_proba(X_test)
  
  outfile.close()
  mlflow.log_artifact("lgbm_model_churn_pe_colteam.pkl")
  
  # generating report
  #generate_report(Y_train, predictions_train, Y_test, \
  #                predictions_test,trained_model , X_train, X_test, params,\
  #                features=features, destination = 'results', \
  #                num_cuts=20, output_wb_name='report_pe.xlsx')
  #mlflow.log_artifact('report_pe.xlsx')