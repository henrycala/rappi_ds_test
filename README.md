# rappi_ds_test
In this folder is include the test use case for the data scientist position at Rappi. In order to run the following codes the next steps are required.

1. Install python 3.8.0
2. Create a virtual enviroment 
```console 
python -m venv .venv 
```
3. Activate the virtual enviroment.
4. upgrade pip
```console 
python -m pip install --upgrade pip
```
5. Install required packages
```console 
 pip install -r requirements.txt
 ```

# Project organization
## Exploratory data analysis
In the folder EDA/ we can found two main EDAs, the initial_data_exploration.ipynb and the price_vs_variables.ipynb.

In <b> initial_data_exploration.ipynb </b> is performed the computation of basic statistics, Also it is computed the histogram and boxplot diagrams for the different variables. Later on is computed the number of null/zero values and and the correlation values between the fields.

In <b>price_vs_variables.ipynb </b> is performed an analysis to mesure the relationship between the different variables and the price. In this notebook we can find scatter plots and box plot analysis under different grouping variables. Also, is include a colinearity analysis and a location analysis to determinate the linear dependencies between the variables and the importance of the location in the price, respectively.

## Model training
In <b>ml_training.ipynb</b> is developed the main data processing flow. In this notebook is imported the training class from the file training_class.py. Two machine learning models were tested: random forest and ensembled boosted models. The performance tracking and the artifacts of the models are stored using mlflow.

For all the notebooks is created a html version to visualizate the results.
 
