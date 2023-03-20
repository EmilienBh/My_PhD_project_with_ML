import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import shapiro,chi2_contingency, f_oneway, mannwhitneyu

from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

from sklearn.metrics import classification_report, f1_score, jaccard_score, mean_absolute_error, mean_squared_error

from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings("ignore")
    
pd.set_option("display.max_row",150) # display 150 lines
pd.set_option("display.max_columns",150) # display 150 columns

pd.set_option("display.max_row",150) # display 150 lines
pd.set_option("display.max_columns",150) # display 150 columns


def explore_dataset(df):
    """
    Give informations about the dataframe (describe(), info(), isnul(), duplicated() functions from pandas)

    Args:
        df (Dataframe): classic dataframe
        
    """
    
    print("\033[1m" + "DESCRIPTION")
    display(df.describe())
    print("*"*100)
    
    print("\033[1m" + "INFORMATIONS")
    display(df.info())
    print("*"*100)
    
    print("\033[1m" + "PRESENCE OF NULL VALUE(S)?")
    display(df.isnull().sum())
    print("*"*100)
    
    print("\033[1m" + "PRESENCE OF DUPLICATED SAMPLE(S)?")
    display(df.duplicated().sum())
    print("*"*100)


def create_datasets(df, only_physio_datas = False):
    """
    We list the useless variables for all cases of creation of datasets.
    - redundant variables  (eg: ID Event et Event)
    - variables used to create the targets (eg: TTC, Collision,etc...)

    Args:
        df (Dataframe): classic dataframe
        only_physio_datas (bool, optional): if is True, it keep only physiological datas. Otherwise, it keep driving datas and socio-demographic. Defaults to False.

    Returns:
        dataframes: Return a dataframe of X_datas
        dictionary: Return a dictionary of y_targets
    """
    
    useless_var = ['Sujet', 
                   'Age category',
                   'Scenario',
                   'ID scenario Event',
                   'Event',
                   'NDRT',
                   'NDRT duration',
                   'Lane changing',
                   'TOT (s)',
                   'TTC (s)',
                   'Brake force (daN)',
                   'Braking',
                   'Standard deviation \nsteering wheel \nrotation speed (° / s)',
                   'Steering wheel rotation speed',
                   'Max Absolute Value Lateral Shift',
                   'ILC',
                   'Collision',
                   'Checking mirrors',
                   'TOQ (Coll a/oTTC < 1,5s)',
                   'TOQ (Coll a/o braking)',
                   'TOQ (Coll a/o steering)',
                   'TOQ (Coll a/o mirrors)',
                   'TOQ (Coll a/o(TTC<1s & braking)',
                   'TOQ (Coll a/o(TTC<1s & steering)',
                   'TOQ (Coll a/o(TTC<1s & mirror)',
                   'TOQ (Coll a/o  ILC)']
    
    if only_physio_datas:
        useless_var.extend(['Age',
                            'Gender', 
                            'ID Event', 
                            'ID NDRT',
                            'NDRT duration (s)', 
                            'NDRT solicitation hands',
                            'NDRT solicitation gaze'])
    
    
    
    # dataset creation for take-over dangerousness prediction with lane changing
    X_lane = df[df['Lane changing'] == 1]
    y_lane = {'TOQ (Coll a/o(TTC<1s & braking)': X_lane['TOQ (Coll a/o(TTC<1s & braking)'],
              'TOQ (Coll a/o(TTC<1s & steering)': X_lane['TOQ (Coll a/o(TTC<1s & steering)'],
              'TOQ (Coll a/o(TTC<1s & mirror)': X_lane['TOQ (Coll a/o(TTC<1s & mirror)']}
    X_lane.drop(useless_var, axis=1, inplace = True)    
    
    # dataset creation for take-over dangerousness prediction without lane changing
    X_no_lane = df[df['Lane changing'] == 0]
    y_no_lane = {'TOQ (Coll a/o  ILC)': X_no_lane['TOQ (Coll a/o  ILC)']}  
    X_no_lane.drop(useless_var, axis=1, inplace = True)
    
    # dataset creation for Take-Over Time
    y_TOT = {'TOT (s)': df['TOT (s)']}
    X_TOT = df.drop(useless_var, axis=1)
    
    return X_lane, y_lane, X_no_lane, y_no_lane, X_TOT, y_TOT 



def discriminate_target_association(X, y):
    """
    Interpretation of the Shapiro-Wilk test
    
     It is a statistical test that checks whether a variable follows a normal distribution or not.
    
     H0: the variable from which the sample comes follows a normal law
     H1: the variable from which the sample comes does not follow a normal law
    
     If the p-value is above the significance level (usually 0.05),
     then we cannot reject the null hypothesis that the variable follows a normal distribution.
     This means that the variable is considered normally distributed.
    
     On the other hand, if the p-value is below the significance level,
     one can reject the null hypothesis that the variable follows a normal distribution.
     This means that the variable does not follow a normal distribution.

    Args:
        X (Dataframe): classic Dataframe without NaNs
        y (Dictionary): dictionary of the targets

    Returns:
        Dictionary: Return a dictionary of different X_datasets in relationship with y_targets.
        Dataframe: Return a dataframe with test results of each variabl for each target.
    """
    
    X_datasets = {}
    df_pvalues = pd.DataFrame(columns = X.columns, index = ['shapiro'])
    df_pvalues = pd.concat([df_pvalues,pd.DataFrame(columns=df_pvalues.columns, index = y.keys())],axis = 1)

    shapiro_results = {}
    for col in X:
        if X[col].dtype == 'object':
            shapiro_results[str(col)] = 'NaN'
        else:
            _, p_value = shapiro(X[col])
            shapiro_results[str(col)] = round(p_value,3)
    df_pvalues.loc['shapiro'] = shapiro_results


    """
    Now that we know for each continuous variable, if it follows a normal distribution or not,
     we can apply a test of association with the target in question.
    
     The goal is to exclude variables that have too weak an association with the target to influence it.

     If the variable follows a normal distribution, we will impose an ANOVA test on it, otherwise a Mann-Whitney test
     Regarding the categorical variables, we will impose a chi-square test on them.
    """

    for i, (key,value) in enumerate(y.items()):
        tmp_df = X.copy()
        for col in X:
            if X[col].dtype == 'object':
                _,p_value,_,_ = chi2_contingency(pd.crosstab(X[col], value))
            else:
                group1 = X[col][value == 0]
                group2 = X[col][value == 1]
                if p_value <= 0.05:
                    _,p_value = mannwhitneyu(group1, group2)
                else:
                    _,p_value = f_oneway(group1, group2)
            df_pvalues.loc[key,col] = round(p_value,3)
            if p_value > 0.05:
                tmp_df.drop(col, axis = 1, inplace = True)
        X_datasets[key] = tmp_df

    return X_datasets, df_pvalues



def encoding_one_hot(X):
    """Encoding object variables

    Args:
        X (Dictionary): A dictionary of X_dataframes

    Returns:
        Dictionary: Return a dictionary of X_dataframes with object variables encoded one hot
    """
    
    for key, value in X.items():
        if any(value.dtypes == 'object'):
            X.update({key: pd.get_dummies(value)})
    return X



def normalize (df, normalization_type: "minmax"):
    """_summary_

    Args:
        df (Dataframe): Classic dataframe
        normalization_type (minmax): "minmax" for ManMaxScaler(); "standard" for Standard_Scaler(); "robust" for Robust_Scaler()

    Returns:
        Dataframe: Return Dataframe with normalized datas
    """
    
    num_var = df.select_dtypes(['int', 'float']).columns.to_list()
    if("minmax"):
        scaler = MinMaxScaler()
        df[num_var] = scaler.fit_transform(df[num_var])
 
    elif("standard"):
        scaler = Standard_Scaler()
        df[num_var] = scaler.fit_transform(df[num_var])

    else:
        scaler = Robust_Scaler()
        df[num_var] = scaler.fit_transform(df[num_var])
        
    return df

        
        
def predict(X,y, method = 'classification'):
    """_summary_

    Args:
        X (Dictionary): Dictionary of X_dataframes
        y (Dictionary): Dictionary of y_dataframes
        
        method (str, optional): 'classification' or 'regression'. Defaults to 'classification'.
    """
    
    if method == 'classification':
        models = {'Random Forest': {'model': RandomForestClassifier(random_state=1805),
                                'hyperparameters': {'n_estimators': [10, 50, 100],
                                                    'max_features': ['auto', 'sqrt', 'log2'],
                                                    'max_depth': [5, 10, None]
                                                    },
                                   },
                  'SVC': {'model': SVC(random_state=1805),
                          'hyperparameters': {'C': [0.1, 1, 10],
                                              'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                                              'gamma': ['auto', 0.1, 1, 10]
                                             },
                         },
                  'KNN Classifier': {'model': KNeighborsClassifier(),
                                     'hyperparameters': {'n_neighbors': [3, 5, 10, 20],
                                                         'weights': ['uniform', 'distance'],
                                                         'p': [1, 2, 3, 4]
                                                        },
                                    },
                  'Logistic Regression': {'model': LogisticRegression(random_state=1805),
                                          'hyperparameters': {'C': [0.1, 1, 10],
                                                              'penalty': ['l1', 'l2', 'elasticnet'],
                                                              'solver': ['liblinear', 'lbfgs', 'saga']
                                                             },
                                         }
                 }
    
        result_df = {'target': [],
                     'model': [],
                     'hyperparameters': [],
                     'grid_score_train': [],
                     'grid_score_test': [],
                     'f1-score': [],
                     'youden': [],
                     'fitting': [],
                     'complexity': []
                    }
        
    else:
        models = {'Random Forest': {'model': RandomForestRegressor(random_state=1805),
                            'hyperparameters': {'n_estimators': [10, 50, 100],
                                                'max_features': ['auto', 'sqrt', 'log2'],
                                                'max_depth': [5, 10, None]
                                                          },
                           },
                  'SVR': {'model': SVR(),
                          'hyperparameters': {'C': [0.1, 1, 10],
                                              'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                                              'gamma': ['auto', 0.1, 1, 10]
                                             },
                         },
                  'KNN Regressor': {'model': KNeighborsRegressor(),
                                     'hyperparameters': {'n_neighbors': [3, 5, 10, 20],
                                                         'weights': ['uniform', 'distance'],
                                                         'p': [1, 2, 3, 4]
                                                        },
                                    },
                  'Linear Regression': {'model': LinearRegression(),
                                          'hyperparameters': {}
                                       }
                 }

        result_df = {'target': [],
                     'model': [],
                     'hyperparameters': [],
                     'grid_score_train': [],
                     'grid_score_test': [],
                     'mse': [],
                     'mae': [],
                     'rmse': [],
                     'mean_error (s)': [],
                     'fitting': [],
                     'complexity': []
                    }
    subplot_count = 0
    
    # for each index, key and value of the target
    for i, (key, value) in enumerate(y.items()):
        
        # We print the name of the target
        print(f'{key}\n')
        
        # Divide the data into training and test sets.
        X_train, X_test, y_train, y_test = train_test_split(list(X.values())[i],value, test_size=0.2, random_state=1805)
        
        # We reset the model performance dictionary of the current target for the next target
        result_df = {key_: [] for key_ in result_df}
        
        plt.figure(figsize = (20,15))
        for model_name, model in models.items():

            # We perform a grid search with a 5-fold cross-validation
            grid = GridSearchCV(model['model'], model['hyperparameters'], cv=5)
            grid.fit(X_train, y_train)
            y_pred = grid.predict(X_test)
            
            # We calculate the performance of the model
            acc_train = round(grid.score(X_train, y_train),2)
            acc_test = round(grid.score(X_test, y_test),2)
            over_under_fitting = round((acc_train - acc_test) / acc_train,2)
            
            if acc_train > acc_test:
                complexity = "too much complex" if acc_train - acc_test > 0.05 else "not complexe enough"
            else:
                complexity = "too much complex" if acc_test - acc_train > 0.05 else "not complexe enough"
                        
            # We store the performance data in the dictionary
            result_df['target'].append(key)
            result_df['model'].append(model_name)
            result_df['hyperparameters'].append(grid.best_params_)
            result_df['grid_score_train'].append(acc_train)
            result_df['grid_score_test'].append(acc_test)
            result_df['fitting'].append(f"Underfitting :{str(over_under_fitting)}" if over_under_fitting < 0 else f"Overfitting :{str(over_under_fitting)}")            
            result_df['complexity'].append(complexity)
            
            
            if method == 'classification':
                # Display of correlation matrix and classification report
                print(f'\033[1m{model_name}')
                print("************************************")
                print('Confusion Matrix')
                display(pd.crosstab(y_test, y_pred, rownames= ['Real class'], colnames=['Predicted classe']))
                print("....................................")
                print('Classification Report')
                print(classification_report(y_test,y_pred))
                print("************************************\n")

                f1 = round(f1_score(y_test, y_pred),2)
                youden = round(jaccard_score(y_test, y_pred,pos_label=1) + jaccard_score(y_test, y_pred, pos_label=0) - 1,2)
                
                # We store the performance data in the dictionary
                result_df['f1-score'].append(f1)
                result_df['youden'].append(youden)
            else:
                mse = round(mean_squared_error(y_test , y_pred),3)
                mae = round(mean_absolute_error(y_test , y_pred),3)
                rmse = round(np.sqrt(mean_squared_error(y_test , y_pred)),3)
                results = round(pd.DataFrame(list(zip(y_test,y_pred))),3)
                mean_error = round(abs(results[0]-results[1]).mean(),3)
               
                # We store the performance data in the dictionary
                result_df['mse'].append(mse)
                result_df['mae'].append(mae)
                result_df['rmse'].append(rmse)
                result_df['mean_error (s)'].append(mean_error)
                
            # Calculate learning curves
            train_sizes, train_scores, val_scores = learning_curve(model['model'], X = X_train, y = y_train, train_sizes= np.linspace(0.1,1,10), cv=5)

            # Calculate means and standard deviations for training and validation scores
            train_scores_mean = train_scores.mean(axis=1)
            train_scores_std = train_scores.std(axis=1)
            val_scores_mean = val_scores.mean(axis=1)
            val_scores_std = val_scores.std(axis=1)

            # Plot the learning curves
            plt.subplot(2,2,subplot_count+1)
            plt.title(f"Learning curves - {model['model']}")
            plt.xlabel("Train set size")
            plt.ylabel("Score")
            plt.grid()

            plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="r")
            plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                             val_scores_mean + val_scores_std, alpha=0.1, color="g")
            plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                     label="Traning score")
            plt.plot(train_sizes, val_scores_mean, 'o-', color="g",
                     label="Validation score")
            plt.legend(loc="best")
            
            subplot_count +=1
        plt.show()
        
        # We convert the performance dictionary to Dataframe, we display it, then we reset it for the next target
        df_results = pd.DataFrame.from_dict(result_df)
        display(df_results)
        df_results.drop(index=df_results.index, inplace=True)
        
        subplot_count = 0
            

def generate_samples(X,y, _sampling_strategy = 'auto', n_samples_created=0):
    """Function to generate new data from existing ones.
    
    Args:
        X (Dictionary): Dictionary of X_dataframes
        y (Dictionary): Dictionary of y_dataframes
        _sampling_strategy (str, optional): number of samples wished if value is indicated (eg:500). Defaults to 'auto'.
        n_samples_created (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: Return a dictiopnary of X_dataframes and y_dataframes
    """
    
    if n_samples_created==0:
        sm = SMOTE(sampling_strategy=_sampling_strategy, random_state=1805)
    else:
        sm = SMOTE(sampling_strategy={0:n_samples_created}, random_state=1805)
    X_resampled = {}
    y_resampled = {}
    for i, (key,value) in enumerate(y.items()):
        tmp_X, tmp_y = sm.fit_resample(list(X.values())[i], value)
        X_resampled[key] = tmp_X
        y_resampled[key] = tmp_y

    return X_resampled, y_resampled