import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from mapie.regression import MapieRegressor
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV
from mapie.subsample import BlockBootstrap
from mapie.regression import MapieTimeSeriesRegressor
from sklearn.preprocessing import StandardScaler
from scalecast.auxmodels import auto_arima
from dieboldmariano import dm_test



os.chdir(rf'//Users/coraline/Desktop/Master2/Big final project macro')
#%%

data = pd.read_excel('finalcomplet.xlsx')
data = data.set_index('DATES')


#%%
def test_trend(series):
    X = sm.add_constant(np.arange(len(series)).reshape(-1, 1))
    y = series.values
    model = sm.OLS(y, X).fit()
    return model.pvalues[1] < 0.05 

def adf_test(series):
    result = adfuller(series,autolag='bic')
    return result[1] <= 0.05  

def make_stationary(df):
    stationary_series = pd.DataFrame()
    transformations = pd.DataFrame(columns=['Series', 'Differencing', 'Trend_Included'])

    for column in df.columns:
        original_series = df[column]
        series = original_series.copy()
        diff_count = 0
        trend_included = test_trend(series)
        while not adf_test(series):
            series = series.diff().dropna()
            series = series.multiply(-1)
            diff_count += 1
            trend_included = test_trend(series) if len(series) > 1 else False

        stationary_series[column] = series
        transformations = transformations.append({'Series': column, 'Differencing': diff_count, 'Trend_Included': trend_included}, ignore_index=True)

    return stationary_series, transformations
#%%


a,b = make_stationary(data)

a = a.dropna(axis = 0)
monnaie = data[['Agrégats monétaires France. M1 [encours]',
'Agrégats monétaires France. M2 [encours]']]

monnaie = monnaie.diff().dropna()
monnaie = monnaie.tail(-1)

a[['Agrégats monétaires France. M1 [encours]',
'Agrégats monétaires France. M2 [encours]']] = monnaie[['Agrégats monétaires France. M1 [encours]',
'Agrégats monétaires France. M2 [encours]']]
                                         
a.index = pd.to_datetime(a.index, format='%d/%m/%Y')
a = a[a.index < pd.Timestamp('2019-12-31')]

#%%


def create_lagged_features(df, n_lags, var_interet):
    for lag in range(1, n_lags + 1):
        df[f'{var_interet}_lag{lag}'] = df[var_interet].shift(lag)
    return df

def rolling_window_forecast_rf(data, window_size, forecast_horizon, pca=False, pls = False,variance=0.9, n_components = 30, var_interet='IPC ENSEMBLE'):
    results = []
    best_params_all = {}
    importances_dict = {}
    best_rf = None
    total_mse = 0
    coverage_counts = 0
    total_interval_width = 0
    total_predictions = 0

    data_with_lags = create_lagged_features(data, 3, var_interet)
    data_with_lags.dropna(inplace=True)
    data_with_lags.reset_index(drop=True, inplace=True)

    y = data_with_lags[var_interet]
    X = data_with_lags.drop(var_interet, axis=1)

    if pca:
        scaler = StandardScaler()
        data_transformed = scaler.fit_transform(data)
        pca_model = PCA(n_components=variance)
        data_transformed = pca_model.fit_transform(data_transformed)

        data_pca = pd.DataFrame(data_transformed, columns=[f'pca_{i}' for i in range(data_transformed.shape[1])])

        for lag in range(1, 4):
            data_pca[f'{var_interet}_lag{lag}'] = data_with_lags[f'{var_interet}_lag{lag}']

        data_pca[var_interet] = y

        data = data_pca
    if pls:
        pls_model = PLSRegression(n_components=n_components)
        X_pls = pls_model.fit_transform(X, y)[0]  

        data_pls = pd.DataFrame(X_pls, columns=[f'pls_component_{i+1}' for i in range(X_pls.shape[1])])

        for lag in range(1, 4):
            data_pls[f'{var_interet}_lag{lag}'] = data_with_lags[f'{var_interet}_lag{lag}']

        data_pls[var_interet] = y

        data = data_pls
    else:
        data = data_with_lags

    param_grid = {
        'n_estimators': [200],
        'max_depth': [5,10, 20],
        'min_samples_split': [ 4,6,8],
        'max_features' :['sqrt', 'log2']
    }

    tscv = TimeSeriesSplit(n_splits=5, test_size = 12)

    for i, start in enumerate(range(len(data) - window_size - forecast_horizon + 1)):
        end = start + window_size
        train = data.iloc[start:end]
        print(train)
        test = data.iloc[end:end + forecast_horizon]

        X_train, y_train = train.drop(var_interet, axis=1), train[var_interet]
        X_test, y_test = test.drop(var_interet, axis=1), test[var_interet]

        rf = RandomForestRegressor()
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=tscv, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        best_rf = grid_search.best_estimator_
        best_params_all[i] = grid_search.best_params_  
        
        feature_importances = best_rf.feature_importances_
        feature_names = data.drop(var_interet, axis=1).columns
        importances_dict = dict(zip(feature_names, feature_importances))


     
        cv_mapietimeseries = BlockBootstrap(
            n_resamplings=10, n_blocks=10, overlapping=False, random_state=59
        )
        
        mapie = MapieTimeSeriesRegressor(
            best_rf,
            method="enbpi",
            cv=cv_mapietimeseries,
            agg_function="mean",
            n_jobs=-1,
        )

        
        mapie.fit(X_train, y_train)

        y_pred, y_pred_interval = mapie.predict(X_test, alpha=0.05)


        mse = mean_squared_error(y_test, y_pred)
        total_mse += mse

        lower_bounds, upper_bounds = y_pred_interval[:, 0, 0], y_pred_interval[:, 1, 0]
        coverage = np.sum((y_test >= lower_bounds) & (y_test <= upper_bounds))
        coverage_counts += coverage

        interval_width = upper_bounds - lower_bounds
        total_interval_width += np.sum(interval_width)

        total_predictions += len(y_test)

        results.append((y_pred, y_test, y_pred_interval, mse))

    average_rmse = np.sqrt(total_mse / len(results))
    coverage_rate = coverage_counts / total_predictions
    average_interval_width = total_interval_width / total_predictions

    return results, best_params_all, importances_dict, average_rmse, coverage_rate, average_interval_width

forecast_results_rf, best_parameters_rf, importances_dict_rf, RMSE_rf, coverage_rate_rf, average_interval_width_rf = rolling_window_forecast_rf(a, len(a)-48-3, 1)
#%%


def rolling_window_forecast_enet(data, window_size, forecast_horizon, pca=True, pls = False, variance=0.95, n_components = 30, var_interet='IPC ENSEMBLE'):
    results = []
    best_params_all = {}
    selected_variables_all = {}
    total_mse = 0
    coverage_counts = 0
    total_interval_width = 0
    total_predictions = 0
    
    
    data_with_lags = create_lagged_features(data, 3, var_interet)
    data_with_lags.dropna(inplace=True)
    data_with_lags.reset_index(drop=True, inplace=True)
    y = data_with_lags[var_interet]
    X = data_with_lags.drop(var_interet, axis=1)

    if pca:
        scaler = StandardScaler()
        data_transformed = scaler.fit_transform(data)
        pca_model = PCA(n_components=variance)
        data_transformed = pca_model.fit_transform(data_transformed)

        data_pca = pd.DataFrame(data_transformed, columns=[f'pca_{i}' for i in range(data_transformed.shape[1])])

        for lag in range(1, 4):
            data_pca[f'{var_interet}_lag{lag}'] = data_with_lags[f'{var_interet}_lag{lag}']

        data_pca[var_interet] = y

        data = data_pca
    if pls:
        pls_model = PLSRegression(n_components=n_components)
        X_pls = pls_model.fit_transform(X, y)[0]  

        data_pls = pd.DataFrame(X_pls, columns=[f'pls_component_{i+1}' for i in range(X_pls.shape[1])])

        for lag in range(1, 4):
            data_pls[f'{var_interet}_lag{lag}'] = data_with_lags[f'{var_interet}_lag{lag}']

        data_pls[var_interet] = y

        data = data_pls
    else:
        data = data_with_lags


    param_grid = {
        'alpha': [0.01,0.025,0.05,0.075, 0.1, 1],
        'l1_ratio': [0, 0.1, 0.25, 0.5,0.75, 0.9, 1]
    }

    tscv = TimeSeriesSplit(n_splits=5, test_size= 12)

    for i, start in enumerate(range(len(data) - window_size - forecast_horizon + 1)):
        end = start + window_size
        train = data.iloc[start:end]
        test = data.iloc[end:end + forecast_horizon] 

        X_train, y_train = train.drop(var_interet, axis=1), train[var_interet]
        X_test, y_test = test.drop(var_interet, axis=1), test[var_interet]

        enet = ElasticNet()
        grid_search = GridSearchCV(estimator=enet, param_grid=param_grid, cv=tscv, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        best_enet = grid_search.best_estimator_
        best_params_all[i] = grid_search.best_params_  
        
        selected_variables = list(X_train.columns[best_enet.coef_ != 0])
        selected_variables_all[i] = selected_variables
        
        
        cv_mapietimeseries = BlockBootstrap(
            n_resamplings=10, n_blocks=10, overlapping=False, random_state=59
        )
        
        mapie = MapieTimeSeriesRegressor(
            best_enet,
            method="enbpi",
            cv=cv_mapietimeseries,
            agg_function="mean",
            n_jobs=-1,
        )

        
        mapie.fit(X_train, y_train)

        y_pred, y_pred_interval = mapie.predict(X_test, alpha=0.05)

        mse = mean_squared_error(y_test, y_pred)
        total_mse += mse

        lower_bounds, upper_bounds = y_pred_interval[:, 0, 0], y_pred_interval[:, 1, 0]
        coverage = np.sum((y_test >= lower_bounds) & (y_test <= upper_bounds))
        coverage_counts += coverage

        interval_width = upper_bounds - lower_bounds
        total_interval_width += np.sum(interval_width)

        total_predictions += len(y_test)

        results.append((y_pred, y_test, y_pred_interval, mse))

    average_rmse = np.sqrt(total_mse / len(results))
    coverage_rate = coverage_counts / total_predictions
    average_interval_width = total_interval_width / total_predictions

    return results, best_params_all, selected_variables_all, average_rmse , coverage_rate, average_interval_width

forecast_results_enet, best_parameters_enet, selected_variables_enet, RMSE_enet, coverage_rate_enet, average_interval_width_enet = rolling_window_forecast_enet(a, len(a)-48-3, 1)

#%%
from collections import Counter

def top_10_variables(importances_dict):
    all_variables = [variable for variables_list in importances_dict.values() for variable in variables_list]

    variable_counter = Counter(all_variables)

    top_10_variables = [var for var, _ in variable_counter.most_common(10)]

    return top_10_variables

top_variables_enet = top_10_variables(selected_variables_enet)
#%%


def rolling_window_forecast_svm(data, window_size, forecast_horizon, pca=False, pls = False, variance=0.95, n_components = 30, var_interet='IPC ENSEMBLE'):
    results = []
    best_params_all = {}
    total_mse = 0
    coverage_counts = 0
    total_interval_width = 0
    total_predictions = 0
    

    data_with_lags = create_lagged_features(data, 3, var_interet)
    data_with_lags.dropna(inplace=True)
    data_with_lags.reset_index(drop=True, inplace=True)
    y = data_with_lags[var_interet]
    X = data_with_lags.drop(var_interet, axis=1)

    if pca:
        scaler = StandardScaler()
        data_transformed = scaler.fit_transform(data)
        pca_model = PCA(n_components=variance)
        data_transformed = pca_model.fit_transform(data_transformed)

        data_pca = pd.DataFrame(data_transformed, columns=[f'pca_{i}' for i in range(data_transformed.shape[1])])

        for lag in range(1, 4):
            data_pca[f'{var_interet}_lag{lag}'] = data_with_lags[f'{var_interet}_lag{lag}']

        data_pca[var_interet] = y

        data = data_pca
    if pls:
        pls_model = PLSRegression(n_components=n_components)
        X_pls = pls_model.fit_transform(X, y)[0]  

        data_pls = pd.DataFrame(X_pls, columns=[f'pls_component_{i+1}' for i in range(X_pls.shape[1])])

        for lag in range(1, 4):
            data_pls[f'{var_interet}_lag{lag}'] = data_with_lags[f'{var_interet}_lag{lag}']

        data_pls[var_interet] = y

        data = data_pls
    else:
        data = data_with_lags

    param_grid = {'estimator_C': [0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], 'estimator_gamma': [1,0.1,0.01, 0.001, 10]}

    tscv = TimeSeriesSplit(n_splits=5, test_size = 12)

    for i, start in enumerate(range(len(data) - window_size - forecast_horizon + 1)):
        print(i)
        end = start + window_size
        train = data.iloc[start:end]
        test = data.iloc[end:end + forecast_horizon]

        X_train, y_train = train.drop(var_interet, axis=1), train[var_interet]
        X_test, y_test = test.drop(var_interet, axis=1), test[var_interet]

        svr = SVR(kernel = 'rbf', degree = 2)
        mapie = MapieRegressor(svr)
        grid_search = GridSearchCV(estimator=mapie, param_grid=param_grid, cv=tscv, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        best_svr = grid_search.best_estimator_
        best_params_all[i] = grid_search.best_params_
        
        cv_mapietimeseries = BlockBootstrap(
            n_resamplings=10, n_blocks=10, overlapping=False, random_state=59
        )
        
        mapie = MapieTimeSeriesRegressor(
            best_svr,
            method="enbpi",
            cv=cv_mapietimeseries,
            agg_function="mean",
            n_jobs=-1,
        )

        
        mapie.fit(X_train, y_train)

        y_pred, y_pred_interval = mapie.predict(X_test, alpha=0.05)

        mse = mean_squared_error(y_test, y_pred)
        total_mse += mse
 

        lower_bounds, upper_bounds = y_pred_interval[:, 0, 0], y_pred_interval[:, 1, 0]
        coverage = np.sum((y_test >= lower_bounds) & (y_test <= upper_bounds))
        coverage_counts += coverage

        interval_width = upper_bounds - lower_bounds
        total_interval_width += np.sum(interval_width)

        total_predictions += len(y_test)

        results.append((y_pred, y_test, y_pred_interval, mse))

    average_rmse = np.sqrt(total_mse / len(results))
    coverage_rate = coverage_counts / total_predictions
    average_interval_width = total_interval_width / total_predictions

    return results, best_params_all, average_rmse, coverage_rate, average_interval_width

# Use the function
forecast_results_svm, best_parameters_svm, RMSE_svm, coverage_rate_svm, average_interval_width_svm = rolling_window_forecast_svm(a, len(a)-48
                                                                                                                                                     -3, 1)



#%%


flat_forecasts = []
flat_actuals = []
lower_bounds = []
upper_bounds = []

for result in forecast_results_rf:
    y_pred, y_test, y_pred_interval, _ = result

    flat_forecasts.append(y_pred)
    flat_actuals.append(y_test)

    lower_bounds.append(y_pred_interval[0, 0, 0])  
    upper_bounds.append(y_pred_interval[0, 1, 0])  

plt.figure(figsize=(12, 6))

plt.plot(flat_forecasts, label='Forecast', color='blue')
plt.plot(flat_actuals, label='Actual', color='orange')

plt.fill_between(range(len(flat_forecasts)), lower_bounds, upper_bounds, color='blue', alpha=0.3, label='Confidence Interval')

plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Random Forest Rolling Window Forecast with Confidence Interval')
plt.legend()

plt.show()



#%%

data = pd.read_excel('finalcomplet.xlsx')
data = data.set_index('DATES')
a,b = make_stationary(data)

a = a.dropna(axis = 0)
monnaie = data[['Agrégats monétaires France. M1 [encours]',
'Agrégats monétaires France. M2 [encours]']]

monnaie = monnaie.diff().dropna()
monnaie = monnaie.tail(-1)

a[['Agrégats monétaires France. M1 [encours]',
'Agrégats monétaires France. M2 [encours]']] = monnaie[['Agrégats monétaires France. M1 [encours]',
'Agrégats monétaires France. M2 [encours]']]
                                         
a.index = pd.to_datetime(a.index, format='%d/%m/%Y')
a = a[a.index < pd.Timestamp('2019-12-31')]
a = a.reset_index()



def rolling_window_forecast_arima(data, window_size, forecast_horizon, var_interet='IPC ENSEMBLE'):
    results = []
    total_mse = 0
    coverage_counts = 0
    total_predictions = 0
    

    y = data[var_interet]

    for start in range(len(y) - window_size - forecast_horizon + 1):
        end = start + window_size
        train = y.iloc[start:end]
        test = y.iloc[end:end + forecast_horizon]

        model = auto_arima(train, seasonal=False, d=0, stepwise=True, trace=False)

        y_forec, conf_int = model.predict(n_periods=forecast_horizon, return_conf_int=True, alpha=0.05)

        mse = mean_squared_error(test, y_forec)
        total_mse += mse

        lower_bounds, upper_bounds = conf_int[:, 0], conf_int[:, 1]
        coverage = np.sum((test.values >= lower_bounds) & (test.values <= upper_bounds))
        coverage_counts += coverage
        total_predictions += len(test)

        results.append((y_forec, test.values, mse, conf_int))

    average_rmse = np.sqrt(total_mse / len(results))
    coverage_rate = coverage_counts / total_predictions

    return results, average_rmse, coverage_rate

forecast_results_arima, RMSE_arima, coverage_rate_arima = rolling_window_forecast_arima(a, window_size=len(a)-48, forecast_horizon=1)
#%%
import matplotlib.pyplot as plt

flat_forecasts = []
flat_actuals = []
lower_bounds = []
upper_bounds = []

for result in forecast_results_arima:
    y_pred, y_test, _, y_pred_interval = result

    for forecast in y_pred:
        flat_forecasts.append(forecast)
    for actual in y_test:
        flat_actuals.append(actual)
    
    for interval in y_pred_interval:
        lower_bounds.append(interval[0])
        upper_bounds.append(interval[1])

plt.figure(figsize=(12, 6))

plt.plot(flat_forecasts, label='Forecast', color='blue')
plt.plot(flat_actuals, label='Actual', color='orange')

plt.fill_between(range(len(flat_forecasts)), lower_bounds, upper_bounds, color='blue', alpha=0.3, label='Confidence Interval')

plt.xlabel('Time')
plt.ylabel('Value')
plt.title('ARIMA Rolling Window Forecast with Confidence Interval')
plt.legend()

plt.show()

#%%
T = []
F = []
G = []
#%%
for result in forecast_results_svm:
    y_pred, y_test, _, _ = result
    
    for forecast in y_pred:
        G.append(forecast)

for result in forecast_results_enet:
    y_pred, y_test, _, _ = result
    
    for forecast in y_pred:
        F.append(forecast)


for result in forecast_results_enet:
    _, y_test, _, _ = result
    
    for actual in y_test:
        T.append(actual)
#%%

dm_test(T, F, G, one_sided=True)
#%%