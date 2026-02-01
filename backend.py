import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from sklearn.linear_model import LinearRegression
import os


# Helper: save forecast DataFrame and figures, log exported file paths
def _save_forecast_and_plots(session_logger, model_name, df, col, resid_clean, forecast_values, fig, fig_res, horizon=24):
    try:
        last_date = pd.to_datetime(df['date'].iloc[-1])
    except Exception:
        last_date = pd.Timestamp.now()

    # assume hourly frequency for PJME_hourly
    future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=horizon, freq='H')
    forecast_df = pd.DataFrame({'date': future_dates, 'forecast': np.asarray(forecast_values).astype(float)})

    # approximate 95% CI using residual std
    try:
        resid_std = float(resid_clean.std()) if len(resid_clean) > 0 else float(np.std(df[col].dropna()))
    except Exception:
        resid_std = 0.0
    z = 1.96
    forecast_df['lower_95'] = forecast_df['forecast'] - z * resid_std
    forecast_df['upper_95'] = forecast_df['forecast'] + z * resid_std

    # Prepare output directory
    out_dir = getattr(session_logger, 'log_dir', 'logs') if session_logger else 'logs'
    os.makedirs(out_dir, exist_ok=True)
    sid = getattr(session_logger, 'session_id', 'nosession') if session_logger else 'nosession'

    safe_name = model_name.replace(' ', '_')
    csv_path = os.path.join(out_dir, f"forecast_{sid}_{safe_name}.csv")
    fig_path = os.path.join(out_dir, f"forecast_{sid}_{safe_name}_plot.png")
    res_path = os.path.join(out_dir, f"forecast_{sid}_{safe_name}_residuals.png")

    try:
        forecast_df.to_csv(csv_path, index=False, encoding='utf-8')
    except Exception:
        csv_path = None

    try:
        if fig is not None:
            fig.savefig(fig_path, bbox_inches='tight')
    except Exception:
        fig_path = None

    try:
        if fig_res is not None:
            fig_res.savefig(res_path, bbox_inches='tight')
    except Exception:
        res_path = None

    files = {'forecast_csv': csv_path, 'figure_png': fig_path, 'residuals_png': res_path}
    try:
        if session_logger and hasattr(session_logger, 'log_exported_files'):
            session_logger.log_exported_files(files)
    except Exception:
        pass

    return files, forecast_df


# ==========================
#  PREPROCESSING FUNCTIONS
# ==========================

def load_and_clean(df, session_logger=None):
    df = df.copy()

    # Convert date
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date').reset_index(drop=True)

    # Set date as index for time interpolation
    df.set_index('date', inplace=True)

    # ============ TRAITEMENT DES VALEURS MANQUANTES ============
    missing_before = df.isnull().sum().to_dict()
    
    # Interpolate missing values
    df.interpolate(method='time', inplace=True)
    
    missing_after = df.isnull().sum().to_dict()

    if session_logger:
        # Détails de l'interpolation
        missing_details = {}
        for col in df.columns:
            before_count = missing_before.get(col, 0)
            after_count = missing_after.get(col, 0)
            interpolated = before_count - after_count
            total_rows = len(df)
            
            missing_details[col] = {
                'before': int(before_count),
                'before_pct': (before_count / total_rows * 100) if total_rows > 0 else 0,
                'after': int(after_count),
                'after_pct': (after_count / total_rows * 100) if total_rows > 0 else 0,
                'interpolated': int(interpolated)
            }
        
        session_logger.log_missing_values_handling(
            "Interpolation temporelle (method='time')",
            missing_details
        )

    # ============ TRAITEMENT DES VALEURS ABERRANTES (IQR) ============
    outlier_stats = {}
    # Target PJME hourly column if present, otherwise operate on numeric columns
    if 'pjme_mw' in df.columns:
        target_cols = ['pjme_mw']
    else:
        target_cols = [c for c in df.select_dtypes(include=[np.number]).columns]

    for col in target_cols:
        try:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            low = Q1 - 1.5*IQR
            high = Q3 + 1.5*IQR
            outliers_mask = ~df[col].between(low, high)
            removed_count = int(outliers_mask.sum())

            # Remplacer les outliers par NaN
            df[col] = df[col].mask(outliers_mask)
            # Interpoler les valeurs supprimées
            df[col].interpolate(method='time', inplace=True)

            total_rows = len(df)
            outlier_stats[col] = {
                "Q1": float(Q1),
                "Q3": float(Q3),
                "IQR": float(IQR),
                "lower_bound": float(low),
                "upper_bound": float(high),
                "outliers_removed": int(removed_count),
                "removal_rate": (removed_count / total_rows * 100) if total_rows > 0 else 0
            }
        except Exception:
            continue

    if session_logger and outlier_stats:
        session_logger.log_outliers_removal(
            "Méthode IQR (Interquartile Range) avec facteur 1.5",
            outlier_stats
        )

    # Reset index to restore 'date' as column
    df.reset_index(inplace=True)

    if session_logger:
        # Résumé final
        summary = f"""
Étapes complétées:
1. Conversion et tri des dates: OK
2. Interpolation temporelle des valeurs manquantes: OK
3. Suppression et interpolation des valeurs aberrantes (IQR): OK
4. Réinitialisation de l'index: OK

Statistiques finales:
- Nombre total d'observations finales: {len(df)}
- Dimensions finales: {len(df)} lignes × {len(df.columns)} colonnes
- Colonnes présentes: {list(df.columns)}
- Valeurs manquantes résiduelles: {int(df.isnull().sum().sum())}
"""
        session_logger.log_preprocessing_summary(summary)

    return df



# ==========================
#  EDA FUNCTIONS
# ==========================

def plot_series(df, col='pjme_mw'):
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(df['date'], df[col])
    ax.set_title(f"Time Series: {col}")
    return fig


def plot_acf_pacf(df, col='pjme_mw'):
    fig, ax = plt.subplots(2,1, figsize=(10,6))
    plot_acf(df[col].dropna(), ax=ax[0])
    plot_pacf(df[col].dropna(), ax=ax[1])
    return fig


def adf_test(df, session_logger=None, col='pjme_mw'):
    result = adfuller(df[col].dropna())
    
    if session_logger:
        test_results = {
            "ADF Statistic": float(result[0]),
            "p-value": float(result[1]),
            "Nombre de lags utilisés": int(result[2]),
            "Nombre d'observations": int(result[3]),
            "Valeurs critiques": {
                "1%": float(result[4]['1%']),
                "5%": float(result[4]['5%']),
                "10%": float(result[4]['10%'])
            }
        }
        session_logger.log_test_result("Augmented Dickey-Fuller (ADF)", test_results)
    
    return result[0], result[1]  # statistic & p-value


# ==========================
#  EDA FUNCTIONS (ANALYSIS)
# ==========================

def compute_descriptive_statistics(df, session_logger=None):
    """Calcule les statistiques descriptives (moyenne, variance, skewness, kurtosis)"""
    stats_dict = {}
    
    # Colonnes numériques (excluant 'date')
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        data = df[col].dropna()
        stats_dict[col] = {
            'count': len(data),
            'mean': float(data.mean()),
            'median': float(data.median()),
            'std': float(data.std()),
            'variance': float(data.var()),
            'min': float(data.min()),
            'max': float(data.max()),
            'q25': float(data.quantile(0.25)),
            'q50': float(data.quantile(0.50)),
            'q75': float(data.quantile(0.75)),
            'skewness': float(stats.skew(data)),
            'kurtosis': float(stats.kurtosis(data))
        }
    
    if session_logger:
        session_logger.log_descriptive_statistics(stats_dict)
    
    return stats_dict


def test_stationarity(df, session_logger=None):
    """Effectue les tests de stationnarité (ADF et KPSS)"""
    tests_results = {'ADF': {}, 'KPSS': {}}
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        data = df[col].dropna()
        
        # Test ADF
        adf_result = adfuller(data)
        tests_results['ADF'][col] = {
            'statistic': float(adf_result[0]),
            'pvalue': float(adf_result[1]),
            'nlags': int(adf_result[2]),
            'nobs': int(adf_result[3]),
            'critical_values': {
                '1%': float(adf_result[4]['1%']),
                '5%': float(adf_result[4]['5%']),
                '10%': float(adf_result[4]['10%'])
            }
        }
        
        # Test KPSS
        try:
            kpss_result = kpss(data, regression='c')
            tests_results['KPSS'][col] = {
                'statistic': float(kpss_result[0]),
                'pvalue': float(kpss_result[1]),
                'nlags': int(kpss_result[2]),
                'nobs': len(data),
                'critical_values': {
                    '10%': float(kpss_result[3]['10%']),
                    '5%': float(kpss_result[3]['5%']),
                    '2.5%': float(kpss_result[3]['2.5%']),
                    '1%': float(kpss_result[3]['1%'])
                }
            }
        except:
            tests_results['KPSS'][col] = {'error': 'Unable to perform KPSS test'}
    
    if session_logger:
        session_logger.log_stationarity_tests(tests_results)
    
    return tests_results


def detect_seasonality(df, col='pjme_mw', session_logger=None):
    """Détecte la saisonnalité dans la série temporelle"""
    seasonality_results = {}
    
    try:
        data = df[col].dropna()
        
        # Décomposition saisonnière (hourly data -> daily seasonality = 24)
        if len(data) >= 48:  # Au moins 2 daily periods
            period = 24
            decomposition = seasonal_decompose(data, model='additive', period=period)

            # Calculer la force de saisonnalité (utiliser nanvar pour robustesse)
            resid_var = np.nanvar(decomposition.resid)
            seasonal_var = np.nanvar(decomposition.seasonal)
            trend_var = np.nanvar(decomposition.trend)

            seasonal_strength = 1 - (resid_var / (seasonal_var + resid_var + 1e-9))
            trend_strength = 1 - (resid_var / (trend_var + resid_var + 1e-9))

            has_seasonality = seasonal_strength > 0.05

            # ACF analysis (inspect up to one week)
            acf_max_lag = min(168, len(data)-1)

            seasonality_results[col] = {
                'has_seasonality': bool(has_seasonality),
                'period': int(period),
                'strength': float(seasonal_strength),
                'trend_strength': float(trend_strength),
                'acf_max_lag': int(acf_max_lag),
                'pacf_significant': 'Check PACF plot'
            }
        else:
            seasonality_results[col] = {
                'has_seasonality': False,
                'period': None,
                'strength': 0,
                'trend_strength': 0,
                'acf_max_lag': None,
                'pacf_significant': None
            }
    except Exception as e:
        seasonality_results[col] = {'error': str(e)}
    
    if session_logger:
        session_logger.log_seasonality_analysis(seasonality_results)
    
    return seasonality_results


def analyze_trend(df, col='pjme_mw', session_logger=None):
    """Analyse la tendance de la série temporelle"""
    trend_results = {}
    
    try:
        data = df[col].dropna()
        
        # Régression linéaire pour la tendance
        X = np.arange(len(data)).reshape(-1, 1)
        y = data.values
        
        model = LinearRegression()
        model.fit(X, y)
        if session_logger:
            try:
                session_logger.log_model_fit_details('Linear Regression Trend', model)
            except Exception:
                pass
        
        slope = float(model.coef_[0])
        intercept = float(model.intercept_)
        r_squared = float(model.score(X, y))
        
        # Test de significativité de la pente
        from scipy.stats import linregress
        slope_test = linregress(np.arange(len(data)), y)
        p_value = float(slope_test.pvalue)
        
        # Déterminer le type de tendance
        if abs(slope) < 0.001:
            trend_type = "Stationnaire"
        elif slope > 0.001:
            trend_type = "Croissante"
        else:
            trend_type = "Décroissante"
        
        direction = "↗ Hausse" if slope > 0 else "↘ Baisse" if slope < 0 else "→ Stable"
        significant = p_value < 0.05
        
        total_change = (y[-1] - y[0]) if len(y) > 0 else 0
        
        trend_results[col] = {
            'trend_type': trend_type,
            'direction': direction,
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'pvalue': p_value,
            'significant': bool(significant),
            'min_value': float(data.min()),
            'max_value': float(data.max()),
            'total_change': float(total_change)
        }
    except Exception as e:
        trend_results[col] = {'error': str(e)}
    
    if session_logger:
        session_logger.log_trend_analysis(trend_results)
    
    return trend_results


# -----------------------------
#  Validation, Grid Search & Helpers
# -----------------------------
def time_series_train_test_split(df, col='pjme_mw', train_ratio=0.8):
    """Split time-ordered data into train/test by proportion."""
    df_sorted = df.sort_values('date').reset_index(drop=True)
    n = len(df_sorted)
    if n == 0:
        return df_sorted.copy(), df_sorted.copy()
    train_n = max(1, int(n * float(train_ratio)))
    train = df_sorted.iloc[:train_n].copy()
    test = df_sorted.iloc[train_n:].copy()
    return train, test


def plot_train_test_split(df, col='pjme_mw', train_ratio=0.8):
    """Create an 80/20 chronological train-test split and plot train (blue) and test (red).

    Returns matplotlib Figure.
    """
    df_sorted = df.sort_values('date').reset_index(drop=True).copy()
    if 'date' not in df_sorted.columns:
        raise ValueError("DataFrame must contain a 'date' column for plotting")

    n = len(df_sorted)
    if n == 0:
        raise ValueError('Empty DataFrame')

    train_n = max(1, int(n * float(train_ratio)))
    train = df_sorted.iloc[:train_n]
    test = df_sorted.iloc[train_n:]

    # Determine split point (vertical line at last training timestamp)
    split_idx = train_n - 1
    split_date = pd.to_datetime(df_sorted['date'].iloc[split_idx])

    fig, ax = plt.subplots(figsize=(12, 4))
    # Plot training data
    ax.plot(pd.to_datetime(train['date']), train[col], color='blue', label='Train', linewidth=1.5)
    # Plot test data
    if len(test) > 0:
        ax.plot(pd.to_datetime(test['date']), test[col], color='red', label='Test', linewidth=1.5)

    # Vertical dashed split line
    ax.axvline(x=split_date, color='black', linestyle='--', linewidth=1)

    ax.set_xlabel('Date')
    ax.set_ylabel(col)
    ax.set_title('Data train and test split')
    ax.legend()
    fig.autofmt_xdate()
    return fig


def _evaluate_model_forecast(model_type, train, horizon, params):
    """Fit a simple model on `train` (pandas DF with date+col) and forecast `horizon` steps.

    Supported model_type: 'ses', 'holt_hw_add', 'holt_hw_mul', 'lr', 'sma'
    Returns forecast (np.array) and residuals on train.
    """
    y = train.iloc[:, train.columns.str.lower() == 'pjme_mw']
    if y.empty:
        # try numeric first numeric column
        numeric = train.select_dtypes(include=[np.number]).columns
        if len(numeric) == 0:
            raise ValueError('No numeric column found for forecasting')
        col = numeric[0]
        y = train[col]
    else:
        col = 'pjme_mw'
        y = train[col]

    y = y.dropna()
    if model_type == 'ses':
        fit = SimpleExpSmoothing(y, initialization_method='estimated').fit(smoothing_level=params.get('alpha') if params and 'alpha' in params else None)
        try:
            forecast = fit.forecast(horizon)
        except Exception:
            last = y.iloc[-1]
            forecast = np.repeat(last, horizon)
        resid = y - fit.fittedvalues
        return np.asarray(forecast), resid

    if model_type == 'holt':
        # Holt's linear trend (no seasonality)
        fit = ExponentialSmoothing(y, trend='add', seasonal=None, initialization_method='estimated').fit()
        try:
            forecast = fit.forecast(horizon)
        except Exception:
            forecast = np.repeat(y.iloc[-1], horizon)
        resid = y - fit.fittedvalues
        return np.asarray(forecast), resid

    if model_type in ('holt_hw_add', 'holt_hw_mul'):
        seasonal = 'add' if model_type == 'holt_hw_add' else 'mul'
        sp = int(params.get('seasonal_periods', 24)) if params else 24
        fit = ExponentialSmoothing(y, trend='add', seasonal=seasonal, seasonal_periods=sp, initialization_method='estimated').fit()
        try:
            forecast = fit.forecast(horizon)
        except Exception:
            forecast = np.repeat(y.iloc[-1], horizon)
        resid = y - fit.fittedvalues
        return np.asarray(forecast), resid

    if model_type == 'lr':
        X = np.arange(len(y)).reshape(-1,1)
        lr = LinearRegression().fit(X, y.values)
        future_idx = np.arange(len(y), len(y) + horizon).reshape(-1,1)
        forecast = lr.predict(future_idx)
        fitted = lr.predict(X)
        resid = y.values - fitted
        return np.asarray(forecast), pd.Series(resid, index=y.index)

    if model_type == 'sma':
        w = int(params.get('window', 24)) if params else 24
        fitted = y.rolling(window=w, min_periods=1).mean()
        last = fitted.iloc[-1] if len(fitted)>0 else y.iloc[-1]
        forecast = np.repeat(last, horizon)
        resid = y - fitted
        return np.asarray(forecast), resid

    raise ValueError(f'Unsupported model_type: {model_type}')


def bayesian_optimization(model_type, search_space, df, col='pjme_mw', n_calls=25, cv_params=None, session_logger=None):
    """Optional Bayesian optimization wrapper using scikit-optimize (skopt).

    search_space: dict of param name -> (low, high, type)
    Example: {'alpha': (0.01,0.5,'real'), 'window': (3,48,'int')}

    This function will attempt to import `skopt` and run `gp_minimize`. If `skopt` is unavailable,
    it will return None and log a warning via session_logger.
    """
    try:
        from skopt import gp_minimize
        from skopt.space import Real, Integer
    except Exception as e:
        if session_logger and hasattr(session_logger, 'log_warning'):
            session_logger.log_warning(f"Bayesian optimization skipped: missing skopt ({e})")
        return None

    # Build skopt space
    sk_space = []
    param_names = []
    for k, v in search_space.items():
        low, high, ptype = v
        param_names.append(k)
        if ptype == 'real':
            sk_space.append(Real(low, high, name=k))
        else:
            sk_space.append(Integer(int(low), int(high), name=k))

    def _objective(x):
        params = {name: val for name, val in zip(param_names, x)}
        # convert ints where needed
        for k, v in search_space.items():
            if v[2] == 'int':
                params[k] = int(params[k])
        # run rolling cv
        res = rolling_origin_cv(model_type, df, col=col, params=params, session_logger=None, **(cv_params or {}))
        return float(res.get('mean_mse', float('inf')))

    res = gp_minimize(_objective, sk_space, n_calls=int(n_calls))
    best_params = {name: (int(val) if search_space[name][2]=='int' else float(val)) for name, val in zip(param_names, res.x)}
    summary = {'best_params': best_params, 'fun': float(res.fun)}
    if session_logger and hasattr(session_logger, 'log_grid_search_results'):
        session_logger.log_grid_search_results(f"bayes_{model_type}", {'tested': [], 'best': {'params': best_params, 'mean_mse': float(res.fun)}})
    return summary


def rolling_origin_cv(model_type, df, col='pjme_mw', initial_train_size=None, horizon=24, step=24, params=None, session_logger=None):
    """Perform rolling-origin evaluation and return per-fold MSE and coverage.

    initial_train_size: number of observations to use for the first train fold. If None, uses 50%.
    step: steps to advance the origin between folds (in observations).
    """
    df_sorted = df.sort_values('date').reset_index(drop=True)
    n = len(df_sorted)
    if initial_train_size is None:
        initial_train_size = max(2, int(0.5 * n))
    folds = []
    start = initial_train_size
    while start < n:
        train = df_sorted.iloc[:start]
        test = df_sorted.iloc[start:start + horizon]
        if len(test) == 0:
            break
        try:
            fc, resid = _evaluate_model_forecast(model_type, train, len(test), params or {})
            true = test[col].values if col in test.columns else test.select_dtypes(include=[np.number]).iloc[:,0].values
            mse = float(np.mean((true - fc) ** 2))
            # Coverage using 95% approx with residual std
            resid_std = float(np.nanstd(resid)) if len(resid)>0 else 0.0
            lower = fc - 1.96 * resid_std
            upper = fc + 1.96 * resid_std
            coverage = float(np.mean((true >= lower) & (true <= upper))) if len(true)>0 else 0.0
        except Exception as e:
            mse = float('inf')
            coverage = 0.0

        folds.append({'train_end': str(df_sorted['date'].iloc[start-1]) if 'date' in df_sorted.columns else start-1,
                      'mse': mse,
                      'params': params or {},
                      'coverage': coverage})
        start += step

    mses = [f['mse'] for f in folds if np.isfinite(f['mse'])]
    mean_mse = float(np.mean(mses)) if len(mses)>0 else float('inf')
    std_mse = float(np.std(mses)) if len(mses)>0 else float('nan')
    result = {'folds': folds, 'mean_mse': mean_mse, 'std_mse': std_mse, 'params': params or {}}
    try:
        if session_logger and hasattr(session_logger, 'log_time_series_cv_results'):
            session_logger.log_time_series_cv_results(model_type, result)
    except Exception:
        pass
    return result


def grid_search(model_type, param_grid, df, col='pjme_mw', cv_method='rolling', cv_params=None, session_logger=None):
    """Grid search over `param_grid` (dict of lists). Returns tested summary and best.

    model_type: string as in rolling_origin_cv
    param_grid: dict of parameter name -> list of values
    cv_params: dict passed to rolling_origin_cv
    """
    from itertools import product
    keys = list(param_grid.keys())
    tested = []
    best = None
    for vals in product(*[param_grid[k] for k in keys]):
        params = dict(zip(keys, vals))
        if cv_method == 'rolling':
            res = rolling_origin_cv(model_type, df, col=col, params=params, session_logger=session_logger, **(cv_params or {}))
            mean_mse = res.get('mean_mse', float('inf'))
            std_mse = res.get('std_mse', float('nan'))
        else:
            # fallback: single holdout
            train, test = time_series_train_test_split(df, col=col, train_ratio=cv_params.get('train_ratio', 0.8) if cv_params else 0.8)
            try:
                fc, resid = _evaluate_model_forecast(model_type, train, len(test), params)
                true = test[col].values if col in test.columns else test.select_dtypes(include=[np.number]).iloc[:,0].values
                mean_mse = float(np.mean((true - fc) ** 2))
                std_mse = 0.0
            except Exception:
                mean_mse = float('inf')
                std_mse = float('nan')

        tested.append({'params': params, 'mean_mse': mean_mse, 'std_mse': std_mse})
        if best is None or mean_mse < best['mean_mse']:
            best = {'params': params, 'mean_mse': mean_mse}

    summary = {'tested': tested, 'best': best}
    try:
        if session_logger and hasattr(session_logger, 'log_grid_search_results'):
            session_logger.log_grid_search_results(model_type, summary)
    except Exception:
        pass
    return summary


def validate_forecast_intervals(forecast_df, true_series, lower_col='lower_95', upper_col='upper_95'):
    """Compute coverage and mean width for forecast intervals.

    forecast_df: DataFrame with forecast values and interval columns
    true_series: 1d array-like of true values aligned to forecast_df
    """
    true = np.asarray(true_series)
    if len(true) == 0 or forecast_df is None:
        return {'coverage': None, 'mean_width': None, 'horizon': 0}
    lower = np.asarray(forecast_df[lower_col]) if lower_col in forecast_df.columns else np.asarray(forecast_df['forecast'] - 1.96 * forecast_df['forecast'].std())
    upper = np.asarray(forecast_df[upper_col]) if upper_col in forecast_df.columns else np.asarray(forecast_df['forecast'] + 1.96 * forecast_df['forecast'].std())
    within = (true >= lower) & (true <= upper)
    coverage = float(np.mean(within))
    mean_width = float(np.mean(upper - lower))
    return {'coverage': coverage, 'mean_width': mean_width, 'horizon': len(true)}


def select_best_model(models_summary, prefer='AIC'):
    """Select best model from a list of model summaries (name+metrics).

    prefer: 'AIC'|'BIC'|'MSE'
    models_summary: list of dicts { 'name': str, 'metrics': {...} }
    Returns selection_summary for logger.
    """
    candidates = []
    for m in models_summary:
        metrics = m.get('metrics', {})
        score = None
        if prefer == 'AIC' and isinstance(metrics.get('AIC'), (int, float)):
            score = metrics.get('AIC')
        elif prefer == 'BIC' and isinstance(metrics.get('BIC'), (int, float)):
            score = metrics.get('BIC')
        elif prefer == 'MSE' and isinstance(metrics.get('MSE'), (int, float)):
            score = metrics.get('MSE')
        else:
            # fallback numeric metric
            score = metrics.get('MSE') if isinstance(metrics.get('MSE'), (int, float)) else float('inf')
        candidates.append({'name': m.get('name'), 'metrics': metrics, 'score': score})

    ranked = sorted(candidates, key=lambda x: x['score'] if x['score'] is not None else float('inf'))
    selected = ranked[0] if ranked else None
    summary = {'selected': selected.get('name') if selected else None, 'by': prefer, 'candidates': candidates}
    return summary



# -----------------------------
# 1. Simple Moving Average
# -----------------------------
def simple_moving_average(df, col='pjme_mw', window=7, session_logger=None):
    start_time = time.time()
    
    try:
        df_sma = df.copy()
        
        initial_params = {'window': window, 'method': 'Simple Moving Average'}
        if session_logger:
            session_logger.log_model_initialization('Simple Moving Average', initial_params)
        
        df_sma[f'SMA_{window}'] = df_sma[col].rolling(window=window).mean()
        
        execution_time = time.time() - start_time
        
        # Calculer les métriques
        mse = np.mean((df_sma[col].dropna() - df_sma[f'SMA_{window}'].dropna()) ** 2)
        mae = np.mean(np.abs(df_sma[col].dropna() - df_sma[f'SMA_{window}'].dropna()))
        
        # Residuals and additional tests
        fitted = df_sma[f'SMA_{window}']
        actual = df_sma[col]
        resid = actual - fitted
        resid_clean = resid.dropna()

        mape = float(np.mean(np.abs(resid_clean / (actual.loc[resid_clean.index].replace(0, np.nan))) ) * 100) if len(resid_clean)>0 else np.nan
        try:
            jb_stat, jb_p = stats.jarque_bera(resid_clean)
        except Exception:
            jb_stat, jb_p = np.nan, np.nan
        try:
            lb = acorr_ljungbox(resid_clean, lags=[10], return_df=True)
            lb_p = float(lb['lb_pvalue'].iloc[-1])
        except Exception:
            lb_p = np.nan

        final_metrics = {
            'MSE': float(mse),
            'MAE': float(mae),
            'MAPE': float(mape) if not np.isnan(mape) else 'N/A',
            'AIC': 'N/A',
            'BIC': 'N/A',
            'JB_stat': float(jb_stat) if not np.isnan(jb_stat) else 'N/A',
            'JB_pvalue': float(jb_p) if not np.isnan(jb_p) else 'N/A',
            'LjungBox_pvalue_lag10': float(lb_p) if not np.isnan(lb_p) else 'N/A',
            'fitted_values_count': int(df_sma[f'SMA_{window}'].notna().sum()),
            'optimal_params': initial_params,
            'converged': True
        }
        
        if session_logger:
            session_logger.log_model_completion(
                'Simple Moving Average',
                initial_params,
                final_metrics,
                execution_time,
                converged=True
            )
        
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(df_sma['date'], df_sma[col], label='Original', linewidth=2)
        ax.plot(df_sma['date'], df_sma[f'SMA_{window}'], label=f'SMA {window}', linewidth=2, alpha=0.7)
        ax.set_title(f"Simple Moving Average ({window})")
        ax.legend()

        # Residuals figure
        fig_res, axes = plt.subplots(1,3, figsize=(15,4))
        axes[0].plot(df_sma['date'], resid)
        axes[0].set_title('Residuals over time')
        axes[1].hist(resid_clean, bins=30)
        axes[1].set_title('Residuals histogram')
        try:
            sm.qqplot(resid_clean, line='s', ax=axes[2])
            axes[2].set_title('QQ-plot')
        except Exception:
            axes[2].text(0.5,0.5,'QQ plot failed', ha='center')

        # Forecast (default horizon = 24 hours)
        try:
            h = 24
            try:
                last_vals = df_sma[col].dropna().iloc[-window:]
                fv = [float(last_vals.mean())] * h
            except Exception:
                fv = [float(df_sma[col].dropna().mean())] * h

            files, forecast_df = _save_forecast_and_plots(session_logger, 'Simple Moving Average', df_sma, col, resid_clean, fv, fig, fig_res, horizon=h)
            final_metrics['exported_files'] = files
        except Exception:
            final_metrics['exported_files'] = {}

        return fig, fig_res, execution_time, final_metrics
    
    except Exception as e:
        execution_time = time.time() - start_time
        if session_logger:
            session_logger.log_model_error('Simple Moving Average', str(e))
        raise

# -----------------------------
# 2. Linear Regression Trend
# -----------------------------
def linear_regression_trend(df, col='pjme_mw', session_logger=None):
    start_time = time.time()
    
    try:
        df_lr = df.copy()
        
        initial_params = {'method': 'Linear Regression Trend'}
        if session_logger:
            session_logger.log_model_initialization('Linear Regression Trend', initial_params)
        
        X = np.arange(len(df_lr)).reshape(-1,1)
        y = df_lr[col].values
        model = LinearRegression().fit(X, y)
        df_lr['Trend'] = model.predict(X)
        
        execution_time = time.time() - start_time
        
        # Calculer les métriques
        mse = np.mean((y - df_lr['Trend']) ** 2)
        mae = np.mean(np.abs(y - df_lr['Trend']))
        r_squared = model.score(X, y)
        
        optimal_params = {
            'slope': float(model.coef_[0]),
            'intercept': float(model.intercept_)
        }
        
        # Residuals and tests
        fitted = df_lr['Trend']
        actual = df_lr[col]
        resid = actual - fitted
        resid_clean = resid.dropna()
        mape = float(np.mean(np.abs(resid_clean / (actual.loc[resid_clean.index].replace(0, np.nan)))) * 100) if len(resid_clean)>0 else np.nan
        try:
            jb_stat, jb_p = stats.jarque_bera(resid_clean)
        except Exception:
            jb_stat, jb_p = np.nan, np.nan
        try:
            lb = acorr_ljungbox(resid_clean, lags=[10], return_df=True)
            lb_p = float(lb['lb_pvalue'].iloc[-1])
        except Exception:
            lb_p = np.nan

        final_metrics = {
            'MSE': float(mse),
            'MAE': float(mae),
            'MAPE': float(mape) if not np.isnan(mape) else 'N/A',
            'R²': float(r_squared),
            'AIC': 'N/A',
            'BIC': 'N/A',
            'JB_stat': float(jb_stat) if not np.isnan(jb_stat) else 'N/A',
            'JB_pvalue': float(jb_p) if not np.isnan(jb_p) else 'N/A',
            'LjungBox_pvalue_lag10': float(lb_p) if not np.isnan(lb_p) else 'N/A',
            'optimal_params': optimal_params,
            'converged': True
        }
        
        if session_logger:
            session_logger.log_model_completion(
                'Linear Regression Trend',
                optimal_params,
                final_metrics,
                execution_time,
                converged=True
            )
        
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(df_lr['date'], df_lr[col], label='Original', linewidth=2)
        ax.plot(df_lr['date'], df_lr['Trend'], label='Trend', color='red', linewidth=2)
        ax.set_title("Linear Regression Trend")
        ax.legend()

        fig_res, axes = plt.subplots(1,3, figsize=(15,4))
        axes[0].plot(df_lr['date'], resid)
        axes[0].set_title('Residuals over time')
        axes[1].hist(resid_clean, bins=30)
        axes[1].set_title('Residuals histogram')
        try:
            sm.qqplot(resid_clean, line='s', ax=axes[2])
            axes[2].set_title('QQ-plot')
        except Exception:
            axes[2].text(0.5,0.5,'QQ plot failed', ha='center')

        # Forecast horizon (24h)
        try:
            h = 24
            X_future = np.arange(len(df_lr), len(df_lr) + h).reshape(-1,1)
            fv = model.predict(X_future).astype(float).tolist()
            files, forecast_df = _save_forecast_and_plots(session_logger, 'Linear Regression Trend', df_lr, col, resid_clean, fv, fig, fig_res, horizon=h)
            final_metrics['exported_files'] = files
        except Exception:
            final_metrics['exported_files'] = {}

        return fig, fig_res, execution_time, final_metrics
    
    except Exception as e:
        execution_time = time.time() - start_time
        if session_logger:
            session_logger.log_model_error('Linear Regression Trend', str(e))
        raise

# -----------------------------
# 4. Simple Exponential Smoothing
# -----------------------------
def ses_model(df, col='pjme_mw', alpha=0.2, session_logger=None):
    start_time = time.time()
    
    try:
        df_ses = df.copy()
        
        initial_params = {'alpha': alpha, 'method': 'Simple Exponential Smoothing'}
        if session_logger:
            session_logger.log_model_initialization('Simple Exponential Smoothing', initial_params)
        
        ses = SimpleExpSmoothing(df_ses[col], initialization_method="estimated").fit(smoothing_level=alpha)
        if session_logger:
            try:
                session_logger.log_model_fit_details('Simple Exponential Smoothing', ses)
            except Exception:
                pass
        df_ses['SES'] = ses.fittedvalues
        
        execution_time = time.time() - start_time
        
        # Calculer les métriques
        mse = np.mean((df_ses[col].dropna() - df_ses['SES'].dropna()) ** 2)
        mae = np.mean(np.abs(df_ses[col].dropna() - df_ses['SES'].dropna()))
        
        optimal_params = {'alpha_optimized': float(alpha)}
        # Residuals and tests
        fitted = ses.fittedvalues
        actual = df_ses[col]
        resid = actual - fitted
        resid_clean = resid.dropna()
        mape = float(np.mean(np.abs(resid_clean / (actual.loc[resid_clean.index].replace(0, np.nan)))) * 100) if len(resid_clean)>0 else np.nan
        try:
            jb_stat, jb_p = stats.jarque_bera(resid_clean)
        except Exception:
            jb_stat, jb_p = np.nan, np.nan
        try:
            lb = acorr_ljungbox(resid_clean, lags=[10], return_df=True)
            lb_p = float(lb['lb_pvalue'].iloc[-1])
        except Exception:
            lb_p = np.nan

        final_metrics = {
            'MSE': float(mse),
            'MAE': float(mae),
            'MAPE': float(mape) if not np.isnan(mape) else 'N/A',
            'AIC': float(ses.aic) if hasattr(ses, 'aic') else 'N/A',
            'BIC': float(ses.bic) if hasattr(ses, 'bic') else 'N/A',
            'JB_stat': float(jb_stat) if not np.isnan(jb_stat) else 'N/A',
            'JB_pvalue': float(jb_p) if not np.isnan(jb_p) else 'N/A',
            'LjungBox_pvalue_lag10': float(lb_p) if not np.isnan(lb_p) else 'N/A'
        }

        # Determine convergence from fit object when possible
        converged_ses = True
        try:
            if hasattr(ses, 'mle_retvals') and isinstance(getattr(ses, 'mle_retvals'), dict):
                converged_ses = bool(ses.mle_retvals.get('converged', True))
            elif hasattr(ses, 'converged'):
                converged_ses = bool(getattr(ses, 'converged'))
        except Exception:
            converged_ses = True

        final_metrics['optimal_params'] = optimal_params
        final_metrics['converged'] = converged_ses

        if session_logger:
            session_logger.log_model_completion(
                'Simple Exponential Smoothing',
                optimal_params,
                final_metrics,
                execution_time,
                converged=converged_ses
            )
        
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(df_ses['date'], df_ses[col], label='Original', linewidth=2)
        ax.plot(df_ses['date'], df_ses['SES'], label='SES', color='green', linewidth=2, alpha=0.7)
        ax.set_title(f"Simple Exponential Smoothing (α={alpha})")
        ax.legend()

        fig_res, axes = plt.subplots(1,3, figsize=(15,4))
        axes[0].plot(df_ses['date'], resid)
        axes[0].set_title('Residuals over time')
        axes[1].hist(resid_clean, bins=30)
        axes[1].set_title('Residuals histogram')
        try:
            sm.qqplot(resid_clean, line='s', ax=axes[2])
            axes[2].set_title('QQ-plot')
        except Exception:
            axes[2].text(0.5,0.5,'QQ plot failed', ha='center')

        # Forecast horizon (24h)
        try:
            h = 24
            fv = None
            try:
                fv = ses.forecast(steps=h) if hasattr(ses, 'forecast') else ses.predict(start=len(df_ses), end=len(df_ses)+h-1)
                fv = np.asarray(fv).astype(float).tolist()
            except Exception:
                fv = [float(df_ses[col].dropna().mean())] * h

            files, forecast_df = _save_forecast_and_plots(session_logger, 'Simple Exponential Smoothing', df_ses, col, resid_clean, fv, fig, fig_res, horizon=h)
            final_metrics['exported_files'] = files
        except Exception:
            final_metrics['exported_files'] = {}

        return fig, fig_res, execution_time, final_metrics
    
    except Exception as e:
        execution_time = time.time() - start_time
        if session_logger:
            session_logger.log_model_error('Simple Exponential Smoothing', str(e))
        raise

# -----------------------------
# 5. Holt (linear trend)
# -----------------------------
def holt_model(df, col='pjme_mw', alpha=0.2, beta=0.1, session_logger=None):
    start_time = time.time()
    
    try:
        df_holt = df.copy()
        
        initial_params = {'alpha': alpha, 'beta': beta, 'method': 'Holt Linear Trend'}
        if session_logger:
            session_logger.log_model_initialization('Holt Linear Trend', initial_params)
        
        holt = ExponentialSmoothing(df_holt[col], trend='add', seasonal=None, initialization_method="estimated").fit(smoothing_level=alpha, smoothing_slope=beta)
        if session_logger:
            try:
                session_logger.log_model_fit_details('Holt Linear Trend', holt)
            except Exception:
                pass
        df_holt['Holt'] = holt.fittedvalues
        
        execution_time = time.time() - start_time
        
        # Calculer les métriques
        mse = np.mean((df_holt[col].dropna() - df_holt['Holt'].dropna()) ** 2)
        mae = np.mean(np.abs(df_holt[col].dropna() - df_holt['Holt'].dropna()))
        
        optimal_params = {'alpha': float(alpha), 'beta': float(beta)}
        # Residuals and tests
        fitted = holt.fittedvalues
        actual = df_holt[col]
        resid = actual - fitted
        resid_clean = resid.dropna()
        mape = float(np.mean(np.abs(resid_clean / (actual.loc[resid_clean.index].replace(0, np.nan)))) * 100) if len(resid_clean)>0 else np.nan
        try:
            jb_stat, jb_p = stats.jarque_bera(resid_clean)
        except Exception:
            jb_stat, jb_p = np.nan, np.nan
        try:
            lb = acorr_ljungbox(resid_clean, lags=[10], return_df=True)
            lb_p = float(lb['lb_pvalue'].iloc[-1])
        except Exception:
            lb_p = np.nan

        final_metrics = {
            'MSE': float(mse),
            'MAE': float(mae),
            'MAPE': float(mape) if not np.isnan(mape) else 'N/A',
            'AIC': float(holt.aic) if hasattr(holt, 'aic') else 'N/A',
            'BIC': float(holt.bic) if hasattr(holt, 'bic') else 'N/A',
            'JB_stat': float(jb_stat) if not np.isnan(jb_stat) else 'N/A',
            'JB_pvalue': float(jb_p) if not np.isnan(jb_p) else 'N/A',
            'LjungBox_pvalue_lag10': float(lb_p) if not np.isnan(lb_p) else 'N/A'
        }

        # Determine convergence from fit object when possible
        converged_holt = True
        try:
            if hasattr(holt, 'mle_retvals') and isinstance(getattr(holt, 'mle_retvals'), dict):
                converged_holt = bool(holt.mle_retvals.get('converged', True))
            elif hasattr(holt, 'converged'):
                converged_holt = bool(getattr(holt, 'converged'))
        except Exception:
            converged_holt = True

        final_metrics['optimal_params'] = optimal_params
        final_metrics['converged'] = converged_holt

        if session_logger:
            session_logger.log_model_completion(
                'Holt Linear Trend',
                optimal_params,
                final_metrics,
                execution_time,
                converged=converged_holt
            )
        
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(df_holt['date'], df_holt[col], label='Original', linewidth=2)
        ax.plot(df_holt['date'], df_holt['Holt'], label='Holt', color='orange', linewidth=2, alpha=0.7)
        ax.set_title("Holt Linear Trend")
        ax.legend()

        fig_res, axes = plt.subplots(1,3, figsize=(15,4))
        axes[0].plot(df_holt['date'], resid)
        axes[0].set_title('Residuals over time')
        axes[1].hist(resid_clean, bins=30)
        axes[1].set_title('Residuals histogram')
        try:
            sm.qqplot(resid_clean, line='s', ax=axes[2])
            axes[2].set_title('QQ-plot')
        except Exception:
            axes[2].text(0.5,0.5,'QQ plot failed', ha='center')

        # Forecast horizon (24h)
        try:
            h = 24
            try:
                fv = holt.forecast(steps=h) if hasattr(holt, 'forecast') else holt.predict(start=len(df_holt), end=len(df_holt)+h-1)
                fv = np.asarray(fv).astype(float).tolist()
            except Exception:
                fv = [float(df_holt[col].dropna().mean())] * h

            files, forecast_df = _save_forecast_and_plots(session_logger, 'Holt Linear Trend', df_holt, col, resid_clean, fv, fig, fig_res, horizon=h)
            final_metrics['exported_files'] = files
        except Exception:
            final_metrics['exported_files'] = {}

        return fig, fig_res, execution_time, final_metrics
    
    except Exception as e:
        execution_time = time.time() - start_time
        if session_logger:
            session_logger.log_model_error('Holt Linear Trend', str(e))
        raise

# -----------------------------
# 6. Holt-Winters (additive)
# -----------------------------
def holt_winters_additive(df, col='pjme_mw', seasonal_periods=24, session_logger=None):
    start_time = time.time()
    
    try:
        df_hw = df.copy()
        
        initial_params = {'seasonal_periods': seasonal_periods, 'model': 'additive', 'column': col}
        if session_logger:
            session_logger.log_model_initialization('Holt-Winters Additive', initial_params)
        
        hw_add = ExponentialSmoothing(df_hw[col], trend='add', seasonal='add', seasonal_periods=seasonal_periods, initialization_method="estimated").fit()
        if session_logger:
            try:
                session_logger.log_model_fit_details('Holt-Winters Additive', hw_add)
            except Exception:
                pass
        df_hw['HW_Add'] = hw_add.fittedvalues
        
        execution_time = time.time() - start_time
        
        # Calculer les métriques
        mse = np.mean((df_hw[col].dropna() - df_hw['HW_Add'].dropna()) ** 2)
        mae = np.mean(np.abs(df_hw[col].dropna() - df_hw['HW_Add'].dropna()))
        
        optimal_params = {
            'alpha': float(hw_add.params['smoothing_level']),
            'beta': float(hw_add.params['smoothing_trend']),
            'gamma': float(hw_add.params['smoothing_seasonal'])
        }
        
        # Residuals and tests
        fitted = hw_add.fittedvalues
        actual = df_hw[col]
        resid = actual - fitted
        resid_clean = resid.dropna()
        mape = float(np.mean(np.abs(resid_clean / (actual.loc[resid_clean.index].replace(0, np.nan)))) * 100) if len(resid_clean)>0 else np.nan
        try:
            jb_stat, jb_p = stats.jarque_bera(resid_clean)
        except Exception:
            jb_stat, jb_p = np.nan, np.nan
        try:
            lb = acorr_ljungbox(resid_clean, lags=[10], return_df=True)
            lb_p = float(lb['lb_pvalue'].iloc[-1])
        except Exception:
            lb_p = np.nan

        final_metrics = {
            'MSE': float(mse),
            'MAE': float(mae),
            'MAPE': float(mape) if not np.isnan(mape) else 'N/A',
            'AIC': float(hw_add.aic) if hasattr(hw_add, 'aic') else 'N/A',
            'BIC': float(hw_add.bic) if hasattr(hw_add, 'bic') else 'N/A',
            'JB_stat': float(jb_stat) if not np.isnan(jb_stat) else 'N/A',
            'JB_pvalue': float(jb_p) if not np.isnan(jb_p) else 'N/A',
            'LjungBox_pvalue_lag10': float(lb_p) if not np.isnan(lb_p) else 'N/A'
        }
        # Determine convergence from fit object when possible
        converged_hw_add = True
        try:
            if hasattr(hw_add, 'mle_retvals') and isinstance(getattr(hw_add, 'mle_retvals'), dict):
                converged_hw_add = bool(hw_add.mle_retvals.get('converged', True))
            elif hasattr(hw_add, 'converged'):
                converged_hw_add = bool(getattr(hw_add, 'converged'))
        except Exception:
            converged_hw_add = True

        final_metrics['optimal_params'] = optimal_params
        final_metrics['converged'] = converged_hw_add

        if session_logger:
            session_logger.log_model_completion(
                'Holt-Winters Additive',
                optimal_params,
                final_metrics,
                execution_time,
                converged=converged_hw_add
            )
        
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(df_hw['date'], df_hw[col], label='Original', linewidth=2)
        ax.plot(df_hw['date'], df_hw['HW_Add'], label='HW Additive', color='purple', linewidth=2, alpha=0.7)
        ax.set_title("Holt-Winters Additive")
        ax.legend()

        fig_res, axes = plt.subplots(1,3, figsize=(15,4))
        axes[0].plot(df_hw['date'], resid)
        axes[0].set_title('Residuals over time')
        axes[1].hist(resid_clean, bins=30)
        axes[1].set_title('Residuals histogram')
        try:
            sm.qqplot(resid_clean, line='s', ax=axes[2])
            axes[2].set_title('QQ-plot')
        except Exception:
            axes[2].text(0.5,0.5,'QQ plot failed', ha='center')

        # Forecast horizon (24h)
        try:
            h = 24
            try:
                fv = hw_add.forecast(steps=h) if hasattr(hw_add, 'forecast') else hw_add.predict(start=len(df_hw), end=len(df_hw)+h-1)
                fv = np.asarray(fv).astype(float).tolist()
            except Exception:
                fv = [float(df_hw[col].dropna().mean())] * h

            files, forecast_df = _save_forecast_and_plots(session_logger, 'Holt-Winters Additive', df_hw, col, resid_clean, fv, fig, fig_res, horizon=h)
            final_metrics['exported_files'] = files
        except Exception:
            final_metrics['exported_files'] = {}

        return fig, fig_res, execution_time, final_metrics
    
    except Exception as e:
        execution_time = time.time() - start_time
        if session_logger:
            session_logger.log_model_error('Holt-Winters Additive', str(e))
        raise

# -----------------------------
# 7. Holt-Winters (multiplicative)
# -----------------------------
def holt_winters_multiplicative(df, col='pjme_mw', seasonal_periods=24, session_logger=None):
    start_time = time.time()
    
    try:
        df_hw = df.copy()
        
        initial_params = {'seasonal_periods': seasonal_periods, 'model': 'multiplicative', 'column': col}
        if session_logger:
            session_logger.log_model_initialization('Holt-Winters Multiplicative', initial_params)
        
        hw_mul = ExponentialSmoothing(df_hw[col], trend='add', seasonal='mul', seasonal_periods=seasonal_periods, initialization_method="estimated").fit()
        if session_logger:
            try:
                session_logger.log_model_fit_details('Holt-Winters Multiplicative', hw_mul)
            except Exception:
                pass
        df_hw['HW_Mul'] = hw_mul.fittedvalues
        
        execution_time = time.time() - start_time
        
        # Calculer les métriques
        mse = np.mean((df_hw[col].dropna() - df_hw['HW_Mul'].dropna()) ** 2)
        mae = np.mean(np.abs(df_hw[col].dropna() - df_hw['HW_Mul'].dropna()))
        
        optimal_params = {
            'alpha': float(hw_mul.params['smoothing_level']),
            'beta': float(hw_mul.params['smoothing_trend']),
            'gamma': float(hw_mul.params['smoothing_seasonal'])
        }
        
        # Residuals and tests
        fitted = hw_mul.fittedvalues
        actual = df_hw[col]
        resid = actual - fitted
        resid_clean = resid.dropna()
        mape = float(np.mean(np.abs(resid_clean / (actual.loc[resid_clean.index].replace(0, np.nan)))) * 100) if len(resid_clean)>0 else np.nan
        try:
            jb_stat, jb_p = stats.jarque_bera(resid_clean)
        except Exception:
            jb_stat, jb_p = np.nan, np.nan
        try:
            lb = acorr_ljungbox(resid_clean, lags=[10], return_df=True)
            lb_p = float(lb['lb_pvalue'].iloc[-1])
        except Exception:
            lb_p = np.nan

        final_metrics = {
            'MSE': float(mse),
            'MAE': float(mae),
            'MAPE': float(mape) if not np.isnan(mape) else 'N/A',
            'AIC': float(hw_mul.aic) if hasattr(hw_mul, 'aic') else 'N/A',
            'BIC': float(hw_mul.bic) if hasattr(hw_mul, 'bic') else 'N/A',
            'JB_stat': float(jb_stat) if not np.isnan(jb_stat) else 'N/A',
            'JB_pvalue': float(jb_p) if not np.isnan(jb_p) else 'N/A',
            'LjungBox_pvalue_lag10': float(lb_p) if not np.isnan(lb_p) else 'N/A'
        }
        # Determine convergence from fit object when possible
        converged_hw_mul = True
        try:
            if hasattr(hw_mul, 'mle_retvals') and isinstance(getattr(hw_mul, 'mle_retvals'), dict):
                converged_hw_mul = bool(hw_mul.mle_retvals.get('converged', True))
            elif hasattr(hw_mul, 'converged'):
                converged_hw_mul = bool(getattr(hw_mul, 'converged'))
        except Exception:
            converged_hw_mul = True

        final_metrics['optimal_params'] = optimal_params
        final_metrics['converged'] = converged_hw_mul

        if session_logger:
            session_logger.log_model_completion(
                'Holt-Winters Multiplicative',
                optimal_params,
                final_metrics,
                execution_time,
                converged=converged_hw_mul
            )
        
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(df_hw['date'], df_hw[col], label='Original', linewidth=2)
        ax.plot(df_hw['date'], df_hw['HW_Mul'], label='HW Multiplicative', color='brown', linewidth=2, alpha=0.7)
        ax.set_title("Holt-Winters Multiplicative")
        ax.legend()

        fig_res, axes = plt.subplots(1,3, figsize=(15,4))
        axes[0].plot(df_hw['date'], resid)
        axes[0].set_title('Residuals over time')
        axes[1].hist(resid_clean, bins=30)
        axes[1].set_title('Residuals histogram')
        try:
            sm.qqplot(resid_clean, line='s', ax=axes[2])
            axes[2].set_title('QQ-plot')
        except Exception:
            axes[2].text(0.5,0.5,'QQ plot failed', ha='center')

        # Forecast horizon (24h)
        try:
            h = 24
            try:
                fv = hw_mul.forecast(steps=h) if hasattr(hw_mul, 'forecast') else hw_mul.predict(start=len(df_hw), end=len(df_hw)+h-1)
                fv = np.asarray(fv).astype(float).tolist()
            except Exception:
                fv = [float(df_hw[col].dropna().mean())] * h

            files, forecast_df = _save_forecast_and_plots(session_logger, 'Holt-Winters Multiplicative', df_hw, col, resid_clean, fv, fig, fig_res, horizon=h)
            final_metrics['exported_files'] = files
        except Exception:
            final_metrics['exported_files'] = {}

        return fig, fig_res, execution_time, final_metrics
    
    except Exception as e:
        execution_time = time.time() - start_time
        if session_logger:
            session_logger.log_model_error('Holt-Winters Multiplicative', str(e))
        raise
