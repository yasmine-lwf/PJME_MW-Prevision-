import logging
import os
from datetime import datetime
import uuid
import json

# Configuration des paramètres de l'application
APP_CONFIG = {
    "app_name": "Time Series Forecasting App",
    "version": "1.0.0",
    "preprocessing_method": "IQR-based outlier removal with time interpolation",
    "models": [
        "Simple Moving Average",
        "Weighted Moving Average",
        "Linear Regression Trend",
        "Simple Exponential Smoothing",
        "Holt Linear Trend",
        "Holt-Winters Additive",
        "Holt-Winters Multiplicative"
    ],
    "target_column": "pjme_mw",
    "stationarity_test": "Augmented Dickey-Fuller (ADF)"
}

class SessionLogger:
    """Gestionnaire de journalisation pour le suivi et l'audit du processus de modélisation"""
    
    def __init__(self):
        self.session_id = str(uuid.uuid4())[:8]
        self.execution_datetime = datetime.now()
        self.log_dir = "logs"
        self.log_file = None
        self.json_log_file = None
        self.json_events = []
        self.logger = None
        self.timeseries_info = {}
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure le système de journalisation"""
        # Créer le répertoire logs s'il n'existe pas
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # Nom du fichier log avec timestamp
        timestamp = self.execution_datetime.strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"session_{self.session_id}_{timestamp}.log")
        # JSON structured log file for integration
        self.json_log_file = os.path.join(self.log_dir, f"session_{self.session_id}_{timestamp}.json")
        
        # Configuration du logger
        self.logger = logging.getLogger(f"SessionLogger_{self.session_id}")
        self.logger.setLevel(logging.DEBUG)
        
        # Handler pour fichier
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Format détaillé
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        # initialize json events list file (will be flushed on important events)
        try:
            # ensure an empty JSON file exists
            with open(self.json_log_file, 'w', encoding='utf-8') as jf:
                json.dump([], jf, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def log_json_event(self, event_type, payload):
        """Append a structured JSON event with timestamp and write to session JSON file."""
        try:
            event = {
                'timestamp': datetime.now().isoformat(),
                'event': event_type,
                'payload': payload
            }
            self.json_events.append(event)
            # flush to disk (overwrite) for integration consumers
            with open(self.json_log_file, 'w', encoding='utf-8') as jf:
                json.dump(self.json_events, jf, ensure_ascii=False, indent=2, default=str)
        except Exception:
            # fallback: log a warning to text log
            try:
                self.logger.warning(f"Could not write JSON event {event_type}")
            except Exception:
                pass

    
    def log_process_header(self):
        """Enregistre l'en-tête du processus de modélisation"""
        header = f"""
{'='*80}
PROCESSUS DE MODÉLISATION - EN-TÊTE
{'='*80}
Date et heure d'exécution: {self.execution_datetime.strftime('%Y-%m-%d %H:%M:%S')}
Identifiant de session: {self.session_id}
{'='*80}

CONFIGURATION DE L'APPLICATION:
- Nom: {APP_CONFIG['app_name']}
- Version: {APP_CONFIG['version']}
- Méthode de prétraitement: {APP_CONFIG['preprocessing_method']}
- Test de stationnarité: {APP_CONFIG['stationarity_test']}
- Colonne cible: {APP_CONFIG['target_column']}

MODÈLES DISPONIBLES:
"""
        for i, model in enumerate(APP_CONFIG['models'], 1):
            header += f"  {i}. {model}\n"
        
        header += f"{'='*80}\n"
        self.logger.info(header)
        # structured JSON event for header
        try:
            header_payload = {
                'execution_datetime': self.execution_datetime.isoformat(),
                'session_id': self.session_id,
                'app_config': APP_CONFIG
            }
            self.log_json_event('process_header', header_payload)
        except Exception:
            pass
    
    def set_timeseries_info(self, df, filename):
        """Enregistre les informations complètes sur la série temporelle analysée"""
        missing_per_col = df.isnull().sum().to_dict()
        total_cells = len(df) * len(df.columns)
        total_missing = df.isnull().sum().sum()
        
        self.timeseries_info = {
            "filename": filename,
            "rows": len(df),
            "columns": list(df.columns),
            "date_range": f"{df['date'].min()} to {df['date'].max()}" if 'date' in df.columns else "N/A",
            "missing_values_total": int(total_missing),
            "missing_values_by_column": missing_per_col,
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
        
        # Construire le message détaillé
        info_msg = f"""
{'='*80}
IMPORTATION ET ANALYSE INITIALE DES DONNÉES
{'='*80}

INFORMATIONS GÉNÉRALES:
- Nom du fichier: {self.timeseries_info['filename']}
- Nombre d'observations (lignes): {self.timeseries_info['rows']}
- Nombre de variables (colonnes): {len(self.timeseries_info['columns'])}
- Plage de dates: {self.timeseries_info['date_range']}

VARIABLES DISPONIBLES ET LEURS TYPES:
"""
        for col, dtype in self.timeseries_info['data_types'].items():
            missing_count = missing_per_col.get(col, 0)
            missing_pct = (missing_count / self.timeseries_info['rows'] * 100) if self.timeseries_info['rows'] > 0 else 0
            info_msg += f"  • {col}: {dtype} (Manquantes: {missing_count} / {missing_pct:.2f}%)\n"
        
        completeness = ((total_cells - total_missing) / total_cells * 100) if total_cells > 0 else 0
        info_msg += f"""
RÉSUMÉ DES VALEURS MANQUANTES:
- Total valeurs manquantes: {self.timeseries_info['missing_values_total']}
- Taux de complétude: {completeness:.2f}%
- Total de cellules de données: {total_cells}

{'='*80}
"""
        self.logger.info(info_msg)
    
    def log_preprocessing_step(self, step_name, details):
        """Enregistre une étape du prétraitement"""
        msg = f"\nPRÉTRAITEMENT - {step_name}:\n{details}"
        self.logger.info(msg)
    
    def log_missing_values_handling(self, method, column_details):
        """Log la gestion des valeurs manquantes par colonne"""
        msg = f"""
{'='*80}
TRAITEMENT DES VALEURS MANQUANTES
{'='*80}
Méthode appliquée: {method}

Détails par colonne:
"""
        for col, info in column_details.items():
            msg += f"\n  • {col}:\n"
            msg += f"    - Valeurs manquantes avant: {info['before']} ({info['before_pct']:.2f}%)\n"
            msg += f"    - Valeurs manquantes après: {info['after']} ({info['after_pct']:.2f}%)\n"
            msg += f"    - Valeurs interpolées: {info['interpolated']}\n"
        
        msg += f"\n{'='*80}\n"
        self.logger.info(msg)
    
    def log_outliers_removal(self, method, outlier_details):
        """Log la suppression des valeurs aberrantes"""
        msg = f"""
{'='*80}
TRAITEMENT DES VALEURS ABERRANTES (OUTLIERS)
{'='*80}
Méthode appliquée: {method}

Détails par colonne:
"""
        for col, info in outlier_details.items():
            msg += f"\n  • {col}:\n"
            msg += f"    - Q1 (25e percentile): {info['Q1']:.4f}\n"
            msg += f"    - Q3 (75e percentile): {info['Q3']:.4f}\n"
            msg += f"    - IQR (Écart interquartile): {info['IQR']:.4f}\n"
            msg += f"    - Bornes: [{info['lower_bound']:.4f}, {info['upper_bound']:.4f}]\n"
            msg += f"    - Outliers détectés et supprimés: {info['outliers_removed']}\n"
            msg += f"    - Taux de suppression: {info['removal_rate']:.2f}%\n"
        
        msg += f"\n{'='*80}\n"
        self.logger.info(msg)
    
    def log_transformation(self, transformation_name, details):
        """Log une transformation (normalisation, différenciation, etc.)"""
        msg = f"""
{'='*80}
TRANSFORMATION - {transformation_name.upper()}
{'='*80}
{details}
{'='*80}
"""
        self.logger.info(msg)
    
    def log_preprocessing_summary(self, summary):
        """Log un résumé complet du prétraitement"""
        msg = f"""
{'='*80}
RÉSUMÉ COMPLET DU PRÉTRAITEMENT
{'='*80}
{summary}
{'='*80}
"""
        self.logger.info(msg)
    
    def log_model_execution(self, model_name, parameters):
        """Enregistre l'exécution d'un modèle"""
        msg = f"\nMODÈLE - {model_name}:\nParamètres: {json.dumps(parameters, ensure_ascii=False, indent=2)}"
        self.logger.info(msg)
    
    def log_test_result(self, test_name, results):
        """Enregistre le résultat d'un test statistique"""
        msg = f"\nTEST STATISTIQUE - {test_name}:\nRésultats: {json.dumps(results, ensure_ascii=False, indent=2)}"
        self.logger.info(msg)
    
    def log_descriptive_statistics(self, stats_dict):
        """Log les statistiques descriptives (moyenne, variance, skewness, kurtosis)"""
        msg = f"""
{'='*80}
STATISTIQUES DESCRIPTIVES - ANALYSE EXPLORATOIRE
{'='*80}

"""
        for col, stats in stats_dict.items():
            msg += f"\n  📊 COLONNE: {col}\n"
            msg += f"    - Nombre d'observations: {stats.get('count', 'N/A')}\n"
            msg += f"    - Moyenne: {stats.get('mean', 'N/A'):.4f}\n"
            msg += f"    - Médiane: {stats.get('median', 'N/A'):.4f}\n"
            msg += f"    - Écart-type: {stats.get('std', 'N/A'):.4f}\n"
            msg += f"    - Variance: {stats.get('variance', 'N/A'):.4f}\n"
            msg += f"    - Minimum: {stats.get('min', 'N/A'):.4f}\n"
            msg += f"    - Maximum: {stats.get('max', 'N/A'):.4f}\n"
            msg += f"    - Quartile 25%: {stats.get('q25', 'N/A'):.4f}\n"
            msg += f"    - Quartile 50%: {stats.get('q50', 'N/A'):.4f}\n"
            msg += f"    - Quartile 75%: {stats.get('q75', 'N/A'):.4f}\n"
            msg += f"    - Skewness (asymétrie): {stats.get('skewness', 'N/A'):.4f}\n"
            msg += f"    - Kurtosis (aplatissement): {stats.get('kurtosis', 'N/A'):.4f}\n"
        
        msg += f"\n{'='*80}\n"
        self.logger.info(msg)
    
    def log_stationarity_tests(self, tests_results):
        """Log les résultats des tests de stationnarité (ADF, KPSS)"""
        msg = f"""
{'='*80}
TESTS DE STATIONNARITÉ - ANALYSE EXPLORATOIRE
{'='*80}

"""
        for test_name, col_results in tests_results.items():
            msg += f"\n  🔍 TEST: {test_name}\n"
            for col, result in col_results.items():
                msg += f"\n    📈 Colonne: {col}\n"
                msg += f"      - Statistique de test: {result.get('statistic', 'N/A'):.6f}\n"
                msg += f"      - p-value: {result.get('pvalue', 'N/A'):.6f}\n"
                msg += f"      - Nombre de lags: {result.get('nlags', 'N/A')}\n"
                msg += f"      - Nombre d'observations: {result.get('nobs', 'N/A')}\n"
                
                if 'critical_values' in result:
                    msg += f"      - Valeurs critiques:\n"
                    for level, value in result['critical_values'].items():
                        msg += f"        • {level}: {value:.6f}\n"
                
                # Interprétation
                pval = result.get('pvalue', 1)
                if test_name == 'ADF':
                    interpretation = "✅ Série STATIONNAIRE (rejeter H0)" if pval < 0.05 else "❌ Série NON-STATIONNAIRE (ne pas rejeter H0)"
                else:  # KPSS
                    interpretation = "❌ Série NON-STATIONNAIRE (rejeter H0)" if pval < 0.05 else "✅ Série STATIONNAIRE (ne pas rejeter H0)"
                
                msg += f"      - Interprétation: {interpretation}\n"
        
        msg += f"\n{'='*80}\n"
        self.logger.info(msg)
    
    def log_seasonality_analysis(self, seasonality_results):
        """Log l'analyse de saisonnalité"""
        msg = f"""
{'='*80}
ANALYSE DE SAISONNALITÉ
{'='*80}

"""
        for col, result in seasonality_results.items():
            msg += f"\n  🌊 COLONNE: {col}\n"
            msg += f"    - Saisonnalité détectée: {'✅ OUI' if result.get('has_seasonality') else '❌ NON'}\n"
            
            if result.get('has_seasonality'):
                msg += f"    - Période de saisonnalité: {result.get('period', 'N/A')} observations\n"
                msg += f"    - Force de saisonnalité: {result.get('strength', 'N/A'):.4f}\n"
                msg += f"    - Trend force: {result.get('trend_strength', 'N/A'):.4f}\n"
            
            msg += f"    - ACF maximal lag: {result.get('acf_max_lag', 'N/A')}\n"
            msg += f"    - PACF significatif: {result.get('pacf_significant', 'N/A')}\n"
        
        msg += f"\n{'='*80}\n"
        self.logger.info(msg)
    
    def log_trend_analysis(self, trend_results):
        """Log l'analyse de tendance"""
        msg = f"""
{'='*80}
ANALYSE DE TENDANCE
{'='*80}

"""
        for col, result in trend_results.items():
            msg += f"\n  📈 COLONNE: {col}\n"
            msg += f"    - Type de tendance: {result.get('trend_type', 'N/A')}\n"
            msg += f"    - Direction: {result.get('direction', 'N/A')}\n"
            msg += f"    - Pente (slope): {result.get('slope', 'N/A'):.6f}\n"
            msg += f"    - Ordonnée à l'origine (intercept): {result.get('intercept', 'N/A'):.6f}\n"
            msg += f"    - Coefficient de détermination (R²): {result.get('r_squared', 'N/A'):.6f}\n"
            msg += f"    - p-value: {result.get('pvalue', 'N/A'):.6f}\n"
            msg += f"    - Tendance significative: {'✅ OUI' if result.get('significant') else '❌ NON'}\n"
            msg += f"    - Valeur min: {result.get('min_value', 'N/A'):.4f}\n"
            msg += f"    - Valeur max: {result.get('max_value', 'N/A'):.4f}\n"
            msg += f"    - Changement total: {result.get('total_change', 'N/A'):.4f}\n"
        
        msg += f"\n{'='*80}\n"
        self.logger.info(msg)
    
    def log_eda_summary(self, summary):
        """Log un résumé de l'analyse exploratoire"""
        msg = f"""
{'='*80}
RÉSUMÉ DE L'ANALYSE EXPLORATOIRE DES DONNÉES (EDA)
{'='*80}
{summary}
{'='*80}
"""
        self.logger.info(msg)
    
    def log_model_initialization(self, model_name, initial_params):
        """Log l'initialisation d'un modèle avec ses paramètres initiaux"""
        msg = f"""
{'='*80}
INITIALISATION DU MODÈLE: {model_name.upper()}
{'='*80}
Date/Heure: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Paramètres initiaux:
"""
        for param, value in initial_params.items():
            msg += f"  • {param}: {value}\n"
        
        msg += f"\n{'='*80}\n"
        self.logger.info(msg)
    
    def log_model_optimization(self, model_name, iteration, current_params, metrics):
        """Log une itération d'optimisation"""
        msg = f"""
{'='*80}
OPTIMISATION - {model_name.upper()} - Itération {iteration}
{'='*80}

Paramètres actuels:
"""
        for param, value in current_params.items():
            msg += f"  • {param}: {value}\n"
        
        msg += f"\nMétriques de performance:\n"
        for metric, value in metrics.items():
            msg += f"  • {metric}: {value}\n"
        
        msg += f"\n{'='*80}\n"
        self.logger.info(msg)

    def log_model_fit_details(self, model_name, fit_obj):
        """Tentatively extract and log optimizer / fit details from a fitted model object."""
        details = {}
        try:
            if hasattr(fit_obj, 'params'):
                try:
                    details['params'] = {k: float(v) for k, v in getattr(fit_obj, 'params').items()} if isinstance(getattr(fit_obj, 'params'), dict) else getattr(fit_obj, 'params').tolist() if hasattr(getattr(fit_obj, 'params'), 'tolist') else str(getattr(fit_obj, 'params'))
                except Exception:
                    details['params'] = str(getattr(fit_obj, 'params'))

            # mle_retvals often contains optimization metadata for some statsmodels objects
            if hasattr(fit_obj, 'mle_retvals'):
                try:
                    mle = getattr(fit_obj, 'mle_retvals')
                    details['mle_retvals'] = mle if isinstance(mle, dict) else str(mle)
                except Exception:
                    details['mle_retvals'] = str(getattr(fit_obj, 'mle_retvals'))

            # common convergence flag
            if hasattr(fit_obj, 'converged'):
                details['converged'] = bool(getattr(fit_obj, 'converged'))

            # Some objects expose fit history or optimizer messages
            for attr in ('fit_details', 'history', 'optimizer_history', 'fit_history'):
                if hasattr(fit_obj, attr):
                    try:
                        details[attr] = getattr(fit_obj, attr)
                    except Exception:
                        details[attr] = str(getattr(fit_obj, attr))

            # AIC / BIC if available
            if hasattr(fit_obj, 'aic'):
                details['aic'] = float(getattr(fit_obj, 'aic'))
            if hasattr(fit_obj, 'bic'):
                details['bic'] = float(getattr(fit_obj, 'bic'))

            msg = f"\n{'='*80}\nFIT DETAILS - {model_name.upper()}\n{'='*80}\n"
            for k, v in details.items():
                msg += f"  • {k}: {v}\n"
            msg += f"\n{'='*80}\n"
            self.logger.info(msg)
        except Exception as e:
            self.logger.info(f"FIT DETAILS - {model_name}: failed to extract details: {e}")
    
    def log_model_completion(self, model_name, optimal_params, final_metrics, execution_time, converged=True):
        """Log la complétion et résultats finaux d'un modèle"""
        status = "✅ CONVERGED" if converged else "⚠️ NON-CONVERGED"
        msg = f"""
{'='*80}
RÉSULTATS FINAUX - {model_name.upper()} - {status}
{'='*80}
Date/Heure: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Temps d'exécution: {execution_time:.4f} secondes

Paramètres optimaux retenus:
"""
        for param, value in optimal_params.items():
            msg += f"  • {param}: {value}\n"
        
        msg += f"\nMétriques finales:\n"
        for metric, value in final_metrics.items():
            msg += f"  • {metric}: {value}\n"
        
        msg += f"\n{'='*80}\n"
        self.logger.info(msg)
        try:
            payload = {
                'model_name': model_name,
                'optimal_params': optimal_params,
                'final_metrics': final_metrics,
                'execution_time': execution_time,
                'converged': converged
            }
            self.log_json_event('model_completion', payload)
        except Exception:
            pass
    
    def log_model_error(self, model_name, error_msg, error_type="ERROR"):
        """Log une erreur ou un avertissement lors de la modélisation"""
        msg = f"""
{'='*80}
{error_type} - {model_name.upper()}
{'='*80}
Date/Heure: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Message: {error_msg}
{'='*80}
"""
        if error_type == "ERROR":
            self.logger.error(msg)
        else:
            self.logger.warning(msg)
        try:
            self.log_json_event('model_error', {'model': model_name, 'error': error_msg, 'type': error_type})
        except Exception:
            pass
    
    def log_models_comparison(self, models_summary):
        """Log une comparaison de tous les modèles testés"""
        msg = f"""
{'='*80}
COMPARAISON ET RÉSUMÉ DE TOUS LES MODÈLES TESTÉS
{'='*80}
Date/Heure: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Modèles testés:
"""
        for model_data in models_summary:
            msg += f"\n  📊 {model_data['name']}:\n"
            msg += f"    - Status: {model_data.get('status', 'N/A')}\n"
            msg += f"    - Temps d'exécution: {model_data.get('execution_time', 0):.4f}s\n"
            msg += f"    - Paramètres optimaux: {model_data.get('optimal_params', {})}\n"
            msg += f"    - Performances: {model_data.get('metrics', {})}\n"
            if model_data.get('errors'):
                msg += f"    - Erreurs: {model_data.get('errors')}\n"
        
        msg += f"\n{'='*80}\n"
        self.logger.info(msg)

    def log_exported_files(self, files_dict):
        """Log paths of exported files (forecasts, plots, etc.)."""
        msg = f"\n{'='*80}\nEXPORTED FILES\n{'='*80}\n"
        for k, v in files_dict.items():
            msg += f"  • {k}: {v}\n"
        msg += f"\n{'='*80}\n"
        self.logger.info(msg)
        try:
            self.log_json_event('exported_files', files_dict)
        except Exception:
            pass

    def log_time_series_cv_results(self, model_name, cv_results):
        """Log rolling-origin or holdout CV results.

        cv_results expected format:
        {
            'folds': [ { 'train_end': str, 'mse': float, 'params': {...} , 'coverage': float }, ... ],
            'mean_mse': float,
            'std_mse': float,
            'params': {...}
        }
        """
        try:
            msg = f"\n{'='*80}\nTIME-SERIES CV RESULTS - {model_name.upper()}\n{'='*80}\n"
            msg += f"Params tried: {cv_results.get('params', {})}\n"
            msg += f"Mean MSE: {cv_results.get('mean_mse', 'N/A')}, Std MSE: {cv_results.get('std_mse', 'N/A')}\n"
            msg += "Folds:\n"
            for i, f in enumerate(cv_results.get('folds', []), 1):
                msg += f"  • Fold {i}: train_end={f.get('train_end')} mse={f.get('mse')} coverage={f.get('coverage', 'N/A')}\n"
            msg += f"\n{'='*80}\n"
            self.logger.info(msg)
        except Exception as e:
            self.logger.warning(f"Failed logging CV results for {model_name}: {e}")

    def log_grid_search_results(self, model_name, grid_summary):
        """Log summary of a grid search.

        grid_summary expected format:
        {
            'tested': [ { 'params': {...}, 'mean_mse': float, 'std_mse': float }, ... ],
            'best': { 'params': {...}, 'mean_mse': float }
        }
        """
        try:
            msg = f"\n{'='*80}\nGRID SEARCH - {model_name.upper()}\n{'='*80}\n"
            msg += f"Tested combinations: {len(grid_summary.get('tested', []))}\n"
            for item in grid_summary.get('tested', []):
                msg += f"  • params={item.get('params')} mean_mse={item.get('mean_mse')} std_mse={item.get('std_mse')}\n"
            best = grid_summary.get('best')
            if best:
                msg += f"\nBEST: params={best.get('params')} mean_mse={best.get('mean_mse')}\n"
            msg += f"\n{'='*80}\n"
            self.logger.info(msg)
        except Exception as e:
            self.logger.warning(f"Failed logging grid search for {model_name}: {e}")

    def log_model_selection(self, selection_summary):
        """Log automatic model selection summary.

        selection_summary expected format:
        { 'selected': 'ModelName', 'by': 'AIC'|'BIC'|'MSE', 'candidates': [...] }
        """
        try:
            msg = f"\n{'='*80}\nMODEL SELECTION SUMMARY\n{'='*80}\n"
            msg += f"Selected model: {selection_summary.get('selected')} by {selection_summary.get('by')}\n"
            msg += "Candidates:\n"
            for c in selection_summary.get('candidates', []):
                msg += f"  • {c.get('name')}: metrics={c.get('metrics')}\n"
            msg += f"\n{'='*80}\n"
            self.logger.info(msg)
        except Exception as e:
            self.logger.warning(f"Failed logging model selection: {e}")

    def log_interval_validation(self, model_name, interval_results):
        """Log validation of forecast intervals (coverage and width).

        interval_results expected format:
        { 'coverage': float, 'mean_width': float, 'horizon': int }
        """
        try:
            msg = f"\n{'='*80}\nINTERVAL VALIDATION - {model_name.upper()}\n{'='*80}\n"
            msg += f"Horizon: {interval_results.get('horizon')}\n"
            msg += f"Coverage (95%): {interval_results.get('coverage')}\n"
            msg += f"Mean interval width: {interval_results.get('mean_width')}\n"
            msg += f"\n{'='*80}\n"
            self.logger.info(msg)
        except Exception as e:
            self.logger.warning(f"Failed logging interval validation for {model_name}: {e}")
    
    def log_error(self, error_msg):
        """Enregistre une erreur"""
        self.logger.error(f"\nERREUR: {error_msg}")
    
    def log_warning(self, warning_msg):
        """Enregistre un avertissement"""
        self.logger.warning(f"\nAVERTISSEMENT: {warning_msg}")
    
    def get_summary(self):
        """Retourne un résumé de la session"""
        return {
            "session_id": self.session_id,
            "execution_datetime": self.execution_datetime.isoformat(),
            "log_file": self.log_file,
            "timeseries_info": self.timeseries_info
        }


def get_session_logger():
    """Fonction utilitaire pour obtenir ou créer le logger de session"""
    if 'session_logger' not in globals():
        globals()['session_logger'] = SessionLogger()
    return globals()['session_logger']
