import streamlit as st
import pandas as pd
import os
from backend import (
    load_and_clean,
    plot_series,
    simple_moving_average,
    linear_regression_trend,
    ses_model,
    holt_model,
    holt_winters_additive,
    holt_winters_multiplicative,
    compute_descriptive_statistics,
    test_stationarity,
    detect_seasonality,
    analyze_trend,
    time_series_train_test_split,
    rolling_origin_cv,
    grid_search,
    validate_forecast_intervals,
    select_best_model,
    bayesian_optimization,
    plot_train_test_split
)
from logger import SessionLogger

st.set_page_config(page_title="Time Series Forecasting App", layout="wide")

# Initialiser le logger de session
if 'session_logger' not in st.session_state:
    st.session_state.session_logger = SessionLogger()
    st.session_state.session_logger.log_process_header()

session_logger = st.session_state.session_logger

st.title("Time Series Forecasting App")

# Afficher l'identifiant de session dans la sidebar
with st.sidebar:
    st.write("---")
    st.subheader("📋 Informations de Session")
    st.write(f"**ID Session:** `{session_logger.session_id}`")
    st.write(f"**Date/Heure:** {session_logger.execution_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    if st.button("📥 Télécharger le journal", key="download_log"):
        with open(session_logger.log_file, 'r', encoding='utf-8') as log:
            st.download_button(
                label="Fichier journal (.log)",
                data=log.read(),
                file_name=f"session_{session_logger.session_id}.log",
                mime="text/plain",
                key=f"download_session_log_{session_logger.session_id}"
            )
    st.write("---")


# Step 1: Upload file
uploaded = st.file_uploader("Upload CSV file", type=['csv'])

if uploaded:

    st.success("File uploaded successfully!")
    df = pd.read_csv(uploaded)
    
    # Log timeseries information
    session_logger.set_timeseries_info(df, uploaded.name)

    # Normalize column names
    df.columns = [c.lower() for c in df.columns]

    if "date" not in df.columns:
        st.error("Your CSV must contain a 'date' column.")
        st.stop()

    st.subheader("Raw Data")
    st.write(df.head())
    
    # Afficher les informations d'importation dans des colonnes
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Nombre d'observations", len(df))
    with col2:
        st.metric("Nombre de variables", len(df.columns))
    with col3:
        missing_count = df.isnull().sum().sum()
        st.metric("Valeurs manquantes", missing_count)
    
    # Afficher les types de données
    st.write("**Types de données:**")
    st.write(df.dtypes)

    # Step 2: Preprocessing
    if st.button("Run Preprocessing"):
        with st.spinner("⏳ Prétraitement en cours..."):
            df_clean = load_and_clean(df, session_logger)
            st.session_state['df_clean'] = df_clean
        
        # Afficher un résumé du prétraitement
        st.success("✅ Prétraitement complété!")
        
        preprocessing_summary = st.expander("📊 Détails du prétraitement", expanded=False)
        with preprocessing_summary:
            st.info(f"""
            **Résumé du prétraitement:**
            - Observations finales: {len(df_clean)}
            - Colonnes: {list(df_clean.columns)}
            - Valeurs manquantes résiduelles: {int(df_clean.isnull().sum().sum())}
            - Plage temporelle: {df_clean['date'].min()} à {df_clean['date'].max()}
            """)
            
            if st.button("📥 Voir le journal complet", key="view_detailed_log"):
                with open(session_logger.log_file, 'r', encoding='utf-8') as log:
                    st.text(log.read())

    # Step 3: EDA (only if preprocessing done)
    if "df_clean" in st.session_state:

        df_clean = st.session_state['df_clean']

        st.subheader("Cleaned Data")
        st.write(df_clean.head())

        st.subheader("Time Series Plot")
        st.pyplot(plot_series(df_clean))

        # ACF/PACF plots removed (not required for PJME_hourly view)
        # ADF test moved into the Stationnarité tab inside EDA
        
        # ============ SECTION ANALYSE EXPLORATOIRE (EDA) ============
        st.subheader("📊 Analyse Exploratoire des Données (EDA)")
        
        # Onglets pour l'EDA
        tab1, tab2, tab3, tab4 = st.tabs(["Statistiques Descriptives", "Stationnarité", "Saisonnalité", "Tendance"])
        
        with tab1:
            st.write("**Statistiques descriptives (moyenne, variance, skewness, kurtosis)**")
            if st.button("Calculer les statistiques", key="compute_stats"):
                with st.spinner("Calcul en cours..."):
                    stats_results = compute_descriptive_statistics(df_clean, session_logger)
                    st.session_state['stats_results'] = stats_results
            
            if 'stats_results' in st.session_state:
                for col, stats in st.session_state['stats_results'].items():
                    with st.expander(f"📈 {col}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Moyenne", f"{stats['mean']:.4f}")
                            st.metric("Variance", f"{stats['variance']:.4f}")
                            st.metric("Min", f"{stats['min']:.4f}")
                            st.metric("Médiane", f"{stats['median']:.4f}")
                        with col2:
                            st.metric("Écart-type", f"{stats['std']:.4f}")
                            st.metric("Skewness", f"{stats['skewness']:.4f}")
                            st.metric("Kurtosis", f"{stats['kurtosis']:.4f}")
                            st.metric("Max", f"{stats['max']:.4f}")
        
        with tab2:
            st.write("**Tests de stationnarité (ADF, KPSS)**")
            if st.button("Effectuer les tests", key="stationarity_tests"):
                with st.spinner("Tests en cours..."):
                    stationarity_results = test_stationarity(df_clean, session_logger)
                    st.session_state['stationarity_results'] = stationarity_results
            
            if 'stationarity_results' in st.session_state:
                for test_name, col_results in st.session_state['stationarity_results'].items():
                    st.write(f"### {test_name}")
                    for col, result in col_results.items():
                        with st.expander(f"📈 {col}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Statistique", f"{result.get('statistic', 'N/A'):.6f}")
                                st.metric("p-value", f"{result.get('pvalue', 'N/A'):.6f}")
                            with col2:
                                pval = result.get('pvalue', 1)
                                if test_name == 'ADF':
                                    interpretation = "✅ STATIONNAIRE" if pval < 0.05 else "❌ NON-STATIONNAIRE"
                                else:  # KPSS
                                    interpretation = "❌ NON-STATIONNAIRE" if pval < 0.05 else "✅ STATIONNAIRE"
                                st.info(interpretation)
        
        with tab3:
            st.write("**Détection de la saisonnalité**")
            if st.button("Détecter la saisonnalité", key="seasonality_detection"):
                with st.spinner("Analyse en cours..."):
                    seasonality_results = detect_seasonality(df_clean, session_logger=session_logger)
                    st.session_state['seasonality_results'] = seasonality_results
            
            if 'seasonality_results' in st.session_state:
                for col, result in st.session_state['seasonality_results'].items():
                    with st.expander(f"🌊 {col}"):
                        if 'error' not in result:
                            st.metric("Saisonnalité détectée", "✅ OUI" if result.get('has_seasonality') else "❌ NON")
                            if result.get('has_seasonality'):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Période", result.get('period', 'N/A'))
                                with col2:
                                    st.metric("Force", f"{result.get('strength', 0):.4f}")
        
        with tab4:
            st.write("**Analyse de la tendance**")
            if st.button("Analyser la tendance", key="trend_analysis"):
                with st.spinner("Analyse en cours..."):
                    trend_results = analyze_trend(df_clean, session_logger=session_logger)
                    st.session_state['trend_results'] = trend_results
            
            if 'trend_results' in st.session_state:
                for col, result in st.session_state['trend_results'].items():
                    with st.expander(f"📈 {col}"):
                        if 'error' not in result:
                            st.metric("Type de tendance", result.get('trend_type', 'N/A'))
                            st.metric("Direction", result.get('direction', 'N/A'))
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Pente", f"{result.get('slope', 0):.6f}")
                                st.metric("R²", f"{result.get('r_squared', 0):.6f}")
                            with col2:
                                st.metric("p-value", f"{result.get('pvalue', 0):.6f}")
                                st.metric("Significative", "✅ OUI" if result.get('significant') else "❌ NON")

st.subheader("📈 Journal et Comparaison des Modèles")

# ============ VALIDATION & OPTIMISATION ============
with st.expander("🔧 Validation et Optimisation", expanded=False):
    st.write("Choisissez la stratégie de validation et lancez une recherche de paramètres (grid / bayes).")
    cv_strategy = st.radio("Stratégie CV", ['Holdout 70/30', 'Holdout 80/20', 'Rolling-origin'], index=2)
    run_grid = st.checkbox("Activer Grid Search", value=True)
    run_bayes = st.checkbox("Activer Bayesian Optimization (optionnel)", value=False)
    pref_metric = st.selectbox("Critère de sélection du meilleur modèle", ['AIC', 'BIC', 'MSE'], index=2)

    # CV params
    if cv_strategy.startswith('Holdout'):
        train_ratio = 0.7 if '70' in cv_strategy else 0.8
        st.write(f"Holdout train ratio: {train_ratio}")
        cv_params = {'train_ratio': train_ratio}
    else:
        init_ratio = st.slider("Initial train ratio for rolling-origin", 0.3, 0.9, 0.5)
        horizon = st.number_input("Horizon (hours) for each fold", min_value=1, max_value=168, value=24)
        step = st.number_input("Step between origins (observations)", min_value=1, max_value=168, value=24)
        cv_params = {'initial_train_size': None, 'horizon': int(horizon), 'step': int(step)}
        # derive absolute initial_train_size later from ratio

    # Automatically show train/test split plot when data is available
    if 'df_clean' in st.session_state:
        try:
            dfc = st.session_state['df_clean']
            plot_ratio = train_ratio if cv_strategy.startswith('Holdout') else init_ratio
            fig_split = plot_train_test_split(dfc, col='pjme_mw', train_ratio=float(plot_ratio))
            st.subheader("Train/Test Split (chronological)")
            st.pyplot(fig_split)
        except Exception as e:
            try:
                session_logger.log_warning(f"Could not plot train/test split: {e}")
            except Exception:
                pass

    st.write("Paramètres de recherche: spécifiez des grilles par modèle ou laissez par défaut.")

    # Default small grids
    grids = {
        'SMA': {'window': [24, 48, 168]},
        'SES': {'alpha': [0.1, 0.2, 0.3]},
        'Holt': {'alpha': [0.1, 0.2], 'beta': [0.05, 0.1]},
        'HW Additive': {'seasonal_periods': [24]},
        'HW Multiplicative': {'seasonal_periods': [24]}
    }

    if st.button("▶️ Run Validation & Model Selection"):
        if 'df_clean' not in st.session_state:
            st.error("Run preprocessing first.")
        else:
            dfc = st.session_state['df_clean']
            models_to_test = ['SMA', 'Linear Regression', 'SES', 'Holt', 'HW Additive', 'HW Multiplicative']
            models_summary = []
            for m in models_to_test:
                st.info(f"Running validation for {m}...")
                if m == 'SMA':
                    mtype = 'sma'
                    param_grid = grids.get('SMA') if run_grid else {'window': [24]}
                elif m == 'Linear Regression':
                    mtype = 'lr'
                    param_grid = {}
                elif m == 'SES':
                    mtype = 'ses'
                    param_grid = grids.get('SES') if run_grid else {'alpha': [0.2]}
                elif m == 'Holt':
                    mtype = 'holt'
                    param_grid = grids.get('Holt') if run_grid else {'alpha': [0.2], 'beta': [0.1]}
                elif m == 'HW Additive':
                    mtype = 'holt_hw_add'
                    param_grid = grids.get('HW Additive') if run_grid else {'seasonal_periods': [24]}
                else:
                    mtype = 'holt_hw_mul'
                    param_grid = grids.get('HW Multiplicative') if run_grid else {'seasonal_periods': [24]}

                # Run grid search
                best_info = None
                if run_grid and param_grid:
                    try:
                        gs = grid_search(mtype, param_grid, dfc, col='pjme_mw', cv_method='rolling' if cv_strategy=='Rolling-origin' else 'holdout', cv_params=(cv_params if cv_strategy=='Rolling-origin' else {'train_ratio': 0.8}), session_logger=session_logger)
                        best_info = gs.get('best')
                    except Exception as e:
                        session_logger.log_model_error(m, str(e))
                elif run_bayes:
                    # assemble a small search_space for bayes
                    try:
                        # transform param_grid to search_space format (low,high,type)
                        search_space = {}
                        for k, vals in param_grid.items():
                            if isinstance(vals[0], int):
                                search_space[k] = (min(vals), max(vals), 'int')
                            else:
                                search_space[k] = (min(vals), max(vals), 'real')
                        bo = bayesian_optimization(mtype, search_space, dfc, col='pjme_mw', n_calls=20, cv_params=cv_params, session_logger=session_logger)
                        if bo:
                            best_info = {'params': bo.get('best_params'), 'mean_mse': bo.get('fun')}
                    except Exception as e:
                        session_logger.log_warning(f"Bayes opt failed for {m}: {e}")

                # If no search ran, run a single CV with default params
                if best_info is None:
                    # choose a representative param set
                    default_params = {k: v[0] for k, v in param_grid.items()} if param_grid else {}
                    res = rolling_origin_cv(mtype, dfc, col='pjme_mw', params=default_params, session_logger=session_logger, **(cv_params if cv_strategy=='Rolling-origin' else {'initial_train_size': None, 'horizon': 24, 'step': 24}))
                    best_info = {'params': default_params, 'mean_mse': res.get('mean_mse')}

                metrics = {'MSE': best_info.get('mean_mse')}
                models_summary.append({'name': m, 'metrics': metrics, 'optimal_params': best_info.get('params', {})})

            # select best
            selection = select_best_model(models_summary, prefer=pref_metric)
            try:
                session_logger.log_model_selection(selection)
            except Exception:
                pass

            st.success(f"Selected model: {selection.get('selected')} by {selection.get('by')}")
            st.json(selection)

# Initialiser le stockage des résultats de modèles
if 'models_results' not in st.session_state:
    st.session_state.models_results = {}

# ============ SECTION MODÈLES ============
st.subheader("🤖 Modèles de Prévision")

# Expander pour chaque catégorie
with st.expander("📊 Modèles Classiques", expanded=True):
    col1, col2, col3 = st.columns(3)
    
    # 1. Simple Moving Average
    with col1:
        if st.button("🔄 Simple Moving Average", key="btn_sma"):
            with st.spinner("Exécution du modèle SMA..."):
                try:
                    fig, fig_res, exec_time, metrics = simple_moving_average(
                        st.session_state['df_clean'], 
                        window=7,
                        session_logger=session_logger
                    )
                    st.session_state.models_results['SMA'] = {
                        'execution_time': exec_time,
                        'metrics': metrics,
                        'status': 'success'
                    }
                    st.pyplot(fig)
                    with st.expander("📉 Résidus"):
                        st.pyplot(fig_res)
                    st.success(f"✅ Exécuté en {exec_time:.4f}s")
                    with st.expander("📋 Détails du modèle"):
                        st.json(metrics)
                        exported = metrics.get('exported_files', {}) if isinstance(metrics, dict) else {}
                        if exported:
                            with st.expander("📁 Fichiers exportés", expanded=False):
                                csvp = exported.get('forecast_csv')
                                if csvp:
                                    with open(csvp, 'rb') as f:
                                        st.download_button(
                                            "Télécharger prévisions (CSV)",
                                            f,
                                            file_name=os.path.basename(csvp),
                                            mime='text/csv',
                                            key=f"download_{session_logger.session_id}_sma_csv_{os.path.basename(csvp)}"
                                        )
                                figp = exported.get('figure_png')
                                if figp:
                                    st.image(figp, caption='Graphique du modèle')
                                    with open(figp, 'rb') as f:
                                        st.download_button(
                                            "Télécharger graphique",
                                            f,
                                            file_name=os.path.basename(figp),
                                            mime='image/png',
                                            key=f"download_{session_logger.session_id}_sma_fig_{os.path.basename(figp)}"
                                        )
                                resp = exported.get('residuals_png')
                                if resp:
                                    st.image(resp, caption='Graphique des résidus')
                                    with open(resp, 'rb') as f:
                                        st.download_button(
                                            "Télécharger résidus",
                                            f,
                                            file_name=os.path.basename(resp),
                                            mime='image/png',
                                            key=f"download_{session_logger.session_id}_sma_res_{os.path.basename(resp)}"
                                        )
                        # exported files already rendered above
                except Exception as e:
                    session_logger.log_model_error('SMA', str(e))
                    st.error(f"❌ Erreur: {str(e)}")
                    st.session_state.models_results['SMA'] = {
                        'status': 'error',
                        'error': str(e)
                    }
    
    # 2. Linear Regression Trend
    with col3:
        if st.button("📐 Linear Regression Trend", key="btn_lr"):
            with st.spinner("Exécution du modèle LR..."):
                try:
                    fig, fig_res, exec_time, metrics = linear_regression_trend(
                        st.session_state['df_clean'],
                        session_logger=session_logger
                    )
                    st.session_state.models_results['Linear Regression'] = {
                        'execution_time': exec_time,
                        'metrics': metrics,
                        'status': 'success'
                    }
                    st.pyplot(fig)
                    with st.expander("📉 Résidus"):
                        st.pyplot(fig_res)
                    st.success(f"✅ Exécuté en {exec_time:.4f}s")
                    with st.expander("📋 Détails du modèle"):
                        st.json(metrics)
                except Exception as e:
                    session_logger.log_model_error('Linear Regression', str(e))
                    st.error(f"❌ Erreur: {str(e)}")
                    st.session_state.models_results['Linear Regression'] = {
                        'status': 'error',
                        'error': str(e)
                    }

with st.expander("🎯 Modèles de Lissage", expanded=True):
    col1, col2 = st.columns(2)
    
    # 4. Simple Exponential Smoothing
    with col1:
        if st.button("📊 Simple Exponential Smoothing", key="btn_ses"):
            with st.spinner("Exécution du modèle SES..."):
                try:
                    fig, fig_res, exec_time, metrics = ses_model(
                        st.session_state['df_clean'], 
                        alpha=0.2,
                        session_logger=session_logger
                    )
                    st.session_state.models_results['SES'] = {
                        'execution_time': exec_time,
                        'metrics': metrics,
                        'status': 'success'
                    }
                    st.pyplot(fig)
                    with st.expander("📉 Résidus"):
                        st.pyplot(fig_res)
                    st.success(f"✅ Exécuté en {exec_time:.4f}s")
                    with st.expander("📋 Détails du modèle"):
                        st.json(metrics)
                except Exception as e:
                    session_logger.log_model_error('SES', str(e))
                    st.error(f"❌ Erreur: {str(e)}")
                    st.session_state.models_results['SES'] = {
                        'status': 'error',
                        'error': str(e)
                    }
    
    # 5. Holt Linear Trend
    with col2:
        if st.button("📈 Holt Linear Trend", key="btn_holt"):
            with st.spinner("Exécution du modèle Holt..."):
                try:
                    fig, fig_res, exec_time, metrics = holt_model(
                        st.session_state['df_clean'], 
                        alpha=0.2, 
                        beta=0.1,
                        session_logger=session_logger
                    )
                    st.session_state.models_results['Holt'] = {
                        'execution_time': exec_time,
                        'metrics': metrics,
                        'status': 'success'
                    }
                    st.pyplot(fig)
                    with st.expander("📉 Résidus"):
                        st.pyplot(fig_res)
                    st.success(f"✅ Exécuté en {exec_time:.4f}s")
                    with st.expander("📋 Détails du modèle"):
                        st.json(metrics)
                except Exception as e:
                    session_logger.log_model_error('Holt', str(e))
                    st.error(f"❌ Erreur: {str(e)}")
                    st.session_state.models_results['Holt'] = {
                        'status': 'error',
                        'error': str(e)
                    }

with st.expander("🌊 Modèles Holt-Winters", expanded=True):
    col1, col2 = st.columns(2)
    
    # 6. Holt-Winters Additive
    with col1:
        if st.button("➕ Holt-Winters Additive", key="btn_hw_add"):
            with st.spinner("Exécution du modèle HW Additive..."):
                try:
                    fig, fig_res, exec_time, metrics = holt_winters_additive(
                        st.session_state['df_clean'], 
                        seasonal_periods=12,
                        session_logger=session_logger
                    )
                    st.session_state.models_results['HW Additive'] = {
                        'execution_time': exec_time,
                        'metrics': metrics,
                        'status': 'success'
                    }
                    st.pyplot(fig)
                    with st.expander("📉 Résidus"):
                        st.pyplot(fig_res)
                    st.success(f"✅ Exécuté en {exec_time:.4f}s")
                    with st.expander("📋 Détails du modèle"):
                        st.json(metrics)
                except Exception as e:
                    session_logger.log_model_error('HW Additive', str(e))
                    st.error(f"❌ Erreur: {str(e)}")
                    st.session_state.models_results['HW Additive'] = {
                        'status': 'error',
                        'error': str(e)
                    }
    
    # 7. Holt-Winters Multiplicative
    with col2:
        if st.button("✖️ Holt-Winters Multiplicative", key="btn_hw_mul"):
            with st.spinner("Exécution du modèle HW Multiplicative..."):
                try:
                    fig, fig_res, exec_time, metrics = holt_winters_multiplicative(
                        st.session_state['df_clean'], 
                        seasonal_periods=12,
                        session_logger=session_logger
                    )
                    st.session_state.models_results['HW Multiplicative'] = {
                        'execution_time': exec_time,
                        'metrics': metrics,
                        'status': 'success'
                    }
                    st.pyplot(fig)
                    with st.expander("📉 Résidus"):
                        st.pyplot(fig_res)
                    st.success(f"✅ Exécuté en {exec_time:.4f}s")
                    with st.expander("📋 Détails du modèle"):
                        st.json(metrics)
                except Exception as e:
                    session_logger.log_model_error('HW Multiplicative', str(e))
                    st.error(f"❌ Erreur: {str(e)}")
                    st.session_state.models_results['HW Multiplicative'] = {
                        'status': 'error',
                        'error': str(e)
                    }

# Résumé et comparaison des modèles
if st.session_state.models_results:
    st.subheader("📊 Résumé des Modèles")
    
    # Créer un tableau de comparaison détaillé
    comparison_data = []
    models_summary = []
    for model_name, result in st.session_state.models_results.items():
        if result.get('status') == 'success':
            metrics = result.get('metrics', {})
            aic = metrics.get('AIC', 'N/A')
            bic = metrics.get('BIC', 'N/A')
            mse = metrics.get('MSE', 'N/A')
            mae = metrics.get('MAE', 'N/A')
            mape = metrics.get('MAPE', 'N/A')

            comparison_data.append({
                'Modèle': model_name,
                'Temps (s)': float(result.get('execution_time', 0)),
                'AIC': aic if isinstance(aic, (int, float)) else 'N/A',
                'BIC': bic if isinstance(bic, (int, float)) else 'N/A',
                'MSE': float(mse) if isinstance(mse, (int, float)) else 'N/A',
                'MAE': float(mae) if isinstance(mae, (int, float)) else 'N/A',
                'MAPE': float(mape) if isinstance(mape, (int, float)) else 'N/A'
            })

            models_summary.append({
                'name': model_name,
                'status': 'success',
                'execution_time': float(result.get('execution_time', 0)),
                'optimal_params': metrics.get('optimal_params', {}),
                'metrics': metrics,
                'errors': None
            })
        else:
            comparison_data.append({
                'Modèle': model_name,
                'Status': '❌ Erreur',
                'Message': result.get('error', 'Unknown error')
            })
            models_summary.append({
                'name': model_name,
                'status': 'error',
                'execution_time': result.get('execution_time', 0),
                'optimal_params': {},
                'metrics': {},
                'errors': result.get('error')
            })

    if comparison_data:
        df_comp = pd.DataFrame(comparison_data)
        # Ranking: prefer AIC if numeric, else MSE
        def rank_key(row):
            aic = row.get('AIC')
            if isinstance(aic, (int, float)):
                return (0, aic)
            mse = row.get('MSE')
            if isinstance(mse, (int, float)):
                return (1, mse)
            return (2, float('inf'))

        ranked = df_comp.copy()
        ranked['rank_key'] = ranked.apply(rank_key, axis=1)
        ranked = ranked.sort_values('rank_key')
        ranked = ranked.drop(columns=['rank_key'])

        st.write("**Tableau comparatif des performances**")
        st.dataframe(ranked, use_container_width=True)

        # Log comparison summary
        try:
            session_logger.log_models_comparison(models_summary)
        except Exception:
            pass

        # Bouton pour voir le journal complet
        if st.button("📥 Télécharger le journal de modélisation complet"):
            with open(session_logger.log_file, 'r', encoding='utf-8') as log:
                st.download_button(
                    label="Fichier journal complet (.log)",
                    data=log.read(),
                    file_name=f"session_{session_logger.session_id}.log",
                    mime="text/plain",
                    key="download_modeling_log"
                )
