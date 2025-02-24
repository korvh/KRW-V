# Import libraries
import streamlit as st
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn. neighbors import KNeighborsRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn_quantile import RandomForestQuantileRegressor, ExtraTreesQuantileRegressor, KNeighborsQuantileRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, OrdinalEncoder, LabelBinarizer, RobustScaler, Normalizer, QuantileTransformer
from category_encoders.woe import WOEEncoder
from category_encoders.cat_boost import CatBoostEncoder
from category_encoders.target_encoder import TargetEncoder
from category_encoders.glmm import GLMMEncoder
from category_encoders.james_stein import JamesSteinEncoder
from category_encoders.target_encoder import TargetEncoder
from category_encoders.wrapper import PolynomialWrapper
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, KBinsDiscretizer
from sklearn.impute import SimpleImputer, KNNImputer
from imblearn import FunctionSampler
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as PipelineImb
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
import shap
from mapie.regression import MapieQuantileRegressor
from mapie.metrics import coverage_width_based, regression_coverage_score, regression_mean_width_score, regression_mwi_score
import MWIS_metric
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import hiplot as hip
import numpy as np
import copy
import warnings
from PIL import Image
import xgboost as xgb
from causallearn.search.FCMBased.lingam import DirectLiNGAM
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.GraphUtils import GraphUtils
import networkx as nx
import io
import os
from openpyxl import load_workbook
from packaging import version
from constrained_linear_regression import ConstrainedLinearRegression
from sklearn.utils.validation import check_is_fitted


# Import custom functions from utils
from krw_utils import plot_causal_graph, initialize_graph_for_background_knowledge, read_excel_with_multiple_sheets
from krw_utils import YeoJohnsonTargetTransformer, LogTargetTransformer, Log1pTransformerWithShift
from krw_utils import drop_columns_by_name, safe_convert_to_str, signed_log1p_transform, inv_signed_log1p_transform, monotonic_constraints
from krw_utils import get_regression_metrics, custom_permutation_importance
from krw_utils import shap_dependence_plots_with_target_as_legend, plot_model_performance


# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*Regressors in active set degenerate.*")
warnings.filterwarnings("ignore", message="X has feature names, but.*")
warnings.filterwarnings("ignore", message="WARNING: The predictions are ill-sorted.*")

# Set XGBoost verbosity to suppress warnings
xgb.set_config(verbosity=0)  # 0 = Silent, 1 = Warning, 2 = Info, 3 = Debug
# Set the page layout
st.set_page_config(layout="wide", page_title="KRW-Verkenner Model Trainer", page_icon=":bar_chart:")

# Set Graphviz path manually
os.environ["PATH"] += os.pathsep + "/home/adminuser/venv/bin/"

# Define paths
path_data = os.getcwd()
path_data = os.path.join(path_data, 'data')

files_data = {
    'Geaggregeerd':'1-s2.0-S0043135421010459-mmc3_clean.xlsx',
    'Gedetailleerd':'Volledige_dataset_versie4_MetLengte_clean.xlsx'
}
file_metadata = 'krw-v_omschrijving_stuurvariabelen.xlsx'
file_logo = 'deltares_logo.png'


# Initialize session state
if 'fit_model' not in st.session_state:
    st.session_state.fit_model = False
if 'render_figures' not in st.session_state:
    st.session_state.render_figures = False
if 'dataset_selected' not in st.session_state:
    st.session_state.dataset_selected = None
if 'new_dataset' not in st.session_state:
    st.session_state.new_dataset = True
if 'new_settings' not in st.session_state:
    st.session_state.new_settings = True
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
# App settings
if 'cluster_name' not in st.session_state:
    st.session_state.cluster_name = None
if 'target_name' not in st.session_state:
    st.session_state.target_name = None
if 'data_type' not in st.session_state:
    st.session_state.data_type = 'Geaggregeerd'
if 'model_name' not in st.session_state:
    st.session_state.model_name = None
if 'conformal_interval' not in st.session_state:
    st.session_state.conformal_interval = None
if 'train_size' not in st.session_state:
    st.session_state.train_size = None
if 'data_split' not in st.session_state:
    st.session_state.data_split = 'Train'
if 'seed_value' not in st.session_state:
    st.session_state.seed_value = None
if 'show_hiplot' not in st.session_state:
    st.session_state.show_hiplot = False
if 'shap_explanation' not in st.session_state:
    st.session_state.shap_explanation = None
# Tables
if 'table_metadata' not in st.session_state:
    st.session_state.table_metadata = None
if 'table_metrics' not in st.session_state:
    st.session_state.table_metrics = None
# Figures
if 'fig_performance' not in st.session_state:
    st.session_state.fig_performance = None
if 'fig_importance' not in st.session_state:
    st.session_state.fig_importance = None
if 'fig_waterfall' not in st.session_state:
    st.session_state.fig_waterfall = None
if 'fig_beeswarm' not in st.session_state:
    st.session_state.fig_beeswarm = None
if 'fig_avg_shap' not in st.session_state:
    st.session_state.fig_avg_shap = None
if 'fig_dependence' not in st.session_state:
    st.session_state.fig_dependence = None
if 'fig_intervals' not in st.session_state:
    st.session_state.fig_intervals = None
if 'fig_causal' not in st.session_state:
    st.session_state.fig_causal = None
# Feature selection
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = []
if 'start_select_features' not in st.session_state:
    st.session_state.start_select_features = True

# General settings
dpi = 150
label_fontsize = 11
tick_fontsize = 9
fig_width = 10
top_n = 12

# Initial slider/listbox settings
conformal_interval_init = 80
train_size_init = 70
data_type_init = 'Geaggregeerd'
data_split_init = 'Train'
seed_value_init = 0

# Model settings
ordinal_columns = ['Meandering','Beschaduwing','Verstuwing','Onderhoud','Peilbeheer','Oeverinrichting',] #'Toxiciteit']
# columns_not_in_model = ['Stikstof totaal']
columns_not_in_model = []
categorical_columns = ['Profielvorm','Vispasseerbaarheid','Ruimtelijke variatie stroomsnelheid']
ohe_drop_first = True
categorical_encode_method = 'CatBoost'
# categorical_encode_method = 'OHE'
# categorical_encode_method = 'GLMM'
# categorical_encode_method = 'Target'

use_categorical_encoding = True
use_ordinal_encoding = False
use_numeric_transform = False
pca_transform_numeric = False
use_target_transform = False
imputation_method = 'drop'
set_monotonic_constraints = False

numeric_transform_method = 'log'
# numeric_transform_method = 'signed_log'
# numeric_transform_method = 'yeojohnson'
target_transform_method = 'log'
# target_transform_method = 'yeojohnson'
target_yeojohnson_lambda = -2.
target_yeojohnson_nan_percentile = 99.5
scaler_quantile_nquantiles = 20
scaler_rowwise_norm = 'l2'
features_monotone_increasing = [None]
features_monotone_decreasing = ['Toxiciteit','Stikstof totaal']
# monotone_constraints_method_lgbm = 'basic'
# monotone_constraints_method_lgbm = 'intermediate'
monotone_constraints_method_lgbm = 'advanced'
continuous_to_ordinal_transform_nbins = {}

# split_method = 'random'
split_method = 'stratified'

scaler_type = 'standard'
# scaler_type = 'minmax'
# scaler_type = 'none'

model_types = {
    'Random Forest':'RF',
    'LGBM - Gradient Boosting':'LGBM',
    # 'XGB - Gradient Boosting':'XGB',
    'Extra Trees':'XT',
    'K-nearest Neighbors':'KNN'
    # 'Linear Regression':'LR',
}

models = {
    "RF": {'model':RandomForestRegressor(n_estimators=30, max_depth=10, min_samples_leaf=3), 'scaler':scaler_type, 'name':'Random Forest Regressor'},
    "LGBM": {'model':LGBMRegressor(objective='regression', max_depth=10, verbose=-1), 'scaler':scaler_type, 'name':'Gradient Boosting Regressor'},	
    "XGB": {'model':XGBRegressor(objective='reg:squarederror', max_depth=10, min_child_weight=20), 'scaler':scaler_type, 'name':'XGBoost Regressor'},
    "XT": {'model':ExtraTreesRegressor(n_estimators=30, min_samples_leaf=3), 'scaler':scaler_type, 'name':'Extra Trees Regressor'},
    "KNN": {'model':KNeighborsRegressor(), 'scaler':'standard', 'name':'K-nearest Neighbors Regressor'},
    "LR": {'model':LinearRegression(), 'scaler':'standard', 'name':'Linear Regression'},
}

cqr_models = {
    "RF": {'model':RandomForestQuantileRegressor(n_estimators=30, max_depth=10, min_samples_leaf=3, q=0.5), 'scaler':scaler_type, 'name':'Random Forest Quantile Regressor'},
    "LGBM": {'model':LGBMRegressor(objective='quantile', max_depth=10,  alpha=0.5, verbose=-1), 'scaler':scaler_type, 'name':'Gradient Boosting Quantile Regressor'},
    "XGB": {'model':XGBRegressor(objective='reg:quantileerror', max_depth=10, min_child_weight=20, quantile_alpha=0.5), 'scaler':scaler_type, 'name':'XGBoost Quantile Regressor'},
    "XT": {'model':ExtraTreesQuantileRegressor(n_estimators=30, min_samples_leaf=3, q=0.5), 'scaler':scaler_type, 'name':'Extra Trees Quantile Regressor'},
    "KNN": {'model':KNeighborsQuantileRegressor(q=0.5), 'scaler':'standard', 'name':'K-nearest Neighbors Quantile Regressor'},
    "LR": {'model':QuantileRegressor(alpha=1., quantile=0.5), 'scaler':'standard', 'name':'Linear Quantile Regressor'},
}


# Causal graph settings
threshold_for_adjmatrix = 1e-5
# threshold_for_adjmatrix = 1e-12


# Build regression model
def regression_model_pipeline():
    global rgr_mdl

    # Define lists to hold preprocessing steps
    num_preprocessor, cat_preprocessor, discrete2ord_preprocessor, continuous2ord_preprocessor, drop_col_preprocessor = [], [], [], [], []

    # Feature type lists
    drop_var_names = columns_not_in_model
    discrete2ord_var_names = [col for col in Xtrain_columns if col in discrete_to_ordinal and col not in drop_var_names] if use_ordinal_encoding else []
    continuous2ord_var_names = [col for col in Xtrain_columns if col in continuous_to_ordinal and col not in drop_var_names] if use_ordinal_encoding else []
    cat_var_names = [col for col in Xtrain_columns if col in categorical_columns and col not in drop_var_names] if use_categorical_encoding else []
    num_var_names = [col for col in Xtrain_columns if col not in cat_var_names and col not in discrete2ord_var_names and col not in continuous2ord_var_names and col not in drop_var_names]

    # Get column indexes of feature types
    num_vars_idx = [Xtrain_columns.index(col) for col in num_var_names if col in Xtrain_columns]
    cat_vars_idx = [Xtrain_columns.index(col) for col in cat_var_names if col in Xtrain_columns]
    discrete2ord_vars_idx = [Xtrain_columns.index(col) for col in discrete2ord_var_names if col in Xtrain_columns]
    continuous2ord_vars_idx = [Xtrain_columns.index(col) for col in continuous2ord_var_names if col in Xtrain_columns]
    drop_vars_idx = [Xtrain_columns.index(col) for col in drop_var_names if col in Xtrain_columns]

    # Drop features
    drop_col_preprocessor.append(('column_dropper', FunctionTransformer(drop_columns_by_name, kw_args={'columns_to_drop': drop_var_names}, validate=False)))

    # Impute missing values
    if imputation_method in ['median', 'mean']:
        num_preprocessor.append(('imputer', SimpleImputer(strategy=imputation_method)))
        if use_ordinal_encoding:
            if discrete2ord_vars_idx:
                discrete2ord_preprocessor.append(('imputer', SimpleImputer(strategy=imputation_method)))
            if continuous2ord_vars_idx:
                continuous2ord_preprocessor.append(('imputer', SimpleImputer(strategy=imputation_method)))
    elif imputation_method == 'knn':
        num_preprocessor.append(('imputer', KNNImputer()))
        if use_ordinal_encoding:
            if discrete2ord_vars_idx:
                discrete2ord_preprocessor.append(('imputer', KNNImputer()))
            if continuous2ord_vars_idx:
                continuous2ord_preprocessor.append(('imputer', KNNImputer()))
    if imputation_method != 'drop':
        if use_categorical_encoding and cat_vars_idx:
            if categorical_encode_method in ['WOE', 'CatBoost']:
                cat_preprocessor.append(('astype_str', FunctionTransformer(safe_convert_to_str, validate=False)))  # WOE will transform all string columns
            cat_preprocessor.append(('imputer', SimpleImputer(strategy='most_frequent')))

    # Transform numeric feature values
    if use_numeric_transform:
        if numeric_transform_method == 'log':
            num_preprocessor.append(('transformer', Log1pTransformerWithShift()))
        elif numeric_transform_method == 'signed_log':
            num_preprocessor.append(('transformer', FunctionTransformer(func=signed_log1p_transform, inverse_func=inv_signed_log1p_transform, validate=False)))
        elif numeric_transform_method == 'yeojohnson':
            num_preprocessor.append(('transformer', PowerTransformer(method='yeo-johnson')))
        else:
            raise ValueError(f"Unsupported numeric_transform_method for numeric features: {numeric_transform_method}")

    # Onehot encode categorical features
    if use_categorical_encoding and cat_vars_idx:
        if categorical_encode_method == 'OHE':
            if ohe_drop_first:
                drop_first = 'first'
            else:
                drop_first = None
            if version.parse(sklearn.__version__) >= version.parse('1.2'):
                cat_preprocessor.append(('onehot_encoder', OneHotEncoder(handle_unknown='ignore', drop=drop_first, sparse_output=False)))
            else:
                cat_preprocessor.append(('onehot_encoder', OneHotEncoder(handle_unknown='ignore', drop=drop_first, sparse=False)))
        elif categorical_encode_method == 'target':
            cat_preprocessor.append(('target_encoder', TargetEncoder(verbose=0, handle_unknown='value', handle_missing='value', return_df=False, drop_invariant=False)))
        elif categorical_encode_method == 'GLMM':
            cat_preprocessor.append(('glmm_encoder', GLMMEncoder(verbose=0, handle_unknown='value', handle_missing='value', return_df=False, drop_invariant=False)))
        elif categorical_encode_method == 'WOE':
            cat_preprocessor.append(('woe_encoder', WOEEncoder(verbose=0, handle_unknown='value', handle_missing='value', return_df=False, drop_invariant=False)))
        elif categorical_encode_method == 'CatBoost':
            cat_preprocessor.append(('catboost_encoder', CatBoostEncoder(verbose=0, handle_unknown='value', handle_missing='value', return_df=False, drop_invariant=False)))
        elif categorical_encode_method == 'JamesStein':
            cat_preprocessor.append(('jamesstein_encoder', JamesSteinEncoder(verbose=0, handle_unknown='value', handle_missing='value', return_df=False, drop_invariant=False)))
        elif categorical_encode_method == 'Target':
            cat_preprocessor.append(('target_encoder', TargetEncoder(verbose=0, handle_unknown='value', handle_missing='value', return_df=False, drop_invariant=False)))
        else:
            raise ValueError(f"Unsupported categorical_encode_method for features: {categorical_encode_method}")

    # Ordinal encode selected numeric features
    if use_ordinal_encoding:
        if discrete2ord_vars_idx:
            discrete2ord_preprocessor.append(('ordinal_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)))
        if continuous2ord_vars_idx:
            nbins_discretizer = [continuous_to_ordinal_transform_nbins.get(name, 10) for name in continuous2ord_var_names]
            continuous2ord_preprocessor.append(('ordinal_discretizer', KBinsDiscretizer(n_bins=nbins_discretizer, encode='ordinal', strategy='uniform')))

    # Scale numeric (and ordinal) feature values
    scl_type = models[mdl_type].get('scaler', None)
    add_pca_to_pipe = models[mdl_type].get('pca_transform', None)
    scaler = None
    if scl_type == 'standard':
        scaler = StandardScaler()
    elif scl_type == 'minmax' and not pca_transform_numeric:
        scaler = MinMaxScaler()
    elif scl_type == 'quantile' and not pca_transform_numeric:
        scaler = QuantileTransformer(n_quantiles=scaler_quantile_nquantiles)
    elif scl_type == 'robust' and not pca_transform_numeric:
        scaler = RobustScaler()
    elif scl_type == 'rowwise' and not pca_transform_numeric:
        scaler = Normalizer(norm=scaler_rowwise_norm)
    if scaler:
        num_preprocessor.append(('scaler', scaler))
        if not use_categorical_encoding and cat_vars_idx:
            cat_preprocessor.append(('scaler', scaler))
        if use_ordinal_encoding:
            if discrete2ord_vars_idx:
                discrete2ord_preprocessor.append(('scaler', scaler))
            if continuous2ord_vars_idx:
                continuous2ord_preprocessor.append(('scaler', scaler))
    if pca_transform_numeric:
        num_preprocessor.append(('scaler', StandardScaler()))
        num_preprocessor.append(('pca', PCA(n_components=len(num_var_names))))

    # Preprocessor part of pipeline
    preprocessor_steps = []
    if num_preprocessor:
        num_transformer = Pipeline(num_preprocessor)
        preprocessor_steps.append(('numeric', num_transformer, num_vars_idx))
    if discrete2ord_preprocessor:
        discrete2ord_transformer = Pipeline(discrete2ord_preprocessor)
        preprocessor_steps.append(('discrete_to_ordinal', discrete2ord_transformer, discrete2ord_vars_idx))
    if continuous2ord_preprocessor:
        continuous2ord_transformer = Pipeline(continuous2ord_preprocessor)
        preprocessor_steps.append(('continuous_to_ordinal', continuous2ord_transformer, continuous2ord_vars_idx))
    if cat_preprocessor:
        cat_transformer = Pipeline(cat_preprocessor)
        preprocessor_steps.append(('categorical', cat_transformer, cat_vars_idx))
    if drop_col_preprocessor:
        drop_col_transformer = Pipeline(drop_col_preprocessor)
        preprocessor_steps.append(('drop_columns', drop_col_transformer, drop_vars_idx))
    preprocessor = ColumnTransformer(transformers=preprocessor_steps, remainder='passthrough')

    # Set random state of regressor
    if 'random_state' in rgr_mdl.get_params():
        rgr_mdl.set_params(random_state=seed_value)
    elif 'random_seed' in rgr_mdl.get_params():
        rgr_mdl.set_params(random_seed=seed_value)

    # Set monotone constraints (only works for gradient-boosting models and linear regression models)
    if set_monotonic_constraints and not pca_transform_numeric:
        monotone_constraints, monotone_columns = monotonic_constraints(
            num_vars=num_var_names,
            ord_vars=discrete2ord_var_names + continuous2ord_var_names, 
            cat_vars=cat_var_names, 
            cat_encode=categorical_encode_method, 
            X=X_train, 
            features_increasing=features_monotone_increasing, 
            features_decreasing=features_monotone_decreasing,
        )
        if mdl_type == 'LGBM':
            rgr_mdl.set_params(monotone_constraints=monotone_constraints, monotone_constraints_method=monotone_constraints_method_lgbm)
        elif mdl_type == 'CatB':
            rgr_mdl.set_params(monotone_constraints=monotone_constraints)
        elif mdl_type == 'HGB':
            rgr_mdl.set_params(monotonic_cst=monotone_constraints)
        elif mdl_type in ['XGB', 'XGB-RF']:
            rgr_mdl.set_params(monotone_constraints=tuple(monotone_constraints))
        elif mdl_type in ['LR', 'Ridge', 'Lasso']:
            rgr_mdl = ConstrainedLinearRegression()
            min_coef = np.repeat(-np.inf, len(monotone_constraints))
            max_coef = np.repeat(np.inf, len(monotone_constraints))
            for j, col in enumerate(monotone_columns):
                if monotone_constraints[j] < 0:
                    max_coef[j] = 0.
                elif monotone_constraints[j] > 0:
                    min_coef[j] = 0.
            if mdl_type == 'Ridge':
                rgr_mdl.set_params(max_coef=max_coef, min_coef=min_coef, ridge=True)
            elif mdl_type == 'Lasso':
                rgr_mdl.set_params(max_coef=max_coef, min_coef=min_coef, lasso=True)
            else:
                rgr_mdl.set_params(max_coef=max_coef, min_coef=min_coef)

    # Transform target variable
    if use_target_transform:
        if target_transform_method == 'yeojohnson':
            rgr_mdl = TransformedTargetRegressor(regressor=rgr_mdl, transformer=YeoJohnsonTargetTransformer(lmbda=target_yeojohnson_lambda, replace_nan_percentile=target_yeojohnson_nan_percentile))
        elif target_transform_method == 'log':
            rgr_mdl = TransformedTargetRegressor(regressor=rgr_mdl, transformer=LogTargetTransformer())

    # Add all steps to final pipeline
    steps = [('preprocessor', preprocessor)] if preprocessor_steps else []
    steps.append(('model', rgr_mdl))
    pipeline = Pipeline(steps=steps)

    return pipeline


# Water board abbreviations
ws_abbr = {
    "De Dommel en Aa en Maas": "WDD&AaM",
    "Limburg en De Dommel": "WL&WDD",
    "Limburg en Aa en Maas": "WL&AaM",
    "Fryslan": "WF",
    "Drents Overijsselse Delta": "WDOD",
    "Vechtstromen": "WVS",
    "Rijn en IJssel": "WRIJ",
    "Rivierenland": "WRL",
    "De Stichtse Rijnlanden": "HDSR",
    "Brabantse Delta": "WBD",
    "De Dommel": "WDD",
    "Hunze en Aa's": "H&A",
    "Noorderzijlvest": "NZV",
    "Aa en Maas": "AaM",
    "Vallei en Veluwe": "V&V",
    "Limburg": "WL",
    "Groot Salland": "WGS",  # Voormalig waterschap, opgegaan in WDOD
    "Peel en Maasvallei": "P&M",  # Voormalig waterschap, opgegaan in WL
    "Regge en Dinkel": "R&D",  # Voormalig waterschap, opgegaan in WVS
    "Vallei en Eem": "V&E",  # Mogelijke variant van V&V
    "Roer en Overmaas": "R&O",  # Voormalig waterschap, opgegaan in WL
    "van Rijnland": "HHR",
    "Hollandse Delta": "WHD",
    "Zuiderzeeland": "ZZL",
    "van Schieland en Krimpenerwaard": "HHSK",
    "van Delfland": "HHD",
    "Hollands Noorderkwartier": "HHNK",
    "Provincie Noord-Holland": "ProvNH",  # Provincie
    "Zeeuws-Vlaanderen": "WZV",  # Voormalig waterschap, opgegaan in WSS
    "Amstel, Gooi en Vecht": "AGV",
    "Scheldestromen": "WSS",
    "Velt en Vecht": "VeltV",  # Alternatieve afkorting om verwarring met V&V te voorkomen
    "Reest en Wieden": "WRW",  # Voormalig waterschap, opgegaan in WDOD
    "Zeeuwse Eilanden": "WZE",  # Voormalig waterschap, opgegaan in WSS
    "Expert data": "Expert",
}

# Read (and clean) datasets, metadata and logo from files
if not st.session_state.data_loaded:
    # Datasets with EKR scores
    alldata = {}
    for key in files_data.keys():
        alldata[key] = read_excel_with_multiple_sheets(path_data, files_data[key], sheets_to_skip=['Meta data'])
        # Rename columns
        alldata[key].columns = alldata[key].columns.str.replace('_EKR', '', regex=False)
        alldata[key].columns = alldata[key].columns.str.replace('_', ' ', regex=False)
        # Standardize names for expert data
        alldata[key]['Waterschap'] = alldata[key]['Waterschap'].where(~alldata[key]['Waterschap'].str.contains('Niels|NEV'), 'Expert data')
        # Add abbreviations to names of waterschappen
        alldata[key]['Waterschap naam'] = alldata[key]['Waterschap']
        alldata[key]['Waterschap'] = alldata[key]['Waterschap'].str.replace('Waterschap ', '', regex=False)
        alldata[key]['Waterschap'] = alldata[key]['Waterschap'].str.replace('Wetterskip ', '', regex=False)
        alldata[key]['Waterschap'] = alldata[key]['Waterschap'].str.replace('Hoogheemraadschap van  ', '', regex=False)
        alldata[key]['Waterschap'] = alldata[key]['Waterschap'].str.replace('Hoogheemraadschap ', '', regex=False)
        for ws in ws_abbr.keys():
            alldata[key]['Waterschap'] = alldata[key]['Waterschap'].str.replace(ws, ws_abbr[ws], regex=False)
    # Add dummy columns for dataset with detailed features
    for col in ['Vis','Overige waterflora','Fytoplankton']:
        alldata['Gedetailleerd'][col] = np.nan
    # Metadata stuurvariabelen
    metadata_df = read_excel_with_multiple_sheets(path_data, file_metadata, sheets_to_skip=[])
    # Deltares logo
    logo = Image.open(path_data + '/' + file_logo)  # Load logo
    
    # Reset session state
    st.session_state.data_loaded = True
    # Add data to session state
    st.session_state.alldata_df = alldata
    st.session_state.metadata_df = metadata_df
    st.session_state.logo_tif = logo
else:
    # Get data from session state
    alldata = st.session_state.alldata_df
    metadata_df = st.session_state.metadata_df
    logo = st.session_state.logo_tif

cluster_types = list(alldata['Geaggregeerd']['Clustertype'].unique())
krw_types = ['Macrofauna','Vis','Overige waterflora','Fytoplankton']

# Setup sidebar with controls
# Adjust sidebar width using CSS in order to fit the texts in a single line
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        min-width: 350px; /* Adjust this width as needed */
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.sidebar.image(logo, use_container_width=True)  # Display logo in sidebar
st.sidebar.title('KRW-Verkenner Regionaal')  # Add title for sidebar
# Dropdown for Cluster Type selection
cluster_name = st.sidebar.selectbox(
    "KRW Cluster",
    cluster_types,
    help="Kies het gewenste clustertype uit de beschikbare opties."
)
# Dropdown for EKR Type selection
target_name = st.sidebar.selectbox(
    "EKR Score",
    krw_types,
    help="Kies de gewenste KRW indicator uit de beschikbare opties."
)
# Radio button for Data Type selection form dataset
data_type = st.sidebar.radio(
    "Soort stuurvariabelen",
    options=["Geaggregeerd", "Gedetailleerd"],
    index=["Geaggregeerd", "Gedetailleerd"].index(data_type_init),
    horizontal=True,
    help="Kies tussen gebruik van geaggregeerde of gedetailleerde stuurvariabelen in het model."
)
# Checkbox for Showing HiPlot
show_hiplot = False
# show_hiplot = st.sidebar.checkbox(
#     "Interactieve Visualisatie van Dataset", 
#     value=False
# )
# Dropdown for Model selection
model_name = st.sidebar.selectbox(
    "Model Type",
    list(model_types.keys()),
    help="Kies het gewenste type ML model uit de beschikbare opties."
)
# Slider for Prediction Interval
conformal_interval = st.sidebar.slider(
    "Nauwkerigheidsinterval van Voorspellingen (%)",
    min_value=50,
    max_value=90,
    step=5,
    value=conformal_interval_init,
    help="Stel het gewenste betrouwbaarheidsinterval in voor de voorspellingen."
)
# Slider for Test Data Size
train_size = st.sidebar.slider(
    "Data voor Training van Model (% van totaal)",
    min_value=60,
    max_value=80,
    step=5,
    value=train_size_init,
    help="Bepaal welk percentage van de dataset wordt gebruikt voor training van het model. De dataset wordt gesplitst in een deel voor training en een deel voor evaluatie. Dit helpt om te voorkomen dat er info uit de testset gebruikt wordt om te trainen, waardoor de nauwkeurigheid van het model mogelijk overschat wordt. "
)
# Radio button for Data Type selection in plots
data_split = st.sidebar.radio(
    "Deel van Dataset tonen in Figuren",
    options=["Train", "Test"],
    index=["Train", "Test"].index(data_split_init),
    horizontal=True,
    help="Selecteer of de figuren de uitkomsten voor de trainings- of testset laten zien."
)
# Slider for Seed Value
max_seed = 1000
seed_value = st.sidebar.slider(
    "Startwaarde voor random generator",
    min_value=0,
    max_value=max_seed,
    step=1,
    value=seed_value_init,
    help="Kies een startwaarde van de random generator."
)


# Get the selected dataset
@st.cache_data
def get_selected_data(data_df, feats_in, cluster, dtype, target, thresh_nan=0.8):
    dataf = data_df[dtype].copy()

    columns_to_keep = feats_in[dtype].copy()
    columns_to_keep.extend(['KRW type', 'Waterschap', 'OWL naam','Clustertype'])
    columns_to_keep = dataf.columns[dataf.columns.isin(columns_to_keep)].tolist()

    dataf = dataf[columns_to_keep]
    dataf[target] = data_df[dtype][target].values
    dataf = dataf.dropna(subset=[target])
    dataf = dataf[dataf['Clustertype'] == cluster]
    dataf = dataf.dropna(axis='columns', thresh=int(thresh_nan*dataf.shape[0]))
    dataf = dataf.drop(columns=['Clustertype'])

    cols_to_move = ['KRW type','Waterschap','OWL naam']
    for col in cols_to_move:
        if col in dataf.columns:
            dataf[col] = dataf.pop(col).astype(str)

    dataf = dataf.dropna()
    dataf = dataf.reset_index(drop=True)
    return dataf

features_in_model = {
    'Geaggregeerd': [
        'BZV', 'Chloride', 'Stikstof totaal', 'Fosfor totaal', 
        'Ammonium', 'Toxiciteit', 'Doorzicht', 'Meandering', 
        'Beschaduwing', 'Oeverinrichting', 'Verstuwing', 
        'Onderhoud', 'Connectiviteit', 'Scheepvaart', 'Peilbeheer', 
    ],
    'Gedetailleerd': [
        'BZV', 'Chloride', 'Stikstof totaal', 'Fosfor totaal', 
        'Ammonium', 'Toxiciteit', 'Percentage beschaduwing',
        'Stroomsnelheid', 'Waterdeel onder invloed verstuwing',
        'Percentage nat profiel gemaaid', 'Sinusoiteit', 'Profielvorm', 
        'Vispasseerbaarheid', 'Ruimtelijke variatie stroomsnelheid',
    ]
}

try:
    df = get_selected_data(alldata, feats_in=features_in_model, target=target_name, cluster=cluster_name, dtype=data_type)
except Exception as e:
    st.error(f"Error loading dataset: {str(e)}")


# Add selected values to session state
if cluster_name != st.session_state.cluster_name:
    st.session_state.cluster_name = cluster_name
    st.session_state.new_settings = True
if target_name != st.session_state.target_name:
    st.session_state.target_name = target_name
    st.session_state.new_settings = True
if data_type != st.session_state.data_type:
    st.session_state.data_type = data_type
    st.session_state.new_settings = True
if model_name != st.session_state.model_name:
    st.session_state.model_name = model_name
    st.session_state.new_settings = True
if conformal_interval != st.session_state.conformal_interval:
    st.session_state.conformal_interval = conformal_interval
    st.session_state.new_settings = True
if train_size != st.session_state.train_size:
    st.session_state.train_size = train_size
    st.session_state.new_settings = True
if data_split != st.session_state.data_split:
    st.session_state.data_split = data_split
    st.session_state.new_settings = True
if seed_value != st.session_state.seed_value:
    st.session_state.seed_value = seed_value
    st.session_state.new_settings = True
if show_hiplot != st.session_state.show_hiplot:
    st.session_state.show_hiplot = show_hiplot


# Check if dataset is changed and reset session state (if true)
if st.session_state.dataset_selected is None:
    st.session_state.dataset_selected = df.copy()
    st.session_state.new_dataset = True
    st.session_state.start_select_features = True
else:
    if not st.session_state.dataset_selected.equals(df):
        st.session_state.dataset_selected = df.copy()
        st.session_state.new_dataset = True
        st.session_state.start_select_features = True

# Reset session state of tables and figures, if new dataset is selected or settings are changed
if (st.session_state.new_dataset) | (st.session_state.new_settings):
    st.session_state.table_metadata = None
    st.session_state.table_metrics = None
    st.session_state.fig_performance = None
    st.session_state.fig_importance = None
    st.session_state.fig_waterfall = None
    st.session_state.fig_beeswarm = None
    st.session_state.fig_avg_shap = None
    st.session_state.fig_dependence = None
    st.session_state.fig_intervals = None
    st.session_state.fig_causal = None


# Display dataset
if len(df) > 0:
    # Show dataset as table
    st.write(f"### {cluster_name}: EKR score '{target_name}' (N={len(df)})")
    st.dataframe(df)
else:
    # Show message if dataset is empty
    st.session_state.dataset_selected = None
    st.write(f"### {cluster_name}: EKR score '{target_name}'")
    st.write("Geen data beschikbaar")


@st.cache_data(ttl=600, max_entries=50)  # Cache expires after 10 minutes, keeps last 50 results
def get_parallel_coordinates(df, target):
    data_hip = df.copy()
    data_hip = data_hip.drop(columns=['KRW type', 'Waterschap naam', 'OWL naam'], errors='ignore')
    data_hip = data_hip.rename(columns={target: f"EKR '{target}'"})
    if 'Expert' in data_hip['Waterschap'].unique():
        data_experts = data_hip.copy().loc[data_hip['Waterschap'] == 'Expert']
        data_hip = data_hip.loc[data_hip['Waterschap'] != 'Expert'].sort_values('Waterschap', ascending=False)
        categories = ['Expert'] + data_hip['Waterschap'].unique().tolist()
        data_hip['Waterschap'] = pd.Categorical(data_hip['Waterschap'], categories=categories, ordered=True)
    data_hip = data_hip.sort_values('Waterschap', ascending=False)
    data_hip = pd.concat([data_experts, data_hip], axis='index', ignore_index=True).reset_index(drop=True)
    xp = hip.Experiment().from_dataframe(data_hip)
    return xp


if len(df) > 0:
    if st.session_state.show_hiplot:
        # Show dataset with interactive parallel coordinates plot
        st.write("")
        st.write("")
        st.markdown(
            "<p style='font-size:14px;'>Onderstaand is een interactieve <b>parallel coordinates plot</b> te zien. Elke verticale as vertegenwoordigt een stuurvariabele of de EKR-score, terwijl de lijnen van links naar rechts individuele datapunten weergeven. Door interactief te filteren, specifieke gebieden te markeren of bepaalde lijnen uit te sluiten, kunnen verborgen relaties en patronen relatief eenvoudig zichtbaar worden gemaakt.</p>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<p style='font-size:14px;'>Per variabele kan een specifieke range geselecteerd worden door links te klikken op een startpunt en vast te houden tot het gewenste bereik. Het geselecteerde bereik wordt weergegeven met een grijs vlak. Deze bereiken kunnen ook worden verplaatst door te klikken, vast te houden en te slepen. Door rechts te klikken op de naam van een variabele kunnen een aantal eigenschappen ervan ingesteld worden (bijv. 'Use for coloring').</p>",
            unsafe_allow_html=True
        )
        st.write("")
        # Render interactive parallel coordinates plot
        xp = get_parallel_coordinates(df, target=target_name)
        # Instead of calling directly '.display()' convert it to a streamlit component with '.to_streamlit()' before
        xp.to_streamlit(key="hipl", ret=None).display();


# Display metadata of features
if len(df) > 0:
    st.write("")
    st.write("")
    st.write("### Stuurvariabelen")
    table_metadata = (
        metadata_df.loc[
            ((metadata_df['Clustertype'].apply(lambda x: cluster_name in x)) & (metadata_df['Stuurvariabele'].isin(df.columns))), 
            ['Deelvariabele','Omschrijving']
        ].rename(columns={'Deelvariabele':'Stuurvariabele'}).set_index('Stuurvariabele')
    )
    st.write("")
    st.table(table_metadata)


# Define callback functions
def update_selected_features(option):
    """
    Toggle the selection of a single option and update selected features.
    """
    # Toggle checkbox state
    st.session_state.checkbox_states[option] = not st.session_state.checkbox_states[option]
    # Update the list of selected features
    st.session_state.selected_features = [
        option for option, selected in st.session_state.checkbox_states.items() if selected
    ]
    # Update state for new settings
    st.session_state.new_settings = True

def select_all_features():
    # Select all options
    for option in define_options:
        st.session_state.checkbox_states[option] = True
    st.session_state.selected_features = define_options.copy()
    # Update state for new settings
    st.session_state.new_settings = True

def deselect_all_features():
    # Deselect all options
    for option in define_options:
        st.session_state.checkbox_states[option] = False
    st.session_state.selected_features = []
    # Update state for new settings
    st.session_state.new_settings = True


# Select input features
if len(df) > 0:
    min_num_features = 2
    st.write("")

    # List of options
    define_options = df.drop(columns=['KRW type', 'Waterschap naam', 'OWL naam', 'Waterschap', st.session_state.target_name], errors='ignore').columns.tolist()

    # Reset selected features if starting a new selection
    if st.session_state.start_select_features:
        st.session_state.selected_features = define_options.copy()  # All selected at start

    # Initialize session state for checkboxes
    if 'checkbox_states' not in st.session_state or st.session_state.start_select_features:
        st.session_state.checkbox_states = {option: True for option in define_options}  # Default: all selected

    # Synchronize checkbox states with define_options (handles new/removed options)
    for option in define_options:
        if option not in st.session_state.checkbox_states:
            st.session_state.checkbox_states[option] = False
    for option in list(st.session_state.checkbox_states.keys()):
        if option not in define_options:
            del st.session_state.checkbox_states[option]

    # Buttons for Select All/Deselect All
    col1, col2, *rest = st.columns(8)
    with col1:
        st.button("Selecteer alle", help="Selecteer tenminste twee stuurvariabelen", on_click=select_all_features)
    with col2:
        st.button("Deselecteer alle", on_click=deselect_all_features)

    # Generate checkboxes for individual options in multiple columns
    num_columns = 4  # Number of columns
    columns = st.columns(num_columns)
    for idx, option in enumerate(define_options):
        col = columns[idx % num_columns]
        with col:
            # Create a checkbox with the callback function
            st.checkbox(
                option,
                value=st.session_state.checkbox_states[option],
                key=f"checkbox_{option}",
                on_change=update_selected_features,
                args=(option,),
            )

    # Update selected features
    updated_selected_features = [
        opt for opt, selected in st.session_state.checkbox_states.items() if selected
    ]

    # Create list of features not included in model training
    columns_not_in_model = [option for option in define_options if option not in st.session_state.selected_features]

    # Show warning if less than the minimum number of features are selected
    if len(st.session_state.selected_features) < min_num_features:
        st.warning(f"Selecteer ten minste {min_num_features} stuurvariabelen om het model te kunnen trainen.")

    # Update session state with the current selections
    st.session_state.selected_features = updated_selected_features
    # Reset start_select_features
    st.session_state.start_select_features = False


# Activate model training (if a dataset is selected and at least three features are selected)
if st.session_state.dataset_selected is not None:
    if len(st.session_state.selected_features) >= min_num_features:
        if st.sidebar.button("Model Trainen en Resultaten Tonen"):
            st.session_state.fit_model = True
            st.session_state.render_figures = True


def stratified_split(X, y, train_size=0.8, random_state=42, binsize=0.1):
    # Add column of ordinal values of y
    y_ordinal = y.copy()
    bins = np.arange(0, 1 + binsize, binsize)
    y_ordinal = pd.cut(y_ordinal, bins=bins, labels=False, include_lowest=True)
    # Split dataset into train and test based on y_ordinal
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=random_state, stratify=y_ordinal)
    return X_train, X_test, y_train, y_test

# Train model and get results
if st.session_state.fit_model:
    # Split dataset into X and y
    X = df.copy().drop(target_name, axis='columns').drop(['Clustertype','KRW type','Waterschap','Waterschap naam', 'OWL naam'], axis='columns', errors='ignore')
    y = df.copy()[target_name]

    # Train/test split on X and y
    if split_method == 'stratified':
        X_train, X_test, y_train, y_test = stratified_split(X, y, train_size=st.session_state.train_size / 100, random_state=st.session_state.seed_value, binsize=0.2)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=st.session_state.train_size / 100, random_state=st.session_state.seed_value)

    # Set figure height
    fig_height = 0.25 * len(X.columns)
    
    # Split ordinal features into discrete and continuous
    discrete_to_ordinal, continuous_to_ordinal = [], []
    for col in ordinal_columns:
        if col in X_train.columns:
            if col not in columns_not_in_model:
                if X_train[col].dtype in ['category', 'object']:
                    discrete_to_ordinal.append(col)
                else:
                    continuous_to_ordinal.append(col)

    # convert X_train columns to list
    Xtrain_columns = X_train.columns.tolist()

    # Initialize regression model
    mdl_type = model_types[model_name]
    name_long = models[mdl_type].get('name',None)
    rgr_mdl = copy.deepcopy(models[mdl_type].get('model'))
    if 'random_state' in rgr_mdl.get_params():
        rgr_mdl.set_params(random_state=seed_value)
    elif 'seed_value' in rgr_mdl.get_params():
        rgr_mdl.set_params(seed_value=seed_value)
    model = regression_model_pipeline()
    
    # Initialize quantile model
    set_monotonic_constraints = False
    rgr_mdl = copy.deepcopy(cqr_models[mdl_type].get('model'))
    if 'random_state' in rgr_mdl.get_params():
        rgr_mdl.set_params(random_state=seed_value)
    elif 'seed_value' in rgr_mdl.get_params():
        rgr_mdl.set_params(seed_value=seed_value)
    cqr_model = regression_model_pipeline()

    # Select the appropriate data part for plotting
    if data_split == "Train":
        X_selected, y_selected = X_train, y_train
    else:
        X_selected, y_selected = X_test, y_test
    
    with st.spinner('Training model...'):
        # Train model
        model.fit(X_train, y_train)

        # Get metrics
        predict_train = model.predict(X_train)
        metrics_train = get_regression_metrics(predict_train, y_train.values, bounds=None)
        predict_test = model.predict(X_test)
        metrics_test = get_regression_metrics(predict_test, y_test.values, bounds=None)
        df_metrics = pd.DataFrame.from_dict(metrics_test, orient='index')
        df_metrics = pd.concat([df_metrics, pd.DataFrame.from_dict(metrics_train, orient='index', columns=['Train data'])], axis='columns')
        metrics_to_show = ['RMSE', 'MAE', 'CCC', 'R2', '+/-0.1']
        df_metrics.columns = ['Test','Training']
        df_metrics = df_metrics.loc[metrics_to_show,:]


        # Calculate permutation importance
        n_repeats = 10
        if isinstance(model, (RandomForestQuantileRegressor, ExtraTreesQuantileRegressor)):
            perm_importances_mean = custom_permutation_importance(model, X_selected, y_selected, n_repeats=n_repeats, random_state=seed_value)
        else:
            mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
            perm_importance = permutation_importance(model, X_selected, y_selected, scoring=mae_scorer, n_repeats=n_repeats, random_state=seed_value)
            perm_importances_mean = perm_importance.importances_mean
        feature_names = X_selected.columns.tolist()
        # Create importance dataframe
        importance_df = pd.DataFrame(perm_importances_mean, index=feature_names, columns=['Importance'])
        importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=False)
        importance_df = importance_df.rename(columns={'index': 'Feature'})
        importance_df = importance_df.loc[~importance_df['Feature'].isin(columns_not_in_model)]
        # Select the top N features by importance
        importance_df = importance_df.iloc[:top_n,:]
        # Revert sorting for barplot
        importance_df = importance_df.sort_values('Importance', ascending=True)
        # Get importances and feature names
        sorted_importances = importance_df['Importance'].tolist()
        sorted_feature_names = importance_df['Feature'].tolist()


        # Calculate SHAP Values
        X_shap, y_shap = X_selected.copy(), y_selected.copy()
        # Transform features
        X_tf = model.named_steps['preprocessor'].transform(X_shap)
        # Define 'column' order in np.array after preprocessing (1. numeric, 2. ordinal, 3. categorical)
        ord_var_names = X_shap.columns[(X_shap.columns.isin(ordinal_columns)) & (~X_shap.columns.isin(columns_not_in_model))].tolist()
        cat_var_names = X_shap.columns[(X_shap.columns.isin(categorical_columns)) & (~X_shap.columns.isin(columns_not_in_model))].tolist()
        num_var_names = X_shap.columns[(~X_shap.columns.isin(cat_var_names)) & (~X_shap.columns.isin(ord_var_names)) & (~X_shap.columns.isin(columns_not_in_model))].tolist()
        new_column_order = num_var_names
        new_column_order.extend(ord_var_names)
        new_column_order.extend(cat_var_names)
        # Choose appropriate explainer
        standard_explainer = True
        try:
            shap_explainer = shap.Explainer(model.named_steps['model'])
        except:
            standard_explainer = False
        # Estimate SHAP values
        if standard_explainer:
            shap_explainer = shap.Explainer(model.named_steps['model'])
            shap_values = shap_explainer.shap_values(X_tf)
        else:
            shap_explainer = shap.PermutationExplainer(model.named_steps['model'].predict, X_tf, silent=True)
            shap_values = shap_explainer.shap_values(X_tf)
        shap_feature_names = X_shap.columns
        # Aggregate all categories of onehot encoded features
        if use_categorical_encoding:
            # Get number of unique categories for each feature 
            n_categories = []
            for feat in new_column_order[:-1]:
                if feat in categorical_columns:
                    n = X_shap[feat].nunique()
                    if ohe_drop_first:
                        n = n - 1
                    n_categories.append(n)
                else:
                    n_categories.append(1)
            # Sum SHAP values of all categories in each feature
            new_shap_values = []
            for values in shap_values:
                # Split shap values into list for each feature
                values_split = np.split(values , np.cumsum(n_categories))
                # Sum values within each list per feature
                values_sum = [sum(l) for l in values_split]
                new_shap_values.append(values_sum)
            new_shap_values = np.array(new_shap_values)
            # Replace SHAP values with new values
            shap_values = new_shap_values
            shap_feature_names = new_column_order
        # Calculate mean absolute SHAP values for each feature
        if isinstance(shap_values, list):
            shap_vals = np.mean(np.abs(shap_values), axis=0)
        else:
            shap_vals = np.abs(shap_values)
        # Calculate mean SHAP values
        mean_shap_values = np.mean(shap_vals, axis=0)
        feature_names_sorted = np.array(shap_feature_names)[np.argsort(mean_shap_values)][-top_n:]  # Top 15 features
        mean_shap_values_sorted = mean_shap_values[np.argsort(mean_shap_values)][-top_n:]
        # Select single item to explain in waterfall plot
        idx_select = int(len(X_shap) * (seed_value + 1) / max_seed)
        explanation = {}
        if 'expected_value' in dir(shap_explainer):
            expected_value = shap_explainer.expected_value
        else:
            expected_value = np.mean(model.named_steps['model'].predict(X_tf))
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[0]
        explanation['expected_value'] = expected_value
        explanation['shap_values'] = shap_values[idx_select,:]
        explanation['feature_names'] = X_shap.columns
        explanation['X_values'] = X_shap.iloc[idx_select]
        explanation['y_true'] = y_shap.iloc[idx_select]
        explanation['y_pred'] = model.predict(X_shap)[idx_select]
        explanation['error'] = explanation['y_pred'] - explanation['y_true']
        explanation['error_text'] = 'lager' if explanation['error'] < 0 else 'hoger'
        explanation['explained_value'] = shap.Explanation(
            values=pd.DataFrame(shap_values, columns=shap_feature_names, index=X_shap.index)[X_shap.drop(columns=columns_not_in_model).columns].values[idx_select,:],  # re-order columns to match X_shap
            base_values=expected_value, 
            data=X_shap.iloc[idx_select], 
            feature_names=X_shap.drop(columns=columns_not_in_model).columns
        )
        # Fix for XGBoost results in SHAP Explanation being stored as float32: SHAP's 'waterfall_plot()' uses
        # model outputs for tick labels, and XGBoost's default 'float32 format causes incorrect ticklabels, 
        # leading Matplotlib to incorrectly display "$f(x)$" as f"$ = {fx_value}$". Converting everything to 
        # 'float64' ensures consistent behavior across all models. Occurs in SHAP.__version__ > '0.46.0'
        if type(explanation['explained_value'].base_values) == np.float32:
        # if mdl_type.lower() == 'xgb':
            explanation['explained_value'].base_values = np.float64(explanation['explained_value'].base_values)
            explanation['explained_value'].values = explanation['explained_value'].values.astype(np.float64)
        st.session_state.explanation = explanation


        # Calculate prediction intervals
        alpha = np.round(1 - (conformal_interval / 100), 3)
        alphas = [alpha/2, 1-(alpha/2), 0.5]
        models_cp = []
        X_tf = model.named_steps['preprocessor'].transform(X_train)
        for a in alphas:
            m = copy.deepcopy(cqr_model).named_steps['model']
            pars = m.get_params().keys()
            if 'q' in pars:
                m = m.set_params(**{'q':np.round(a,3)})
            elif 'alpha' in pars:
                m = m.set_params(**{'alpha':np.round(a,3)})
            elif 'quantile' in pars:
                m = m.set_params(**{'quantile':np.round(a,3)})
            elif 'quantile_alpha' in pars:
                m = m.set_params(**{'quantile_alpha':np.round(a,3)})
            else:
                print('Quantile parameters not found')
            m.fit(X_tf, y_train)
            models_cp.append(m)
        # Calibrate mapie quantile regressor
        mapie_cqr = MapieQuantileRegressor(models_cp, alpha=alpha, cv='prefit')
        calib_fraction = 0.5
        idx_calib = X_test.sample(frac=calib_fraction, random_state=seed_value).index
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            mapie_cqr.fit(model.named_steps['preprocessor'].transform(X_test.loc[idx_calib,:]), y_test.loc[idx_calib].values)
        # Predict mapie intervals
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            y_pred, y_pis = mapie_cqr.predict(model.named_steps['preprocessor'].transform(X_test))
        for i in range(y_pis.shape[1]):
            y_pis[:,i,0] = np.clip(y_pis[:,i,0], 0, 1)

        # Aggregate prediction intervals
        if model_name != 'Linear Regression':
            prediction_intervals = pd.DataFrame(
                {'y_new':y_test.values, 'y_pred':y_pred, 'q_lower':y_pis[:,0].flatten(), 'q_upper':y_pis[:,1].flatten()}
            )
        else:
            prediction_intervals = pd.DataFrame(
                {'y_new':y_test.values, 'y_pred':predict_test, 'q_lower':y_pis[:,0].flatten(), 'q_upper':y_pis[:,1].flatten()}
            )
        prediction_intervals = prediction_intervals.sort_values(by='y_new', ascending=True).reset_index(drop=True)
        digits = 3
        conformal_metrics = {
            'mean_coverage':float(np.round(regression_coverage_score(y_test.to_numpy(), y_pis[:, 0, 0], y_pis[:, 1, 0]), digits)),
            'mean_width':float(np.round(regression_mean_width_score(y_pis[:, 0, 0], y_pis[:, 1, 0]), digits)),
            'winkler_score':float(np.round(MWIS_metric.score(y_test.to_numpy(), y_pis[:, 0, 0], y_pis[:, 1, 0], alpha=alpha)[0], digits)),
            'cwc_score':float(np.round(coverage_width_based(y_test.to_numpy(), y_pis[:, 0, 0], y_pis[:, 1, 0], eta=10, alpha=alpha), digits)),
        }
        # print(conformal_metrics)


        # Causal discovery using DirectLiNGAM
        apply_prior_knowledge_softly = False
        remove_links_originating_from_target = True
        measure = 'pwling'
        # Create a copy of the dataframe for causal discovery
        df_causal = X.copy()
        # Create list of columns after transforming features
        cols_to_drop = columns_not_in_model.copy()
        if (use_ordinal_encoding) & (len(ordinal_columns)>0):
            for col in ordinal_columns:
                if col in df_causal.columns:
                    cols_to_drop.append(col)
        if (use_categorical_encoding) & (len(categorical_columns)>0):
            for col in categorical_columns:
                if col in df_causal.columns:
                    cols_to_drop.append(col)
        cols_result = df_causal.drop(columns=cols_to_drop).columns.tolist()
        if (use_ordinal_encoding) & (len(ordinal_columns)>0):
            cols_result.extend(df_causal.columns[(df_causal.columns.isin(ordinal_columns)) & (~df_causal.columns.isin(columns_not_in_model))].tolist())
        if (use_categorical_encoding) & (len(categorical_columns)>0):
            if categorical_encode_method in ['OHE']:
                if ohe_drop_first:
                    drop_first = 'first'
                else:
                    drop_first = None
                for col in categorical_columns:
                    if col not in columns_not_in_model:
                        if version.parse(sklearn.__version__) >= version.parse('1.2'):
                            ohe = OneHotEncoder(handle_unknown='ignore', drop=drop_first, sparse_output=False)
                        else:
                            ohe = OneHotEncoder(handle_unknown='ignore', drop=drop_first, sparse=False)
                        ohe_result = ohe.fit_transform(X_train[[col]])
                        ohe_categories = list(ohe.categories_[0])
                        if ohe_drop_first:
                            ohe_categories = ohe_categories[1:]
                        ohe_categories = [col + '__' + str(item) if isinstance(item, int) else col + '__' + item for item in ohe_categories]
                        cols_result.extend(ohe_categories)
            else:
                cols_result.extend(X_train.columns[(X_train.columns.isin(categorical_columns)) & (~X_train.columns.isin(columns_not_in_model))].tolist())
        cols_result = [str(item if isinstance(item, int) else item) for item in cols_result]
        # Transform features prior to causal discovery
        df_causal = pd.DataFrame(model.named_steps['preprocessor'].transform(df_causal), columns=cols_result)
        # Add target variable to dataframe
        df_causal[target_name] = y.values
        # Set index of dataframe
        df_causal.index = X.index
        
        # Define list of impossible (and predefined) links
        impossible_links, predefined_links = [], []
        if remove_links_originating_from_target:
            impossible_links = impossible_links + [(target_name,col) for col in df_causal.drop(columns=target_name).columns]
        impossible_links = [(cause, effect) for cause, effect in impossible_links if (cause in df_causal.columns) and (effect in df_causal.columns)]  
        # Get node names
        node_names = df_causal.columns.tolist()
        # Initialize prior knowledge matrix with -1 (no prior knowledge)
        n_vars = len(node_names)
        prior_knowledge = np.full((n_vars, n_vars), -1)
        # Update prior knowledge matrix with 0 for physically impossible causal links
        for cause, effect in impossible_links:
            cause_idx = node_names.index(cause)
            effect_idx = node_names.index(effect)
            prior_knowledge[cause_idx, effect_idx] = 0  # Forbid the causal link
        # Update prior knowledge matrix with 1 for correct causal links
        for cause, effect in predefined_links:
            cause_idx = node_names.index(cause)
            effect_idx = node_names.index(effect)
            prior_knowledge[cause_idx, effect_idx] = 1  # Pre-define the causal link
        # Set up and fit causal model
        model_dling = DirectLiNGAM(prior_knowledge=prior_knowledge, apply_prior_knowledge_softly=apply_prior_knowledge_softly, measure=measure, random_state=seed_value)
        model_dling.fit(df_causal)
        # Get causal order and adjacency matrix
        causal_order = model_dling.causal_order_
        adjmatrix_dling = model_dling.adjacency_matrix_
        # Create directed graph using networkx
        G_dling = nx.DiGraph()
        # Add edges based on the adjacency matrix
        for i in range(adjmatrix_dling.shape[0]):
            for j in range(adjmatrix_dling.shape[1]):
                if np.abs(adjmatrix_dling[i, j]) > threshold_for_adjmatrix:  # If there's a causal link
                    G_dling.add_edge(node_names[i], node_names[j])

        # Causal discovery using FCI (with background knowledge)
        indep_test = 'kci'
        # indep_test = 'fisherz'
        # indep_test = 'rcit'
        alpha_fci = 0.05
        kernelX = 'Gaussian'
        kernelY = 'Gaussian'
        # Initialize graph with subset of dataset and PC algorithm
        bk = initialize_graph_for_background_knowledge(df_causal, forbidden_links=impossible_links, required_links=predefined_links)
        # Run FCI Algorithm with background knowledge
        G_fci, _ = fci(df_causal.values, alpha=alpha_fci, indep_test=indep_test, background_knowledge=bk, kernelX=kernelX, kernelY=kernelY, verbose=0, show_progress=False)


    # Show message when model is trained
    st.success('Model succesvol getraind')

    # Reset session state after training
    st.session_state.fit_model = False
    st.session_state.new_settings = False
    st.session_state.new_dataset = False


# Re-render all tables and figures
if st.session_state.render_figures:
    with st.spinner('Rendering figures...'):
        # Add metrics table to session state
        st.session_state.table_metrics = df_metrics.transpose()

        # Plot model performance
        fig = plot_model_performance(y_train.values, predict_train, y_test.values, predict_test, ncols=2, width=fig_width, height=fig_height, binsize=0.1)
        # Add model performance to session state
        st.session_state.fig_performance = fig

        # Plot Feature Importances
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        # Plot horizontal bar chart of feature importances
        ax.barh(sorted_feature_names, sorted_importances, height=0.7, color='royalblue')
        # Update axis
        ax.set_xlabel("Permutation Importance", fontsize=label_fontsize)
        ax.tick_params(axis='both', labelsize=tick_fontsize)  # Adjust label size
        ax.tick_params(axis='y', which='both', length=0)  # Make y-axis ticks invisible
        ax.tick_params(axis='x', width=0.5)  # Adjust width of x-axis ticks
        ax.spines['top'].set_visible(False)  # Remove top spine
        ax.spines['right'].set_visible(False)  # Remove right spine
        ax.spines['left'].set_linewidth(0.5)  # Adjust the thickness as desired
        ax.spines['bottom'].set_linewidth(0.5)  # Adjust the thickness as desired
        # Add importance to session state
        st.session_state.fig_importance = fig

        # Plot SHAP waterfall (single instance)
        fig = plt.figure()  # Create figure (to fill with waterfall_plot result)
        # Convert XGBoost SHAP values to float64 for consistency
        shap.waterfall_plot(explanation['explained_value'], max_display=len(explanation['feature_names']), show=False)  # generate SHAP plot without showing it
        # Adjust figure size
        fig = plt.gcf()
        fig.set_size_inches(fig_width, ((fig_height) / 8) * len(explanation['feature_names']) * 1.4)
        fig.patch.set_facecolor('xkcd:white')
        # Add explanation to session state
        st.session_state.fig_waterfall = fig
        

        # Plot SHAP beeswarm
        fig = plt.figure()  # Create figure (to fill with summary_plot result)
        shap.summary_plot(shap_values, X_shap[shap_feature_names], plot_type="dot", show=False)
        ax = plt.gca()
        ax.collections[0].set_sizes([6])
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        ax.set_xlabel("SHAP-waarde (invloed op modeluitvoer)", fontsize=label_fontsize)
        # change colorbar labels
        colorbar = plt.gcf().axes[-1]
        colorbar.set_ylabel('Relatieve waarde\nstuurvariabele', fontsize=tick_fontsize)
        colorbar.set_yticklabels(["Laag", "Hoog"], fontsize=tick_fontsize-2)
        # Change size of dots
        dot_size = 8
        ax = plt.gca()
        for collection in ax.collections:
            collection.set_sizes([dot_size])
        fig = plt.gcf()
        fig.set_size_inches(fig_width, fig_height)
        fig.patch.set_facecolor('xkcd:white')
        # Add explanation to session state
        st.session_state.fig_beeswarm = fig
        
        # Plot average SHAP Values
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        # Plot horizontal bar chart of SHAP values
        ax.barh(feature_names_sorted, mean_shap_values_sorted, height=0.7, color='royalblue')
        # Update axis
        ax.set_xlabel("Gemiddelde (|SHAP-waarde|) (gemiddelde invloed op modeluitvoer)", fontsize=label_fontsize)
        ax.tick_params(axis='both', labelsize=tick_fontsize)  # Adjust label size
        ax.tick_params(axis='y', which='both', length=0)  # Remove y-axis ticks
        ax.tick_params(axis='x', width=0.5)  # Adjust width of x-axis ticks
        ax.spines['top'].set_visible(False)  # Remove top spine
        ax.spines['right'].set_visible(False)  # Remove right spine
        ax.spines['left'].set_linewidth(0.5)  # Adjust the thickness as desired
        ax.spines['bottom'].set_linewidth(0.5)  # Adjust the thickness as desired
        # Add explanation to session state
        st.session_state.fig_avg_shap = fig
        
        # Plot SHAP dependence
        colors = ["firebrick", "darkorange", "lightgray", "royalblue", "darkblue"]
        cmap = LinearSegmentedColormap.from_list("blue_gray_red", colors)
        fig = shap_dependence_plots_with_target_as_legend(shap_values, X_shap, y_shap, shap_feature_names, share_yaxes=False, cmap=cmap)
        # Add explanation to session state
        st.session_state.fig_dependence = fig

        # Plot conformal intervals
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        # Ensure q_lower is always less than or equal to q_upper
        prediction_intervals[['q_lower', 'q_upper']] = np.sort(prediction_intervals[['q_lower', 'q_upper']].values, axis=1)
        # Calculate the bounds for the error bars
        lower_errors = prediction_intervals['y_pred'] - prediction_intervals['q_lower']
        upper_errors = prediction_intervals['q_upper'] - prediction_intervals['y_pred']
        lower_errors = np.where(lower_errors < 0, np.nan, lower_errors)
        upper_errors = np.where(upper_errors < 0, np.nan, upper_errors)
        # Plot true values and predicted median
        ax.plot(
            prediction_intervals.index, 
            prediction_intervals['y_new'], 
            'o', label='Waarneming', 
            color='royalblue', markersize=3, zorder=10
        )
        ax.plot(
            prediction_intervals.index, 
            prediction_intervals['y_pred'], 
            'x', label='Voorspelling',
            color='firebrick', markersize=4, zorder=20
        )
        # Plot error bars for conformal intervals
        ax.errorbar(
            prediction_intervals.index,
            prediction_intervals['y_pred'],
            yerr=[lower_errors, upper_errors],
            fmt='none', color='lightsteelblue', ecolor='lightsteelblue', capsize=2, linewidth=0.6,
            label=f"{conformal_interval}% Interval", zorder=0
        )
        # Update axis
        ax.set_ylabel(f"EKR {target_name.title()}", fontsize=label_fontsize*0.8)
        xlims = ax.get_xlim()
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels([])
        ax.set_xlim(xlims)
        # ax.legend(loc="upper right", fontsize=tick_fontsize)
        ax.tick_params(axis='x', which='both', length=0)  # Remove x-axis ticks
        ax.tick_params(axis='y', labelsize=tick_fontsize, width=0.5)  # Adjust label size
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=0.8*tick_fontsize, frameon=False)  # Put legend in the top center
        ax.spines['top'].set_visible(False)  # Remove top spine 
        ax.spines['right'].set_visible(False)  # Remove right spine
        ax.spines['left'].set_linewidth(0.5)  # Adjust the thickness as desired
        ax.spines['bottom'].set_linewidth(0.5)  # Adjust the thickness as desired
        # Add conformal result to session state
        st.session_state.fig_intervals = fig

        # # Plot causal graph
        # fig, ax = plt.subplots(figsize=(12,6), ncols=2)
        # ax[0] = plot_causal_graph(
        #     G_fci, 
        #     node_names=node_names, 
        #     ax=ax[0],
        #     ci_type='fci',
        #     color_direction=False, 
        #     show_edge_width=False,
        #     max_line_width=5,
        #     pyd_layout='dot',
        #     title="Fast Causal Inference (FCI)",
        #     fontsize=11,
        #     bbox_to_anchor=(1.,1.),
        # )
        # if G_dling.number_of_edges() > 0:
        #     ax[1] = plot_causal_graph(
        #         G_dling, 
        #         node_names=node_names,
        #         ax=ax[1],
        #         ci_type='directlingam',
        #         adj_matrix=adjmatrix_dling,
        #         color_direction=True, 
        #         show_edge_width=True,
        #         max_line_width=5,
        #         pyd_layout='dot',
        #         title="DirectLiNGAM",
        #         fontsize=11,
        #         bbox_to_anchor=(1.,1.),
        #     )
        # else:
        #     st.write("No causal links detected with DirectLiNGAM")
        # ax[0].axis('off')
        # ax[1].axis('off')
        # # Add causal graphs to session state
        # st.session_state.fig_causal = fig
        
        # Reset session state
        st.session_state.render_figures = False


# Show results
# Display metadata
if st.session_state.table_metadata is not None:
    st.write("")
    st.write("")
    if len(st.session_state.table_metadata) is not None:
        st.dataframe(st.session_state.table_metadata, width=1000,)

# Display metrics with markdown
if st.session_state.table_metrics is not None:
    st.write("")
    st.write("")
    st.write("### Model Prestaties")
    st.markdown(
        "<p style='font-size:14px;'><b>RMSE (Root Mean Squared Error):</b> De gemiddelde grootte van de kwadratische fouten. Grotere fouten werken harder door in het gemiddelde doordat ze gekwadrateerd worden. Een lagere waarde betekent een betere nauwkeurigheid. </p>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='font-size:14px;'><b>MAE (Mean Absolute Error):</b> De gemiddelde absolute fout tussen voorspelde en waargenomen waarden, zonder onderscheid tussen positieve of negatieve afwijkingen. Een lagere waarde betekent een betere nauwkeurigheid. </p>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='font-size:14px;'><b>CCC (Concordance Correlation Coefficient):</b> Combineert correlatie en verschillen in gemiddelden om de overeenstemming tussen voorspelde en waargenomen waarden te meten. Een waarde dichtbij 1 duidt op sterke overeenstemming, een waarde rond 0 op zwakke of geen overeenkomst en een waarde dichtbij -1 op een sterke negatieve relatie, waarbij hogere voorspelde waarden corresponderen met lagere waargenomen waarden, en omgekeerd. </p>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='font-size:14px;'><b>R² (R-squared):</b> Laat zien welk percentage van de variatie in de waarnemingen kan worden verklaard met het model. Een hoger percentage betekent een betere nauwkeurigheid.</p>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='font-size:14px;'><b>+/- 0,1:</b> Het percentage voorspellingen dat binnen een bandbreedte van 0,2 (dus ± 0,1) van de waargenomen waarde ligt. Hogere percentages duiden op betere nauwkeurigheid.</p>",
        unsafe_allow_html=True
    )
    st.write("")
    st.dataframe(st.session_state.table_metrics)
    st.write("")

# Display model performance with markdown
if st.session_state.fig_performance is not None:
    st.markdown(
        "<p style='font-size:14px;'>De <b>linker grafiek</b> toont de relatie tussen de voorspelde scores van het model en de werkelijke EKR-scores voor zowel de trainings- als testdata. De blauwe punten vertegenwoordigen de trainingsdata en de oranje punten de testdata. De diagonale lijn geeft de ideale situatie weer waarin de voorspelde score precies gelijk is aan de werkelijke score. De grijze band rond de lijn toont een foutmarge van ±0,1.</p>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='font-size:14px;'>De <b>rechter grafiek</b> toont de verdeling van de voorspellingsfouten per EKR-scorecategorie voor de testdata. Elke balk vertegenwoordigt het aantal voorspellingen binnen een specifieke EKR-scoregroep. De kleuren geven de nauwkeurigheid van de voorspelling weer: oranje balken staan voor voorspellingen met een foutmarge binnen ±0,1, groene balken tonen overschattingen van meer dan 0,1, en blauwe balken geven onderschattingen van meer dan -0,1. Deze grafiek laat zien hoe het model presteert over verschillende EKR-klassen en waar eventuele afwijkingen optreden.</p>",
        unsafe_allow_html=True
    )
    st.pyplot(st.session_state.fig_performance)

# Display feature importances with markdown
if st.session_state.fig_importance is not None:
    st.write("")
    st.write("")
    st.write("### Feature Importance")
    st.markdown(
        "<p style='font-size:14px;'>Feature importance laat zien welke stuurvariabelen het meest bijdragen aan de voorspellingen van het model. Hoe groter het belang van een stuurvariabelen, hoe meer impact het heeft op de uitkomst. Dit helpt om te begrijpen welke factoren cruciaal zijn voor de voorspelling en waar we ons op moeten richten.</p>",
        unsafe_allow_html=True
    )
    st.pyplot(st.session_state.fig_importance)

# Display SHAP waterfall (single instance) with markdown
if st.session_state.fig_waterfall is not None:
    explanation = st.session_state.explanation
    st.write("")
    st.write("")
    st.write("### SHAP Waarden")
    st.markdown(
        "<p style='font-size:14px;'>SHAP-waarden maken inzichtelijk hoe het model tot zijn voorspelling komt door de bijdrage van individuele stuurvariabelen te kwantificeren. Deze waarden geven zowel de richting (positief of negatief) als de sterkte van de invloed van een stuurvariabele op de voorspelling weer. Dit helpt om te begrijpen welke stuurvariabelen een grote rol spelen bij de voorspelling en in welke richting zij deze beïnvloeden. Onderstaand diagram illustreert hoe de stuurvariabelen voor één specifieke waarneming bijdragen aan de voorspelling van het model.</p>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<p style='font-size:14px;'>De volgende visualisatie helpt om de impact van stuurvariabelen beter te begrijpen. Het diagram wordt van onderaf gelezen. De voorspelling begint met een basiswaarde: de gemiddelde uitkomst van het model (<b><i>E[f(X)]</i> = {explanation['expected_value']:.3f}</b>). Vervolgens worden de stuurvariabelen weergegeven in volgorde van hun bijdrage aan de voorspelling, van klein naar groot. Stuurvariabelen die de voorspelling verhogen, zijn weergegeven met rode pijlen naar rechts, terwijl stuurvariabelen die de voorspelling verlagen, zijn weergegeven met blauwe pijlen naar links. De lengte van een pijl geeft de grootte van de bijdrage aan. Helemaal bovenaan staat de uiteindelijke voorspelde EKR-score (<b><i>f(X)</i> = {explanation['y_pred']:.3f}</b>). Dit is de som van de verwachtingswaarde en alle SHAP-bijdragen samen. De voorspelde EKR-score is {np.abs(explanation['error']):.3f} (={(100 * np.abs(explanation['error']) / explanation['y_true']):.1f}%) {explanation['error_text']} dan de waargenomen waarde (<b><i>y</i> = {explanation['y_true']:.3f}</b>).</p>",
        unsafe_allow_html=True
    )
    st.pyplot(st.session_state.fig_waterfall)

# Display SHAP beeswarm with markdown
if st.session_state.fig_beeswarm is not None:
    st.write("")
    st.write("")
    st.markdown(
        "<p style='font-size:14px;'>De <b>'beeswarm'-plot</b> laat zien hoe consistent of variabel de stuurvariabelen bijdragen aan de voorspellingen van het model. Elke stip in de grafiek vertegenwoordigt de SHAP-waarde voor een specifieke waarneming. De x-as toont de grootte van de bijdrage, waarbij negatieve waarden een verlagende invloed hebben en positieve waarden een verhogende invloed. De kleur van de stip geeft de relatieve waarde van de stuurvariabele weer: blauw staat voor lage waarden, en rood voor hoge waarden. Deze visualisatie helpt om inzicht te krijgen in de rol van stuurvariabelen en hoe hun waarde de richting en sterkte van de voorspelling beïnvloedt.</p>",
        unsafe_allow_html=True
    )
    st.pyplot(st.session_state.fig_beeswarm)

# Display average SHAP with markdown
if st.session_state.fig_avg_shap is not None:
    st.write("")
    st.write("")
    st.markdown(
        "<p style='font-size:14px;'><b>Gemiddelde SHAP-waarden</b>. Door de absolute waarde van de SHAP-waarden per stuurvariabele over alle waarnemingen te middelen, wordt duidelijk welke stuurvariabelen gemiddeld het meest bepalend zijn voor het model. Hiermee wordt de bijdrage van de stuurvariabelen per punt vertaald naar een gemiddelde voor de hele dataset. De volgorde van de stuurvariabelen in deze figuur is meestal redelijk vergelijkbaar met de 'Feature Importance' van het model.</p>",
        unsafe_allow_html=True
    )
    st.pyplot(st.session_state.fig_avg_shap)

# Display SHAP dependence with markdown
if st.session_state.fig_dependence is not None:
    st.write("")
    st.write("")
    # st.write("### SHAP Depencence Plots")
    st.markdown(
        "<p style='font-size:14px;'><b>SHAP-dependence plots</b> laten op een gedetailleerde manier zien hoe elke waarde van een stuurvariabele bijdraagt aan de voorspelde uitkomst. Dit geeft een indruk van de 'kennisregels' die door het model uit de data zijn gedestilleerd. De kleur van de punten weerspiegelt de waarde van de  EKR score, waardoor het effect van de stuurvariabele op de voorspelling eenvoudig en inzichtelijk wordt. Deze visualisatie helpt om trends en patronen in de invloed van stuurvariabelen beter te begrijpen. </p>",
        unsafe_allow_html=True
    )
    st.pyplot(st.session_state.fig_dependence)

# Display conformal intervals with markdown
if st.session_state.fig_intervals is not None:
    st.write("")
    st.write("")
    st.write("### Voorspellingsintervallen")
    st.markdown(
        "<p style='font-size:14px;'>Voorspellingsintervallen geven een betrouwbaarheidsinterval rond de voorspellingen van een model. Ze bieden een schatting van de onzekerheid, zodat je kunt zien binnen welke grenzen de werkelijke waarde waarschijnlijk zal vallen. Bijvoorbeeld, een 80% voorspellingsinterval betekent dat 80% van de werkelijke waarden binnen dit interval vallen. Dit helpt om niet alleen een voorspelling te maken, maar ook de nauwkeurigheid en betrouwbaarheid ervan te beoordelen.</p>",
        unsafe_allow_html=True
    )
    st.pyplot(st.session_state.fig_intervals)

# Display causal graphs with markdown
if st.session_state.fig_causal is not None:
    st.write("")
    st.write("")
    st.write("### Causale Relaties")
    st.markdown(
        "<p style='font-size:14px;'>Causale relaties helpen om te begrijpen welke variabelen direct invloed hebben op andere variabelen. In tegenstelling tot correlaties, die alleen samenhang tonen, laten causale relaties zien welke variabelen veranderingen veroorzaken. Dit is essentieel voor het nemen van effectieve beslissingen, omdat het inzicht geeft in onderliggende oorzaken. De weergegeven relaties zijn afgeleid uit de beschikbare data met behulp van twee verschillende methoden: FCI en DirectLiNGAM.</p>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='font-size:14px;'><b>FCI</b> ontdekt causale verbanden, zelfs als er variabelen ontbreken (niet gemeten zijn). Het maakt een overzicht van mogelijke oorzaken, maar houdt ook rekening met onzekerheden, bijvoorbeeld als de richting van invloed niet zeker is. FCI gebruikt statistische testen om verbanden uit te sluiten.Het eindresultaat is een zogenaamde Partial Ancestral Graph (PAG), waarin zowel directe als indirecte causale verbanden (en eventuele onzekerheden) worden weergegeven.</p>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='font-size:14px;'><b>DirectLiNGAM</b> vindt directe oorzaken in datasets uitgaand van lineaire verbanden, waarbij de data niet normaal verdeeld hoeft te zijn (bijv. scheef of asymmetrisch). Het werkt alleen als er geen kringetjes (cycli) in de oorzaken zitten; elke oorzaak leidt in één richting naar één gevolg. DirectLiNGAM gebruikt patronen in de data om de richting van oorzaken en gevolgen te bepalen en laat alleen de directe relaties zien.</p>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='font-size:14px;'>Beide grafen tonen causale relaties, maar met een andere focus. De <b>DirectLiNGAM</b>-graaf geeft een overzicht van directe verbanden, waarbij de dikte van de pijlen de sterkte van de invloed aangeeft. De <b>FCI</b>-graaf biedt een breder beeld door ook mogelijke latente variabelen en niet-lineaire verbanden te tonen, maar kan bij kleinere datasets, zoals de KRW-V data, minder zekerheid bieden. Overeenkomsten tussen de grafen wijzen op sterke en betrouwbare relaties. <b>DirectLiNGAM</b> is praktischer voor directe invloeden, terwijl <b>FCI</b> aanvullend inzicht biedt in niet-lineaire en indirecte effecten, evenals verborgen variabelen.</p>",
        unsafe_allow_html=True
    )
    st.pyplot(st.session_state.fig_causal)
