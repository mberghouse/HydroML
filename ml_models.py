


from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, RidgeCV, Lasso, LassoLars, BayesianRidge, TweedieRegressor, SGDRegressor, SGDClassifier, Perceptron, TheilSenRegressor, HuberRegressor, ElasticNet, OrthogonalMatchingPursuit
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch, DBSCAN, BisectingKMeans, MiniBatchKMeans, SpectralClustering
from sklearn.cluster import affinity_propagation, cluster_optics_dbscan, cluster_optics_xi, dbscan, estimate_bandwidth, k_means, kmeans_plusplus, mean_shift, spectral_clustering, ward_tree
from sklearn.compose import ColumnTransformer
from sklearn.covariance import EmpiricalCovariance, GraphicalLasso, GraphicalLassoCV, MinCovDet, OAS, empirical_covariance, graphical_lasso, ledoit_wolf, oas
from sklearn.cross_decomposition import CCA, PLSCanonical, PLSRegression, PLSSVD
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.decomposition import FactorAnalysis, FastICA, IncrementalPCA, MiniBatchSparsePCA, PCA
from xgboost import XGBRegressor as xgbr

from sklearn.naive_bayes import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.neural_network import BernoulliRBM, MLPClassifier, MLPRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor, ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor, RandomForestRegressor, RandomForestClassifier, StackingClassifier, StackingRegressor, VotingClassifier, VotingRegressor, HistGradientBoostingRegressor,HistGradientBoostingClassifier

class ModelSelector:
    def __init__(self):
        self.regression_models = {
        "Ridge": Ridge, 
        "RidgeCV": RidgeCV, 
        "Lasso": Lasso, 
        "LassoLars": LassoLars,
        "BayesianRidge": BayesianRidge, 
        "TweedieRegressor": TweedieRegressor, 
        "SGDRegressor": SGDRegressor, 
        "SGDClassifier": SGDClassifier, 
        "Perceptron": Perceptron, 
        "TheilSenRegressor": TheilSenRegressor,
        "HuberRegressor": HuberRegressor,
        "ElasticNet": ElasticNet, 
        "OrthogonalMatchingPursuit": OrthogonalMatchingPursuit,
            "RandomForestRegressor": RandomForestRegressor,
            "LinearRegression": LinearRegression,
            "SVR": SVR,
            "KNeighborsRegressor": KNeighborsRegressor,
            "XGBRegressor": xgbr
        }
        self.classification_models = {
            "RandomForestClassifier": RandomForestClassifier,
            "LogisticRegression": LogisticRegression,
            "SVC": SVC,
            "KNeighborsClassifier": KNeighborsClassifier
        }
        self.model_parameters = {
        "RandomForestRegressor": {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1
        },
        "Ridge": {
            "alpha": 1.0,
            "fit_intercept": True,
            "copy_X": True,
            "max_iter": 1000,
            "tol": 0.0001,
            "solver": "auto",
            "random_state": 42
        },
        "RidgeCV": {
            "alphas": (0.1, 1.0, 10.0),
            "fit_intercept": True,
            "scoring": None,
            "cv": None,
            "gcv_mode": None,
            "store_cv_values": False
        },
        "Lasso": {
            "alpha": 1.0,
            "fit_intercept": True,
            "precompute": False,
            "copy_X": True,
            "max_iter": 1000,
            "tol": 0.0001,
            "warm_start": False,
            "positive": False,
            "random_state": 42,
            "selection": "cyclic"
        },
        "LassoLars": {
            "alpha": 1.0,
            "fit_intercept": True,
            "verbose": False,
            "precompute": "auto",
            "max_iter": 500,
            "eps": 2.220446049250313e-16,
            "copy_X": True,
            "fit_path": True,
            "positive": False,
            "jitter": None,
            "random_state": 42
        },
        "BayesianRidge": {
            "n_iter": 300,
            "tol": 0.001,
            "alpha_1": 1e-06,
            "alpha_2": 1e-06,
            "lambda_1": 1e-06,
            "lambda_2": 1e-06,
            "compute_score": False,
            "fit_intercept": True,
            "copy_X": True,
            "verbose": False
        },
        "TweedieRegressor": {
            "power": 0.0,
            "alpha": 1.0,
            "fit_intercept": True,
            "link": "auto",
            "max_iter": 100,
            "tol": 0.0001,
            "warm_start": False,
            "verbose": False
        },
        "SGDRegressor": {
            "loss": "squared_error",
            "penalty": "l2",
            "alpha": 0.0001,
            "l1_ratio": 0.15,
            "fit_intercept": True,
            "max_iter": 1000,
            "tol": 0.001,
            "shuffle": True,
            "verbose": 0,
            "epsilon": 0.1,
            "random_state": 42,
            "learning_rate": "invscaling",
            "eta0": 0.01,
            "power_t": 0.25,
            "early_stopping": False,
            "validation_fraction": 0.1,
            "n_iter_no_change": 5,
            "warm_start": False,
            "average": False
        },
        "SGDClassifier": {
            "loss": "hinge",
            "penalty": "l2",
            "alpha": 0.0001,
            "l1_ratio": 0.15,
            "fit_intercept": True,
            "max_iter": 1000,
            "tol": 0.001,
            "shuffle": True,
            "verbose": 0,
            "epsilon": 0.1,
            "n_jobs": 2,
            "random_state": 42,
            "learning_rate": "optimal",
            "eta0": 0.0,
            "power_t": 0.5,
            "early_stopping": False,
            "validation_fraction": 0.1,
            "n_iter_no_change": 5,
            "class_weight": None,
            "warm_start": False,
            "average": False
        },
        "Perceptron": {
            "penalty": None,
            "alpha": 0.0001,
            "fit_intercept": True,
            "max_iter": 1000,
            "tol": 0.001,
            "shuffle": True,
            "verbose": 0,
            "eta0": 1.0,
            "n_jobs": 2,
            "random_state": 0,
            "early_stopping": False,
            "validation_fraction": 0.1,
            "n_iter_no_change": 5,
            "class_weight": None,
            "warm_start": False
        },
        "TheilSenRegressor": {
            "fit_intercept": True,
            "copy_X": True,
            "max_subpopulation": 10000,
            "n_subsamples": None,
            "max_iter": 300,
            "tol": 0.001,
            "random_state": None,
            "n_jobs": 2,
            "verbose": False
        },
        "HuberRegressor": {
            "epsilon": 1.35,
            "max_iter": 100,
            "alpha": 0.0001,
            "warm_start": False,
            "fit_intercept": True,
            "tol": 1e-05
        },
        "ElasticNet": {
            "alpha": 1.0,
            "l1_ratio": 0.5,
            "fit_intercept": True,
            "precompute": False,
            "max_iter": 1000,
            "copy_X": True,
            "tol": 0.0001,
            "warm_start": False,
            "positive": False,
            "random_state": 42,
            "selection": "cyclic"
        },
        "OrthogonalMatchingPursuit": {
            "n_nonzero_coefs": None,
            "tol": None,
            "fit_intercept": True,
            "precompute": "auto"
        },
            "SVR": {
                "kernel": "rbf",
                "C": 1.0,
                "epsilon": 0.1
            },
            "KNeighborsRegressor": {
                "n_neighbors": 5,
                "weights": "uniform",
                "algorithm": "auto"
            },
            "XGBRegressor": {
                "n_estimators": 100,
                "eta": 0.3,
                "gamma": 0.0,
                "lambda": 1.0,
                "alpha": 0.0,
                "min_child_weight" : 1.0,
                "learning_rate": 0.1,
                "max_depth": 3,
                "subsample": 1.0,
                "colsample_bytree": 1.0
            },
            "RandomForestClassifier": {
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1
            },
            "LogisticRegression": {
                "penalty": "l2",
                "C": 1.0,
                "solver": "lbfgs",
                "max_iter": 100
            },
            "SVC": {
                "C": 1.0,
                "kernel": "rbf",
                "degree": 3,
                "gamma": "scale"
            },
            "KNeighborsClassifier": {
                "n_neighbors": 5,
                "weights": "uniform",
                "algorithm": "auto"
            }
        }

    def get_model(self, model_name):
        if model_name in self.regression_models:
            return self.regression_models[model_name]
        elif model_name in self.classification_models:
            return self.classification_models[model_name]
        else:
            raise ValueError(f"Model '{model_name}' not found.")

    def get_model_parameters(self, model_name):
        if model_name in self.model_parameters:
            return self.model_parameters[model_name]
        else:
            raise ValueError(f"Parameters for model '{model_name}' not found.")

