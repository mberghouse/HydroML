


from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, RidgeCV, Lasso, LassoLars, BayesianRidge, TweedieRegressor, SGDRegressor, SGDClassifier, Perceptron, TheilSenRegressor, HuberRegressor, ElasticNet, OrthogonalMatchingPursuit
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch, DBSCAN, BisectingKMeans, MiniBatchKMeans, SpectralClustering
#from sklearn.cluster import affinity_propagation, cluster_optics_dbscan, cluster_optics_xi, dbscan, estimate_bandwidth, k_means, kmeans_plusplus, mean_shift, spectral_clustering, ward_tree
#from sklearn.covariance import EmpiricalCovariance, GraphicalLasso, GraphicalLassoCV, MinCovDet, OAS, empirical_covariance, graphical_lasso, ledoit_wolf, oas
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
    
        self.decomp_models = {
        "CCA":CCA,
        "PLSCanonical":PLSCanonical,
        "PLSRegression":PLSRegression,
        "PLSSVD":PLSSVD,
        "FactorAnalysis":FactorAnalysis,
        "FastICA":FastICA,
        "IncrementalPCA":IncrementalPCA,
        "MiniBatchSparsePCA":MiniBatchSparsePCA,  
        "PCA":PCA
        }     
        
        self.cluster_models = {
        "KMeans": KMeans,
        "AgglomerativeClustering": AgglomerativeClustering,
        "Birch": Birch,
        "DBSCAN": DBSCAN,
        "BisectingKMeans": BisectingKMeans,
        "SpectralClustering": SpectralClustering,
        "MiniBatchKMeans": MiniBatchKMeans
        }
        
        self.regression_models = {
        "BaggingRegressor": BaggingRegressor,
        "ExtraTreesRegressor":ExtraTreesRegressor,
        "GradientBoostingRegressor": GradientBoostingRegressor,
        "StackingRegressor": StackingRegressor,
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
        "VotingRegressor": VotingRegressor,
        "MLPRegressor": MLPRegressor,
        "Ridge": Ridge, 
        "RidgeCV": RidgeCV, 
        "Lasso": Lasso, 
        "LassoLars": LassoLars,
        "BayesianRidge": BayesianRidge, 
        "TweedieRegressor": TweedieRegressor, 
        "SGDRegressor": SGDRegressor,        
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
            "BernoulliNB": BernoulliNB,
            "MLPClassifier": MLPClassifier,
            "BernoulliRBM": BernoulliRBM,
            "CategoricalNB":CategoricalNB,
            "ComplementNB": ComplementNB,
            "GaussianNB": GaussianNB,
            "MultinomialNB": MultinomialNB,
            "BaggingClassifier": BaggingClassifier,
            "ExtraTreesClassifier": ExtraTreesClassifier,
            "GradientBoostingClassifier": GradientBoostingClassifier,
            "StackingClassifier": StackingClassifier,
            "VotingClassifier": VotingClassifier,
            "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
            "RandomForestClassifier": RandomForestClassifier,
            "LogisticRegression": LogisticRegression,
            "SVC": SVC,
            "KNeighborsClassifier": KNeighborsClassifier,
            "SGDClassifier": SGDClassifier
        }
        
        self.model_parameters = {
        "CCA": {
            "n_components": 2,
            "max_iter": 500,
            "tol": 1e-06,
        },
        "PLSCanonical": {
            "n_components": 2,
            "algorithm": "nipals",
            "max_iter": 500,
            "tol": 1e-06,
        },
        "PLSRegression": {
            "n_components": 2,
            "max_iter": 500,
            "tol": 1e-06,
        },
        "PLSSVD": {
            "n_components": 2,
            "algorithm": "svd"
        },
        "FactorAnalysis": {
            "tol": 1e-2,
            "max_iter": 1000,
            "svd_method": "randomized",
            "iterated_power": 3,
            "random_state": 0
        },
        "FastICA": {
            "algorithm": "parallel",
            "fun": "logcosh",
            "max_iter": 200,
            "tol": 1e-4
        },
        "IncrementalPCA": {
            
        },
        "MiniBatchSparsePCA": {
            "alpha": 1,
            "ridge_alpha": 0.01,
            "n_iter": 100,
            "batch_size": 3,
            "method": "lars",
        },
        "PCA": {
            "svd_solver": "auto",
            "tol": 1e-3
        },
        "KMeans": {
            "n_clusters": 8,
            "init": "k-means++",
            "n_init": 10,
            "max_iter": 300,
            "tol": 1e-4,
            "verbose": 0,
            "algorithm": "auto"
        },
        "AgglomerativeClustering": {
            "n_clusters": 2,
            "affinity": "euclidean",
            "linkage": "ward"
        },
        "Birch": {
            "threshold": 0.5,
            "branching_factor": 50,
            "n_clusters": 3,

        },
        "DBSCAN": {
            "eps": 0.5,
            "min_samples": 5,
            "metric": "euclidean",
            "algorithm": "auto",
            'n_jobs':-1,
            "leaf_size": 30
        },
        "BisectingKMeans": {
            "n_clusters": 8,
            "init": "k-means++",
            "n_init": 1,
            "max_iter": 300,
            "tol": 1e-4,
            'n_jobs':-1,
            "verbose": 0,
            "bisecting_strategy": "biggest_inertia"
        },
        "SpectralClustering": {
            "n_clusters": 8,
            "n_init": 10,
            "gamma": 1.0,
            "affinity": "rbf",
            "n_neighbors": 10,
            'n_jobs':-1,
            "assign_labels": "kmeans",
            "degree": 3,
            "coef0": 1,
        },
        "MiniBatchKMeans": {
            "n_clusters": 8,
            "init": "k-means++",
            "max_iter": 100,
            "batch_size": 100,
            "verbose": 0,
            "tol": 0.0,
            "max_no_improvement": 10,
            "n_init": 3,
            'n_jobs':-1,
            "reassignment_ratio": 0.01
        },
        "BaggingRegressor": {
            "n_estimators": 10,
            "max_samples": 1.0,
            "max_features": 1.0,
            'n_jobs':-1,

            "verbose": 0
        },
        "ExtraTreesRegressor": {
            "n_estimators": 100,
            'min_impurity_decrease':0.0,
            'n_jobs':-1,
            "criterion": "squared_error",
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "min_weight_fraction_leaf": 0.0,
            "max_features": 1.0,

        },
        "GradientBoostingRegressor": {
            "loss": "squared_error",
            "learning_rate": 0.1,
            "n_estimators": 100,
            "subsample": 1.0,
            "criterion": "friedman_mse",
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "min_weight_fraction_leaf": 0.0,
            "max_depth": 3,
            "alpha": 0.9,
            "verbose": 0,
            "validation_fraction": 0.1,
            "tol": 1e-4
        },
        "StackingRegressor": {
            "verbose": 0,
        },
        "HistGradientBoostingRegressor": {
            "loss": "squared_error",
            "learning_rate": 0.1,
            "max_iter": 100,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "l2_regularization": 0.0,
            "max_bins": 255,
            "early_stopping": "auto",
            "scoring": "loss",
            "validation_fraction": 0.1,
            "n_iter_no_change": 10,
            "tol": 1e-7
        },
        "VotingRegressor": {
        },
        "MLPRegressor": {
            "hidden_layer_sizes": 100,
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.0001,
            "batch_size": "auto",
            "learning_rate": "constant",
            "learning_rate_init": 0.001,
            "power_t": 0.5,
            "max_iter": 200,
            "tol": 1e-4,
            "momentum": 0.9,
            "validation_fraction": 0.1,
            "beta_1": 0.9,
            "beta_2": 0.999,
            "epsilon": 1e-8,
            "n_iter_no_change": 10
        },
        "BernoulliNB": {
            "alpha": 1.0,
            "binarize": 0.0,

        },
        "MLPClassifier": {
            "hidden_layer_sizes": 100,
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.0001,
            "batch_size": "auto",
            "learning_rate": "constant",
            "learning_rate_init": 0.001,
            "power_t": 0.5,
            "max_iter": 200,
            "tol": 1e-4,
            "momentum": 0.9,
            "validation_fraction": 0.1,
            "beta_1": 0.9,
            "beta_2": 0.999,
            "epsilon": 1e-8,
            "n_iter_no_change": 10
        },
        "BernoulliRBM": {
            "n_components": 256,
            "learning_rate": 0.1,
            "batch_size": 10,
            'n_jobs':-1,
            "n_iter": 10,
            "verbose": 0
        },
        "CategoricalNB": {
            "alpha": 1.0,
        },
        "ComplementNB": {
            "alpha": 1.0,
        },
        "GaussianNB": {
            "var_smoothing": 1e-9
        },
        "MultinomialNB": {
            "alpha": 1.0,
        },
        "BaggingClassifier": {
            "n_estimators": 10,
            "max_samples": 1.0,
            'n_jobs':-1,
            "max_features": 1.0,
            "verbose": 0
        },
        "ExtraTreesClassifier": {
            "n_estimators": 100,
            "criterion": "gini",
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            'n_jobs':-1,
            "min_weight_fraction_leaf": 0.0,
            "max_features": 1.0,
            "verbose": 0,
        },
        "GradientBoostingClassifier": {
            "loss": "deviance",
            "learning_rate": 0.1,
            "n_estimators": 100,
            "subsample": 1.0,
            "criterion": "friedman_mse",
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "min_weight_fraction_leaf": 0.0,
            "max_depth": 3,
            "verbose": 0,
            "validation_fraction": 0.1,
            "tol": 1e-4,
        },
        "StackingClassifier": {
            "verbose": 0,
        },
        "VotingClassifier": {
        },
        "HistGradientBoostingClassifier": {
            "loss": "log_loss",
            "learning_rate": 0.1,
            "max_iter": 100,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "l2_regularization": 0.0,
            "max_bins": 255,
            "early_stopping": "auto",
            "scoring": "loss",
            "validation_fraction": 0.1,
            "n_iter_no_change": 10,
            "tol": 1e-7
        },
        "RandomForestRegressor": {
        "n_estimators": 100,
        "max_depth": 100,
        'min_weight_fraction_leaf':0.0, 
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        'n_jobs':-1,
        },
        "Ridge": {
            "alpha": 1.0,
            "max_iter": 1000,
            "tol": 0.0001,
            "solver": "auto",
            "random_state": 42
        },
        "RidgeCV": {
        },
        "Lasso": {
            "alpha": 1.0,
            "max_iter": 1000,
            "tol": 0.0001,
            "random_state": 42,
            "selection": "cyclic"
        },
        "LassoLars": {
            "alpha": 1.0,
            "max_iter": 500,
            "eps": 2.220446049250313e-16,
            "random_state": 42
        },
        "BayesianRidge": {
            "n_iter": 300,
            "tol": 0.001,
            "alpha_1": 1e-06,
            "alpha_2": 1e-06,
            "lambda_1": 1e-06,
            "lambda_2": 1e-06,

        },
        "TweedieRegressor": {
            "power": 0.0,
            "alpha": 1.0,
            "max_iter": 100,
            "tol": 0.0001,
    
        },
        "SGDRegressor": {
            "loss": "squared_error",
            "penalty": "l2",
            "alpha": 0.0001,
            "l1_ratio": 0.15,
            "max_iter": 1000,
            "tol": 0.001,
            "verbose": 0,
            "epsilon": 0.1,
            "random_state": 42,
            "eta0": 0.01,
            "power_t": 0.25,
            "validation_fraction": 0.1,
            "n_iter_no_change": 5,

        },
        "SGDClassifier": {
            "loss": "hinge",
            "penalty": "l2",
            "alpha": 0.0001,
            "l1_ratio": 0.15,
            "max_iter": 1000,
            "tol": 0.001,
            "verbose": 0,
            "epsilon": 0.1,
            "random_state": 42,
            "eta0": 0.0,
            "power_t": 0.5,
            "validation_fraction": 0.1,
            "n_iter_no_change": 5,
        },
        "Perceptron": {
            "alpha": 0.0001,
            "max_iter": 1000,
            "tol": 0.001,
            "verbose": 0,
            "eta0": 1.0,
            'n_jobs':-1,
            "random_state": 0,
            "validation_fraction": 0.1,
            "n_iter_no_change": 5,
        },
        "TheilSenRegressor": {
            "max_subpopulation": 10000,
            "max_iter": 300,
            "tol": 0.001,
            'n_jobs':-1,
        },
        "HuberRegressor": {
            "epsilon": 1.35,
            "max_iter": 100,
            "alpha": 0.0001,
            "tol": 1e-05
        },
        "ElasticNet": {
            "alpha": 1.0,
            "l1_ratio": 0.5,
            "max_iter": 1000,
            "tol": 0.0001,
            "random_state": 42,
            "selection": "cyclic"
        },
        "OrthogonalMatchingPursuit": {

            "precompute": "auto"
        },
            "SVR": {
                "kernel": "rbf",
                "C": 1.0,
                "epsilon": 0.1
            },
            "KNeighborsRegressor": {
                "n_neighbors": 5,
                'n_jobs':-1,
                "weights": "uniform",
                "algorithm": "auto"
            },
            "XGBRegressor": {
                "n_estimators": 100,
                "eta": 0.3,
                "gamma": 0.0,
                "lambda": 1.0,
                'n_jobs':-1,
                "alpha": 0.0,
                "min_child_weight" : 1.0,
                "learning_rate": 0.1,
                "max_depth": 3,
                "subsample": 1.0,
                "colsample_bytree": 1.0
            },
            "RandomForestClassifier": {
                "n_estimators": 100,
                "max_depth": 100,
                'n_jobs':-1,
                "min_samples_split": 2,
                "min_samples_leaf": 1
            },
            "LogisticRegression": {
                "penalty": "l2",
                "C": 1.0,
                'n_jobs':-1,
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
                'n_jobs':-1,
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

