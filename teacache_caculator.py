
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error


def get_L1_relative_dis(prev, current):
    return np.mean(np.abs(prev - current).mean() / np.abs(prev).mean())

class TeaCacheCoefficientCaculator:
    def __init__(self, num_timesteps = 28):
        self.candidate_features = {}
        self.y_values = []

    def add_feature(self, feature_name : str, feature_value : float):
        if feature_name not in self.candidate_features:
            self.candidate_features[feature_name] = [feature_name]
        pass
    def add_target_noise_residual(self, target_noise_residual : float):
        self.y_values.append(target_noise_residual)

    def calculate_coefficients(self, order=4):
        _, key = self.analyze_trend_similarity(self.candidate_features, self.y_values)
        return np.polyfit(self.candidate_features[key], self.y_values, order)
    
    def analyze_trend_similarity(self, x_variables, y_variable):

        results = {}
        
        for name, x_var in x_variables.items():
            x = []
            corr, _ = pearsonr(x_var, y_variable)
            
            
            results[name] = {'correlation': corr}
            print(f" [{name}]'s pearsonr correlation: {corr:.4f}")

        
        best_corr_name = max(results, key=lambda name: abs(results[name]['correlation']))
        
        return results, best_corr_name