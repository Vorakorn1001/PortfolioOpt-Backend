from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np

class optimizeService:
    def __init__(self):
        pass

    def optimizeFixedReturn(self, target_return, mean_returns, cov_matrix):
        if isinstance(mean_returns, dict):
            mean_returns = list(mean_returns.values())

        num_assets = len(mean_returns)
        args = (cov_matrix,)

        constraints = (
            {'type': 'eq', 'fun': lambda weights: self.getPortfolioReturn(weights, mean_returns) - target_return},
            {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
        )

        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_weights = num_assets * [1. / num_assets,]

        result = minimize(self.getPortfolioVariance, initial_weights, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x
    
    def optimizeFixedVariance(self, target_variance, mean_returns, cov_matrix):
        if isinstance(mean_returns, dict):
            mean_returns = list(mean_returns.values())
        num_assets = len(mean_returns)
        args = (mean_returns,)

        def objective(weights, mean_returns):
            return -self.getPortfolioReturn(weights, mean_returns)

        constraints = (
            {'type': 'eq', 'fun': lambda weights: self.getPortfolioVariance(weights, cov_matrix) - target_variance},
            {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
        )

        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_weights = num_assets * [1. / num_assets,]

        result = minimize(objective, initial_weights, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x

    def optimizeSharpeRatio(self, mean_returns, cov_matrix, risk_free_rate):
        if isinstance(mean_returns, dict):
            mean_returns = list(mean_returns.values())
        num_assets = len(mean_returns)
        args = (mean_returns, cov_matrix, risk_free_rate)

        def objective(weights, mean_returns, cov_matrix, risk_free_rate):
            return -self.calculateSharpeRatio(weights, mean_returns, cov_matrix, risk_free_rate)

        constraints = (
            {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
        )

        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_weights = num_assets * [1. / num_assets,]

        result = minimize(objective, initial_weights, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x
    
