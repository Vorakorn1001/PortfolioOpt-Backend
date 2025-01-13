from scipy.optimize import minimize
from app.services.PortfolioService import PortfolioService
from pypfopt import EfficientFrontier
import matplotlib.pyplot as plt
import numpy as np
from typing import List

class OptimizeService:
    def __init__(self):
        self.portfolioService = PortfolioService()

    def optimizeFixedReturn(self, target_return: float, mean_returns: List[float], cov_matrix: np.ndarray) -> List[float]:
        
        # Initialize the Efficient Frontier object
        ef = EfficientFrontier(mean_returns, cov_matrix, solver="OSQP")

        # Optimize for minimum variance at the given target return
        ef.efficient_return(target_return)

        # Retrieve the optimized weights
        optimal_weights = ef.clean_weights()

        return list(optimal_weights.values())
        
        # mean_returns = np.array(mean_returns)
        # cov_matrix_inv = np.linalg.inv(cov_matrix)
        # ones = np.ones(len(mean_returns))

        # # Calculate weights
        # A = ones @ cov_matrix_inv @ mean_returns
        # B = mean_returns @ cov_matrix_inv @ mean_returns
        # C = ones @ cov_matrix_inv @ ones

        # lambda1 = (target_return * C - A) / (B * C - A ** 2)
        # lambda2 = (B - target_return * A) / (B * C - A ** 2)

        # weights = lambda1 * (cov_matrix_inv @ mean_returns) + lambda2 * (cov_matrix_inv @ ones)
        # return weights.tolist()

    def optimizeFixedRisk(self, target_volatility: float, mean_returns: List[float], cov_matrix: np.ndarray) -> List[float]:
        print(target_volatility)
        ef = EfficientFrontier(mean_returns, cov_matrix)
        ef.efficient_risk(target_volatility)
        optimal_weights = ef.clean_weights()
        return list(optimal_weights.values())
    
    def optimizeRangeRisk(self, min_volatility, max_volatility, step, mean_returns, cov_matrix, riskFreeRate):   
        results = []
        ef = EfficientFrontier(mean_returns, cov_matrix)
        
        target_volatility = min_volatility
        while (target_volatility <= max_volatility):
            ef.efficient_risk(target_volatility)
            optimal_weights = ef.clean_weights()
            optimal_weights = list(optimal_weights.values())
            
            results.append({
                "weight": optimal_weights,
                "return": self.portfolioService.getPortfolioReturn(optimal_weights, mean_returns),
                "volatility": self.portfolioService.getPortfolioStdDev(optimal_weights, cov_matrix),
                "sharpeRatio": self.portfolioService.getPortfolioSharpeRatio(optimal_weights, mean_returns, cov_matrix, riskFreeRate)
            })
            target_volatility += step
        
        return results

    def optimizeSharpeRatio(self, mean_returns: List[float], cov_matrix: np.ndarray, risk_free_rate: float) -> List[float]:
        mean_returns = np.array(mean_returns)
        cov_matrix_inv = np.linalg.inv(cov_matrix)
        ones = np.ones(len(mean_returns))
        excess_returns = mean_returns - risk_free_rate
        weights = cov_matrix_inv @ excess_returns / (ones @ cov_matrix_inv @ excess_returns)
        return weights.tolist()
