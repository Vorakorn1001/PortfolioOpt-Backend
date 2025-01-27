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
        ef.efficient_return(target_return)
        optimal_weights = ef.clean_weights()
        return list(optimal_weights.values())

        # Can't be use due to negative weights/overweight
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
        ef = EfficientFrontier(mean_returns, cov_matrix)
        ef.efficient_risk(target_volatility)
        optimal_weights = ef.clean_weights()
        return list(optimal_weights.values())
    
    def optimizeRangeRisk(self, min_volatility, max_volatility, step, mean_returns, cov_matrix, riskFreeRate, threshold_pct = 0.1) -> List[dict]:   
        results = []
        ef = EfficientFrontier(mean_returns, cov_matrix)
        target_volatility = min_volatility
        
        while target_volatility <= max_volatility:
            ef.efficient_risk(target_volatility)
            optimal_weights = ef.clean_weights()
            optimal_weights = list(optimal_weights.values())
            
            portfolioReturn = self.portfolioService.getPortfolioReturn(optimal_weights, mean_returns)
            portfolioVolatility = self.portfolioService.getPortfolioStdDev(optimal_weights, cov_matrix)
            portfolioSharpeRatio = self.portfolioService.getPortfolioSharpeRatio(optimal_weights, mean_returns, cov_matrix, riskFreeRate)
            
            # Check if we have enough results to calculate percentage change
            if len(results) >= 2:
                prev_return = results[-1]["return"]
                
                # Calculate percentage change between the last two returns and the current return
                pct_change = abs((portfolioReturn - prev_return) / prev_return) * 100
                
                if pct_change < threshold_pct:  # If the change is less than the threshold, stop
                    break
            
            # Append the current result
            results.append({
                "weight": optimal_weights,
                "return": portfolioReturn,
                "volatility": portfolioVolatility,
                "sharpeRatio": portfolioSharpeRatio
            })
            
            # Break if the weight is fully allocated to one asset
            if 1.0 in optimal_weights:
                break
            
            target_volatility += step
            
        return results


    def optimizeSharpeRatio(self, mean_returns: List[float], cov_matrix: np.ndarray, risk_free_rate: float) -> List[float]:
        ef = EfficientFrontier(mean_returns, cov_matrix)
        ef.max_sharpe(risk_free_rate)
        optimal_weights = ef.clean_weights()
        return list(optimal_weights.values())
