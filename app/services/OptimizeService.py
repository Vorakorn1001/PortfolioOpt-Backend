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
    
    def optimizeRangeRisk(self, min_volatility, max_volatility, step, mean_returns, cov_matrix, riskFreeRate, threshold_pct=5, maxPortfolios=12) -> List[dict]:
        results = []
        ef = EfficientFrontier(mean_returns, cov_matrix)
        target_volatility = min_volatility

        while target_volatility <= max_volatility and len(results) < maxPortfolios:
            ef.efficient_risk(target_volatility)
            optimal_weights = ef.clean_weights()
            optimal_weights = list(optimal_weights.values())

            portfolioReturn = self.portfolioService.getPortfolioReturn(optimal_weights, mean_returns)
            portfolioVolatility = self.portfolioService.getPortfolioStdDev(optimal_weights, cov_matrix)
            portfolioSharpeRatio = self.portfolioService.getPortfolioSharpeRatio(optimal_weights, mean_returns, cov_matrix, riskFreeRate)

            if len(results) >= 2:
                prev_vol = results[-1]["volatility"]
                pct_change = abs((portfolioVolatility - prev_vol) / prev_vol) * 100
                if pct_change < threshold_pct:
                    break

            results.append({
                "weight": optimal_weights,
                "return": portfolioReturn,
                "volatility": portfolioVolatility,
                "sharpeRatio": portfolioSharpeRatio
            })

            if 1.0 in optimal_weights:
                break

            target_volatility += step

        return results


    def optimizeSharpeRatio(self, mean_returns: List[float], cov_matrix: np.ndarray, risk_free_rate: float) -> List[float]:
        ef = EfficientFrontier(mean_returns, cov_matrix)
        ef.max_sharpe(risk_free_rate)
        optimal_weights = ef.clean_weights()
        return list(optimal_weights.values())
