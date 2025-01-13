from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
from typing import List, Tuple
import pandas as pd

class PortfolioService:
    def getPortfolioReturn(self, weights, meanReturns):
        return np.dot(weights, meanReturns)

    def getPortfolioVariance(self, weights, covMatrix):
        weights = np.array(weights)
        return np.dot(weights.T, np.dot(covMatrix, weights))
    
    def getPortfolioStdDev(self, weights, covMatrix):
        return np.sqrt(self.getPortfolioVariance(weights, covMatrix))

    def getPortfolioVaR(self, weights, meanReturns, covMatrix, confidenceLevel=0.95):
        weights = np.array(weights)
        
        portfolioMean = np.dot(weights, meanReturns)
        portfolioStdDev = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights)))
        zScore = norm.ppf(confidenceLevel)
        VaR = portfolioMean - zScore * portfolioStdDev
        return VaR
    
    def getPortfolioES(self, weights, meanReturns, covMatrix, confidenceLevel=0.95):
        weights = np.array(weights)
        
        portfolioMean = np.dot(weights, meanReturns)
        portfolioStdDev = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights)))
        zScore = norm.ppf(confidenceLevel)
        pdfZScore = norm.pdf(zScore)
        EstimateShortfall = portfolioMean - (portfolioStdDev * pdfZScore) / (1 - confidenceLevel)
        return EstimateShortfall
    
    def getHistoricalVaR(self, data: pd.Series, confidenceLevel: float = 0.95):
        """
        Calculate the Value at Risk (VaR) using the historical method.

        Parameters:
        - data: pd.Series of returns (e.g., daily returns)
        - confidence_level: float, confidence level for VaR (e.g., 0.95)

        Returns:
        - VaR value (negative for losses)
        """
        if not 0 < confidenceLevel < 1:
            raise ValueError("Confidence level must be between 0 and 1.")

        var = np.percentile(data.dropna(), (1 - confidenceLevel) * 100)
        return var
    
    def getHistoricalES(self, data: pd.Series, confidenceLevel):
        """
        Calculate the Expected Shortfall (ES) using the historical method.

        Parameters:
        - data: pd.Series of returns (e.g., daily returns)
        - confidence_level: float, confidence level for ES (e.g., 0.95)

        Returns:
        - Expected Shortfall (negative for losses)
        """
        if not 0 < confidenceLevel < 1:
            raise ValueError("Confidence level must be between 0 and 1.")

        var = self.getHistoricalVaR(data, confidenceLevel)
        es = data[data <= var].mean()
        return es
    
    def getMaxDrawdown(self, data: pd.Series) -> float:
        """
        Calculate the Maximum Drawdown (MDD) from a series of returns.

        Parameters:
        - data: pd.Series of returns (e.g., daily returns)

        Returns:
        - Maximum Drawdown (as a negative percentage)
        """
        # Step 1: Calculate cumulative returns
        cumulative_returns = (1 + data).cumprod()

        # Step 2: Calculate the running maximum
        running_max = cumulative_returns.cummax()

        # Step 3: Compute drawdowns
        drawdown = (cumulative_returns - running_max) / running_max

        # Step 4: Get the maximum drawdown
        max_drawdown = drawdown.min()
        return max_drawdown

    def getPortfolioSharpeRatio(self, weights, meanReturns, covMatrix, riskFreeRate):
        expectedReturn = self.getPortfolioReturn(weights, meanReturns)
        portfolioVariance = self.getPortfolioVariance(weights, covMatrix)
        portfolioStdDev = np.sqrt(portfolioVariance)
        sharpeRatio = (expectedReturn - riskFreeRate) / portfolioStdDev
        return sharpeRatio
    
    def getPortfolioMetrics(self, weights, meanReturn, covMatrix, data, confidentLevel=0.95, riskFreeRate=0.02):
        return [
            { "label": "Return", "value": f"{(self.getPortfolioReturn(weights, meanReturn) * 100):.2f}"},
            { "label": "Variance", "value": f"{(self.getPortfolioVariance(weights, covMatrix) * 100):.2f}"},
            { "label": "Sharpe Ratio", "value": f"{self.getPortfolioSharpeRatio(weights, meanReturn, covMatrix, riskFreeRate):.2f}"},
            { "label": f"Value at Risk ({confidentLevel*100}%)", "value": f"{(self.getHistoricalVaR(data, confidentLevel) * 100):.2f}" },
            { "label": f"Expected Shortfall ({confidentLevel*100}%)", "value": f"{(self.getHistoricalES(data, confidentLevel) * 100):.2f}" },
            { "label": "Max Dropdown", "value": f"{(self.getMaxDrawdown(data) * 100):.2f}" }
        ]
        
    def getPortfolioSectorWeights(self, weights, stocks, stockDataList):
        assetWeights = dict(zip(stocks, weights))
        # Filter stockDataList and calculate sector weights
        sectorWeights = {}
        for stockData in stockDataList:
            symbol = stockData.get('symbol')
            sector = stockData.get('sector')
            if symbol in assetWeights and sector:
                weight = assetWeights[symbol]
                if sector in sectorWeights:
                    sectorWeights[sector] += weight
                else:
                    sectorWeights[sector] = weight
        
        sortedSectorWeights = dict(sorted(sectorWeights.items(), key=lambda x: x[1], reverse=True))
        return sortedSectorWeights

    def getPriorReturns(self, marketCap: np.array | List, covMatrix, sharpeRatio: float = 0.5) -> List[float]:
        marketWeights = marketCap / np.sum(marketCap)
        portfolioStdDev = self.getPortfolioStdDev(marketWeights, covMatrix)
        lambdaValue = sharpeRatio * 1 / portfolioStdDev
        priorReturns = lambdaValue * np.dot(covMatrix, marketWeights)
        return priorReturns
    
    def getPosteriorReturns(self, P: np.array | List[List[int]], Q: np.array | List[float], Omega: np.array | List[List[float]], priorReturns: np.array | List[float], covMatrix: np.array | List[List[float]], tau: float = 0.01) -> List[float]:
        P = np.array(P)
        Q = np.array(Q)
        Omega = np.array(Omega)
        covMatrix = np.array(covMatrix)
        
        newCovMatrix = tau * covMatrix
        posteriorReturns = np.dot(
            np.linalg.inv(np.linalg.inv(newCovMatrix) + np.dot(np.dot(P.T, np.linalg.inv(Omega)), P)),
            np.dot(np.linalg.inv(newCovMatrix), priorReturns) + np.dot(np.dot(P.T, np.linalg.inv(Omega)), Q))
        return posteriorReturns

    def getPosteriorVariables(self, P: np.array | List[List[int]], Q: np.array | List[float], Omega: np.array | List[List[float]], priorReturns: np.array | List[float], covMatrix: np.array | List[List[float]], tau: float = 0.01) -> Tuple[List[float], List[float]]:
        P = np.array(P)
        Q = np.array(Q)
        Omega = np.array(Omega)
        covMatrix = np.array(covMatrix)
        
        newCovMatrix = tau * covMatrix
        posteriorReturns = np.dot(
            np.linalg.inv(np.linalg.inv(newCovMatrix) + np.dot(np.dot(P.T, np.linalg.inv(Omega)), P)),
            np.dot(np.linalg.inv(newCovMatrix), priorReturns) + np.dot(np.dot(P.T, np.linalg.inv(Omega)), Q))
        posteriorCovariance = np.linalg.inv(np.linalg.inv(newCovMatrix) + np.dot(np.dot(P.T, np.linalg.inv(Omega)), P))
        return posteriorReturns, posteriorCovariance
