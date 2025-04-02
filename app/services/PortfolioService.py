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
        if not 0 < confidenceLevel < 1:
            raise ValueError("Confidence level must be between 0 and 1.")

        var = np.percentile(data.dropna(), (1 - confidenceLevel) * 100)
        return var
    
    def getHistoricalES(self, data: pd.Series, confidenceLevel):
        if not 0 < confidenceLevel < 1:
            raise ValueError("Confidence level must be between 0 and 1.")

        var = self.getHistoricalVaR(data, confidenceLevel)
        es = data[data <= var].mean()
        return es
    
    def getMaxDrawdown(self, data: pd.Series) -> float:
        cumulative_returns = (1 + data).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        return max_drawdown
    
    def getPortfolioBeta(self, portfolio: pd.Series, benchmark: pd.Series) -> float:
        cov_matrix = np.cov(portfolio, benchmark)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]
        return beta

    def getPortfolioSharpeRatio(self, weights, meanReturns, covMatrix, riskFreeRate):
        expectedReturn = self.getPortfolioReturn(weights, meanReturns)
        portfolioVariance = self.getPortfolioVariance(weights, covMatrix)
        portfolioStdDev = np.sqrt(portfolioVariance)
        sharpeRatio = (expectedReturn - riskFreeRate) / portfolioStdDev
        return sharpeRatio
    
    def getPortfolioMetrics(self, dataDf, cummulativeDf, confidentLevel=0.95, riskFreeRate=0.02):
        portfolioReturn = cummulativeDf['portfolioReturn'].iloc[-1]
        portfoloVolatility = dataDf['portfolioReturn'].std() * np.sqrt(252)
        portfolioSharpeRatio = (portfolioReturn - riskFreeRate) / portfoloVolatility
        return [
            { "label": "Return", "value": f"{(portfolioReturn * 100):.2f}"},
            { "label": "Volatility", "value": f"{(portfoloVolatility * 100):.2f}"},
            { "label": "Sharpe Ratio", "value": f"{portfolioSharpeRatio:.2f}"},
            { "label": f"Value at Risk ({confidentLevel*100}%)", "value": f"{(self.getHistoricalVaR(dataDf['portfolioReturn'], confidentLevel) * 100):.2f}" },
            { "label": f"Expected Shortfall ({confidentLevel*100}%)", "value": f"{(self.getHistoricalES(dataDf['portfolioReturn'], confidentLevel) * 100):.2f}" },
            { "label": "Beta", "value": f"{(self.getPortfolioBeta(dataDf['portfolioReturn'], dataDf['marketReturn'])):.2f}" }
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
