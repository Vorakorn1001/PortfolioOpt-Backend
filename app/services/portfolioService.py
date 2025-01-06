from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
from typing import List, Tuple

class portfolioService:
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

    def getPortfolioSharpeRatio(self, weights, meanReturns, covMatrix, riskFreeRate):
        expectedReturn = self.getPortfolioReturn(weights, meanReturns)
        portfolioVariance = self.getPortfolioVariance(weights, covMatrix)
        portfolioStdDev = np.sqrt(portfolioVariance)
        sharpeRatio = (expectedReturn - riskFreeRate) / portfolioStdDev
        return sharpeRatio

    def getPriorReturns(self, marketCap: np.array | List, covMatrix, sharpeRatio: float = 0.5) -> List[float]:
        marketWeights = marketCap / np.sum(marketCap)
        portfolioStdDev = self.getPortfolioStdDev(marketWeights, covMatrix)
        lambdaValue = sharpeRatio * 1 / portfolioStdDev
        priorReturns = lambdaValue * np.dot(covMatrix, marketWeights)
        return priorReturns

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
