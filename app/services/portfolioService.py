from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

class portfolioService:
    def getPortfolioReturn(self, weights, meanReturns):
        return np.dot(weights, meanReturns)

    def getPortfolioVariance(self, weights, covMatrix):
        return np.dot(weights.T, np.dot(covMatrix, weights))
    
    def getPortfolioStdDev(self, weights, covMatrix):
        return np.sqrt(self.getPortfolioVariance(weights, covMatrix))

    def getPortfolioVarES(self, weights, meanReturns, covMatrix, confidenceLevel=0.95):
        portfolioMean = np.dot(weights, meanReturns)
        portfolioStdDev = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights)))
        zScore = norm.ppf(confidenceLevel)
        VaR = portfolioMean - zScore * portfolioStdDev
        pdfZScore = norm.pdf(zScore)
        ES = portfolioMean - (portfolioStdDev * pdfZScore) / (1 - confidenceLevel)
        return VaR, ES

    def getPortfolioSharpeRatio(self, weights, meanReturns, covMatrix, riskFreeRate):
        expectedReturn = self.getPortfolioReturn(weights, meanReturns)
        portfolioVariance = self.getPortfolioVariance(weights, covMatrix)
        portfolioStdDev = np.sqrt(portfolioVariance)
        sharpeRatio = (expectedReturn - riskFreeRate) / portfolioStdDev
        return sharpeRatio

    def getPriorReturns(self, marketCap: np.array | list, covMatrix, sharpeRatio: float = 0.5) -> np.array:
        marketWeights = marketCap / np.sum(marketCap)
        portfolioStdDev = self.getPortfolioStdDev(marketWeights, covMatrix)
        lambdaValue = sharpeRatio * 1 / portfolioStdDev
        priorReturns = lambdaValue * np.dot(covMatrix, marketWeights)
        return priorReturns

    def getPosteriorVariables(self, P: np.array, Q: np.array, Omega: np.array, covMatrix: np.array, tau: float = 0.01) -> np.array:
        newCovMatrix = tau * covMatrix
        posteriorReturns = np.dot(
            np.linalg.inv(np.linalg.inv(newCovMatrix) + np.dot(np.dot(P.T, np.linalg.inv(Omega)), P)),
            np.dot(np.linalg.inv(newCovMatrix), newCovMatrix) + np.dot(np.dot(P.T, np.linalg.inv(Omega)), Q))
        posteriorCovariance = np.linalg.inv(np.linalg.inv(newCovMatrix) + np.dot(np.dot(P.T, np.linalg.inv(Omega)), P))
        return posteriorReturns, posteriorCovariance
