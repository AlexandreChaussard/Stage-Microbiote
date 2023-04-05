import scipy.stats as stats


def pdf_gaussian(x, mu, sigma):
    return stats.multivariate_normal(mean=mu, cov=sigma ** 2).pdf(x)
