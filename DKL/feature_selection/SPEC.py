import numpy as np
def spec(X, G, style=0):
    """
    SPEC function for feature selection based on similarity matrix G.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data.
    G : array-like, shape (n_samples, n_samples)
        Combined similarity matrix.
    style : int, default=0
        SPEC style: -1 for all eigenvalues, 0 for all except the first,
        >=2 for top-k eigenvalues except the first.

    Returns
    -------
    w_fea : array-like, shape (n_features,)
        SPEC scores for each feature.
    """
    n_samples, n_features = X.shape

    # Build degree matrix D and Laplacian matrix L
    X_sum = np.array(G.sum(axis=1)).flatten()  # Row sum of G
    D = np.diag(X_sum)
    L = D - G

    # Normalized Laplacian L_hat
    d_inv_sqrt = np.power(X_sum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L_hat = D_inv_sqrt @ L @ D_inv_sqrt

    # Spectral decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(L_hat)
    eigenvalues = np.flipud(eigenvalues)
    eigenvectors = np.fliplr(eigenvectors)

    # Feature selection process
    w_fea = np.ones(n_features) * 1000

    for i in range(n_features):
        f = X[:, i]
        F_hat = D_inv_sqrt @ f  # Weighted feature vector
        norm_F_hat = np.linalg.norm(F_hat)
        
        if norm_F_hat < 100 * np.spacing(1):  # Ignore near-zero features
            w_fea[i] = 1000
            continue
        else:
            F_hat /= norm_F_hat
        # helps quantify how well each feature aligns with the smooth patterns on the graph induced by G
        a = np.square(F_hat.T @ eigenvectors)  # Projection of F_hat onto eigenvectors
        # Compute feature score based on selected style
        if style == -1:
            w_fea[i] = np.sum(a * eigenvalues)
        elif style == 0:
            a1 = a[1:]
            w_fea[i] = np.sum(a1 * eigenvalues[1:]) / (1 - np.power(F_hat.T @ np.ones(n_samples), 2))
        else:
            a1 = a[-style:]
            w_fea[i] = np.sum(a1 * (2 - eigenvalues[-style:]))

    if style != -1 and style != 0:
        w_fea[w_fea == 1000] = -1000

    return w_fea
def feature_ranking(score, style=0):
    '''
    Lower scores correspond to features that are smoother and more consistent with the graph, 
    which is especially helpful in biomarker discovery and identifying features relevant for 
    Alzheimer's disease diagnosis.'''
    if style in (-1, 0):
        idx = np.argsort(score)[::-1]  # Descending order
    else:
        idx = np.argsort(score)  # Ascending order
    return idx