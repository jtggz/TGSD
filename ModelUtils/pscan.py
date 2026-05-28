"""
Parallel scan implementation for Mamba SSM
"""
import torch
import torch.nn as nn


def pscan(A, X):
    """
    Parallel scan operation for selective scan

    Args:
        A: (B, L, D, N) - state transition matrices
        X: (B, L, D, N) - inputs

    Returns:
        output: (B, L, D, N) - scan results
    """
    B, L, D, N = A.shape

    # Scan in forward direction
    Y = torch.zeros_like(X)
    h = torch.zeros(B, D, N, device=A.device, dtype=A.dtype)

    for i in range(L):
        h = A[:, i] * h + X[:, i]
        Y[:, i] = h

    return Y


def pscan_bwd(A, X, dY):
    """
    Backward pass for parallel scan

    Args:
        A: (B, L, D, N)
        X: (B, L, D, N)
        dY: (B, L, D, N) - gradient at output

    Returns:
        dA, dX: gradients
    """
    B, L, D, N = A.shape

    dX = torch.zeros_like(X)
    dh = torch.zeros(B, D, N, device=A.device, dtype=A.dtype)

    for i in reversed(range(L)):
        dh = dh + dY[:, i]
        dX[:, i] = dh
        dh = A[:, i] * dh

    return dX