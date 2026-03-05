import numpy as np

def state_to_index(B, W, C):
    return B * 15 + W *3 + C

def index_to_state(index):
    B = index // 15
    W = (index % 15) // 3
    C = index % 3
    return B, W, C

def weather_transitions(W):
    trans = {}

    # boundary reflection
    if W == 0 or W == 4:
        trans[W] = trans.get(W, 0) + 0.2

    if W > 0:
        trans[W - 1] = 0.2
    if W < 4:
        trans[W + 1] = 0.2

    # ficar
    trans[W] = trans.get(W, 0) + 0.6

    return trans

def consumption_transitions(C):
    trans = {}

    if C == 0 or C == 2:
        trans[C] = trans.get(C, 0) + 0.15

    if C > 0:
        trans[C - 1] = 0.15
    if C < 2:
        trans[C + 1] = 0.15

    trans[C] = trans.get(C, 0) + 0.7

    return trans

def build_matrix(scenario):
    P = np.zeros((76, 76))
    for B in range(5):
        for W in range(5):
            for C in range(3):
                i = state_to_index(B, W, C)

                # Atualização determinística da bateria
                Bp = max(0, min(4, B + W - C))

                W_trans = weather_transitions(W)
                C_trans = consumption_transitions(C)

                # Construção da linha base
                for Wn, pW in W_trans.items():
                    for Cn, pC in C_trans.items():
                        j = state_to_index(Bp, Wn, Cn)
                        P[i, j] += pW * pC

                # ---------- Scenario B ----------
                if scenario == 1:
                    if P[i, 0] > 0.10:
                        P[i, 0] -= 0.10
                        P[i, 75] += 0.10

                # ---------- Scenario C ----------
                if scenario == 2 and (Bp == 0 or Bp == 4):
                    for j in range(75):
                        if P[i, j] > 0.05:
                            P[i, j] -= 0.05
                            P[i, 75] += 0.05
        
    
    P[75, 75] = 1.0
    return P

def check_matrix(P, tol=1e-10):
    assert P.shape == (76, 76)
    assert np.all(P >= -tol)
    assert np.allclose(P.sum(axis=1), 1.0, atol=tol)
    print("✓ Matriz válida")


def power_iteration(P, s0_index, N):
    """Compute the distribution mu_N after N steps using power iteration.
    
    Starting from a deterministic initial state s0_index, computes
    mu_N = mu_0 * P^N by iteratively multiplying mu by P.
    
    Parameters
    ----------
    P : ndarray (76, 76) - transition matrix
    s0_index : int - index of the initial state
    N : int - number of steps
    
    Returns
    -------
    mu : ndarray (76,) - distribution after N steps
    """
    mu = np.zeros(P.shape[0])
    mu[s0_index] = 1.0

    for _ in range(N):
        mu = mu @ P

    return mu


def stationary_eigen(P):
    """Compute the stationary distribution using eigen-decomposition.
    
    The stationary distribution pi satisfies pi * P = pi, which means
    pi is a left eigenvector of P with eigenvalue 1, or equivalently
    a right eigenvector of P^T with eigenvalue 1.
    
    Parameters
    ----------
    P : ndarray (76, 76) - transition matrix
    
    Returns
    -------
    pi : ndarray (76,) or None - stationary distribution (normalized),
         or None if no eigenvalue 1 is found.
    """
    eigenvalues, eigenvectors = np.linalg.eig(P.T)

    # Find eigenvalue(s) close to 1
    idx = np.where(np.isclose(eigenvalues, 1.0))[0]

    if len(idx) == 0:
        print("No eigenvalue equal to 1 found.")
        return None

    # Take the first eigenvector associated with eigenvalue 1
    pi = np.real(eigenvectors[:, idx[0]])

    # Normalize so it sums to 1 (eigenvectors are determined up to a scalar)
    pi = pi / pi.sum()

    return pi


def expected_time_to_failure(P):
    Q = P[:75, :75]
    I = np.eye(75)
    N = np.linalg.inv(I - Q)
    t = N @ np.ones(75)
    return t


if __name__ == "__main__":
    P_A = build_matrix(0)
    P_B = build_matrix(1)
    P_C = build_matrix(2)

    check_matrix(P_A)
    check_matrix(P_B)
    check_matrix(P_C)

    # Initial state s0 = (B:2, W:2, C:1)
    s0 = state_to_index(2, 2, 1)
    print(f"\nInitial state index: {s0}  ->  (B, W, C) = {index_to_state(s0)}")

    # ---- Compare methods for each scenario ----
    for name, P in [("A", P_A), ("B", P_B), ("C", P_C)]:
        print(f"\n{'='*60}")
        print(f"Scenario {name}")
        print(f"{'='*60}")

        # Power iteration for increasing N
        for N in [100, 1000, 10000]:
            mu_N = power_iteration(P, s0, N)
            print(f"  Power iteration (N={N:>5d}): sum={mu_N.sum():.6f}, "
                  f"top state={np.argmax(mu_N)} (p={mu_N.max():.6f})")

        # Eigen-decomposition
        pi = stationary_eigen(P)
        if pi is not None:
            print(f"  Eigen-decomposition:        sum={pi.sum():.6f}, "
                  f"top state={np.argmax(pi)} (p={pi.max():.6f})")

            # Compare power iteration (large N) with eigen
            mu_large = power_iteration(P, s0, 10000)
            diff = np.max(np.abs(mu_large - pi))
            print(f"  Max |power_iter(N=10000) - eigen| = {diff:.2e}")
        else:
            print("  Eigen-decomposition: no stationary distribution found.")

    # ---- Expected time to failure ----
    print(f"\n{'='*60}")
    print("Expected Time to Failure (from s0 = (B:2, W:2, C:1))")
    print(f"{'='*60}")
    for name, P in [("B", P_B), ("C", P_C)]:
        t = expected_time_to_failure(P)
        print(f"  Scenario {name}: E[T_failure | s0] = {t[s0]:.2f} steps")

        # Also show min/max across all transient states
        print(f"    Min over all states: {t.min():.2f} steps "
              f"(state {np.argmin(t)}, {index_to_state(np.argmin(t))})")
        print(f"    Max over all states: {t.max():.2f} steps "
              f"(state {np.argmax(t)}, {index_to_state(np.argmax(t))})")
