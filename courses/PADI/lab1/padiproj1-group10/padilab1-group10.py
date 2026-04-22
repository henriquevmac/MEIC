import numpy as np
import matplotlib.pyplot as plt

def state_to_index(B, W, C):
    return B * 15 + W *3 + C

def index_to_state(index):
    B = index // 15
    W = (index % 15) // 3
    C = index % 3
    return B, W, C

def weather_transitions(W):
    trans = {}

    if W == 0 or W == 4:
        trans[W] = trans.get(W, 0) + 0.2

    if W > 0:
        trans[W - 1] = 0.2
    if W < 4:
        trans[W + 1] = 0.2

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

                Bp = max(0, min(4, B + W - C))

                W_trans = weather_transitions(W)
                C_trans = consumption_transitions(C)

                for Wn, pW in W_trans.items():
                    for Cn, pC in C_trans.items():
                        j = state_to_index(Bp, Wn, Cn)
                        P[i, j] += pW * pC

                if scenario == 1:
                    if P[i, 0] > 0.10:
                        P[i, 0] -= 0.10
                        P[i, 75] += 0.10

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
    mu = np.zeros(P.shape[0])
    mu[s0_index] = 1.0

    for _ in range(N):
        mu = mu @ P

    return mu


def stationary_eigen(P):
    eigenvalues, eigenvectors = np.linalg.eig(P.T)

    idx = np.where(np.isclose(eigenvalues, 1.0))[0]

    if len(idx) == 0:
        print("No eigenvalue equal to 1 found.")
        return None

    pi = np.real(eigenvectors[:, idx[0]])

    pi = pi / pi.sum()

    return pi


def expected_time_to_failure(P):
    Q = P[:75, :75]
    I = np.eye(75)
    N = np.linalg.inv(I - Q)
    t = N @ np.ones(75)
    return t

def theoretical_failure(P, s0, Ns):
    mu = np.zeros(76)
    mu[s0] = 1.0
    probs = []
    for t in range(Ns + 1):
        probs.append(mu[75])
        mu = mu @ P
    return probs

def simulate_trajectories(P, s0, Nr, Ns, seed=42):
    rng = np.random.default_rng(seed)
    states = np.full((Nr, Ns + 1), s0, dtype=int)
    for t in range(Ns):
        for i in range(Nr):
            current = states[i, t]
            states[i, t + 1] = rng.choice(76, p=P[current])
    return states

def empirical_failure(trajectories, Ns):
    return [(trajectories[:, t] == 75).mean() for t in range(Ns + 1)]

def plot_q2(Nr=5000, Ns=40, s0=0):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    scenario_names = ['A', 'B', 'C']
    
    for scenario in range(3):
        P = build_matrix(scenario)
        
        theoretical = theoretical_failure(P, s0, Ns)
        trajs = simulate_trajectories(P, s0, Nr, Ns)
        empirical = empirical_failure(trajs, Ns)
        
        ax = axes[scenario]
        ax.plot(theoretical, label='Theoretical Î¼_t[75]', lw=2)
        ax.plot(empirical, label=f'Empirical ({Nr} runs)', lw=2, linestyle='--')
        ax.set_xlabel('Time step t')
        ax.set_ylabel('P(failure)')
        ax.set_title(f'Scenario {scenario_names[scenario]}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('q2_failure_prob.png', dpi=150)
    plt.show()

def stationary_distribution(P, n_steps=100):
    mu = np.ones(76) / 76
    for _ in range(n_steps):
        mu = mu @ P
    return mu

def build_matrix_custom_weather(scenario, p_stay):
    """Rebuild matrix with custom weather stay probability."""
    from Lab1_Scenario import consumption_transitions
    
    p_move = (1 - p_stay) / 2

    def custom_weather(W):
        trans = {}
        if W > 0:
            trans[W - 1] = p_move
        else:
            trans[W] = trans.get(W, 0) + p_move
        if W < 4:
            trans[W + 1] = p_move
        else:
            trans[W] = trans.get(W, 0) + p_move
        trans[W] = trans.get(W, 0) + p_stay
        return trans

    P = np.zeros((76, 76))
    for B in range(5):
        for W in range(5):
            for C in range(3):
                i = state_to_index(B, W, C)
                Bp = max(0, min(4, B + W - C))
                W_trans = custom_weather(W)
                C_trans = consumption_transitions(C)
                for Wn, pW in W_trans.items():
                    for Cn, pC in C_trans.items():
                        j = state_to_index(Bp, Wn, Cn)
                        P[i, j] += pW * pC
                if scenario == 1:
                    if P[i, 0] > 0.10:
                        P[i, 0] -= 0.10
                        P[i, 75] += 0.10
    P[75, 75] = 1.0
    return P

def plot_q5():
    p_stay_values = [0.2, 0.3, 0.4, 0.5, 0.6]
    fail_probs = []

    for p_stay in p_stay_values:
        P = build_matrix_custom_weather(scenario=1, p_stay=p_stay)
        mu = stationary_distribution(P)
        fail_probs.append(mu[75])

    plt.figure(figsize=(8, 5))
    plt.plot(p_stay_values, fail_probs, marker='o', lw=2)
    plt.xlabel('P(stay same weather)')
    plt.ylabel('Stationary P(failure)')
    plt.title('Scenario B – Weather volatility vs failure probability')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('q5_volatility.png', dpi=150)
    plt.show()

def q6_battery_comparison():
    P = build_matrix(2)  # Scenario C
    
    # Compare B=2 vs B=4, fixing W=2 (neutral) and C=1 (neutral)
    states = {
        'B=2': state_to_index(2, 2, 1),
        'B=4': state_to_index(4, 2, 1),
    }
    
    print("Scenario C – Failure probability from each state (W=2, C=1):")
    for name, i in states.items():
        print(f"  {name}: P(failure next step) = {P[i, 75]:.4f}")

if __name__ == "__main__":
    P_A = build_matrix(0)
    P_B = build_matrix(1)
    P_C = build_matrix(2)

    check_matrix(P_A)
    check_matrix(P_B)
    check_matrix(P_C)

    plot_q2()
    plot_q5()
    q6_battery_comparison()

    s0 = state_to_index(2, 2, 1)
    print(f"\nInitial state index: {s0}  ->  (B, W, C) = {index_to_state(s0)}")

    for name, P in [("A", P_A), ("B", P_B), ("C", P_C)]:
        print(f"\n{'='*60}")
        print(f"Scenario {name}")
        print(f"{'='*60}")

        for N in [100, 1000, 10000]:
            mu_N = power_iteration(P, s0, N)
            print(f"  Power iteration (N={N:>5d}): sum={mu_N.sum():.6f}, "
                  f"top state={np.argmax(mu_N)} (p={mu_N.max():.6f})")

        pi = stationary_eigen(P)
        if pi is not None:
            print(f"  Eigen-decomposition:        sum={pi.sum():.6f}, "
                  f"top state={np.argmax(pi)} (p={pi.max():.6f})")

            mu_large = power_iteration(P, s0, 10000)
            diff = np.max(np.abs(mu_large - pi))
            print(f"  Max |power_iter(N=10000) - eigen| = {diff:.2e}")
        else:
            print("  Eigen-decomposition: no stationary distribution found.")

    print(f"\n{'='*60}")
    print("Expected Time to Failure (from s0 = (B:2, W:2, C:1))")
    print(f"{'='*60}")
    for name, P in [("B", P_B), ("C", P_C)]:
        t = expected_time_to_failure(P)
        print(f"  Scenario {name}: E[T_failure | s0] = {t[s0]:.2f} steps")

        print(f"    Min over all states: {t.min():.2f} steps "
              f"(state {np.argmin(t)}, {index_to_state(np.argmin(t))})")
        print(f"    Max over all states: {t.max():.2f} steps "
              f"(state {np.argmax(t)}, {index_to_state(np.argmax(t))})")
