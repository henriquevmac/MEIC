import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Define state space dimensions
B_STATES = 5 # for B = 0, 1, 2, 3, 4
W_STATES = 5 # for W = 0, 1, 2
C_STATES = 3 # for C = 0, 1, 2, 3, 4

# Total number of regular states
NUM_REGULAR_STATES = B_STATES * W_STATES * C_STATES # 5 * 3 * 5 = 75

# Total number of states including the failure state (index 75)
NUM_TOTAL_STATES = NUM_REGULAR_STATES + 1 # 75 + 1 = 76

# Number of actions
NUM_ACTIONS = 3

# Define action indices
AC = 0 # Action Charge
AS = 1 # Action Solar
AE = 2 # Action Export


def factors_to_state(b, w, c):
    """
    Converts (b, w, c) factors to a unique state index.
    Raises ValueError if factors are outside defined ranges.
    """
    if not (0 <= b < B_STATES and 0 <= w < W_STATES and 0 <= c < C_STATES):
        raise ValueError(f"Invalid factors: b={b}, w={w}, c={c} must be within defined space dimensions.")
        
    return b * (W_STATES * C_STATES) + w * C_STATES + c

def state_to_factors(state_index):
    """
    Converts a state index to (b, w, c) factors.
    Handles the failure state (index NUM_REGULAR_STATES) by returning (-1, -1, -1).
    Raises ValueError for invalid regular state indices.
    """
    if state_index == NUM_REGULAR_STATES: # State 75 is the failure state
        return (-1, -1, -1) # Indicating a non-factorable failure state
    
    if not (0 <= state_index < NUM_REGULAR_STATES):
        raise ValueError(f"Invalid state_index: {state_index}. Must be between 0 and {NUM_TOTAL_STATES}")
        
    b = state_index // (W_STATES * C_STATES)
    remaining_index = state_index % (W_STATES * C_STATES)
    w = remaining_index // C_STATES
    c = remaining_index % C_STATES
    return (b, w, c)

def create_complex_transition_matrix():
    """
    Creates and populates the 3D transition tensor (T) with probabilistic transitions.
    T has shape (NUM_ACTIONS, NUM_TOTAL_STATES, NUM_TOTAL_STATES).
    """
    T = np.zeros((NUM_ACTIONS, NUM_TOTAL_STATES, NUM_TOTAL_STATES))
    failure_state_index = NUM_TOTAL_STATES - 1
    
    # Weather Transition Matrix P(W_next | W_current)
    # P_W[W_current, W_next]
    P_W = np.zeros((W_STATES, W_STATES))
    P_W[0, 1] = 0.3
    P_W[0, 0] = 0.7
    P_W[1, 2] = 0.4
    P_W[1, 0] = 0.3
    P_W[1, 1] = 0.3
    P_W[2, 3] = 0.4
    P_W[2, 2] = 0.3
    P_W[2, 1] = 0.3
    P_W[3, 4] = 0.4
    P_W[3, 3] = 0.3
    P_W[3, 2] = 0.3
    P_W[4, 4] = 0.7
    P_W[4, 3] = 0.3
    
    # Consumption Probability Function P(C_next | W_next)
    def get_prob_c_next(c_next_val, w_next_val):
        if w_next_val in [0, 1]: # If W_next is 1 or 2, C uniform over [1, 2, 3, 4]
            if c_next_val >= 1 and c_next_val <= 2:
                return 0.5
            else:
                return 0.0
        elif w_next_val == 2: # If W_next is 0, C uniform over [0, 1, 2]
            if c_next_val >= 0 and c_next_val <= 2:
                return 1 / 3
            else:
                return 0.0
        else:  # w_next_val in [3, 4]
            if c_next_val >= 0 and c_next_val <= 1:
                return 0.5
            else:
                return 0.0

    # Failure state transitions to itself for all actions with probability 1
    for action_idx in range(NUM_ACTIONS):
        T[action_idx, failure_state_index, failure_state_index] = 1.0

    # Populate transitions for regular states (0 to NUM_REGULAR_STATES - 1)
    for s_current in range(NUM_REGULAR_STATES):
        b_current, w_current, c_current = state_to_factors(s_current)

        # --- Handle Action 0 (Charge) ---
        action_idx = AC

        # Calculate deterministic B_next for normal charge outcome
        b_next_AC_normal = min(max(b_current - c_current, 0) + 1, B_STATES - 1)

        # Distribute the remaining probability (1 - prob_failure_AC) over possible W_next and C_next
        for w_next_val in range(W_STATES):
            for c_next_val in range(C_STATES):
                prob_w_c_transition = P_W[w_current, w_next_val] * get_prob_c_next(c_next_val, w_next_val)
                
                if prob_w_c_transition > 0:
                    s_next = factors_to_state(b_next_AC_normal, w_next_val, c_next_val)
                    T[action_idx, s_current, s_next] += prob_w_c_transition


        # --- Handle Action 1 (Solar - AS) ---
        action_idx = AS
        failure_prob_AS = 0.0
        
        # Determine charging probability based on current weather
        charge_prob_AS = 0.0
        
        if w_current == 4:
            charge_prob_AS = 0.95
        elif w_current == 3:
            charge_prob_AS = 0.8
        elif w_current == 2:
            charge_prob_AS = 0.5
        elif w_current == 1:
            charge_prob_AS = 0.2
        elif w_current == 0:
            charge_prob_AS = 0.05
            failure_prob_AS = 0.02

        # Calculate deterministic B_next for charged and uncharged outcomes
        b_next_AS_uncharged = max(b_current - c_current, 0)
        b_next_AS_charged = min(b_next_AS_uncharged + 1, B_STATES - 1)

        # Distribute probabilities over possible W_next and C_next for both outcomes
        for w_next_val in range(W_STATES):
            for c_next_val in range(C_STATES):
                prob_w_c_transition = P_W[w_current, w_next_val] * get_prob_c_next(c_next_val, w_next_val)
                
                if prob_w_c_transition > 0:
                    # Add probability for the 'charged' outcome
                    s_next_charged = factors_to_state(b_next_AS_charged, w_next_val, c_next_val)
                    T[action_idx, s_current, s_next_charged] += charge_prob_AS * prob_w_c_transition

                    # Add probability for the 'uncharged' outcome
                    s_next_uncharged = factors_to_state(b_next_AS_uncharged, w_next_val, c_next_val)
                    T[action_idx, s_current, s_next_uncharged] += (1 - charge_prob_AS - failure_prob_AS) * prob_w_c_transition

                    # Add probability for the 'failure' outcome
                    T[action_idx, s_current, failure_state_index] += failure_prob_AS * prob_w_c_transition


        # --- Handle Action 2 (Export - AE) ---
        action_idx = AE
        failure_prob_AE = 0.0
        
        # Calculate deterministic B_next based on current B and C
        b_next_AE = max(b_current - c_current - 1, 0)

        if b_next_AE == 0:
            failure_prob_AE = 0.05

        T[action_idx, s_current, failure_state_index] += failure_prob_AE

        # Distribute probabilities over possible W_next and C_next
        for w_next_val in range(W_STATES):
            for c_next_val in range(C_STATES):
                prob_w_c_transition = P_W[w_current, w_next_val] * get_prob_c_next(c_next_val, w_next_val) * (1 - failure_prob_AE)
                
                if prob_w_c_transition > 0:
                    s_next = factors_to_state(int(b_next_AE), w_next_val, c_next_val)
                    T[action_idx, s_current, s_next] += prob_w_c_transition
                    

    for a in range(T.shape[0]):
        for s in range(T.shape[1]):
            assert np.all(np.isclose(T[a, s].sum(), 1.0)), (s, a, T[a, s].sum())
            
    return T

# --- Main execution --- 
# Create the complex transition matrix
T_complex = create_complex_transition_matrix()

GRID_COST      = 10   
SOLAR_REVENUE  = -5    
FAIL_PENALTY   = 500  
FAILURE_STATE = NUM_TOTAL_STATES - 1
GAMMA = 0.99

def build_cost_matrix(grid_cost=10, solar_revenue=-5, fail_penalty=500):
    C_sa = np.zeros((NUM_TOTAL_STATES, NUM_ACTIONS))
    C_sa[FAILURE_STATE, :] = fail_penalty
    for s in range(NUM_REGULAR_STATES):
        C_sa[s, AC] = grid_cost    
        C_sa[s, AS] = 0             
        C_sa[s, AE] = solar_revenue  
    return C_sa


C_sa = build_cost_matrix()

# test cost matrix values for some states and all actions
print("Cost matrix spot-checks  C(s, [AC, AS, AE])")
for s in [0, 5, 32, 49, 75]:
    print(f"  C({s:>2}, :) = {C_sa[s]}")

def value_iteration(T, C_sa, gamma=GAMMA, tol=1e-6, max_iter=10_000):
    """
    Standard Value Iteration.

    Returns
    -------
    V       : (NUM_TOTAL_STATES,)  optimal value function
    policy  : (NUM_TOTAL_STATES,)  greedy policy
    history : dict with keys
                'bellman_errors'        – ||V_{k+1} - V_k||_inf per iteration
                'policy_changes'        – # states where policy changed per iter
                'n_iter'                – total iterations until convergence
    """
    n_states  = T.shape[1]
    n_actions = T.shape[0]

    V       = np.zeros(n_states)
    policy  = np.zeros(n_states, dtype=int)

    bellman_errors  = []
    policy_changes  = []

    for k in range(max_iter):
        # Q(s,a) = C(s,a) + γ · Σ_{s'} T(a,s,s') · V(s')
        # Shape: (n_states, n_actions)
        Q = C_sa + gamma * np.einsum('aij,j->ia', T, V)

        V_new      = Q.min(axis=1)
        new_policy = Q.argmin(axis=1)

        bellman_err   = np.max(np.abs(V_new - V))
        policy_change = np.sum(new_policy != policy)

        bellman_errors.append(bellman_err)
        policy_changes.append(policy_change)

        V      = V_new
        policy = new_policy

        if bellman_err < tol:
            print(f"[VI] Converged in {k+1} iterations  (Bellman error = {bellman_err:.2e})")
            break
    else:
        print(f"[VI] Did NOT converge within {max_iter} iterations")

    history = {
        'bellman_errors': bellman_errors,
        'policy_changes': policy_changes,
        'n_iter':         k + 1,
    }
    return V, policy, history

def policy_evaluation(T, C_sa, policy, gamma=GAMMA, tol=1e-9):
    """
    Exact policy evaluation by solving the linear system:
        (I - γ P_π) V = C_π
    This is exact (no iterative inner loop) and therefore cheap per PI step.
    """
    n_states = T.shape[1]

    # Gather per-state cost and transition row for the current policy
    C_pi = C_sa[np.arange(n_states), policy]                  # (n_states,)
    T_pi = T[policy, np.arange(n_states), :]                  # (n_states, n_states)

    # Solve  (I - γ T_π) V = C_π
    A = np.eye(n_states) - gamma * T_pi
    V = np.linalg.solve(A, C_pi)
    return V


def policy_iteration(T, C_sa, gamma=GAMMA, max_iter=500):
    """
    Policy Iteration with exact policy evaluation.

    Returns
    -------
    V, policy, history   (same structure as VI)
    """
    n_states  = T.shape[1]
    n_actions = T.shape[0]

    # Initialise with a uniform policy (always AC)
    policy = np.zeros(n_states, dtype=int)

    bellman_errors = []
    policy_changes = []

    for k in range(max_iter):
        # --- Evaluation step (exact) ---
        V = policy_evaluation(T, C_sa, policy, gamma)

        # --- Improvement step ---
        Q          = C_sa + gamma * np.einsum('aij,j->ia', T, V)
        new_policy = Q.argmin(axis=1)

        # Track diagnostics
        V_new          = Q.min(axis=1)
        bellman_err    = np.max(np.abs(V_new - V))
        n_changed      = np.sum(new_policy != policy)

        bellman_errors.append(bellman_err)
        policy_changes.append(n_changed)

        policy = new_policy

        if n_changed == 0:
            print(f"[PI] Converged in {k+1} iterations  (policy stable, Bellman error = {bellman_err:.2e})")
            break
    else:
        print(f"[PI] Did NOT converge within {max_iter} iterations")

    history = {
        'bellman_errors': bellman_errors,
        'policy_changes': policy_changes,
        'n_iter':         k + 1,
    }
    return V, policy, history

V_vi, pi_vi, hist_vi = value_iteration(T_complex, C_sa)
V_pi, pi_pi, hist_pi = policy_iteration(T_complex, C_sa)

# --- Are the policies the same? ---
policy_match = np.sum(pi_vi != pi_pi)
print(f"\nStates where VI and PI policies differ: {policy_match}")
print(f"Max |V_VI - V_PI| = {np.max(np.abs(V_vi - V_pi)):.4e}")

# --- Spot-check values ---
ACTION_NAMES = {AC: 'AC', AS: 'AS', AE: 'AE'}
print("\nValue / policy spot-checks")
print(f"{'State':>6}  {'Factors':>12}  {'V_VI':>10}  {'V_PI':>10}  {'π_VI':>5}  {'π_PI':>5}")
for s in [0, 5, 32, 49, 75]:
    f = state_to_factors(s)
    print(f"  {s:>4}  {str(f):>12}  {V_vi[s]:>10.2f}  {V_pi[s]:>10.2f}"
          f"  {ACTION_NAMES[pi_vi[s]]:>5}  {ACTION_NAMES[pi_pi[s]]:>5}")
    
    
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# --- Bellman error ---
ax = axes[0]
ax.semilogy(hist_vi['bellman_errors'], label='VI', color='steelblue')
ax.semilogy(hist_pi['bellman_errors'], label='PI', color='tomato', linestyle='--')
ax.set_xlabel('Iteration')
ax.set_ylabel('Bellman Error  ||V_{k+1} − V_k||∞  (log scale)')
ax.set_title('Convergence: Bellman Error')
ax.legend()
ax.grid(True, which='both', alpha=0.4)

# --- Policy changes ---
ax = axes[1]
ax.plot(hist_vi['policy_changes'], label='VI', color='steelblue')
ax.plot(hist_pi['policy_changes'], label='PI', color='tomato', linestyle='--')
ax.set_xlabel('Iteration')
ax.set_ylabel('# states with changed policy')
ax.set_title('Convergence: Policy Changes per Iteration')
ax.legend()
ax.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig('vi_vs_pi_convergence.png', dpi=150)
plt.show()
print("Plot saved → vi_vs_pi_convergence.png")

def print_transition_examples(action_idx, action_name, max_initial_states_to_show=5, max_transitions_per_initial_state=3):
    print(f"\n--- Examples for Action {action_idx} ({action_name}) ---")
    
    # Define some interesting example states (b, w, c) to showcase different scenarios
    example_states = [
        (0, 0, 0),   # Low battery, no sun, no consumption
        (4, 2, 0),   # Full battery, sunny, no consumption
        (1, 4, 0),   # Low battery, no sun, high consumption (potential for 0 battery)
        (3, 1, 2),   # Medium battery, cloudy, medium consumption
        (0, 4, 2)    # Empty battery, sunny, high consumption (challenging state)
    ]
    
    states_shown_count = 0
    
    for b_cur, w_cur, c_cur in example_states:
        if states_shown_count >= max_initial_states_to_show:
            break
            
        s_current = factors_to_state(b_cur, w_cur, c_cur)
        print(f"  From Initial State {s_current} ({b_cur},{w_cur},{c_cur}):")
        transitions_for_state = []
        
        for s_next in range(NUM_TOTAL_STATES):
            prob = T_complex[action_idx, s_current, s_next]
            
            if prob > 0:
                transitions_for_state.append((prob, s_next))

        # Sort by probability descending
        transitions_for_state.sort(key=lambda x: x[0], reverse=True)
        transitions_printed_for_this_state = 0
        
        for prob, s_next in transitions_for_state:
            if transitions_printed_for_this_state >= max_transitions_per_initial_state:
                break
            
            if s_next == NUM_TOTAL_STATES - 1: # Failure state
                print(f"    to Failure State {s_next} with P = {prob:.4f}")
            else:
                b_next, w_next, c_next = state_to_factors(s_next)
                print(f"    to State {s_next} ({b_next},{w_next},{c_next}) with P = {prob:.4f}")
                
            transitions_printed_for_this_state += 1
        
        if transitions_printed_for_this_state == 0:
            print("    No transitions with P > 0 found (this should not happen for valid MDPs unless states are terminal).")
            
        states_shown_count += 1

# Print examples for each action
print_transition_examples(AC, "Charge", max_initial_states_to_show=5, max_transitions_per_initial_state=3)
print_transition_examples(AS, "Solar", max_initial_states_to_show=5, max_transitions_per_initial_state=3)
print_transition_examples(AE, "Export", max_initial_states_to_show=5, max_transitions_per_initial_state=3)

# Demonstrate state conversion functions with examples (unchanged from before)
print("\n--- Demonstrating state conversion functions (unchanged) ---")
print(f"Factors (0, 0, 0) -> State: {factors_to_state(0, 0, 0)}")
print(f"Factors (4, 4, 2) -> State: {factors_to_state(4, 4, 2)}") # Max regular state
print(f"State 0 -> Factors: {state_to_factors(0)}")
print(f"State 74 -> Factors: {state_to_factors(74)}") # Max regular state
print(f"State 75 (Failure) -> Factors: {state_to_factors(75)}") # Failure state


# =============================================================================
# QUESTION: Importance of weather information in the state
# =============================================================================
# We create a new MDP where weather is marginalized out of the state.
# The new state is just (B, C), and transitions are averaged over the
# stationary distribution of weather.
# =============================================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Step 1: Compute the stationary distribution of weather ---
P_W = np.zeros((W_STATES, W_STATES))
P_W[0, 1] = 0.3; P_W[0, 0] = 0.7
P_W[1, 2] = 0.4; P_W[1, 0] = 0.3; P_W[1, 1] = 0.3
P_W[2, 3] = 0.4; P_W[2, 2] = 0.3; P_W[2, 1] = 0.3
P_W[3, 4] = 0.4; P_W[3, 3] = 0.3; P_W[3, 2] = 0.3
P_W[4, 4] = 0.7; P_W[4, 3] = 0.3

# Stationary distribution: pi * P_W = pi, sum(pi) = 1
# Solve (P_W^T - I) pi = 0 with sum constraint
A_stat = P_W.T - np.eye(W_STATES)
A_stat[-1, :] = 1.0
b_stat = np.zeros(W_STATES)
b_stat[-1] = 1.0
pi_weather = np.linalg.solve(A_stat, b_stat)
print(f"\nStationary distribution of weather: {pi_weather}")
print(f"  Sum = {pi_weather.sum():.6f}")

# --- Step 2: Build a reduced MDP without weather ---
# New state: (B, C) -> index = b * C_STATES + c
# Plus a failure state at index B_STATES * C_STATES

NUM_REDUCED_REGULAR = B_STATES * C_STATES  # 5 * 3 = 15
NUM_REDUCED_TOTAL = NUM_REDUCED_REGULAR + 1  # 16 (failure = 15)

def reduced_factors_to_state(b, c):
    return b * C_STATES + c

def reduced_state_to_factors(s):
    if s == NUM_REDUCED_REGULAR:
        return (-1, -1)
    b = s // C_STATES
    c = s % C_STATES
    return (b, c)

def create_marginalized_transition_matrix():
    """
    Creates a transition matrix marginalized over weather.
    For each (b, c) state and action, we average the battery transitions
    over all possible weather states weighted by the stationary distribution,
    and similarly average the consumption transitions.
    """
    T_marg = np.zeros((NUM_ACTIONS, NUM_REDUCED_TOTAL, NUM_REDUCED_TOTAL))
    fail_idx = NUM_REDUCED_TOTAL - 1
    
    # Failure state stays failure
    for a in range(NUM_ACTIONS):
        T_marg[a, fail_idx, fail_idx] = 1.0
    
    # Consumption transition: P(C_next | W_next)
    def get_prob_c_next(c_next_val, w_next_val):
        if w_next_val in [0, 1]:
            return 0.5 if 1 <= c_next_val <= 2 else 0.0
        elif w_next_val == 2:
            return 1/3 if 0 <= c_next_val <= 2 else 0.0
        else:  # 3, 4
            return 0.5 if 0 <= c_next_val <= 1 else 0.0
    
    # Precompute marginalized P(C_next = c') = sum_w sum_w' pi(w) * P_W(w'|w) * P(c'|w')
    # Actually since C_next depends on W_next and W transitions depend on W_current,
    # we marginalize over W_current with stationary dist, then sum over W_next.
    # P(C_next=c') = sum_{w} pi(w) * sum_{w'} P_W(w,w') * P(c'|w')
    P_C_marg = np.zeros(C_STATES)
    for c_next in range(C_STATES):
        for w_cur in range(W_STATES):
            for w_next in range(W_STATES):
                P_C_marg[c_next] += pi_weather[w_cur] * P_W[w_cur, w_next] * get_prob_c_next(c_next, w_next)
    
    print(f"\nMarginalized consumption distribution P(C_next):")
    for c in range(C_STATES):
        print(f"  P(C_next={c}) = {P_C_marg[c]:.4f}")
    print(f"  Sum = {P_C_marg.sum():.6f}")
    
    # Solar charge probabilities per weather state
    p_charge_solar = {0: 0.05, 1: 0.2, 2: 0.5, 3: 0.8, 4: 0.95}
    
    # Marginalized solar charge probability
    avg_p_charge = sum(pi_weather[w] * p_charge_solar[w] for w in range(W_STATES))
    print(f"\nMarginalized solar charge probability: {avg_p_charge:.4f}")
    
    # For AS action: failure only when w=0 and max(B-C,0)=0 -> p_fail=0.02
    # Marginalized: p_fail conditional on battery depletion = pi(w=0) * 0.02
    # But actually p_fail depends on BOTH w_current AND (b,c). When w=0, p_fail=0.02.
    # For other weather states, the remaining prob 1-p_charge goes to discharge, not failure.
    # We need to handle this carefully.
    
    # Actually re-reading the spec more carefully:
    # For AS: p_fail depends on whether max(B-C, 0) = 0
    # - If max(B-C,0) = 0: p_fail = 0.02, otherwise p_fail = 0
    # Wait, let me re-read... The spec says for AS:
    # "If max(B_t - C_t, 0) = 0, then p_fail = 0.02, otherwise p_fail = 0"
    # And this is separate from the weather-dependent p_charge.
    
    # Let me re-read the original code more carefully...
    # In the original code: failure_prob_AS = 0.0 by default
    # Only set to 0.02 when w_current == 0.
    # So failure for AS depends on weather, not on battery level!
    
    # Marginalized p_fail for AS = pi(w=0) * 0.02
    avg_p_fail_AS = pi_weather[0] * 0.02
    
    for s_red in range(NUM_REDUCED_REGULAR):
        b_cur, c_cur = reduced_state_to_factors(s_red)
        
        # === Action 0: AC (Charge) - deterministic charge ===
        a = AC
        b_next_ac = min(max(b_cur - c_cur, 0) + 1, B_STATES - 1)
        for c_next in range(C_STATES):
            s_next = reduced_factors_to_state(b_next_ac, c_next)
            T_marg[a, s_red, s_next] += P_C_marg[c_next]
        
        # === Action 1: AS (Solar) ===
        a = AS
        b_after_consumption = max(b_cur - c_cur, 0)
        b_next_charged = min(b_after_consumption + 1, B_STATES - 1)
        b_next_uncharged = b_after_consumption
        
        # Marginalized: average over weather
        # p_charge averaged, p_fail averaged, p_discharge = 1 - p_charge - p_fail (averaged)
        # Note: In the original, p_fail is only nonzero for w=0, where p_charge=0.05
        # So p_discharge(w=0) = 1 - 0.05 - 0.02 = 0.93
        # For w!=0, p_fail=0, p_discharge = 1 - p_charge
        # Marginalized p_discharge = sum_w pi(w) * (1 - p_charge(w) - p_fail(w))
        avg_p_discharge = 0
        for w in range(W_STATES):
            p_fail_w = 0.02 if w == 0 else 0.0
            avg_p_discharge += pi_weather[w] * (1 - p_charge_solar[w] - p_fail_w)
        
        for c_next in range(C_STATES):
            s_charged = reduced_factors_to_state(b_next_charged, c_next)
            s_uncharged = reduced_factors_to_state(b_next_uncharged, c_next)
            T_marg[a, s_red, s_charged] += avg_p_charge * P_C_marg[c_next]
            T_marg[a, s_red, s_uncharged] += avg_p_discharge * P_C_marg[c_next]
        T_marg[a, s_red, fail_idx] += avg_p_fail_AS
        
        # === Action 2: AE (Export) ===
        a = AE
        b_next_ae = max(b_cur - c_cur - 1, 0)
        p_fail_ae = 0.05 if b_next_ae == 0 else 0.0
        
        T_marg[a, s_red, fail_idx] += p_fail_ae
        for c_next in range(C_STATES):
            s_next = reduced_factors_to_state(b_next_ae, c_next)
            T_marg[a, s_red, s_next] += (1 - p_fail_ae) * P_C_marg[c_next]
    
    # Validate
    for a in range(NUM_ACTIONS):
        for s in range(NUM_REDUCED_TOTAL):
            row_sum = T_marg[a, s].sum()
            assert np.isclose(row_sum, 1.0), f"Row sum issue: a={a}, s={s}, sum={row_sum}"
    
    return T_marg

T_marg = create_marginalized_transition_matrix()

# Build cost matrix for reduced state space
def build_reduced_cost_matrix():
    C_red = np.zeros((NUM_REDUCED_TOTAL, NUM_ACTIONS))
    C_red[NUM_REDUCED_REGULAR, :] = FAIL_PENALTY  # failure state
    for s in range(NUM_REDUCED_REGULAR):
        C_red[s, AC] = GRID_COST
        C_red[s, AS] = 0
        C_red[s, AE] = SOLAR_REVENUE
    return C_red

C_red = build_reduced_cost_matrix()

# --- Run Value Iteration on the reduced (no-weather) MDP ---
print("\n" + "="*70)
print("VALUE ITERATION ON MARGINALIZED (NO-WEATHER) MDP")
print("="*70)

def value_iteration_general(T, C_sa, gamma=GAMMA, tol=1e-6, max_iter=10_000):
    n_states = T.shape[1]
    V = np.zeros(n_states)
    policy = np.zeros(n_states, dtype=int)
    bellman_errors = []
    policy_changes = []

    for k in range(max_iter):
        Q = C_sa + gamma * np.einsum('aij,j->ia', T, V)
        V_new = Q.min(axis=1)
        new_policy = Q.argmin(axis=1)
        bellman_err = np.max(np.abs(V_new - V))
        policy_change = np.sum(new_policy != policy)
        bellman_errors.append(bellman_err)
        policy_changes.append(policy_change)
        V = V_new
        policy = new_policy
        if bellman_err < tol:
            print(f"[VI] Converged in {k+1} iterations  (Bellman error = {bellman_err:.2e})")
            break
    else:
        print(f"[VI] Did NOT converge within {max_iter} iterations")

    return V, policy, {'bellman_errors': bellman_errors, 'policy_changes': policy_changes, 'n_iter': k+1}

V_marg, pi_marg, hist_marg = value_iteration_general(T_marg, C_red)

# --- Compare policies ---
print("\n" + "="*70)
print("COMPARISON: FULL MDP (with weather) vs MARGINALIZED (no weather)")
print("="*70)

ACTION_NAMES = {0: 'AC', 1: 'AS', 2: 'AE'}

print(f"\n{'(B,C)':>8}  {'V_marg':>10}  {'π_marg':>7}")
for s_red in range(NUM_REDUCED_REGULAR):
    b, c = reduced_state_to_factors(s_red)
    print(f"  ({b},{c})    {V_marg[s_red]:>10.2f}  {ACTION_NAMES[pi_marg[s_red]]:>7}")
print(f"  FAIL     {V_marg[NUM_REDUCED_REGULAR]:>10.2f}  {ACTION_NAMES[pi_marg[NUM_REDUCED_REGULAR]]:>7}")

# --- Compare: For the full MDP, show policy for each (B,W,C) and see how it varies with W ---
print("\n\nFull MDP policy breakdown by weather (showing how policy changes with W):")
print(f"{'(B,C)':>8}  {'W=0':>5}  {'W=1':>5}  {'W=2':>5}  {'W=3':>5}  {'W=4':>5}  {'Marg':>5}  {'Varies?':>8}")

n_weather_dependent = 0
n_total_bc = 0
for b in range(B_STATES):
    for c in range(C_STATES):
        policies_by_w = []
        for w in range(W_STATES):
            s_full = factors_to_state(b, w, c)
            policies_by_w.append(pi_vi[s_full])
        
        s_red = reduced_factors_to_state(b, c)
        pi_m = pi_marg[s_red]
        
        varies = len(set(policies_by_w)) > 1
        if varies:
            n_weather_dependent += 1
        n_total_bc += 1
        
        w_strs = [ACTION_NAMES[p] for p in policies_by_w]
        marker = "  <<<" if varies else ""
        print(f"  ({b},{c})    {w_strs[0]:>5}  {w_strs[1]:>5}  {w_strs[2]:>5}  {w_strs[3]:>5}  {w_strs[4]:>5}  {ACTION_NAMES[pi_m]:>5}  {'YES' if varies else 'no':>7}{marker}")

print(f"\n  States where optimal action depends on weather: {n_weather_dependent}/{n_total_bc}")

# --- Compare expected costs ---
print("\n\nExpected cost comparison (full MDP values averaged over weather with stationary dist):")
print(f"{'(B,C)':>8}  {'V_full_avg':>12}  {'V_marg':>10}  {'Diff':>10}")
total_diff = 0
for b in range(B_STATES):
    for c in range(C_STATES):
        v_full_avg = 0
        for w in range(W_STATES):
            s_full = factors_to_state(b, w, c)
            v_full_avg += pi_weather[w] * V_vi[s_full]
        s_red = reduced_factors_to_state(b, c)
        diff = V_marg[s_red] - v_full_avg
        total_diff += abs(diff)
        print(f"  ({b},{c})    {v_full_avg:>12.2f}  {V_marg[s_red]:>10.2f}  {diff:>+10.2f}")

print(f"\n  Total absolute value difference: {total_diff:.2f}")

# --- Create comparison plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Policy heatmap for full MDP across weather states
ax = axes[0]
policy_grid = np.zeros((B_STATES * C_STATES, W_STATES))
y_labels = []
for b in range(B_STATES):
    for c in range(C_STATES):
        row = b * C_STATES + c
        y_labels.append(f"B={b},C={c}")
        for w in range(W_STATES):
            s = factors_to_state(b, w, c)
            policy_grid[row, w] = pi_vi[s]

im = ax.imshow(policy_grid, aspect='auto', cmap='viridis', vmin=0, vmax=2)
ax.set_xticks(range(W_STATES))
ax.set_xticklabels([f'W={w}' for w in range(W_STATES)])
ax.set_yticks(range(len(y_labels)))
ax.set_yticklabels(y_labels, fontsize=7)
ax.set_title('Full MDP: Optimal Action by (B,C) and W')
ax.set_xlabel('Weather State')
ax.set_ylabel('(Battery, Consumption) State')

# Add text annotations
for i in range(policy_grid.shape[0]):
    for j in range(policy_grid.shape[1]):
        ax.text(j, i, ACTION_NAMES[int(policy_grid[i, j])], ha='center', va='center', fontsize=6, color='white')

# Plot 2: Marginalized policy
ax = axes[1]
marg_policy_col = np.zeros((B_STATES * C_STATES, 1))
for b in range(B_STATES):
    for c in range(C_STATES):
        row = b * C_STATES + c
        s_red = reduced_factors_to_state(b, c)
        marg_policy_col[row, 0] = pi_marg[s_red]

im2 = ax.imshow(marg_policy_col, aspect=0.3, cmap='viridis', vmin=0, vmax=2)
ax.set_xticks([0])
ax.set_xticklabels(['No W'])
ax.set_yticks(range(len(y_labels)))
ax.set_yticklabels(y_labels, fontsize=7)
ax.set_title('Marginalized MDP: Optimal Action (no weather)')

for i in range(marg_policy_col.shape[0]):
    ax.text(0, i, ACTION_NAMES[int(marg_policy_col[i, 0])], ha='center', va='center', fontsize=6, color='white')

plt.tight_layout()
plt.savefig('weather_comparison.png', dpi=150)
plt.show()
print("\nPlot saved → weather_comparison.png")
