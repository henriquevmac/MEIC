"""
Runner for Fishing Game AI
Covers Q1.1 (Q-Learning vs SARSA comparison) and Q1.2 (Convergence Analysis).
"""

import random
import numpy as np
import matplotlib.pyplot as plt

from fishing_logic import FishingGameLogic, FISH_TYPES
from agents import PredictiveAgent, QLearningAgent, SarsaLearningAgent, ImprovedQLearningAgent


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_agent(
    agent,
    fish_types=None,
    num_episodes=2500,
    do_learning=True,
    verbose=True,
    visualize=False,
):
    """
    Run the agent in the environment for a set number of episodes.
    If do_learning=True, the agent will explore and update its policy.
    If do_learning=False, the agent will strictly exploit its current policy.
    """
    assert fish_types or num_episodes

    if fish_types:
        num_episodes = len(fish_types)

    mode_str = "Training" if do_learning else "Testing"

    # Handle agent state for testing
    if not do_learning and hasattr(agent, "set_training_mode"):
        agent.set_training_mode(False)
    elif verbose:
        print(f"\n{'=' * 60}")
        print(f"{mode_str} {agent.__class__.__name__}")
        print(f"{'=' * 60}")

    visualizer = None

    if visualize:
        from visualize import GameVisualizer

        visualizer = GameVisualizer(
            f"{agent.__class__.__name__} ({mode_str})", "Random"
        )

    wins = 0
    total_cost = 0
    total_steps = 0
    costs_history = []

    for episode in range(num_episodes):
        game = FishingGameLogic(fish_name=fish_types[episode] if fish_types else None)
        done = False
        episode_cost = 0
        steps = 0

        while not done and steps < 2500:
            state = game.get_state()
            action = agent.get_action(state)

            next_state, cost, done = game.step_physics(action)
            next_action = agent.get_action(next_state)

            if visualizer:
                visualizer.update(game.get_state(), done)

            if do_learning:
                agent.learn(state, action, -cost, next_state, next_action, done)

            episode_cost += cost
            steps += 1

        if do_learning:
            agent.end_episode()

        costs_history.append(episode_cost)
        total_cost += episode_cost
        total_steps += steps

        if game.catch_timer > 0:
            wins += 1

        # Print progress only when training and verbose
        if do_learning and verbose and (episode + 1) % 100 == 0:
            win_rate = wins / (episode + 1) * 100
            epsilon = getattr(agent, "epsilon", "N/A")
            print(
                f"Episode {episode + 1}/{num_episodes} | "
                f"Win Rate: {win_rate:5.1f}% | "
                f"ε: {epsilon if epsilon == 'N/A' else f'{epsilon:.3f}'}"
            )

    # Re-enable training if it was disabled for testing
    if not do_learning and hasattr(agent, "set_training_mode"):
        agent.set_training_mode(True)

    if visualizer:
        import time
        time.sleep(1)
        visualizer.close()

    win_rate = wins / num_episodes * 100
    avg_cost = total_cost / num_episodes
    avg_steps = total_steps / num_episodes

    return {
        "wins": wins,
        "win_rate": win_rate,
        "avg_cost": avg_cost,
        "avg_steps": avg_steps,
        "costs_history": costs_history,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_experiment(agent_factory, num_runs, num_train_episodes, test_fish_types,
                   train_fish_types=None, verbose_first_run=True):
    """
    Run an agent factory num_runs times, each time training then testing.

    Returns dicts with lists of per-run arrays/scalars:
        train_costs   : list of cumulative-cost arrays (one per run)
        test_costs    : list of cumulative-cost arrays (one per run)
        win_rates     : list of floats
        avg_costs     : list of floats
        avg_steps     : list of floats
    """
    train_costs, test_costs = [], []
    win_rates, avg_costs, avg_steps = [], [], []

    for run in range(num_runs):
        seed = run * 42
        random.seed(seed)
        np.random.seed(seed)

        agent = agent_factory()

        train_stats = run_agent(
            agent,
            fish_types=train_fish_types,
            num_episodes=num_train_episodes,
            do_learning=True,
            verbose=(verbose_first_run and run == 0),
        )
        train_costs.append(np.cumsum(train_stats["costs_history"]))

        test_stats = run_agent(
            agent,
            fish_types=test_fish_types,
            do_learning=False,
            verbose=False,
        )
        test_costs.append(np.cumsum(test_stats["costs_history"]))
        win_rates.append(test_stats["win_rate"])
        avg_costs.append(test_stats["avg_cost"])
        avg_steps.append(test_stats["avg_steps"])

        print(
            f"  Run {run + 1}/{num_runs} | "
            f"Win Rate: {test_stats['win_rate']:.1f}% | "
            f"Avg Cost: {test_stats['avg_cost']:.1f} | "
            f"Avg Steps: {test_stats['avg_steps']:.1f}"
        )

    return {
        "train_costs": train_costs,
        "test_costs": test_costs,
        "win_rates": win_rates,
        "avg_costs": avg_costs,
        "avg_steps": avg_steps,
    }


def plot_mean_std(ax, runs_data, label, color=None):
    """Plot mean ± std dev of a list of cumulative-cost arrays."""
    matrix = np.array(runs_data)
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    p = ax.plot(mean, label=label, color=color)
    c = p[0].get_color()
    ax.fill_between(range(len(mean)), mean - std, mean + std, color=c, alpha=0.25)
    return c


def print_summary(label, res):
    m_win = np.mean(res["win_rates"])
    m_cost = np.mean(res["avg_costs"])
    m_steps = np.mean(res["avg_steps"])
    print(f"  {label:<35} Win: {m_win:5.1f}%  AvgCost: {m_cost:7.1f}  AvgSteps: {m_steps:7.1f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    NUM_TRAIN_EPISODES = 5000
    NUM_RUNS_Q11 = 1   # Q1.1: single run is fine for a visual comparison
    NUM_RUNS_Q12 = 5   # Q1.2: 5 seeds for robustness analysis

    # Standard test suite: 50 episodes per fish type (covers all fish)
    test_fish_types = [fish.name for fish in FISH_TYPES for _ in range(50)]

    # Base Q-Learning factory (used throughout Q1.2)
    BASE_EPSILON       = 1.0
    BASE_EPSILON_DECAY = 0.999
    BASE_ALPHA         = 0.1
    BASE_GAMMA         = 0.99

    def base_qlearning():
        return QLearningAgent(
            alpha=BASE_ALPHA,
            gamma=BASE_GAMMA,
            epsilon=BASE_EPSILON,
            epsilon_decay=BASE_EPSILON_DECAY,
        )

    # -----------------------------------------------------------------------
    # Q1.1 — Q-Learning vs SARSA vs PredictiveAgent
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Q1.1  Q-Learning vs SARSA vs PredictiveAgent")
    print("=" * 70)

    q11_configs = [
        ("PredictiveAgent", PredictiveAgent),
        ("Q-Learning",      lambda: QLearningAgent(epsilon=BASE_EPSILON, epsilon_decay=BASE_EPSILON_DECAY)),
        ("SARSA",           lambda: SarsaLearningAgent(epsilon=BASE_EPSILON, epsilon_decay=BASE_EPSILON_DECAY)),
    ]

    q11_results = {}
    for name, factory in q11_configs:
        print(f"\n--- {name} ---")
        q11_results[name] = run_experiment(
            factory, NUM_RUNS_Q11, NUM_TRAIN_EPISODES, test_fish_types,
            verbose_first_run=True,
        )

    print("\nQ1.1 Summary:")
    print(f"  {'Agent':<35} {'Win%':>6}  {'AvgCost':>9}  {'AvgSteps':>9}")
    print("  " + "-" * 65)
    for name in q11_results:
        print_summary(name, q11_results[name])

    # Q1.1 Figure
    fig11, (ax11a, ax11b) = plt.subplots(1, 2, figsize=(14, 5))
    fig11.suptitle("Q1.1 — Training & Testing Cumulative Costs", fontsize=13)
    for name, res in q11_results.items():
        plot_mean_std(ax11a, res["train_costs"], name)
        plot_mean_std(ax11b, res["test_costs"],  name)
    for ax, title in [(ax11a, "Training"), (ax11b, "Testing")]:
        ax.set_title(title + ": Cumulative Cost")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative Cost")
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig("q11_comparison.png", dpi=120)
    print("\nSaved: q11_comparison.png")

    # -----------------------------------------------------------------------
    # Q1.2.1 — Multi-seed robustness of Q-Learning
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Q1.2.1  Q-Learning — Multi-seed Robustness (5 seeds)")
    print("=" * 70)

    res_q121 = run_experiment(
        base_qlearning, NUM_RUNS_Q12, NUM_TRAIN_EPISODES, test_fish_types,
        verbose_first_run=False,
    )

    print("\nQ1.2.1 Summary (Q-Learning, 5 seeds):")
    print_summary("Q-Learning (base)", res_q121)

    fig121, (ax121a, ax121b) = plt.subplots(1, 2, figsize=(14, 5))
    fig121.suptitle("Q1.2.1 — Q-Learning Robustness (5 seeds, mean ± std)", fontsize=13)
    plot_mean_std(ax121a, res_q121["train_costs"], f"Q-Learning (ε={BASE_EPSILON}, decay={BASE_EPSILON_DECAY})")
    plot_mean_std(ax121b, res_q121["test_costs"],  f"Q-Learning (ε={BASE_EPSILON}, decay={BASE_EPSILON_DECAY})")
    for ax, title in [(ax121a, "Training"), (ax121b, "Testing")]:
        ax.set_title(title + ": Cumulative Cost (mean ± std)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative Cost")
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig("q121_robustness.png", dpi=120)
    print("Saved: q121_robustness.png")

    # -----------------------------------------------------------------------
    # Q1.2.3a — Epsilon parameter experiments
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Q1.2.3a  Impact of epsilon (initial exploration rate)")
    print("=" * 70)

    epsilon_configs = {
        f"ε=1.0 (base)":  lambda: QLearningAgent(alpha=BASE_ALPHA, gamma=BASE_GAMMA, epsilon=1.0,  epsilon_decay=BASE_EPSILON_DECAY),
        f"ε=0.5":         lambda: QLearningAgent(alpha=BASE_ALPHA, gamma=BASE_GAMMA, epsilon=0.5,  epsilon_decay=BASE_EPSILON_DECAY),
        f"ε=0.1":         lambda: QLearningAgent(alpha=BASE_ALPHA, gamma=BASE_GAMMA, epsilon=0.1,  epsilon_decay=BASE_EPSILON_DECAY),
    }

    eps_results = {}
    for label, factory in epsilon_configs.items():
        print(f"\n--- {label} ---")
        eps_results[label] = run_experiment(
            factory, NUM_RUNS_Q12, NUM_TRAIN_EPISODES, test_fish_types,
            verbose_first_run=False,
        )

    print("\nQ1.2.3a Summary:")
    print(f"  {'Config':<35} {'Win%':>6}  {'AvgCost':>9}  {'AvgSteps':>9}")
    print("  " + "-" * 65)
    for label, res in eps_results.items():
        print_summary(label, res)

    fig123a, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig123a.suptitle("Q1.2.3a — Impact of Initial Epsilon", fontsize=13)
    ax_tr, ax_te, ax_st = axes
    for label, res in eps_results.items():
        c = plot_mean_std(ax_tr, res["train_costs"], label)
        plot_mean_std(ax_te, res["test_costs"],  label, color=c)
        steps_mean = np.mean(res["avg_steps"])
        steps_std  = np.std(res["avg_steps"])
        ax_st.bar(label, steps_mean, yerr=steps_std, capsize=5, alpha=0.7)
    ax_tr.set_title("Training: Cumulative Cost")
    ax_tr.set_xlabel("Episode"); ax_tr.set_ylabel("Cumulative Cost")
    ax_tr.legend(); ax_tr.grid(True)
    ax_te.set_title("Testing: Cumulative Cost")
    ax_te.set_xlabel("Episode"); ax_te.set_ylabel("Cumulative Cost")
    ax_te.legend(); ax_te.grid(True)
    ax_st.set_title("Testing: Avg Steps")
    ax_st.set_ylabel("Avg Steps"); ax_st.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig("q123a_epsilon.png", dpi=120)
    print("Saved: q123a_epsilon.png")

    # -----------------------------------------------------------------------
    # Q1.2.3b — Epsilon decay parameter experiments
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Q1.2.3b  Impact of epsilon_decay")
    print("=" * 70)

    decay_configs = {
        f"decay=0.999 (base)": lambda: QLearningAgent(alpha=BASE_ALPHA, gamma=BASE_GAMMA, epsilon=BASE_EPSILON, epsilon_decay=0.999),
        f"decay=0.99 (fast)":  lambda: QLearningAgent(alpha=BASE_ALPHA, gamma=BASE_GAMMA, epsilon=BASE_EPSILON, epsilon_decay=0.99),
        f"decay=0.9995 (slow)":lambda: QLearningAgent(alpha=BASE_ALPHA, gamma=BASE_GAMMA, epsilon=BASE_EPSILON, epsilon_decay=0.9995),
    }

    decay_results = {}
    for label, factory in decay_configs.items():
        print(f"\n--- {label} ---")
        decay_results[label] = run_experiment(
            factory, NUM_RUNS_Q12, NUM_TRAIN_EPISODES, test_fish_types,
            verbose_first_run=False,
        )

    print("\nQ1.2.3b Summary:")
    print(f"  {'Config':<35} {'Win%':>6}  {'AvgCost':>9}  {'AvgSteps':>9}")
    print("  " + "-" * 65)
    for label, res in decay_results.items():
        print_summary(label, res)

    fig123b, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig123b.suptitle("Q1.2.3b — Impact of Epsilon Decay", fontsize=13)
    ax_tr, ax_te, ax_st = axes
    for label, res in decay_results.items():
        c = plot_mean_std(ax_tr, res["train_costs"], label)
        plot_mean_std(ax_te, res["test_costs"],  label, color=c)
        steps_mean = np.mean(res["avg_steps"])
        steps_std  = np.std(res["avg_steps"])
        ax_st.bar(label, steps_mean, yerr=steps_std, capsize=5, alpha=0.7)
    ax_tr.set_title("Training: Cumulative Cost")
    ax_tr.set_xlabel("Episode"); ax_tr.set_ylabel("Cumulative Cost")
    ax_tr.legend(); ax_tr.grid(True)
    ax_te.set_title("Testing: Cumulative Cost")
    ax_te.set_xlabel("Episode"); ax_te.set_ylabel("Cumulative Cost")
    ax_te.legend(); ax_te.grid(True)
    ax_st.set_title("Testing: Avg Steps")
    ax_st.set_ylabel("Avg Steps"); ax_st.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig("q123b_decay.png", dpi=120)
    print("Saved: q123b_decay.png")

    # -----------------------------------------------------------------------
    # Q1.2.4 — Single fish type (Sturgeon) vs mixed training
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Q1.2.4  Single-fish training (Sturgeon) vs Mixed training")
    print("=" * 70)

    # Sturgeon-only training episodes (same count as mixed)
    sturgeon_train = ["Sturgeon"] * NUM_TRAIN_EPISODES

    q124_configs = {
        "Mixed training (base)":    (base_qlearning, None),
        "Sturgeon-only training":   (base_qlearning, sturgeon_train),
    }

    q124_results = {}
    for label, (factory, train_types) in q124_configs.items():
        print(f"\n--- {label} ---")
        q124_results[label] = run_experiment(
            factory, NUM_RUNS_Q12, NUM_TRAIN_EPISODES, test_fish_types,
            train_fish_types=train_types,
            verbose_first_run=False,
        )

    print("\nQ1.2.4 Summary:")
    print(f"  {'Config':<35} {'Win%':>6}  {'AvgCost':>9}  {'AvgSteps':>9}")
    print("  " + "-" * 65)
    for label, res in q124_results.items():
        print_summary(label, res)

    fig124, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig124.suptitle("Q1.2.4 — Single-fish vs Mixed Training", fontsize=13)
    ax_tr, ax_te, ax_st = axes
    for label, res in q124_results.items():
        c = plot_mean_std(ax_tr, res["train_costs"], label)
        plot_mean_std(ax_te, res["test_costs"],  label, color=c)
        steps_mean = np.mean(res["avg_steps"])
        steps_std  = np.std(res["avg_steps"])
        ax_st.bar(label, steps_mean, yerr=steps_std, capsize=5, alpha=0.7)
    ax_tr.set_title("Training: Cumulative Cost")
    ax_tr.set_xlabel("Episode"); ax_tr.set_ylabel("Cumulative Cost")
    ax_tr.legend(); ax_tr.grid(True)
    ax_te.set_title("Testing: Cumulative Cost (all fish)")
    ax_te.set_xlabel("Episode"); ax_te.set_ylabel("Cumulative Cost")
    ax_te.legend(); ax_te.grid(True)
    ax_st.set_title("Testing: Avg Steps (all fish)")
    ax_st.set_ylabel("Avg Steps"); ax_st.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig("q124_single_fish.png", dpi=120)
    print("Saved: q124_single_fish.png")

    q3_configs = [
        ("Improved Q-Learning", lambda: ImprovedQLearningAgent(epsilon=1.0, epsilon_decay=0.999)),
    ]

    q3_results = {}
    for name, factory in q3_configs:
        print(f"\n--- {name} ---")
        q3_results[name] = run_experiment(
            factory, NUM_RUNS_Q11, NUM_TRAIN_EPISODES, test_fish_types,
            verbose_first_run=True,
        )

    print("\nQ3 Summary:")
    print(f"  {'Agent':<35} {'Win%':>6}  {'AvgCost':>9}  {'AvgSteps':>9}")
    print("  " + "-" * 65)
    for name in q3_results:
        print_summary(name, q3_results[name])

    # Q3 Figure
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))
    fig3.suptitle("Q3 — Improvement", fontsize=13)
    for name, res in q3_results.items():
        plot_mean_std(ax3a, res["train_costs"], name)
        plot_mean_std(ax3b, res["test_costs"],  name)
    for ax, title in [(ax3a, "Training"), (ax3b, "Testing")]:
        ax.set_title(title + ": Cumulative Cost")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative Cost")
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig("q3_improvement.png", dpi=120)
    print("\nSaved: q3_improvement.png")

    # -----------------------------------------------------------------------
    # Show all figures
    # -----------------------------------------------------------------------
    print("\nAll experiments complete. Displaying plots...")
    plt.show()
