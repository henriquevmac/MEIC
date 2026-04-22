import random
import pickle

"""
Agent classes for the fishing game.
Students should create their own agent classes by inheriting from the base Agent class.
"""


class Agent:
    """
    Base class for all fishing game agents.
    Students should inherit from this class and implement their own strategies.
    """

    def __init__(self):
        """Initialize the agent. Override this method if you need to set up any state."""
        pass

    def get_action(self, state):
        """
        Decide what action to take given the current game state.

        Args:
            state (dict): Dictionary containing:
                - 'fish_y': Y position of the fish
                - 'bar_y': Y position of the green bar
                - 'bar_vel': Velocity of the bar
                - 'catch_timer': Current catch timer value

        Returns:
            bool: True to thrust upward, False to let gravity pull down
        """
        pass

    def learn(self, state, action, reward, next_state, next_action, done):
        """
        Optional: Update internal state/policy based on experience.

        Args:
            state:       The state in which the action was taken
            action:      The action that was taken
            reward:      Reward received after taking the action
            next_state:  The state observed after the action
            next_action: The action chosen in next_state (used by on-policy methods like SARSA)
            done:        Whether the episode has ended
        """
        pass

    def end_episode(self):
        """
        Optional: Called at the end of each episode.
        """
        pass

    def set_training_mode(self, training):
        """
        Optional: Switch between training (exploration) and testing (exploitation) modes.
        """
        pass

    def save_q_table(self, filename):
        """
        Optional: Save Q-table to file.
        """
        pass

    def load_q_table(self, filename):
        """
        Optional: Load Q-table from file.
        """
        pass


class PredictiveAgent(Agent):
    """
    Example: An agent that tries to predict where the fish will be.
    This considers the bar's velocity to make smoother movements.
    """

    def __init__(self, reaction_distance=20):
        """
        Initialize the predictive agent.

        Args:
            reaction_distance: How far ahead to start reacting (in pixels)
        """
        super().__init__()
        self.reaction_distance = reaction_distance

    def get_action(self, state):
        """
        Thrust if fish is above the bar center, accounting for reaction distance.

        Args:
            state (dict): Current game state

        Returns:
            bool: True to thrust, False to fall
        """
        bar_center = state["bar_y"] + 40  # Bar is 80 pixels tall, so center is +40

        # If fish is above bar center (with buffer), thrust
        if state["fish_y"] < bar_center - self.reaction_distance:
            return True
        return False


class TDAgent(Agent):
    """
    Base class for tabular Temporal-Difference agents (Q-Learning, SARSA, …).

    Provides the shared machinery:
      - Hyperparameters (alpha, gamma, epsilon / decay)
      - Q-table storage and lookup
      - Epsilon-greedy action selection
      - State discretization
      - Episode bookkeeping (epsilon decay, episode counter)
      - Training-mode toggle
      - Q-table save / load

    Derived classes only need to implement learn().
    """

    def __init__(
        self,
        alpha=0.1,  # Learning rate
        gamma=0.99,  # Discount factor
        epsilon=0.1,  # Exploration rate
        epsilon_decay=0.995,  # Epsilon decay per episode
        epsilon_min=0.01,  # Minimum epsilon
    ):
        """
        Initialize the TD agent.

        Args:
            alpha:         Learning rate (how much to update Q-values)
            gamma:         Discount factor (how much to value future rewards)
            epsilon:       Initial exploration rate (probability of random action)
            epsilon_decay: Factor to decay epsilon after each episode
            epsilon_min:   Minimum epsilon value
        """
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table: maps (discrete_state, action) -> Q-value
        self.q_table = {}

        # Training mode flag
        self.training = True

        # Statistics
        self.episodes_trained = 0

    # ------------------------------------------------------------------
    # State discretization
    # ------------------------------------------------------------------

    def discretize_state(self, state):
        """
        Convert continuous state to discrete bins for the Q-table.

        Args:
            state (dict[str, float]): Continuous game state

        Returns:
            tuple[int|float, ...]: Discretized state as a hashable tuple
        """
        
        rel_pos = state["fish_y"] - (state["bar_y"] + 40)  # Relative position to bar center
        bar_vel = state["bar_vel"]
        
        rel_pos_clipped = max(-460, min(460, rel_pos))
        vel_clipped = max(-17, min(11.5, bar_vel))
        
        n_rel_bins = 20
        n_vel_bins = 10
        
        rel_bin = int((rel_pos_clipped + 460) / 920 * n_rel_bins)
        vel_bin = int((vel_clipped + 17) / 28.5 * n_vel_bins)
        
        rel_bin=min(n_rel_bins - 1, max(0, rel_bin))
        vel_bin=min(n_vel_bins - 1, max(0, vel_bin))
        return (rel_bin, vel_bin)

    # ------------------------------------------------------------------
    # Q-value helpers
    # ------------------------------------------------------------------

    def get_q_value(self, state, action):
        """
        Get Q-value for a (discrete) state-action pair.

        Args:
            state:  Discretized state tuple
            action: Action (True or False)

        Returns:
            float: Q-value (0.0 if not seen before)
        """
        return self.q_table.get((state, action), 0.0)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def get_action(self, state):
        """
        Choose action, e.g., using epsilon-greedy strategy.

        Args:
            state (dict[str, float]): Original game state (not yet discretized).

        Returns:
            bool: True to thrust, False to fall
        """
        disc_state = self.discretize_state(state)
        
        if self.training and random.random() < self.epsilon:
            return random.choice([True, False])
        else:
                # Exploit: choose action with highest Q-value
                q_true = self.get_q_value(disc_state, True)
                q_false = self.get_q_value(disc_state, False)
                if q_true > q_false:
                    return True
                elif q_false > q_true:
                    return False
                else:
                    return random.choice([True, False])

    # ------------------------------------------------------------------
    # Episode bookkeeping
    # ------------------------------------------------------------------

    def end_episode(self):
        """Decay epsilon and increment episode counter."""
        self.episodes_trained += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Training mode
    # ------------------------------------------------------------------

    def set_training_mode(self, training):
        """
        Set whether the agent is in training or evaluation mode.

        Args:
            training (bool): If True, use epsilon-greedy. If False, always exploit.
        """
        self.training = training

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_q_table(self, filepath):
        """Save Q-table to file."""
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "q_table": self.q_table,
                    "episodes_trained": self.episodes_trained,
                    "epsilon": self.epsilon,
                },
                f,
            )
        print(f"Q-table saved to {filepath}")

    def load_q_table(self, filepath):
        """Load Q-table from file."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.q_table = data["q_table"]
            self.episodes_trained = data.get("episodes_trained", 0)
            self.epsilon = data.get("epsilon", self.epsilon)
        print(
            f"Q-table loaded from {filepath} ({len(self.q_table)} entries, {self.episodes_trained} episodes)"
        )


class QLearningAgent(TDAgent):
    """
    Tabular Q-Learning agent (off-policy TD control).

    Update rule:

    The max over next actions makes this off-policy: it always learns towards
    the greedy policy regardless of which action is actually taken next.
    """

    def learn(self, state, action, reward, next_state, next_action, done):
        """
        Update Q-table using the Q-learning (off-policy) rule.

        Args:
            state:       State in which the action was taken (dict)
            action:      Action that was taken
            reward:      Reward received
            next_state:  State observed after the action (dict)
            next_action: Action chosen in next_state (ignored by Q-learning)
            done:        Whether the episode has ended
        """
        if not self.training:
            return
        
        s = self.discretize_state(state)
        s_next = self.discretize_state(next_state)
        
        # Get current Q-value
        q_current = self.get_q_value(s, action)
        if done:
            td_target = reward
        else:
            q_next_max = max(self.get_q_value(s_next, True), self.get_q_value(s_next, False))
            td_target = reward + self.gamma * q_next_max

        # Update Q-value
        td_error = td_target - q_current
        self.q_table[(s, action)] = q_current + self.alpha * td_error


class SarsaLearningAgent(TDAgent):
    """
    SARSA Agent (on-policy TD control).

    Update rule:

    Uses the action actually chosen in the next state (a'), making this
    on-policy: the learned values reflect the agent's own behaviour policy.
    """

    def learn(self, state, action, reward, next_state, next_action, done):
        """
        Update Q-table using the SARSA (on-policy) rule.

        Args:
            state:       State in which the action was taken (dict)
            action:      Action that was taken
            reward:      Reward received
            next_state:  State observed after the action (dict)
            next_action: Action chosen in next_state (used by SARSA)
            done:        Whether the episode has ended
        """
        if not self.training:
            return
        
        s = self.discretize_state(state)
        s_next = self.discretize_state(next_state)
        
        # Get current Q-value
        q_current = self.get_q_value(s, action)
        if done:
            td_target = reward
        else:
            q_next = self.get_q_value(s_next, next_action)
            td_target = reward + self.gamma * q_next

        # Update Q-value
        td_error = td_target - q_current
        self.q_table[(s, action)] = q_current + self.alpha * td_error

class ImprovedQLearningAgent(QLearningAgent):
    """
    Improved Q-Learning agent that extends the state representation with
    an estimated fish velocity, computed from consecutive fish positions.
 
    Motivation
    ----------
    The base agent's state (rel_pos, bar_vel) captures where the fish
    currently is relative to the bar and how fast the bar is moving, but
    it has no information about where the fish is *going*. A fish moving
    rapidly upward requires a different response than a stationary fish at
    the same position. By estimating fish velocity from two consecutive
    observations, the agent can anticipate future fish positions and act
    proactively rather than reactively.
 
    Since fishing_logic.py cannot be modified, fish velocity is not
    available directly from get_state(). Instead, it is approximated as:
 
        fish_vel_est = fish_y(t) - fish_y(t-1)
 
    This estimate is maintained internally across steps and reset at the
    start of each episode via end_episode().
 
    State space: 20 x 10 x 7 = 1400 discrete states
      - rel_pos bin  (20): signed distance from bar center to fish
      - bar_vel bin  (10): current bar velocity
      - fish_vel bin  (7): estimated fish velocity [-30, +30]
    """
 
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prev_fish_y = None  # Previous fish position for velocity estimation
 
    def discretize_state(self, state):
        """
        Extended discretization including estimated fish velocity.
 
        Args:
            state (dict[str, float]): Continuous game state.
 
        Returns:
            tuple[int, int, int]: (rel_bin, vel_bin, fish_vel_bin)
        """
        rel_pos = state["fish_y"] - (state["bar_y"] + 40)
        bar_vel = state["bar_vel"]
 
        # Estimate fish velocity from consecutive positions
        if self.prev_fish_y is None:
            fish_vel_est = 0.0
        else:
            fish_vel_est = state["fish_y"] - self.prev_fish_y
 
        # Update previous fish position
        self.prev_fish_y = state["fish_y"]
 
        # Clip to known ranges
        rel_pos_clipped = max(-460, min(460, rel_pos))
        vel_clipped = max(-17, min(11.5, bar_vel))
        fish_vel_clipped = max(-30, min(30, fish_vel_est))
 
        # Bin counts
        n_rel_bins = 20
        n_vel_bins = 10
        n_fish_vel_bins = 7
 
        rel_bin = int((rel_pos_clipped + 460) / 920 * n_rel_bins)
        vel_bin = int((vel_clipped + 17) / 28.5 * n_vel_bins)
        fish_vel_bin = int((fish_vel_clipped + 30) / 60 * n_fish_vel_bins)
 
        rel_bin = min(n_rel_bins - 1, max(0, rel_bin))
        vel_bin = min(n_vel_bins - 1, max(0, vel_bin))
        fish_vel_bin = min(n_fish_vel_bins - 1, max(0, fish_vel_bin))
 
        return (rel_bin, vel_bin, fish_vel_bin)
 
    def end_episode(self):
        """Reset fish velocity estimate at the end of each episode."""
        super().end_episode()
        self.prev_fish_y = None