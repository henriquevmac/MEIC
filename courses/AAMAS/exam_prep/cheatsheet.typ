// ============================================================================
//  EXAM CHEAT SHEET TEMPLATE  —  front + back A4, maximum density
//  Compile:  typst compile cheatsheet.typ
//  Watch:    typst watch cheatsheet.typ   (live recompile while editing)
// ============================================================================

// ----- DENSITY DIALS --------------------------------------------------------
// Tune these four to make the content "just" fill both sides of the A4.
#let body-size   = 6pt     // drop to 7pt / 6.5pt to cram more in
#let num-columns = 4       // 3 = readable, 4 = dense, 5 = extreme
#let margin-size = 0.4cm   // page margin all around
#let col-gutter  = 0.45em  // horizontal gap between columns
// ----------------------------------------------------------------------------

#set page(
  paper: "a4",
  flipped: false,          // false = portrait. Set true for landscape columns.
  margin: margin-size,
  columns: num-columns,
)
#set text(size: body-size, font: "New Computer Modern", hyphenate: true)
#set par(justify: true, leading: 0.42em, spacing: 0.55em)
#show math.equation: set text(size: body-size)

// Tighter lists
#set list(indent: 0.6em, body-indent: 0.35em, spacing: 0.42em, marker: [‣])
#set enum(indent: 0.6em, body-indent: 0.35em, spacing: 0.42em)

// ----- HELPERS --------------------------------------------------------------

// Section header: colored bar, jumps out when scanning mid-exam.
#let section(title, fill: rgb("#1f4e79")) = block(
  width: 100%,
  inset: (x: 3pt, y: 2pt),
  radius: 1.5pt,
  fill: fill,
  above: 0.55em, below: 0.4em,
  text(fill: white, weight: "bold", size: body-size + 0.5pt, title),
)

// Sub-heading inside a section.
#let sub(title) = text(weight: "bold", fill: rgb("#1f4e79"), title)

// Boxed "remember this" callout (e.g. a key formula or gotcha).
#let key(body) = block(
  width: 100%,
  inset: 3pt,
  radius: 1.5pt,
  stroke: 0.5pt + rgb("#c00000"),
  fill: rgb("#fdeaea"),
  above: 0.4em, below: 0.4em,
  body,
)

// Compact definition: bold term + description on one line.
#let def(term, body) = [#text(weight: "bold")[#term:] #body]

// Keep a block from splitting across columns (good for formulas/tables).
#let nobreak(body) = block(breakable: false, body)

// ============================================================================
//  CONTENT  —  replace everything below this line.
//  Order top-to-bottom, left column first, then it flows to the next column.
// ============================================================================

#section(fill: rgb("#1f4e79"))[Normal Form Games]

#sub[Formal model]
#def[NF game][tuple $(N, A, u)$: $N={1..n}$ agents; $A=A_1 times ... times A_n$ action profiles; $u=(u_1..u_n)$, $u_i: A -> bb(R)$.] A profile $a=(a_i, a_(-i))$; $a_(-i)$ = others' actions. Payoff $u_i (a_i, a_(-i))$. Finite + 2 players $=>$ payoff *matrix* (row=P1, col=P2, cell $= (u_1, u_2)$).

#sub[Modeling a scenario]
Checklist: + How many agents $N$? + Each action set $A_i$? + Payoffs $u_i$ (one number per agent per profile)? Discrete $->$ matrix; continuous $A_i$ $->$ calculus.

#sub[Best response & dominance]
#def[Best response][$a_i in "BR"_i (a_(-i))$ iff $u_i (a_i, a_(-i)) >= u_i (a'_i, a_(-i)) forall a'_i$.]
#def[Strictly dominated][$a_i$ str. dom. by $a'_i$ if $u_i (a'_i, a_(-i)) > u_i (a_i, a_(-i))$ for *every* $a_(-i)$.]
#def[Weakly dominated][$>=$ for all $a_(-i)$ and $>$ for at least one.]

#sub[IESDS — procedure]
+ Find any *strictly* dominated action for some agent.
+ Delete it; reduced game.
+ Repeat (new dominations may appear after deletions).
+ Stop when none remain $->$ surviving profile(s) = prediction.
#key[Use *STRICT* dominance only. Eliminating *weakly* dominated actions can destroy real NE and is order-dependent.]

#sub[Pure-strategy Nash equilibrium]
#def[Pure NE][profile $a^*$ s.t. no agent gains by unilateral deviation: $u_i (a_i^*, a_(-i)^*) >= u_i (a_i, a_(-i)^*) forall i, forall a_i$.]
*Underline / best-response method:* + per column, mark P1's best payoff(s); + per row, mark P2's best payoff(s); + cell with *both* marked = pure NE (may be 0, 1, or many).

#sub[Mixed strategies & mixed NE]
#def[Mixed strategy][prob. distribution $sigma_i$ over $A_i$; pure = degenerate. 2-action: $(q, 1-q)$.] Expected payoff $u_i (sigma) = sum_a (product_j sigma_j (a_j)) u_i (a)$.
#def[Mixed NE][each $sigma_i^*$ maximizes expected payoff vs $sigma_(-i)^*$.]
*Nash's theorem:* every *finite* game has $>=1$ NE (possibly mixed).

#sub[Indifference principle — find mixed NE (2×2)]
+ Let opponent play action w.p. $p$. + Write your $"EU"("act"_1)$, $"EU"("act"_2)$ as fns of $p$. + Set them *equal* (indifference) $->$ solve $p$. + Repeat for other player.
#key[The $p$ that solves "make ME indifferent" is the prob. the *OPPONENT* must play. Mixing probs that make the opponent indifferent are what *YOU* play. Easy to swap players.]

#sub[Canonical 2×2 games]
#nobreak[
*Prisoner's Dilemma (advertising):* C str. dominates $->$ unique $("Adv","Adv")=(3,3)$, both worse than $(5,5)$.
#table(columns: 3, [], [*Adv*], [*Don't*], [*Adv*], [3, 3], [6, 2], [*Don't*], [2, 6], [5, 5])
*Matching Pennies* (zero-sum, outguess): *no pure NE*; mixed NE $(1/2,1/2),(1/2,1/2)$.
#table(columns: 3, [], [*H*], [*T*], [*H*], [-1, 1], [1, -1], [*T*], [1, -1], [-1, 1])
*Battle of Sexes* (coordination): 2 pure NE (Ballet,Ballet),(Fight,Fight); mixed: man$=(1/3,2/3)$, woman$=(2/3,1/3)$ over (Ballet,Fight) — players mix with *different* probs.
#table(columns: 3, [], [*Ballet*], [*Fight*], [*Ballet*], [1, 2], [0, 0], [*Fight*], [0, 0], [2, 1])
]
BoS mixed: woman plays Ballet w.p. $p$; man indiff $1 dot p = 2(1-p) => p=2/3$.

#sub[Cournot duopoly (continuous)]
2 firms, homogeneous good, no collusion, choose quantities $q_i in [0, infinity)$ simultaneously (one-shot). Inverse demand $P(Q)=a-Q$ for $Q<a$ ($Q=q_1+q_2$), const. marg. cost $c<a$, no fixed cost.
Profit: $pi_i (q_i, q_j) = q_i [a - (q_i + q_j) - c]$.
*FOC* (firm 1): $(partial pi_1)/(partial q_1) = a - 2 q_1 - q_2 - c = 0$.
*Best-response (reaction) fns:* $b_i (q_j) = 1/2 (a - q_j - c)$.
NE = intersection of $b_1, b_2$:
#key[$q_1^* = q_2^* = 1/3 (a - c)$ — symmetric Cournot–Nash equilibrium.]
*Discrete version:* enumerate $Q_i in {0..5}$, $P=max(0, 12-2Q)$, $pi_i=(P-1)Q_i$, then apply BR/NE method.

#sub[Interpreting mixed strategies]
Confuse/outguess opponent; uncertainty about other's action; limiting frequency in repeated play; population state (fraction playing each pure strategy). Apps: penalties, patrols, audits — "unpredictable in a precise way."

#section(fill: rgb("#1f4e79"))[Extensive Form Games]

Tree/sequential representation: makes *temporal structure* explicit (normal-form = simultaneous, one-shot). Treat perfect-info first, then imperfect.

#sub[Perfect-info game (tuple)]
$G=(N,A,H,Z,chi,rho,sigma,u)$: $N$ agents; $A$ actions; $H$ choice nodes; $Z$ terminals; $chi: H -> 2^A$ actions-at-node; $rho: H -> N$ who-moves; $sigma: H times A -> H union Z$ successor (each node unique predecessor $=>$ tree); $u_i: Z -> bb(R)$.
- node = choice, edge = action, leaf = payoff vector.

#sub[Pure strategy = complete contingent plan]
Action at *every* own node (perfect) / *every own info set* (imperfect) — even unreachable ones. Pure strategies of $i$:
$ product_(h in H, rho(h)=i) chi(h) quad "(imperfect:" product_(I_(i,j) in I_i) chi(I_(i,j)) ")" $
#key[Strategy specifies actions even at nodes never reached. This causes normal-form *redundancy* and lets some NE rest on *non-credible threats*.]

#sub[Extensive $->$ Normal form (build matrix)]
+ Rows $= S_1$, cols $= S_2$ (full strategy sets, incl. contingent actions).
+ For each pair $(s_1,s_2)$: follow the path it induces from root to a leaf.
+ Cell $=$ payoff vector at that leaf.
+ Find NE the usual (normal-form) way.

Running game: 1 at root $A|B$; 2 plays $C|D$ after $A$, $E|F$ after $B$; 1 plays $G|H$ after $F$. Leaves: $A C(3,8)$, $A D(8,3)$, $B E(5,5)$, $B F G(2,10)$, $B F H(1,0)$. $S_1$: $(A G), (A H), (B G), (B H)$; $S_2$: $(C E), (C F), (D E), (D F)$.
#nobreak[
#table(columns: 5,
[*1\\2*],[CE],[CF],[DE],[DF],
[(A,G)],[3,8],[3,8],[8,3],[8,3],
[(A,H)],[3,8],[3,8],[8,3],[8,3],
[(B,G)],[5,5],[2,10],[5,5],[2,10],
[(B,H)],[5,5],[1,0],[5,5],[1,0],
)
]
Redundancy: 16 normal-form cells vs only 5 leaves — $(A,G)=(A,H)$ (G/H node unreached after $A$). Pure NE: $\{(A G),(C F)\}, \{(A H),(C F)\}, \{(B H),(C E)\}$.
*Thm:* every finite perfect-info game has a *pure-strategy NE* (turns + full observation $=>$ no randomness needed).

#sub[Subgame-perfect equilibrium (SPE)]
#def[Subgame at $h$][restriction of $G$ to descendants of $h$ ($G$ itself is a subgame).]
#def[SPE][profile $s$ s.t. its restriction to *every* subgame is a NE of that subgame. Rules out non-credible threats.]
- Every SPE is a NE; *not* every NE is SPE; $>= 1$ SPE always exists.
- To *disprove* SPE: find one subgame (usually bottom-most) where restricted profile is not a NE. Above: $\{(B H),(C E)\}$ fails (1 plays $H$, but $G$: $2>1$) $=>$ only $\{(A G),(C F)\}$ is SPE.

#sub[Backward induction (computes a SPE)]
Single depth-first pass, *linear* in tree size.
+ Recurse to terminal; return its payoff vector.
+ At node $h$, evaluate each child via recursion.
+ Pick child maximizing the *mover $rho(h)$'s coordinate*; return that whole vector up.
```
BI(h): if h∈Z return u(h)
  best ← −∞
  for a∈χ(h): c ← BI(σ(h,a))
    if c[ρ(h)] > best[ρ(h)]: best ← c
  return best
```
#key[Compare on the moving agent's component $rho(h)$ only; `best` is a full vector $(u_1,..,u_n)$.]
Zero-sum: BI $=$ *minimax* (one number/node, speed up with *alpha-beta pruning*).

#sub[Imperfect information + info sets]
Add $I$: $G=(...,u,I)$, $I_i$ partitions $i$'s nodes into *information sets*. Nodes in same set are *indistinguishable*; must have *same action set* $chi(I_(i,j))$ (else distinguishable).
- Models *simultaneous* moves / hidden info (PD, Poker) — perfect-info trees *cannot*.
#key[Converting a normal-form (simultaneous) game needs an info set linking the 2nd mover's nodes — else the tree wrongly lets him observe move 1.]
- Normal $arrow.l.r$ imperfect-info mappings both exist but are *not bijective*; round-trip gives a bigger but *strategically equivalent* game (same strategy space + equilibria).

#sub[Perfect vs imperfect & randomization]
#table(columns: 2,
[*Perfect info*],[*Imperfect info*],
[knows current node + all past moves],[some own-nodes indistinguishable],
[pure-strat NE exists],[may need mixed (coordination)],
[BI direct],[only over proper subgames],
)
- *Mixed*: one lottery over pure strategies at start. *Behavioral*: independent draw at each info set.
- *Kuhn (1953)*: under *perfect recall* mixed $arrow.l.r$ behavioral are equivalent. Under *imperfect recall* they can differ (behavioral may do strictly better).

#section(fill: rgb("#1f4e79"))[Bayesian Games]
#def[Bayesian game][game of *incomplete information*: agents are uncertain about *which game is played*. Uncertainty = prob. distribution over possible games.]
Two assumptions: (1) games differ *only in payoffs* (same $N$, same action sets); (2) beliefs are *posteriors* from conditioning a *common prior* on private signals.

#sub[Two equivalent definitions]
- *Info-sets:* $(N, G, P, I)$ — agents, set of games (same actions, diff payoffs), common prior $P in Pi(G)$, partitions $I=(I_1,...,I_N)$ of $G$ (each $I_i$ = $i$'s indistinguishable-game classes).
- *Types:* $(N, A, Theta, p, u)$ — $A=A_1 times...times A_n$; type space $Theta=Theta_1 times...times Theta_n$; common prior $p:Theta->[0,1]$; $u_i: A times Theta -> bb(R)$.
- *Equivalence:* each partition cell $<->$ one type ($I_(i,k) -> theta_(i,k)$); prior over games $P -> $ prior over type profiles $p$.

#def[Type $theta_i$][agent $i$'s *private info* (indexes its payoff fn). $i$ *knows own type*, not others'. $(theta,a)$ fully determines all payoffs.]

#sub[Strategies & timing]
- *Pure strategy:* $alpha_i : Theta_i -> A_i$ — maps *every* type of $i$ to an action (contingent plan, one action per type). Equilibrium checked *per type*.
- *Timing of info:* #box[*ex-ante* knows no type (not even own); *interim* knows own $theta_i$ only; *ex-post* knows all types.]

#sub[Interim expected utility]
$ E U_i (alpha mid theta_i) = sum_(theta_(-i) in Theta_(-i)) p(theta_(-i) mid theta_i) thin u_i (alpha, theta_i, theta_(-i)) $
$p(theta_(-i) mid theta_i)$ = *posterior* over others' types given own type; $alpha$ evaluated = action each type plays.

#sub[Bayesian Nash Equilibrium (BNE)]
Profile $alpha$ s.t. *for each $i$ and each $theta_i in Theta_i$*:
$ alpha_i in arg max_(alpha'_i) E U_i (alpha'_i, alpha_(-i) mid theta_i) $
Every *type* best-responds, expecting over *both actions and types* of others.

#key[BNE is an *interim* concept (each type best-responds). Use posterior $p(theta_(-i)mid theta_i)$, NOT raw joint $p(theta_i,theta_(-i))$ — only equal up to normalization when types independent. A type with a *strictly dominant action* must play it in any BNE.]

#sub[Compute BNE — procedure]
+ List each agent's types; pure strat = action choice per type.
+ Per type, check *strict dominance*: if one action beats all others for that type regardless of others, fix it.
+ For remaining types, compute $E U_i(dot mid theta_i)$ via posterior over $theta_(-i)$; pick best response.
+ Profile is BNE iff *all* type best-responses hold simultaneously.

#sub[Canonical ex: Firm hires Worker]
$N={F,W}$; $A_F={"hire","dont"}$, $A_W={"work","shirk"}$; $Theta_F={t_f}$, $Theta_W={"high","low"}$; prior $p("high")=p$.
#nobreak[#table(columns: 3, [*$theta_W="high"$*],[work],[shirk],[hire],[$1,2$],[$0,1$],[dont],[$0,0$],[$0,0$])]
#nobreak[#table(columns: 3, [*$theta_W="low"$*],[work],[shirk],[hire],[$1,1$],[$-1,2$],[dont],[$0,0$],[$0,0$])]
Worker high: work($2$)>shirk($1$). Worker low: shirk($2$)>work($1$). So $alpha_W^*=("work","shirk")$. Firm vs $alpha_W^*$: $E U_F("hire")=p(1)+(1-p)(-1)=2p-1$, $E U_F("dont")=0$. Hire iff $p>=1/2$. At $p=3/4$: $1/2>0 =>$ hire $=> alpha^*$ is BNE.

#sub[Modeling tip]
Nature draws types from prior; each agent observes own signal, then acts. Hide Nature's choice from uninformed agents via an *information set* (extensive form) $<=>$ private types.

#section(fill: rgb("#1f4e79"))[Repeated Games]

A *stage game* = a normal-form game played repeatedly. Key questions: can agents *observe* others' actions, *remember* the past, and what is *total utility*?

#sub[Finitely-Repeated $G(T)$]
- Modelled as *extensive-form, imperfect info* (don't see this round's simultaneous move, but recall past). Total payoff = *sum* of stage payoffs.
- #def[Proposition][If stage game $G$ has a *unique* NE, then for any finite $T$, $G(T)$ has a *unique* outcome: that NE played *every* stage (by backward induction).]
- *Backward induction:* last stage has no future $=>$ play stage-NE; knowing that, 2nd-last unravels too, back to round 1.
- #key[Finitely-repeated PD: cooperation is *impossible*. Rational agents defect every round. "Always defect" *is* a NE.]

#sub[Infinitely-Repeated]
Infinite tree; naive sum $= infinity$, so redefine utility over payoffs $r_1, r_2, dots$:
#nobreak[
$ "Average:" lim_(k->infinity) sum_(j=1)^k (r_j)/k quad quad "Discounted:" sum_(j=1)^infinity beta^j r_j, #h(4pt) 0<beta<1 $
]
#key[*Geometric series* (the workhorse for infinite streams): $sum_(t=0)^infinity beta^t = 1/(1-beta)$. So a constant $c$ every period *from now* $= c/(1-beta)$; starting *next* period $= beta c/(1-beta)$.]
- #def[Discount factor $beta$][weight on the future. Low $beta$ = impatient (cares about near term); high $beta$ = patient. Also = prob. game continues.]
- Pure strategy maps history to action: $s_i: H -> A_i$, $H = union_t H^t$.
- *Tit-for-tat:* cooperate; if opp defects, defect *one* round then forgive.
- *Grim trigger:* cooperate; if opp *ever* defects, defect *forever*.
- #def[SPE][NE in *every* subgame (after every history). Repeating a stage-game NE is *always* an SPE.]

#sub[Folk Theorem]
Let $a$ = NE of stage game $G$. If $a'$ gives $u_i (a') > u_i (a)$ #emph[for all] $i$ (Pareto-beats the NE), then $exists beta in (0,1)$ s.t. for patient enough players ($beta_i >= beta$), there is an *SPE* of the infinite game playing $a'$ every period. Enforced by *trigger* strategies.

#sub[Sustaining cooperation: deviation algebra]
Trigger: play $a'$; if anyone deviates, play NE $a$ forever (punishment is SPE since $a$ is stage-NE). Define one-shot temptation and per-period punishment loss:
#nobreak[
$ M = max_(i, a''_i) u_i (a''_i, a'_(-i)) - u_i (a'), quad m = min_i u_i (a') - u_i (a) $
]
Net gain from deviating $= M - m (beta_i)/(1-beta_i)$. Deviation *not* worth it iff:
#key[ $ M - m (beta_i)/(1-beta_i) <= 0 #h(4pt) <=> #h(4pt) beta_i >= M/(M+m) quad forall i $ ]

#sub[Canonical example — PD, sustain $(C,C)$]
#nobreak[
#table(columns: 3, align: center,
[], [*C*], [*D*],
[*C*], [$3,3$], [$0,5$],
[*D*], [$5,0$], [$1,1$],
)]
Want $(C,C)=3$; stage NE $=(D,D)=1$. Compare discounted streams (first term *undiscounted*):
- Cooperate: $3/(1-beta)$ #h(8pt) Defect (then punished at $1$): $5 + beta/(1-beta)$
- Diff $= beta (2)/(1-beta) - 2 >= 0 #h(4pt) => #h(4pt) bold(beta >= 1/2)$.
- *Interpretation:* must value tomorrow at least half as much as today.
- Bigger temptation ($D$ payoff $10$ not $5$): diff $= -7 + beta 2/(1-beta) => beta >= 7/9$ (more patience needed).

#key[*Trap:* finite + unique NE $=>$ no cooperation (unravels from last round); infinite + high $beta$ $=>$ cooperation possible. TFT forgives after 1 round; Trigger punishes forever (Folk proof uses Trigger).]

#section(fill: rgb("#548235"))[Multiagent Learning]

#sub[Stochastic (Markov) Game]
#def[Tuple $(Q,N,A,P,r)$][$Q$ states/games, $N$ = $n$ agents, $A=A_1 times dots.c times A_n$ joint actions, $P(q,a,hat(q)) in [0,1]$ transition, $r_i: Q times A -> bb(R)$.] Generalizes: 1 state $=>$ *repeated game*; 1 agent $=>$ *MDP*.

#sub[Learning taxonomy (who do you model?)]
#nobreak[
#table(columns: 2,
[*Type*],[*Q-function / how others treated*],
[Independent],[$Q_i (s_i,a_i)$; others = environment],
[Centralized],[one entity decides all],
[JAL],[$Q_i (s, a_1..a_n)$; need solution concept $C$],
[JAL-AM],[JAL + opponent *model* $hat(pi)_j$],
)]
Decision tree: intentions? No$->$Indep. Yes$->$single agent? Yes$->$Central. No$->$model others? Yes$->$JAL-AM, No$->$JAL.

#key[*Q size:* Indep $N_a$ (own action). JAL $N_a times N_a$ ($|S| times |A|^n$, *exp in n*; $n$ tables). *$max$ / $epsilon$-greedy only valid for Independent* — JAL best value depends on others $=>$ use $C$.]

#sub[Q-learning update (Independent)]
$ Q(s_i,a_i) <- Q(s_i,a_i) + alpha[r + gamma max_b Q(s'_i,b) - Q(s_i,a_i)] $
Flaw: others learning $=>$ transition model *non-stationary*, breaks convergence. Claus&Boutilier: may reach NE but *not Pareto-optimal*.

#sub[JAL value iteration / solution concept $C$]
$ V_i (s):=C(Q_1(s,a),..,Q_n(s,a)); quad Q_i(s,a):=R_i(s,a)+gamma sum_(s') P(s'|s,a) V_i(s') $
#def[minimaxQ][*zero-sum only*; LP for maximin: $pi[s,dot]=arg max min_(o') sum_(a') pi[s,a'] Q[s,a',o']$, $V[s]=$ that min.]
#def[NashQ][general-sum; bootstrap with $"Nash"Q^j(s')$ = value of stage-game NE at $s'$, not $max$.]

#sub[Fictitious Play (repeated games) — RECIPE]
Model-based; needs *only own payoff matrix*. Belief = empirical freq of opponent's actions.
$ P(a) = w(a) \/ sum_(a') w(a') $
+ Init beliefs = action counts $w$ (*must be nonzero*, not all-0; sensitive to prior).
+ Best-respond to assessed mixed strategy $P$.
+ Observe opponent's real play, increment its $w(a)$. Repeat.

#def[Worked ex][Opp plays C,C,N,C,N $=>$ $w=(3,2) => (0.6,0.4)$.] Matching Pennies: empirical dist $-> (0.5,0.5)$ = unique NE (per-round still oscillates).

#sub[JAL-AM (fictitious play for stochastic games)]
$ hat(pi)_j (a_j|s)=(C(s,a_j))/(sum_(a'_j) C(s,a'_j)); quad "AV"_i (s,a_i)=sum_(a_(-i)) Q_i (s,chevron.l a_i,a_(-i) chevron.r) product_(j!=i) hat(pi)_j (a_j|s) $
Act: $arg max_(a_i) "AV"_i$ ($epsilon$-greedy); update $Q_i$ with $gamma max_(a'_i) "AV"_i (s',a'_i)$.

#sub[Stag Hunt / coordination]
Game collection in stochastic games (PD, Stag Hunt, Battle of Sexes). Reward depends on others' actions (bears: both right=$+5$ each, both left=$+1$, split: mover-left $+2$, mover-right $0$) $=>$ motivates observing state (distributed) then actions (JAL).

#section(fill: rgb("#7030a0"))[Auctions]

#def[Auction][mechanism allocating scarce resources among self-interested agents.] All bidders know #emph[number] of bidders.

#sub[Four Canonical Auctions]
#nobreak[
#table(columns: 4,
  [*Type*], [*Format*], [*Winner pays*], [*Bids*],
  [English], [open ascending], [own (highest) bid], [many, public],
  [Dutch], [open descending clock], [price at accept], [1 accept, public],
  [1st-price SB], [sealed 1-shot], [own highest bid], [1, private],
  [2nd-price SB (Vickrey)], [sealed 1-shot], [2nd-highest bid], [1, private],
)]
- *English:* auctioneer opens at reserve; bidders shout ascending; highest standing bid wins if unchallenged. Art, wine.
- *Dutch:* clock starts high, drops each step until someone accepts and pays that ask. Flowers, produce.
- Strategic links: *English $approx$ 2nd-price* (truthful), *Dutch $approx$ 1st-price* (shade).

#sub[First-Price Sealed-Bid]
Highest bidder wins, pays own bid $=>$ incentive to *shade* ($b_i < v_i$). *No dominant strategy* (best response depends on others). Tradeoff: higher $b$ = win more often but less profit $v_i - b_i$.

#sub[BNE of 1st-Price (risk-neutral, $v_i$ iid $~U(0,1)$)]
#key[General $N$ bidders: $ b_i = (N-1)/N med v_i $ As $N -> infinity$, $(N-1)/N -> 1$ (shade *less* w/ more rivals). $N=2$: $b_i = 1/2 v_i$.]
*Derivation ($N=2$, given $b_2 = 1/2 v_2$):* bidder 1 wins iff $b_2 < b_1 <=> v_2 < 2 b_1$.
$ bb(E)[u_1] = P(v_2 < 2b_1)(v_1 - b_1) = F(2b_1)(v_1 - b_1) $
$U(0,1) => F(x)=x$, so $bb(E)[u_1] = 2b_1(v_1 - b_1) = 2 v_1 b_1 - 2 b_1^2$.
FOC: $partial / (partial b_1) = 2 v_1 - 4 b_1 = 0 => b_1 = v_1 / 2$.

#sub[Second-Price (Vickrey)]
Highest wins, pays *2nd-highest* bid. *Truthful $b_i = v_i$ is weakly dominant* (optimal regardless of others) and a NE.
#key[Trap: 2nd-price winner pays the *second*-highest bid, NOT own bid. Ex: bids 100, 95, 80 $->$ winner pays *95, not 100*. 1st-price winner of same bids pays 100.]
*Why truthful (no profitable deviation):*
- _Loser_ raising $b$: still lose (pay 0) or win at price $> v_i$ (negative payoff). Lowering: still 0. $=>$ no gain.
- _Winner_ raising $b$: still pays 2nd price (unchanged). Lowering: stays above 2nd $->$ unchanged, or drops below $->$ loses (0). $=>$ no gain.

#sub[Equilibrium Concept & Revenue Equivalence]
- 1st-price: *Bayesian-Nash* (incomplete info, values from distribution). 2nd-price: *(weakly dominant) Nash* for any realized values.
- #def[Revenue Equivalence][under risk-neutral bidders, iid private values from common distribution, symmetric eq., all 4 canonical formats give the seller the *same expected revenue*.]
- Uses: 1st-price $->$ privatization; 2nd-price $->$ digital ad tech (Google, Facebook).

#section(fill: rgb("#7030a0"))[Coordination]

#def[Coordination][process where a group of agents choose *a single Pareto-optimal NE*. Many NE exist $=>$ real problem = *equilibrium selection*.]
#def[Pure coordination / team game][all share one payoff: $u_1(a)=dots=u_n(a) equiv u(a)$.]
#def[Pareto dominance][$a$ dominates $a'$ iff $forall i: u_i (a)>=u_i (a')$ and $exists j: u_j (a)>u_j (a')$. *Pareto-optimal* = not dominated by any $a'$.]

#sub[Social Conventions (Boutilier 1996)]
Recipe (common knowledge) constraining action choice so all pick the *same* NE; once established, no one gains by deviating.
+ Fix a *unique ordering scheme of joint actions*: order *agents* + order *actions*.
+ Each agent computes *all* equilibria.
+ Each selects the *first* equilibrium in the ordering.
#key[Must be *common knowledge*. If agents disagree on the ordering they may pick different NE (both cross $->$ crash).]
Ex (movie): order $1 succ 2$, $"Thriller" succ "Comedy"$ $=>$ pick (Thriller,Thriller). Car right-of-way: right has priority.

#sub[With Communication]
Only need an *ordering of agents* (no full joint-action order). Sequential/synchronized:
+ agent $i$ waits until all $1,dots,i-1$ broadcast their actions;
+ picks $a_i^*$ = component of a NE *consistent* with predecessors;
+ broadcasts $a_i^*$ to those not yet chosen.

#sub[Roles] mask action set (deactivate actions) $->$ shrink $m$, simplify selection. Assign greedily by *potential* (fit of agent to role) in fixed role order; one role/agent; $O(n^2)$.

#sub[Coordination Graphs (Guestrin 2002)] reduce *number of agents*.
#key[Joint actions = $m^n$ (exp). *Roles cut $m$; coordination graphs cut $n$.*]
Global payoff = sum of local pairwise factors:
$ u(a) = sum_(j=1)^k f_j (dot) , quad "e.g. " u(a)=f_1(a_1,a_2)+f_2(a_1,a_3)+f_3(a_3,a_4) $
Graph: nodes = agents, edges = local factors $f_j$. Solve many small subgames vs one huge one.

#sub[Variable Elimination] (max over joint actions, exact)
+ Pick agent $i$; collect all factors $f_j$ containing $a_i$.
+ Compute its best response: $e_i (a_("neigh")) = max_(a_i) sum_(j) f_j$, recording $arg max a_i$.
+ Replace those factors with new factor $e_i$ over $i$'s neighbours; remove $i$.
+ Repeat until one agent left $->$ $max$ gives optimal value; back-substitute $arg max$ to recover joint action $a^*$.
#nobreak[#table(columns: 2, [*Roles*], [single reduced subgame, cuts $m$], [*Coord. graph*], [many subgames, cuts $n$, $u=sum f_j$])]

#section(fill: rgb("#c55a11"))[Agent Architectures]
#def[Architecture][principles describing agent behavior: formal/abstract view + map of internals (control flow). Goal: guide agent design.]

#sub[Abstract model]
Env states $E={e_0,e_1,...}$; actions $"Ac"={alpha_0,...}$. Run $r: e_0 ->^(alpha_0) e_1 ->^(alpha_1) e_2 ...$
- State transformer $tau: R^("Ac") -> "pow"(E)$: history-dependent, non-deterministic; $tau(r)=nothing$ => run ended.
- Env $= chevron.l E, e_0, tau chevron.r$. Agent $"Ag": R^E -> "Ac"$ (maps run ending in state to action).
#key[Signatures (trap): general $"Ag": R^E->"Ac"$ (history) vs purely reactive $"Ag": E->"Ac"$ (present only) vs $tau: R^("Ac")->"pow"(E)$. Deliberation fns use power sets $2^("Bel"),2^("Des"),2^("Int")$.]

#sub[Deductive (symbolic) 1956-85]
Explicit symbolic world model (predicate logic); decide by theorem proving $"DB" tack.r_rho "Do"(a)$.
2 problems: *transduction* (world->symbols), *representation/reasoning*. Bad: complexity, no time constraints, env can't change while deciding, hard for dynamic envs.

#sub[Deliberative / BDI (intentional stance)]
#key[*Practical Reasoning = Deliberation (WHAT) + Means-Ends Reasoning (HOW)*]
*Beliefs* (info), *Desires* (goals), *Intentions* (commitments). Intentions: persist, drive means-ends, constrain future deliberation (*filter of admissibility*), influence beliefs.
Functions: $"brf": 2^("Bel") times "Per" -> 2^("Bel")$; $"options": 2^("Bel") times 2^("Int") -> 2^("Des")$; $"filter": 2^("Bel") times 2^("Des") times 2^("Int") -> 2^("Int")$; $"plan": 2^("Bel") times 2^("Int") times 2^("Ac") -> "Plan"$.
Loop: 1.observe+update beliefs 2.deliberate (options->filter)=intention 3.plan 4.execute 5.repeat.
*Commitment (trap):* Blind=until *achieved*; Single-minded=until achieved OR *impossible*; Open-minded=while not achieved AND *still desired*.

#sub[Reactive (Brooks subsumption)]
No symbolic reasoning; intelligence *distributed*, *emerges* from behavior+env interaction. Behavior modules = FSMs (action-selection); *lower layers inhibit higher* (lower = priority). Many fire at once.
*Limits:* only local/short-term info; needs enough local info; *no learning*; hard to engineer complex behavior; *unpredictable with many layers*.

#sub[Hybrid (reactive+deliberative, reactive precedence)]
*Horizontal*: every layer touches input+output (mediator reconciles $m^n$; conflict/coherence problem). *Vertical*: at most 1 layer per input/output (bottleneck, fault-intolerant); *one-pass* (up only) vs *two-pass* (up then down). Ex: *Stanley* DARPA 2005.
#nobreak[#table(columns:4,
[ ],[*Reactive*],[*Deliberative*],[*Hybrid*],
[Reasoning],[none/rules],[symbolic/plan],[both layered],
[Speed],[fast],[slow],[real-time+goals],
[Learning],[no],[—],[layered abstraction],
[Weakness],[short-term,scale],[complexity,slow],[layer interaction])]

#section(fill: rgb("#c55a11"))[Introduction to Agents]
#def[Agent][computer system capable of *autonomous action* in some environment to meet its *design objectives*, on behalf of its user/owner. *Autonomy* = freedom from external (human) control: figures out _what_ to do.]
#def[MAS][set of agents that *interact*, often for users with *different goals* $=>$ need cooperation, coordination, negotiation.]
- *Loop:* sense (sensors/perceptors) $->$ decide $->$ act (effectors/actuators) $->$ ... ; closed-coupled, continual.
- *Two problems:* agent design (micro: single-agent decision making) vs society design (macro: interaction).
- *Autonomy levels:* operational (choose _actions_) vs motivational (choose _goals_). Bounded by: environment, resources, other agents, norms.
#sub[Three Intelligent-Agent Behaviours]
- *Reactivity:* maintains ongoing interaction with a *dynamic* environment, responds to changes _in time to be useful_ (fixed env $=>$ blind execution).
- *Pro-activeness:* goal-directed; generates & attempts to achieve goals, takes initiative, not driven solely by events.
- *Social ability:* interact with other agents via coordination/cooperation/negotiation + modeling others.
#key[Open problem: balancing *reactive* (timely response) vs *pro-active* (long-term goals) — they can be at odds.]
#sub[Social Abilities (define + example)]
- *Coordination* = managing interdependencies between activities. _Ex:_ two agents share a non-sharable resource.
- *Cooperation* = working as a team toward a *shared* goal. _Ex:_ goal no single agent can achieve alone.
- *Negotiation* = reach agreement on matters of *common interest* (possibly conflicting). _Ex:_ roommates — he watches football tonight, you watch a movie tomorrow.
#key[*Trap:* coordination = interdependencies; cooperation = shared goal; negotiation = agreement on conflicting interests. Don't conflate.]
#sub[Environment Properties (chess = "easy" side of all)]
#nobreak[#table(columns: 3, [*Dichotomy*],[*"Good" side*],[*Contrast*],
[Accessible/Inaccessible],[complete, accurate, up-to-date state (= fully observable)],[chess / real world],
[Deterministic/Non-det],[action has single guaranteed effect],[chess / football],
[Static/Dynamic],[world unchanged while deliberating],[chess / Counter-Strike],
[Discrete/Continuous],[finite, countable actions/percepts],[chess / landscape],
[Episodic/Non-episodic],[independent intervals, no cross-influence],[MRI scan / persistent game])]
#key[*Trap:* accessible = fully observable (state); determinism = single action effect — distinct axes. Most complex envs are inaccessible + dynamic + non-deterministic.]
- *Other props:* adaptivity (learn), rationality (maximize utility fn), mobility (change location), curiosity, believability.

#section(fill: rgb("#548235"))[Markov Decision Processes]
#def[MDP][single-agent sequential decision making; *Markovian* (next $s,r$ depend only on current $s,a$), *fully observable*, horizon may be infinite.] Tuple $(S,A,P,R,gamma)$: states, actions, transition $P(s'|s,a)$, reward $R(s,a)$, discount $gamma in [0,1)$.

#sub[Value functions]
$ V^pi (s) = bb(E)[sum_(t=0)^infinity gamma^t R(s_t,a_t) | s_0=s, a_t=pi(s_t)] $
$Q^pi (s,a)$: same but force first action $a$, then follow $pi$. Optimal: $V^*(s)=max_pi V^pi (s)$, $Q^*$ analog; $>= 1$ optimal $pi^*$ always exists (may be many, all share $V^*,Q^*$). Greedy extract: $pi^*(s) in arg max_a Q^*(s,a)$.

#key[*Bellman optimality:* $Q^*(s,a) = R(s,a) + gamma sum_(s') P(s'|s,a) max_(a') Q^*(s',a')$ , $V^*(s)=max_a Q^*(s,a)$.]

#sub[Solving (model known: need $P,R$)]
- *Value Iteration:* repeat $Q(s,a):=R(s,a)+gamma sum_(s') P(s'|s,a)V(s')$ then $V(s):=max_a Q(s,a)$ until stable; -> $Q^*$ for any init.
- *Policy Iteration:* alternate _policy evaluation_ (solve $V^pi$) + _policy improvement_ (greedy on $Q^pi$) until $pi$ stable.

#sub[Markov / stochastic game]
Multiagent MDP: tuple $(S, {A_i}, P, {R_i}, gamma)$; joint action $a=(a_1,...,a_n)$, $P(s'|s,a)$, each $i$ has own $R_i$; solution = equilibrium policies (e.g. Nash) not a single optimum.

#section(fill: rgb("#c55a11"))[Rational Agents]
#def[Rational agent][acts to *maximize (expected) utility*. Decisions are made *algorithmically* over the agent's preferences.]
#sub[Preferences & utility]
Relations on outcomes: strict $x succ y$, indiff. $x ~ y$, pref-indiff. $x succ.eq y$ (= $succ$ or $~$). A preference is *rational* iff *complete* ($x R y or y R x$) and *transitive* ($x R y, y R z => x R z$); $succ.eq$ is rational.
#key[*Utility thm:* if $succ.eq$ rational on $X$, exists $u: X -> bb(R)$ with $u(x) >= u(y) <=> x succ.eq y$.]
#sub[Decision rule]
No uncertainty: $Q(a) = u(O(a))$. Under uncertainty: maximize *expected utility* $ a^* = arg max_(a in A) sum_(o in O) u(o) P(o | a) $
#def[PEAS][task-environment spec: *P*erformance measure, *E*nvironment, *A*ctuators, *S*ensors.]
#def[Decision tree][_decision_ nodes (diamond, agent chooses) + _chance_ nodes (circle, prob.); eval each branch by exp. utility, pick max.]

#section(fill: rgb("#c55a11"))[Hybrid MAS (rarely tested)]
#def[Hybrid MAS][collective of *human + AI* agents sharing an env; both sense/act, have goals.] #def[Layering][reactive+deliberative layers: *horizontal*=all layers see input (needs mediator, e.g. TouringMachines), fault-tolerant/parallel; *vertical*=pipeline (e.g. InteRRaP).]
- *Delegation* = control $->$ influence; user gives *intent*, agent fills details. Trust dims: Competence, Predictability, Benevolence, Integrity. Autonomy: in-loop / on-loop / out-of-loop.
- *Contract-Net:* manager broadcasts call-for-proposals $->$ agents bid/decline $->$ award lowest-cost bid.

#section(fill: rgb("#7030a0"))[Voting (not seen on past exams)]
#def[Social choice fn][maps all agents' (truthful) preferences -> one enforced outcome.]
#sub[Schemes] *Plurality:* most first-place votes wins. *Plurality+elim (IRV):* drop fewest-votes, transfer to next, repeat. *Borda:* top gets $n-1$, next $n-2$, ..., last $0$ ($n$=\#cands); sum over voters, max wins. *Pairwise elim:* fixed agenda; majority runoffs eliminate loser.
#def[Condorcet winner][beats every other candidate in pairwise majority. Desirable but *need not exist*: cycle A>B, B>C, C>A (Condorcet paradox).]
#key[Profile 499:A>B>C, 3:B>C>A, 498:C>B>A -> Plurality=*A* (499), IRV=*C* (B out, 3->C=501), Condorcet=*B* (beats A 501-499, C 502-498). 3 methods, 3 winners.]
#sub[Defects] *IIA violated:* dropping a losing candidate flips winner (Borda/plurality). *Agenda:* in pairwise elim the order alone picks the winner (last-compared has edge). *Pareto:* pairwise elim can pick D even though all prefer B>D.
#key[*Arrow:* with $>= 3$ candidates, Pareto + IIA $=>$ dictatorship. No fair non-dictatorial ranked system has both. (Pareto: all prefer A>B $=>$ B not winner; IIA: pairwise pref independent of other alternatives.)]
#sub[Gibbard-Satterthwaite] $>= 3$ outcomes: any non-dictatorial, onto social choice fn is *manipulable* (some voter gains by lying). Truthful + non-dictatorial impossible.
#def[Seats][D'Hondt quotient $V/(s+1)$, $s$=seats won so far (not total); largest quotient wins next seat. Sainte-Laguë: divisors 1,3,5,... (fairer to small parties); Hare-LR: largest remainder, most proportional. Gerrymandering: boundaries can let minority win.]
