// ============================================================================
//  HANDWRITING MASTER — content sized to copy onto BOTH sides of ONE A4 by hand
//  This is a READ-WHILE-YOU-WRITE guide, not the exam artifact. Hand-copy it.
//  Compile: typst compile cheatsheet_handwriting.typ
//  Watch:   typst watch cheatsheet_handwriting.typ
//  Priorities come from past-exam frequency: game theory = 61% of questions.
// ============================================================================

#let body-size   = 9pt     // readable while copying; not the handwriting size
#let num-columns = 3
#let margin-size = 0.5cm

#set page(paper: "a4", margin: margin-size, columns: num-columns)
#set text(size: body-size, font: "New Computer Modern", hyphenate: true)
#set par(justify: true, leading: 0.5em, spacing: 0.6em)
#show math.equation: set text(size: body-size)
#set list(indent: 0.6em, body-indent: 0.35em, spacing: 0.45em, marker: [‣])
#set enum(indent: 0.6em, body-indent: 0.35em, spacing: 0.45em)

#let section(title, fill: rgb("#1f4e79")) = block(
  width: 100%, inset: (x: 3pt, y: 2pt), radius: 1.5pt, fill: fill,
  above: 0.6em, below: 0.45em,
  text(fill: white, weight: "bold", size: body-size + 0.5pt, title),
)
#let sub(title) = text(weight: "bold", fill: rgb("#1f4e79"), title)
#let key(body) = block(
  width: 100%, inset: 3pt, radius: 1.5pt, stroke: 0.5pt + rgb("#c00000"),
  fill: rgb("#fdeaea"), above: 0.45em, below: 0.45em, body,
)
#let def(term, body) = [#text(weight: "bold")[#term:] #body]

// ============================================================================

#section(fill: rgb("#1f4e79"))[Normal Form Games  ·  #1 topic]
#def[Game][$(N,A,u)$: agents, actions $A=A_1 times ... times A_n$, payoffs $u_i: A->bb(R)$. 2 players + finite $=>$ matrix (row=P1, col=P2).]
#def[Best response][$a_i in "BR"(a_(-i))$ iff $u_i (a_i,a_(-i)) >= u_i (a'_i,a_(-i)) forall a'_i$.]
#def[Strictly dominated][$exists a'_i$: $u_i (a'_i,a_(-i)) > u_i (a_i,a_(-i))$ for *every* $a_(-i)$.]

#sub[IESDS] repeatedly delete *strictly* dominated actions; survivors = prediction.
#key[STRICT only. Deleting *weakly* dominated can kill real NE + is order-dependent.]

#sub[Pure NE] profile where nobody gains by unilateral deviation.
*Find:* mark each player's best-response payoff per opponent action; cell with *all* marks = NE (can be 0, 1, many).

#sub[Mixed NE — indifference method (2×2)]
+ Opponent plays action w.p. $p$.
+ Write your $"EU"("act"_1)$, $"EU"("act"_2)$ as fns of $p$.
+ Set *equal* $->$ solve $p$. Repeat for other player.
#key[The $p$ making ME indifferent = prob. the OPPONENT plays. (Nash: every finite game has $>= 1$ NE, maybe mixed.)]
#sub[Worked example] Rows $a,b$; cols $c,d$; payoffs $a c=3","4$, $a d=5","1$, $b c=2","2$, $b d=8","3$.
P2 plays $c$ w.p. $p$, P1 indiff: $3p+5(1-p)=2p+8(1-p) => p=3/4$. P1 plays $a$ w.p. $q$, P2 indiff: $4q+2(1-q)=q+3(1-q) => q=1/4$. Mixed NE: $(a:1/4, b:3/4),(c:3/4, d:1/4)$. Each prob $in [0,1]$ or you slipped.

#sub[Canonical games]
- *Prisoner's Dilemma:* dominance $=>$ unique bad NE (defect,defect).
- *Matching Pennies* (zero-sum): no pure NE; mixed $(1/2,1/2)$.
- *Battle of Sexes:* 2 pure NE + 1 mixed; players mix with *different* probs.

#sub[Cournot duopoly]
$P(Q)=a-Q$, $Q=q_1+q_2$, cost $c$. $pi_i = q_i (a-q_1-q_2-c)$.
FOC $=> b_i (q_j)=1/2(a-q_j-c)$.
#key[Symmetric NE: $q_1^*=q_2^*=1/3(a-c)$.]

#section(fill: rgb("#1f4e79"))[Extensive Form Games  ·  #2]
Tree = sequential moves. Node=choice, edge=action, leaf=payoff vector.
#def[Strategy][*complete contingent plan*: an action at every own node / info set, even unreached ones.]
#key[Specifying unreached nodes $=>$ normal-form redundancy + lets NE rest on *non-credible threats*.]

#sub[Extensive $->$ Normal] rows/cols = full strategy sets; each cell = leaf reached by that profile; then find NE normally.

#sub[Subgame-perfect equilibrium (SPE)]
profile that is a NE in *every* subgame. Rules out non-credible threats. Every SPE is a NE (not vice-versa); $>= 1$ always exists.
*Disprove SPE:* find one subgame where the restriction isn't a NE.

#sub[Backward induction (gives a SPE)]
from leaves up: at node $h$, mover $rho(h)$ picks the child maximizing *their own* payoff coordinate; propagate that whole vector up.
#key[Compare on the *moving* player's component only. Zero-sum: BI = minimax (+ alpha-beta).]

#section(fill: rgb("#1f4e79"))[Bayesian Games  ·  #3]
#def[Game][$(N,A,Theta,p,u)$: types $Theta_i$, common prior $p(theta)$, payoffs $u_i (a,theta)$ depend on types.]
#def[BNE][each type $theta_i$ best-responds in *expectation* over opponents' types:
$ max_(a_i) sum_(theta_(-i)) p(theta_(-i)|theta_i) u_i (a_i,a_(-i)(theta_(-i)),theta) $]
#sub[Compute] for each of your types, compute expected utility of each action over opponent types (use conditional prob.), pick best. Check no type wants to deviate. Strict dominance can be checked *per type*.

#section(fill: rgb("#1f4e79"))[Repeated Games  ·  #4]
#def[Discount $beta in (0,1)$][weight on future; $sum beta^t r_t$. Higher $beta$ = more patient.]
#key[*Geometric series:* $sum_(t=0)^infinity beta^t = 1/(1-beta)$. Constant $c$ from now $= c/(1-beta)$; from next period $= beta c/(1-beta)$.]
- *Finitely* repeated: backward induction $=>$ play stage-NE every round (cooperation unravels).
- *Infinitely* repeated: cooperation sustainable via threats.
#sub[Grim trigger (infinite PD)]
cooperate until a defection, then defect forever. Cooperation is a NE iff patient enough:
#key[$ beta >= (T - R)/(T - P) $ ($T$=temptation, $R$=reward, $P$=punish). i.e. future loss from punishment $>=$ one-shot gain from defecting.]
*Folk thm:* any feasible payoff $>$ minimax is a NE for $beta$ near 1.

#section(fill: rgb("#548235"))[Multiagent Learning  ·  #5]
#sub[Fictitious play]
+ Count opponent's past actions $C(a)$.
+ Belief = empirical freq. $hat(p)(a)=C(a)/sum C$.
+ Best-respond to belief; observe; update counts. Repeat.
#key[If FP converges, the point is a NE. Converges for zero-sum + 2×2 games; can cycle (Shapley).]
#sub[Multiagent Q-learning]
- *Independent* learners: own $Q_i (s,a_i)$, ignore others (non-stationary).
- *Joint-action* learners: $Q_i (s,arrow(a))$ over joint actions $=>$ table is $|A|^n$ vs $|A|$.
- Update: $Q(s,a) <- Q + alpha[r + gamma max_(a') Q(s',a') - Q]$. Minimax-Q / Nash-Q replace $max$ by game value.

#section(fill: rgb("#7030a0"))[Auctions  ·  #7]
#def[Second-price / Vickrey][highest bid wins, pays 2nd price. *Truthful bidding is dominant.*]
#def[First-price sealed][highest wins, pays own bid $=>$ *shade* your bid.]
#key[First-price, $n$ bidders, valuations iid $U(0,1)$, risk-neutral: symmetric eq. bid $b(v)=(n-1)/n dot v$.]
- *English* (ascending) $approx$ second-price; *Dutch* (descending) $approx$ first-price.
- *VCG:* each winner pays the *externality* it imposes (others' lost welfare) $=>$ truthful.
- *Revenue equivalence:* under iid + risk-neutral, all these auctions give the same expected revenue.

#section(fill: rgb("#7030a0"))[Coordination  ·  #8]
- *Social conventions:* fixed ordering over agents+actions $=>$ each picks deterministically, no conflict.
- *Coordination graph / factor graph:* global payoff $= sum_j f_j$ of local (often pairwise) factors.
#key[Optimal joint action $arrow(a)^* = arg max_(arrow(a)) sum_j f_j$ via *variable elimination*: eliminate agents one by one, each maximizing out over its factors (best-response fn of neighbours).]

#section(fill: rgb("#c55a11"))[Agents & Architectures  ·  #6,#9]
#def[Agent][autonomous action in an environment to meet design objectives.] Behaviours: *reactive* (timely response to dynamic env), *pro-active* (goal-directed), *social*.
#key[Social: *coordination* = manage interdependencies; *cooperation* = shared goal; *negotiation* = agree on (conflicting) interests. Don't conflate.]
#sub[Environment props] accessible/inaccessible, deterministic/non-det, static/dynamic, discrete/continuous, episodic/non. (chess = "easy" side of each.)
#sub[Reactive (subsumption)] no symbolic model; behaviour layers, *lower inhibits higher*. *Limits:* local/short-term info only, *no learning*, hard to engineer many layers.
#sub[Deliberative / BDI]
#key[*Practical reasoning = deliberation (WHAT goal) + means-ends reasoning (HOW = planning).*]
Beliefs / Desires / Intentions. Intentions persist + filter future options. Commitment: *blind* (until achieved) / *single-minded* (until achieved or impossible) / *open-minded* (while still desired).
#sub[Hybrid layering] *horizontal* = all layers see I/O (needs mediator, fault-tolerant) vs *vertical* = pipeline.

#section(fill: rgb("#548235"))[MDP (safety box)]
$(S,A,P,R,gamma)$, Markov + fully observable.
#key[Bellman opt.: $Q^*(s,a)=R(s,a)+gamma sum_(s') P(s'|s,a) max_(a') Q^*(s',a')$; $V^*=max_a Q^*$.]
*Value iteration*: iterate Bellman to $Q^*$. *Policy iteration*: evaluate + greedily improve. Markov game = MDP with joint actions + per-agent rewards $=>$ equilibrium policies.

#section(fill: rgb("#c55a11"))[Misc safety one-liners]
- *Rational agent* = maximizes expected utility $a^*=arg max_a sum_o u(o) P(o|a)$; spec via *PEAS*.
- *Voting* (not seen on exams): Plurality / Borda ($n-1,...,0$) / Condorcet winner (may not exist—cycle). *Arrow:* $>= 3$ options, Pareto+IIA $=>$ dictatorship. *Gibbard-Satterthwaite:* non-dictatorial onto rule is manipulable.
