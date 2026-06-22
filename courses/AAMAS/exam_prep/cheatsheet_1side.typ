// ============================================================================
//  ONE-SIDE HANDWRITING MASTER — sized to copy onto ONE SINGLE SIDE of A4 by hand.
//  Exam rule: ONE A4 page, single side, your OWN handwriting. Typed notes banned.
//  This is the READ-WHILE-YOU-WRITE guide — hand-copy it; it is NOT the artifact.
//  Compile: typst compile cheatsheet_1side.typ
//  Watch:   typst watch cheatsheet_1side.typ
//  Priorities = past-exam frequency: the 4 game-theory blocks are ~61% of points.
//  CONSERVATIVE on space — leaves open room at column bottoms to pencil in extras.
// ============================================================================

#let body-size   = 9pt
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
#def[Game][$(N,A,u)$: agents, $A=A_1 times ... times A_n$, payoffs $u_i: A->bb(R)$. 2 players + finite $=>$ matrix (row=P1, col=P2).]
#def[Best response][$a_i in "BR"(a_(-i))$ iff $u_i (a_i,a_(-i)) >= u_i (a'_i,a_(-i)) forall a'_i$.]
#def[Strictly dominated][$exists a'_i$: $u_i (a'_i,a_(-i)) > u_i (a_i,a_(-i))$ for *every* $a_(-i)$.]

#sub[IESDS] repeatedly delete *strictly* dominated actions; survivors = prediction.
#key[STRICT only. Deleting *weakly* dominated can kill real NE + is order-dependent.]

#sub[Pure NE] nobody gains by unilateral deviation.
*Find:* mark each player's best-response payoff per opponent action; cell with *all* marks = NE (can be 0, 1, many).
#def[Pareto optimal][no other outcome makes everyone $>=$ as well off AND someone strictly better. (Coordination/BoS pure NE are PO; PD's (D,D) is NOT.)]
#def[Story $->$ game][list $N$ (players), $A_i$ (each one's options), fill $u$ from text: "want to match" $=>$ match cells $+$, mismatch $0$; "each prefers own option" $=>$ asymmetric $(2,1)/(1,2)$.]

#sub[Mixed NE — indifference (2×2)]
+ Opponent plays an action w.p. $p$.
+ Write your $"EU"("act"_1)$, $"EU"("act"_2)$ as fns of $p$.
+ Set *equal* $->$ solve $p$. Repeat for other player.
#key[The $p$ making ME indifferent = prob. the OPPONENT plays. Every finite game has $>= 1$ NE (maybe mixed).]
#sub[Worked — Battle of Sexes] (the recurring "model + solve" game). Max=row, Ann=col; $I I=(1,2)$, $I T=(0,0)$, $T I=(0,0)$, $T T=(2,1)$.
*Pure:* underline BRs $=> (I,I),(T,T)$, both Pareto optimal.
*Mixed:* Ann plays $I$ w.p. $q$, Max indiff: $1q+0=0+2(1-q)=>q=2/3$. Max plays $I$ w.p. $r$, Ann indiff: $2r+0=0+1(1-r)=>r=1/3$. Mixed NE $((2/3,1/3),(1/3,2/3))$.

#sub[Canonical games]
- *Prisoner's Dilemma:* dominance $=>$ unique bad NE (defect,defect).
- *Matching Pennies* (zero-sum): no pure NE; mixed $(1/2,1/2)$.
- *Battle of Sexes:* 2 pure NE + 1 mixed; players mix with *different* probs.

#sub[Cournot duopoly]
$P(Q)=a-Q$, $Q=q_1+q_2$, cost $c$. $pi_i=q_i (a-q_1-q_2-c)$.
FOC $=> b_i (q_j)=1/2(a-q_j-c_i)$.
#key[Symmetric NE: $q_1^*=q_2^*=1/3(a-c)$. *Asymmetric costs:* solve BOTH reaction fns simultaneously (don't assume $q_1=q_2$). Demand $P=a-b Q$: $b_i=(a-c_i-b q_j)/(2b)$. Plug back to check.]

#section(fill: rgb("#1f4e79"))[Extensive Form Games  ·  #2]
Tree = sequential moves. Node=choice, edge=action, leaf=payoff vector.
#def[Strategy][*complete contingent plan*: an action at every own node / info set, even unreached ones.]
#key[Unreached-node actions $=>$ normal-form redundancy + let NE rest on *non-credible threats*.]

#sub[Ext $->$ Normal] rows/cols = full strategy sets; cell = leaf reached; find NE normally. Round-trip need not redraw the same tree but games are *strategically equivalent* (same equilibria).
#key[*Imperfect info:* nodes the mover can't tell apart joined in one info set (dashed line). Simultaneous matrix game $=>$ single info set.]

#sub[SPE] NE in *every* subgame. Rules out non-credible threats. Every SPE is a NE (not vice-versa); $>= 1$ exists.
*Disprove:* find one subgame where the restriction isn't a NE.

#sub[Backward induction (gives SPE)]
leaves up: at node $h$, mover picks child maximizing *their own* payoff coordinate; propagate the whole vector up.
#key[Compare on the *moving* player's component only. Zero-sum: BI = minimax (+ alpha-beta).]

#section(fill: rgb("#1f4e79"))[Bayesian Games  ·  #3]
#def[Game][$(N,A,Theta,p,u)$: types $Theta_i$, prior $p(theta)$, payoffs $u_i (a,theta)$ depend on types.]
#def[BNE][each type best-responds in *expectation* over opponents' types:
$ max_(a_i) sum_(theta_(-i)) p(theta_(-i)|theta_i) u_i (a_i,a_(-i)(theta_(-i)),theta) $]
#sub[Compute] per own type, expected utility of each action over opponent types (conditional prob.), pick best. Check no type deviates. Strict dominance checkable *per type* (often pins one player first).
#key[*Solve-for-p:* write $"EU"(a_1|"type")$, $"EU"(a_2|"type")$ as linear fns of prior $p$; solve $"EU"(a_2)>"EU"(a_1)$ for $p$-interval. An agent that KNOWS its type $=>$ its BR is per-type, does NOT depend on $p$.]

#section(fill: rgb("#1f4e79"))[Repeated Games  ·  #4]
#def[Discount $beta in (0,1)$][weight on future; $sum beta^t r_t$. Higher = more patient.]
#key[Geom: $sum_(t=0)^infinity beta^t=1/(1-beta)$. Constant $c$ from now $=c/(1-beta)$; from next period $=beta c/(1-beta)$.]
#key[*As a tree:* draw stage game once per period; a path's payoff = SUM of stage payoffs (discounted by $beta^t$ if asked).]
- *Finite:* backward induction $=>$ play stage-NE every round (cooperation unravels).
- *Infinite:* cooperation sustainable via threats.
#sub[Grim trigger (infinite PD)] cooperate until a defection, then defect forever. NE iff patient enough:
#key[$beta >= (T-R)/(T-P)$ ($T$=temptation, $R$=reward, $P$=punish): future punishment loss $>=$ one-shot defect gain.]
*Folk thm:* any feasible payoff $>$ minimax is a NE for $beta$ near 1.

#section(fill: rgb("#548235"))[Multiagent Learning  ·  #5 (12%!)]
#sub[Fictitious play] belief = COUNT vector of opp's past actions (init counts given).
+ $"EU"("act")=sum_"opp" ("count"_"opp"/"total") dot u("act","opp")$.
+ Play $arg max$; THEN $+1$ to the action opp actually played. Ties: stated rule.
#key[Init counts: magnitude = confidence, ratio = prior freq. If it converges $=>$ that point is a NE. Can cycle (Shapley).]
#sub[Q-learning / Nash-Q]
- *Indep* learners: own $Q_i (s,a_i)$, ignore others (non-stationary).
- *Joint-action* learners: $Q_i (s,arrow(a))$. Table size: indep $=|A_i|$; joint/central $=|A_1| dot ... dot |A_n|$.
- *Matrix game* (no state, $gamma=0$): $Q_i [arrow(a)] <- Q_i + alpha(r_i - Q_i)$. Greedy picks NE of current $Q$ (init 0 $=>$ first step ties). Minimax-Q / Nash-Q replace $max$ by game value.
#def[Markov game][$(N,S,{A_i},P,{R_i},gamma)$: MDP with joint actions + per-agent rewards; solve model-free w/ (minimax/Nash) Q-learning.]

#section(fill: rgb("#7030a0"))[Auctions  ·  #6]
#def[Second-price/Vickrey][highest wins, pays 2nd price $=>$ *truthful bidding dominant*.]
#def[First-price sealed][highest wins, pays own bid $=>$ *shade*. Sym. eq. ($n$ bidders, iid $U(0,1)$): $b(v)=(n-1)/n dot v$.]
#key[*Payoff* = valuation $-$ price paid (add commission to winner's cost). *Efficient* iff winner = highest-valuation agent. 2nd-price under uncertainty: bid the EXPECTED value of winning.]
English (asc) $approx$ 2nd-price; Dutch (desc) $approx$ 1st-price.

#section(fill: rgb("#c55a11"))[Agents & Coordination  ·  #7]
- *Behaviours:* reactive (timely) / pro-active (goal) / social. Social = coordination (interdependencies) vs cooperation (shared goal) vs negotiation (conflicting interests). Don't conflate.
- *Subsumption:* behaviour layers, *lower inhibits higher*; no model, *no learning*; hard to add layers.
- *Deductive/logic agent cons:* theorem-proving costly; *calculative-rationality* (world changes mid-decision); hard to model dynamic worlds.
- *BDI:* practical reasoning = deliberation (WHAT goal) + means-ends (HOW = plan). Commitment: blind / single-minded / open-minded.
- *Hybrid:* horizontal layering (all layers see I/O, fault-tolerant, needs mediator) vs vertical (pipeline).
- *Rational agent:* $a^*=arg max_a sum_o u(o)P(o|a)$; spec via PEAS.
- *Coordination:* social convention = fixed agent+action ordering $=>$ deterministic, no conflict. Factor graph: $arrow(a)^*=arg max sum_j f_j$ via *variable elimination* (eliminate agents one by one).

#section(fill: rgb("#7f7f7f"))[EXTRA — worked examples (copy top-down till full)]
#sub[1. Fictitious play (2025 exam)] game $U\/D × L\/R$, $(P_1,P_2)$: $U L=(1,2),U R=(5,6),D L=(4,6),D R=(7,1)$. ($D$ dominates for P1; unique NE $(D,L)$.) Init counts $(1,1)$ each.
- R1: P1 belief $(1/2,1/2)$: $"EU"(U)=1/2+5/2=3$, $"EU"(D)=2+7/2=5.5=>D$. P2: $"EU"(L)=1+3=4$, $"EU"(R)=3+1/2=3.5=>L$. Saw $(D,L)$.
- R2: P1 belief $L=2/3$: $"EU"(U)=7/3,"EU"(D)=5=>D$. P2 $=>L$.
#key[Converges to the unique NE $(D,L)$ — *regardless of initialization*.]

#sub[2. Nash-Q joint-action (2025 exam)] same game, $gamma=0$. Current $Q_R=[U:1,4;D:3,7]$, $Q_C=[U:2,5;D:5,1]$.
- Greedy = NE of current $Q$: $(D,L)$ ($Q_R=3,Q_C=5$; deviating lowers both). Play it; observe $r_R=4,r_C=6$.
- Update: $Q_R(D,L)=3+alpha(4-3)=3+alpha$; $Q_C(D,L)=5+alpha(6-5)=5+alpha$. NE unchanged $=>$ converged. Sizes: indep $=|A_i|$; joint $=|A_1||A_2|$.

#sub[3. BNE — solve-for-$p$ (2023 exam)] P1 type $t_1$; P2 type $A$ (w.p. $p$) or $B$. If $A$: $X$ dominates for P2; if $B$: $Y$ dominates. So $a_2^*=(X "if" A, Y "if" B)$.
- P1: $"EU"(X)=p dot 0+(1-p)2=2(1-p)$; $"EU"(Y)=p dot 3+(1-p)1=2p+1$.
- P1 plays $Y$ iff $2p+1>2(1-p) => p>1/4$.

#sub[4. Extensive form — 3 P1-nodes (2025 exam)] P1${A,B}$; $A->$P2${C,D}$ ($C->$P1${G,H}: G(3,8),H(8,3)$; $D->(6,6)$); $B->$P2${E,F}$ ($E->(4,3)$; $F->$P1${I,J}: I(4,5),J(5,7)$).
- *BI:* P1 leaves: $H$ ($8>3$), $J$ ($7>5$). P2 left: $D$ ($6>3$)$->(6,6)$. P2 right: $F$ ($7>3$)$->(5,7)$. Root: $A$ ($6>5$).
- *SPE:* ${(A,H,J),(D,F)}$ (P1 strategy lists all 3 nodes).

#sub[5. Grim trigger — $beta$ threshold (2022 exam)] PD $(C,C)=6,6$; defect $T=15$ once then punished $P=3$; $R=6$.
- Coop forever $6/(1-beta)$ vs defect $15+beta(3/(1-beta))$.
- Diff $-9+beta(3/(1-beta)) >= 0 => beta >= (T-R)/(T-P)=(15-6)/(15-3)=3/4$. "Value tomorrow $>= 3/4$ of today."

#sub[6. Cournot asymmetric (2023 exam)] $P=100-2Q$, $c_A=20,c_B=10$. $pi_A=-2q_A^2-2q_A q_B+80q_A$; FOC $q_A=(40-q_B)/2$, $q_B=(45-q_A)/2$. Solve $=> q_A=35/3, q_B=50/3$, $P=130/3$.

#sub[7. Pareto on PD] NE $(D,D)$ is NOT Pareto optimal — $(C,C)$ beats it for both. NE $!=$ Pareto.

// --- leave any remaining column space blank: pencil-in room for anything missing ---
