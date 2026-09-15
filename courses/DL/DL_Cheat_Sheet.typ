// Deep Learning — Exam Cheat Sheet (A4, printable)
// Compile:  typst compile DL_Cheat_Sheet.typ
#set document(title: "Deep Learning — Cheat Sheet", author: "")
#set page(
  paper: "a4",
  margin: (x: 1cm, y: 1cm),
  numbering: "1 / 1",
  footer: context [
    #set text(size: 6.5pt, fill: gray)
    Deep Learning — IST — Exam Cheat Sheet
    #h(1fr)
    #counter(page).display("1 / 1", both: true)
  ],
)
#set text(font: ("New Computer Modern", "Libertinus Serif"), size: 8pt, hyphenate: true)
#set par(justify: true, leading: 0.42em, spacing: 0.55em)
#show math.equation: set text(size: 8pt)

// ---- Colours ----
#let cRec = rgb("#2f6db0")      // recipe blue
#let cRecBg = rgb("#eaf2fb")
#let cTrap = rgb("#b03a2e")     // trap red
#let cTrapBg = rgb("#fdecea")
#let cKey = rgb("#1e7a46")      // key green
#let cHead = rgb("#1a3a63")

// ---- Heading styling ----
#show heading.where(level: 1): it => block(width: 100%, above: 0.7em, below: 0.45em)[
  #set text(size: 9.5pt, fill: white, weight: "bold")
  #block(fill: cHead, inset: (x: 5pt, y: 3pt), radius: 2pt, width: 100%)[#it.body]
]
#show heading.where(level: 2): it => block(above: 0.5em, below: 0.3em)[
  #set text(size: 8.5pt, fill: cHead, weight: "bold")
  #it.body
  #v(-0.35em)
  #line(length: 100%, stroke: 0.4pt + cHead)
]

// ---- Helper boxes ----
#let recipe(title, body) = block(
  fill: cRecBg, inset: 5pt, radius: 3pt, width: 100%, above: 0.5em, below: 0.5em,
  stroke: 0.6pt + cRec,
)[
  #text(fill: cRec, weight: "bold", size: 8.2pt)[#title] \
  #body
]
#let trap(body) = block(
  fill: cTrapBg, inset: 5pt, radius: 3pt, width: 100%, above: 0.5em, below: 0.5em,
  stroke: 0.6pt + cTrap,
)[#body]
#let kw(x) = text(fill: cKey, weight: "bold")[#x]

// ================= TITLE =================
#align(center)[
  #text(size: 15pt, weight: "bold", fill: cHead)[Deep Learning — Exam Cheat Sheet] \
  #text(size: 7.5pt, fill: gray)[IST · Open-book final · Part 1 (8 MCQ ·32) + Part 2 (3 short ·18) + Part 3 (2 problems ·50) · 120 min]
]
#v(2pt)
#text(size: 7.5pt)[*Part 3 is fixed:* #kw[Problem 1 = CNN] · #kw[Problem 2 = sequence/attention]. Those two = 50 pts. The recipe boxes below win the exam.]
#v(3pt)
#line(length: 100%, stroke: 0.8pt + cHead)

// ================= BODY (2 columns) =================
#columns(2, gutter: 12pt)[

= ★ Exam Recipes (highest yield)

#recipe("CNN forward pass  (conv → ReLU → pool)")[
Input $X$, kernel $K$ ($k times k$), stride $S$, pad $P$, bias $b$: \
1. *Pad* $X$ with $P$ zero-rings. \
2. Each output cell $=$ (element-wise $K dot.o$ patch)$ + b$. No kernel flip (it's cross-correlation). \
3. Out size $M' = floor.l (M - k + 2P)/S floor.r + 1$. \
4. *ReLU* $max(0, dot)$ elementwise. \
5. *Max-pool* $p times p$, stride $s$: max per window; out $floor.l (M'-p)/s floor.r + 1$. Avg-pool $=$ mean.
]

#recipe("CNN parameter count — whole net (step by step)")[
*Track the shape $H times W times C$ layer by layer, then sum params:* \
1. *Conv/pool spatial size:* $M' = floor.l (M - k + 2P)\/S floor.r + 1$. (pad $P{=}1, k{=}3, S{=}1$ ⇒ *same* size; pool $2 times 2$ stride $2$ ⇒ *halves* $H,W$.) \
2. *Channels* $C$ = the number of filters of the last conv (pool keeps channels). \
3. *Conv params:* $(k dot k dot C_"in" + 1) dot C_"out"$ — $+1$ = bias/filter; $C_"in"$ = input *channels* (not spatial size!). Pool & activations: *0 params*. \
4. *Flatten* $= H_"last" dot W_"last" dot C_"last"$ (product of the last feature-map dims). \
5. *FC (linear)* $n_"in" -> n_"out"$: $(n_"in" + 1) dot n_"out"$ — $+1$ = bias/output neuron. \
*Total* = sum of all conv + FC params. Conv params are independent of input size; the FC count needs the flatten (so track shapes!). \
$1{times}1$ conv = channel mixing; factorized ($k{times}1$ then $1{times}k$) = fewer params, same field.
]

#recipe("RNN forward pass  (Prob 2, 2021–23)")[
$ h_t = g(W_(h x) x_t + W_(h h) h_(t-1) + b_h) $
$ "logits " o_t = W_(y h) h_t + b_y $
$h_0 = bold(0)$, usually $g = "ReLU"$. Compute each $h_t$. Greedy: predict $op("argmax")$ of logits. \
*Linear trick:* if $g = "id"$, $h_t = W_(h h)^t h_0 + sum_(j=1)^t W_(h h)^(t-j) W_(h x) x_j$ ⇒ classifier is *linear* in embeddings.
]

#recipe("Scaled dot-product attention  (Prob 2, 2023–25)")[
Form $X = E + P$ (token + positional), then
$ Q=X W_Q, quad K=X W_K, quad V=X W_V $
$ S = (Q K^top)/sqrt(d_k), quad P = "softmax"(S), quad Z = P V $
*Last token only:* $s = q_t K^top\/sqrt(d_k)$, $p="softmax"(s)$, $z = p V$; logits $=W_(y h) z$, predict $op("argmax")$. \
*Causal mask:* set $S_(i j)=-infinity$ for $j>i$ *before* softmax (decoder). \
*Multi-head:* concat heads then project $W_O$. Scaling a head's $W_Q$ by $c$ scales scores ⇒ *changes $P$ and $Z$*.
]


= Math & ML foundations
- #kw[Linear algebra]: inner product $x^top y$; norms $||x||_2 = sqrt(x^top x)$, $||x||_1 = sum|x_i|$, $||x||_infinity = max_i |x_i|$. #kw[Eigen]: $A v = lambda v$; symmetric $A$ ⇒ real eigenvalues, orthogonal eigenvectors. #kw[PSD] $A succ.eq 0$ iff $x^top A x >= 0 forall x$ iff all $lambda_i >= 0$. Gradient $nabla f$, Hessian $nabla^2 f$; convex iff $nabla^2 f succ.eq 0$.
- #kw[Probability]: Bayes $P(y|x) = P(x|y)P(y)\/P(x)$. $bb(E)[x]$, $"Var"(x) = bb(E)[x^2] - bb(E)[x]^2$. Entropy $H(p) = -sum p log p$; KL $"KL"(p||q) = sum p log(p\/q) >= 0$ (Gibbs). Gaussian $cal(N)(mu, sigma^2)$.
- #kw[ML]: learn $f: X -> Y$ from data (Mitchell: improve at task T on metric P with experience E). #kw[Supervised] (labels) / #kw[unsupervised] / #kw[RL]. Regression (continuous $y$) vs classification (discrete $y$). Naive Bayes = estimate class-conditionals by counting (+ smoothing).
- #kw[Over/underfit]: overfit = low train / high test error (high variance) → more data, regularize, dropout, early stop; underfit = both high (high bias) → more capacity/features. Test err $approx "bias"^2 + "variance" + "noise"$.

= Softmax, sigmoid & temperature
$ "softmax"(z)_i = e^(z_i) / (sum_j e^(z_j)) in (0,1), quad sum_i "softmax"(z)_i = 1 $
- For *finite* logits every component is *strictly* in $(0,1)$ — never exactly $0$/$1$ (since $e^x > 0$); those are reached only in a *limit*.
- #kw[Shift-invariance]: $"softmax"(z + c bold(1)) = "softmax"(z)$ (the $e^c$ cancels). Used for numerical stability: subtract $max_i z_i$ first.
- #kw[Temperature] $"softmax"(z\/T)$, $T>0$: $T -> 0^+$ → one-hot at $op("argmax") z$; $T -> infinity$ → *uniform*. For finite $T$ with *distinct* logits, output is strictly between the two; it is exactly uniform *iff all logits are equal* (then any $T$ gives uniform).
- #kw[Jacobian]: $partial p_i \/ partial z_j = p_i(delta_(i j) - p_j)$. With cross-entropy target $y$: $partial L\/partial z = p - y$.
- #kw[Sigmoid] = 2-class softmax: $sigma(z) = "softmax"([z,0])_1 = 1\/(1+e^(-z))$. Facts: $sigma(-z){=}1{-}sigma(z)$, $sigma(0){=}0.5$, $sigma' = sigma(1{-}sigma) <= 0.25$ (⇒ vanishing-gradient contributor).

= Losses, MLE & MAP
- #kw[Squared loss] $= $ Gaussian-noise MLE; #kw[cross-entropy/NLL] $=$ categorical MLE. Min NLL $=$ min $"KL"(hat(p)_"data" || p_theta)$.
- #kw[MAP] $=$ MLE $+$ log-prior $=$ regularized MLE: Gaussian prior → $ell_2$ (ridge/weight decay), Laplace prior → $ell_1$ (sparsity).
- Cross-entropy $H(p,q) = H(p) + "KL"(p||q)$. Jensen: $bb(E)[f(x)] >= f(bb(E)[x])$ for convex $f$.
- Why CE not MSE for classification: CE is convex in the logits, punishes confident-wrong hard, and gives calibrated probabilities; MSE+sigmoid saturates (near-zero gradient).
- Common losses: MSE (regression), MAE (robust), cross-entropy/NLL (classification), hinge $max(0, 1 - y f)$ (margin/SVM). Convex in linear params: squared, logistic, hinge.

= Linear regression & regularization
- Model $hat(y) = w^top x$ (bias via $x_0 {=} 1$). Squared loss $1/2 ||X w - y||^2$; #kw[normal equations] $hat(w) = (X^top X)^(-1) X^top y$. This LS solution $=$ MLE under Gaussian noise.
- #kw[Ridge] ($ell_2$) $hat(w) = (X^top X + lambda I)^(-1) X^top y$ = MAP with Gaussian prior; shrinks weights, always invertible for $lambda > 0$, reduces variance. #kw[Lasso] ($ell_1$) → sparse weights (feature selection), no closed form.
- #kw[SGD] step (one example): $w <- w - eta (w^top x - y) x$. Minibatch trades variance (noise) vs compute.
- #kw[Feature maps] $phi(x)$ let a linear model fit nonlinear targets (e.g. polynomial); one-hot encode categoricals.

= Linear classifiers: perceptron, logistic, softmax
- #kw[Perceptron]: predict $"sign"(w^top x)$; on a mistake $y(w^top x) <= 0$ do $w <- w + y x$. #kw[Novikoff]: if separable with unit-norm $w^*$, margin $gamma = min_i y_i w^(*top) x_i > 0$, $||x_i|| <= R$, then the number of mistakes is $<= (R\/gamma)^2$ (independent of $N$, dim). ⇒ 100 mistakes on unit-norm data ⇒ $gamma <= 0.1$.
- #kw[Logistic regression]: $P(y{=}1|x) = sigma(w^top x)$; binary cross-entropy; $nabla_w L = (sigma(w^top x) - y) x$. Loss convex ⇒ unique min *if not separable*; on *separable* data the min is at $||w|| -> infinity$ (no finite minimizer) — $ell_2$ restores one. Unlike the perceptron, GD gives *no finite-step* separation guarantee.
- #kw[Softmax/multinomial regression]: $P(y{=}k|x) = "softmax"(W x)_k$; $nabla_(w_k) L = (p_k - bb(1)[y{=}k]) x$. Outputs live on the simplex.
- #kw[Separability]: AND, OR are linearly separable; #kw[XOR is not] (needs a hidden layer).

= Neural networks & expressivity
- MLP: $h^((ℓ)) = g(W^((ℓ)) h^((ℓ-1)) + b^((ℓ)))$, output layer softmax/linear. Nonlinearity essential — stacked linear layers collapse to one linear map.
- #kw[Universal approximation]: one hidden layer with a non-polynomial activation approximates any continuous $f$ on a compact set arbitrarily well — but width can be *exponential*; #kw[depth] gives exponential efficiency for many functions. XOR needs only 1 hidden layer (2 ReLU units).
- #kw[Activation trade-offs]: sigmoid/tanh *saturate* (grad→0); tanh is zero-centered (better-conditioned than sigmoid); ReLU is cheap/sparse, no positive saturation, but *dead* if pre-activation always $<0$; Leaky/PReLU/ELU/GELU keep gradient for $z<0$; GELU/SiLU are smooth (Transformers).

== Activation derivatives
#table(columns: (1fr, 1fr), inset: 2.5pt, stroke: 0.3pt + gray, align: left,
  [*$g(z)$*], [*$g'(z)$*],
  [$sigma(z) = 1\/(1+e^(-z))$], [$sigma(z)(1-sigma(z))$],
  [$tanh(z)$], [$1 - tanh^2(z)$],
  [ReLU $max(0,z)$], [$bb(1)[z>0]$],
  [Softplus $log(1+e^z)$], [$sigma(z)$],
  [ELU ($z$; $e^z-1$)], [$1$; $e^z = min(e^z,1)$],
  [Leaky ReLU], [$1$; $alpha$],
)

= Backpropagation & autodiff
Layer $z^((ℓ)) = W^((ℓ)) h^((ℓ-1)) + b^((ℓ))$, $h^((ℓ)) = g(z^((ℓ)))$. With $delta^((ℓ)) = partial L\/partial z^((ℓ))$:
$ delta^((ℓ)) = (W^((ℓ+1) top) delta^((ℓ+1))) dot.o g'(z^((ℓ))), quad delta^((L)) = partial L\/partial h^((L)) dot.o g'(z^((L))) $
$ partial L\/partial W^((ℓ)) = delta^((ℓ)) (h^((ℓ-1)))^top, quad partial L\/partial b^((ℓ)) = delta^((ℓ)), quad partial L\/partial h^((ℓ-1)) = W^((ℓ) top) delta^((ℓ)) $
- Backprop = *reverse-mode autodiff*: one forward pass caches activations, one backward pass reuses shared sub-gradients ⇒ cost $approx 2 times$ forward.
- Chain-rule MCQ: multiply local derivatives along each path, *sum* over paths.
- #kw[Key gradients]: softmax+CE $partial L\/partial z = p - y$; sigmoid+BCE $sigma(z) - y$; MSE $hat(y)-y$; linear $z{=}W h$ ⇒ $partial L\/partial W = delta h^top$.

= Optimization, initialization & regularization
- #kw[Optimizers]: SGD $theta <- theta - eta nabla$; #kw[Momentum] $v <- beta v - eta nabla$, $theta <- theta + v$; Nesterov (look-ahead); AdaGrad ($1\/sqrt(sum g^2)$ per-param, decays LR); RMSProp (EMA of $g^2$); #kw[Adam] = momentum + RMSProp + bias-corrected $hat(m), hat(v)$: $theta <- theta - eta hat(m)\/(sqrt(hat(v)) + epsilon)$. LR schedules: warmup (Transformers), step/cosine decay. Minibatch: large = smoother/parallel/more memory; small = noisier (regularizing).
- #kw[Init] (variance preservation): #kw[Xavier/Glorot] $"Var"(W) = 2\/(n_"in" + n_"out")$ (tanh); #kw[He] $2\/n_"in"$ (ReLU). *Zero/equal init = symmetry problem*: all neurons get identical gradients and never differentiate.
- #kw[Regularization]: $ell_2$/weight decay ($+lambda/2 ||w||^2$ ⇒ grad $+lambda w$); $ell_1$ → sparse; #kw[dropout] (zero units w.p. $p$ at train, scale $1\/(1{-}p)$; off at test; ≈ ensembling); data augmentation; early stopping; *label smoothing* (softens targets → better calibration).
- #kw[Vanishing/exploding gradients]: products of Jacobians shrink/grow. Fix: ReLU, good init, residuals, normalization, and *gradient clipping* $g <- g dot min(1, tau\/||g||)$ (clipping fixes *exploding* only). Loss is *non-convex*; in high-dim *saddle points* dominate — SGD noise helps escape.

= Normalization
- #kw[BatchNorm]: $hat(x) = (x - mu_B)\/sqrt(sigma_B^2 + epsilon)$, then $gamma hat(x) + beta$. Smooths the landscape, allows higher LR, mild regularizer; *batch-size dependent*; uses running stats at test (train ≠ test behavior).
- #kw[LayerNorm]: normalize over *features per example* — batch-independent, standard in RNNs/Transformers. #kw[RMSNorm] = LayerNorm without mean subtraction.

= Convolutional networks
- Convolution = sparse, weight-*shared* linear map (cross-correlation): each output = $sum$ (filter $dot.o$ patch) $+ b$. #kw[Output size] $M' = floor.l (M - k + 2P)\/S floor.r + 1$; with dilation $d$, effective kernel $k_e = d(k{-}1)+1$.
- #kw[Params] $(k^2 C_"in" + 1) C_"out"$; pooling has none; conv $<<$ FC in params (sharing + locality).
- #kw[Equivariance]: shift in → shift out (translation only, *not* rotation/scale). #kw[Invariance]: conv *+ pooling* → ~unchanged under small shifts. #kw[Inductive bias]: translation equivariance + local receptive fields + parameter sharing. #kw[Receptive field] grows with depth/stride/dilation.
- #kw[Backprop] through conv is *also a convolution* (flipped kernel); max-pool routes gradient to the arg-max cell only, avg-pool spreads equally.
- #kw[Building blocks]: $1{times}1$ conv (per-pixel channel mixing / bottleneck); dilated (larger field, same params); transposed (upsampling); depthwise-separable (MobileNet, fewer params); global avg pool (replaces FC head). #kw[ResNet] skip $y = cal(F)(x) + x$ ⇒ $partial y\/partial x = cal(F)'(x)+1$ (gradient highway, enables very deep nets — *not* to skip layers).
- #kw[Architectures]: LeNet → AlexNet (ReLU+dropout+GPU, 2012) → VGG (stacked $3{times}3$) → GoogLeNet/Inception (parallel multi-scale + $1{times}1$ bottlenecks) → ResNet (skip, very deep).

= Recurrent networks & BPTT
- $h_t = g(W_(h h) h_(t-1) + W_(h x) x_t + b)$, output $o_t = W_(y h) h_t$. #kw[Parameter sharing] across time; trained by #kw[BPTT] (unroll then backprop, summing gradients).
- #kw[Vanishing/exploding]: $partial h_t\/partial h_k = product_(j=k+1)^t W_(h h)^top "diag"(g'(z_j))$ — governed by the *spectral radius* of $W_(h h)$ (and $|g'| <= 1$ for tanh/sigmoid): $<1$ vanish, $>1$ explode.
- #kw[LSTM]: gates $f,i,o = sigma(dot)$, $tilde(c)_t = tanh(dot)$, cell $c_t = f_t dot.o c_(t-1) + i_t dot.o tilde(c)_t$ (additive path ⇒ near-constant gradient), $h_t = o_t dot.o tanh(c_t)$. #kw[GRU]: update+reset gates, no separate cell state (fewer params). #kw[BiLSTM]: reads both directions.
- Autoregressive RNN makes *no Markov assumption* (full history in $h_t$). #kw[Teacher forcing] (feed gold prev token) speeds training but causes *exposure bias*. Variants: recursive nets / Tree-LSTMs (tree structure), PixelRNN (images).

= Sequence-to-sequence & attention
- #kw[Encoder–decoder]: encode source → decode target. A single fixed context vector is an #kw[information bottleneck] → #kw[attention] lets the decoder read *all* encoder states. Background: SMT / noisy-channel, IBM word alignments.
- #kw[Attention]: query $s$ (decoder state), keys/values $h_j$ (encoder). Scores → softmax weights $alpha_j$ → context $c = sum_j alpha_j h_j$. Score variants: #kw[additive] (Bahdanau) $v^top tanh(W_1 q + W_2 k)$; #kw[dot/scaled-dot] (Luong) $q^top k \/ sqrt(d)$.
- #kw[Decoding]: greedy (argmax); #kw[beam search] (keep top-$B$, *length-normalize* to avoid short bias); sampling with temperature / top-$k$ / top-$p$ (nucleus). #kw[BLEU] = geometric mean of $n$-gram precisions $times$ brevity penalty; LM → perplexity.

= Transformers & self-attention
$ "Attn"(Q,K,V) = "softmax"(Q K^top \/ sqrt(d_k)) V, quad Q{=}X W_Q, K{=}X W_K, V{=}X W_V $
- #kw[Why $sqrt(d_k)$]: iid unit-variance $q,k$ give $q dot k$ variance $d_k$; dividing by $sqrt(d_k)$ restores unit variance so softmax stays out of the low-gradient saturated regime.
- #kw[Multi-head]: $h$ heads on $d\/h$-dim subspaces, concat + $W_O$ — attends to different subspaces/relations. Params/block $approx 4 d^2$ (Q,K,V,O) + FFN $approx 8 d^2$. #kw[Block]: (multi-head self-attn → residual+LayerNorm → position-wise FFN [$approx 4 times$ wider] → residual+LayerNorm); *pre-LN* trains more stably than post-LN.
- #kw[Positional encoding] needed (self-attention is permutation-equivariant): sinusoidal $"PE"("pos",2i) = sin("pos"\/10000^(2i\/d))$ (relative offsets = linear maps, extrapolates) or learned. Deliberately omitted for permutation-invariant tasks.
- #kw[Masking]: causal/decoder sets future scores to $-infinity$ *before* softmax (autoregressive); padding masks ignore pad tokens. Decoder also has *cross-attention* to the encoder.
- #kw[Complexity]: self-attention $O(L^2 d)$ but *parallel* over positions; RNN $O(L d^2)$ but *sequential*.

= Representation learning & embeddings
- #kw[Autoencoder]: encoder → bottleneck → decoder, minimize reconstruction (MSE). #kw[Linear AE + squared loss ≡ PCA]: optimum spans the top-variance subspace (via SVD of centered $X$ / Eckart–Young). Variants: denoising, sparse, stochastic AEs; used for pretraining/transfer.
- #kw[PCA]: top eigenvectors of the covariance maximize retained variance (equivalently top singular vectors).
- #kw[word2vec]: #kw[skip-gram] (predict context from word) / #kw[CBOW] (predict word from context); #kw[negative sampling] replaces the full softmax with $log sigma(u_c^top v_w) + sum_"neg" log sigma(-u_n^top v_w)$. Captures analogies ($"king"-"man"+"woman" approx "queen"$). GloVe factorizes log co-occurrence; FastText adds subwords.
- Static (word2vec/GloVe) = one vector/word; #kw[contextual] (ELMo/BERT) = vector depends on the sentence.

= Self-supervised & pretrained models
- #kw[Self-supervised pretraining]: learn from unlabeled text via a proxy objective, then fine-tune. #kw[BPE] tokenization iteratively merges the most frequent adjacent pair ⇒ open-vocabulary subwords (rare words = sequences of subwords).
- #kw[BERT]: encoder-only, #kw[masked LM] (mask ~15%), bidirectional [+ next-sentence pred] → understanding/classification. #kw[GPT]: decoder-only, causal/next-token LM → generation. #kw[T5/BART]: encoder-decoder, span corruption (text-to-text).
- #kw[Adaptation]: full fine-tune vs parameter-efficient — #kw[adapters] (small bottleneck layers), #kw[LoRA] ($W + B A$, rank $r << d$, train only $B,A$), prompt/prefix-tuning. #kw[In-context learning] = few/zero-shot, *no weight update*.
- #kw[Scaling laws]: loss falls as a power law in params/data/compute (compute-optimal balances them); emergent abilities at scale. #kw[RLHF/RLVR]: SFT → reward model from human preferences → PPO; aligns behavior (≠ capability). Risks: "stochastic parrots" / hallucination / bias.

= Generative models
- #kw[Autoregressive] $p(x) = product_t p(x_t | x_(<t))$: exact likelihood, slow sequential sampling (PixelRNN, GPT).
- #kw[VAE]: encoder outputs a *distribution* $q(z|x) = cal(N)(mu, sigma^2)$. Maximize the #kw[ELBO] (lower bound on $log p(x)$; gap $= "KL"(q || "true posterior")$):
$ cal(L) = underbrace(bb(E)_q [log p(x|z)], "reconstruction") - underbrace("KL"(q(z|x) || p(z)), "regularizer") $
#kw[Reparameterization] $z = mu + sigma dot.o epsilon$, $epsilon ~ cal(N)(0,I)$ (differentiable sampling). $"KL"(cal(N)(mu,sigma^2)||cal(N)(0,I)) = 1/2 sum_i (sigma_i^2 + mu_i^2 - 1 - ln sigma_i^2)$. Risk: *posterior collapse*.
- #kw[GAN]: $min_G max_D bb(E)[log D(x)] + bb(E)[log(1 - D(G(z)))]$; implicit density, sharp samples, *mode collapse* / instability; non-saturating & Wasserstein (WGAN) losses stabilize.
- #kw[Diffusion]: forward adds Gaussian noise over $T$ steps; model learns to denoise (predict $epsilon$). High quality, slow sampling. #kw[Energy-based]: $p(x) prop e^(-E(x))$. #kw[Normalizing flows]: invertible maps, *exact* likelihood via change-of-variables.
#table(columns: (auto, 1fr), inset: 2.5pt, stroke: 0.3pt + gray, align: left,
  [*Model*], [*Likelihood / sampling / failure*],
  [Autoregressive], [exact likelihood; slow sequential sampling],
  [VAE], [ELBO (approx.); fast; posterior collapse],
  [GAN], [implicit (none); fast; mode collapse / unstable],
  [Diffusion], [approx.; slow (iterative); high quality],
  [Norm. flow], [exact; fast; architecture-constrained],
)

= Uncertainty, explainability & fairness
- #kw[Aleatoric] uncertainty = data noise (irreducible); #kw[epistemic] = model uncertainty (reducible with data). Total = aleatoric + epistemic; predictive entropy = total, mutual information (BALD) isolates epistemic. Estimate epistemic via #kw[deep ensembles], #kw[MC-dropout], BNNs. #kw[Distribution shift] (covariate/label/concept) inflates epistemic uncertainty.
- #kw[Calibration]: confidence ≈ accuracy. $"ECE" = sum_b (n_b\/N) |"acc"(b) - "conf"(b)|$ over confidence bins; reliability diagram plots conf vs acc; #kw[temperature scaling] (one logit param) is a post-hoc fix.
- #kw[Explainability]: *intrinsic* (interpretable models) vs *post-hoc*. Gradient-based: saliency, integrated gradients, Grad-CAM. Surrogates: LIME, SHAP; counterfactuals. Attention-as-explanation is *contested*.
- #kw[Fairness]: bias originates in data (e.g. co-occurrence imbalance → MT gender bias, fixed by data augmentation). Metrics: #kw[demographic parity] ($P(hat(y){=}1|A)$ equal), #kw[equalized odds] (equal TPR *and* FPR), #kw[equal opportunity] (equal TPR) — *mutually incompatible in general* (impossibility). Also: privacy (membership inference / PII), sustainability (compute cost). WEAT measures embedding bias.

= Information theory & metrics
- $H(p) = -sum p log p$; cross-entropy $H(p,q) = -sum p log q$; $"KL"(p||q) = H(p,q) - H(p) >= 0$. #kw[Perplexity] $= e^"NLL"$ (LM, lower better).
- Classification: $"Prec" = "TP"\/("TP"+"FP")$, $"Rec" = "TP"\/("TP"+"FN")$, $F_1 = 2 "PR"\/(P+R)$, accuracy $= ("TP"+"TN")\/"all"$; ROC-AUC. Generation: BLEU (MT), perplexity (LM).

= Generalization & double descent
- Test error $approx "bias"^2 + "variance" + "noise"$: high bias = underfit, high variance = overfit; regularization trades variance for bias.
- Deep nets show #kw[double descent]: past the interpolation threshold test error can *fall again*. Overparameterized nets generalize despite fitting noise — attributed to SGD's *implicit regularization* (bias toward low-norm / flat minima). Classical capacity bounds (VC / Rademacher) are loose here. More parameters than samples is fine.

= ⚠ "False-statement" traps
#trap[
1. GD for logistic reg finds a separator in finite steps — *FALSE* (perceptron does). \
2. Masking is used in the *encoder* to block future tokens — *FALSE* (decoder; future = output seq). \
3. GPT-3 is encoder-only masked-LM — *FALSE* (decoder-only, causal). \
4. Gradient *clipping* prevents *vanishing* grads — *FALSE* (prevents exploding). \
5. Adding a constant to softmax logits changes output — *FALSE* (shift-invariant); scaling *does*. \
6. Convolution is *rotation*-equivariant — *FALSE* (translation only). \
7. Autoregressive RNN makes a *Markov* assumption — *FALSE* (full history via $h_t$). \
8. Skip connections exist to *drop/skip layers* — *FALSE* (gradient flow). \
9. Linear AE differs from PCA — *FALSE* (equivalent). \
10. More params than samples ⇒ can't generalize — *FALSE* (deep nets often do). \
11. VAE encodes to a fixed point — *FALSE* (a distribution). \
12. BatchNorm behaves the same at train and test — *FALSE* (batch vs running stats).
]

]

#pagebreak()

// ================= SOLVED PROBLEMS (full width) =================
#let psol(body) = block(fill: rgb("#f2f7f2"), inset: 5pt, radius: 3pt, width: 100%,
  stroke: 0.5pt + cKey, above: 0.4em, below: 0.7em)[#text(fill: cKey, weight: "bold")[Solution.] #body]
#let ans(body) = block(inset: (left: 4pt, top: 1pt, bottom: 1pt), stroke: (left: 1.5pt + cKey),
  above: 0.15em, below: 0.55em)[#text(fill: cKey, size: 7.6pt)[*✓ Ans.* #body]]

= 📘 Worked Exam Problems — from past exams (official solutions)

== R1 · CNN forward pass + equivariance / invariance  (2024, Prob 1)
Conv layer, $3 times 3$ filter, stride $1$, pad $1$, ReLU, bias $b=0$:
$ K = mat(1,0,-1; 2,0,-2; 1,0,-1), quad x = mat(0,0,1,0; 0,1,0,0; 0,1,0,0; 0,1,0,0) $
*(a)* Compute the layer output $h$.  *(b)* Repeat for the input translated one column right.  *(c)* Apply a $2 times 2$ max-pool (stride 2) to each; identify the property.

#psol[
*(a)* Zero-pad $x$ to $6 times 6$ (a ring of zeros). Each output cell = sum of the elementwise product of $K$ with the $3 times 3$ padded window whose centre is that pixel.
- Cell $z_(1,1)$ (top-left): window $mat(0,0,0; 0,0,0; 0,0,1)$, so $z_(1,1) = 1 dot (-1) = -1$ (only the bottom-right entry is non-zero, aligned with $K_(3,3){=}{-}1$).
- Cell $z_(3,3)$: window $mat(1,0,0; 1,0,0; 1,0,0)$, so $z_(3,3) = 1 dot 1 + 1 dot 2 + 1 dot 1 = 4$ (the "1" column meets $K$'s first column $[1,2,1]$).

Doing all 16 cells:
$ z = mat(-1,-2,1,2; -3,-1,3,1; -4,0,4,0; -3,0,3,0), quad h = "ReLU"(z) = mat(0,0,1,2; 0,0,3,1; 0,0,4,0; 0,0,3,0) $
(ReLU zeros every negative entry.)
*(b)* Translating the input one column right translates the whole output one column right ⇒ #kw[equivariance]: $f("shift"(x)) = "shift"(f(x))$.
*(c)* Max-pool of $h$ with $2 times 2$ windows (stride 2): top-left window $mat(0,0;0,0) -> 0$, top-right $mat(1,2;3,1) -> 3$, bottom-left $mat(0,0;0,0) -> 0$, bottom-right $mat(4,0;3,0) -> 4$, giving $mat(0,3; 0,4)$. The translated input yields the *same* pooled output $mat(0,3;0,4)$ ⇒ #kw[invariance] (conv + pooling is approximately shift-invariant).
]

== R2 · CNN parameter table + Inception forward  (2023, Prob 1, GoogLeNet)
A $1 times 1$ conv on an $H times W times C$ image applies a filter of shape $1 times 1 times C$ (a per-pixel FC over channels; used to change channel count). For an Inception block with single-output-channel convs, stride 1, pad 0 unless stated:

#table(columns: (auto, auto, auto), inset: 3pt, stroke: 0.35pt + gray, align: (left, center, center),
  [*Element*], [*\# params*], [*Dimension*],
  [$h_1$ ($1 times 1$ conv)], [$C+1$], [$H times W$],
  [$h_2$ ($1 times 1$ conv)], [$C+1$], [$H times W$],
  [$h'_2$ ($3 times 3$ conv, pad 1)], [$3 dot 3 + 1 = 10$], [$H times W$],
  [$h_3$ ($1 times 1$ conv)], [$C+1$], [$H times W$],
  [$h'_3$ ($5 times 5$ conv, pad 2)], [$5 dot 5 + 1 = 26$], [$H times W$],
  [$h_4$ ($3 times 3$ maxpool, pad 1)], [$0$], [$H times W times C$],
  [$h'_4$ ($1 times 1$ conv)], [$C+1$], [$H times W$],
  [$h$ (concat)], [–], [$H times W times 4$],
)
Forward: the $1 times 1$ conv $K=[0.5, 0.5]$ averages the 2 channels → $h_2 = mat(1.0,1.0,1.0; 0.8,0.5,0.8; 0.9,0.8,0.9)$. Then $3 times 3$ conv $K_(3 times 3)=mat(-1,-1,-1; 0,0,0; 1,1,1)$ (pad 1):

#psol[
$K_(3 times 3)$ has $-1$ on its top row, $0$ middle, $+1$ bottom ⇒ each output = (sum of bottom row of window) − (sum of top row).
- Cell $z'_(2,(1,1))$ (pad the $3 times 3$ $h_2$ to $5 times 5$; window = top-left corner) $= mat(0,0,0; 0,1.0,1.0; 0,0.8,0.5) $: bottom row $0+0.8+0.5 = 1.3$ minus top row $0 ⇒ 1.3$.
- Cell $z'_(2,(2,1))$: window $mat(0,1.0,1.0; 0,0.8,0.5; 0,0.9,0.8)$: bottom $1.7$ − top $2.0 = -0.3$.
$ z'_2 = mat(1.3, 2.1, 1.3; -0.3, -0.4, -0.3; -1.3, -2.1, -1.3), quad h'_2 = "ReLU"(z'_2) = mat(1.3, 2.1, 1.3; 0,0,0; 0,0,0) $
]

== R3 · RNN forward pass + greedy prediction  (2024, Prob 2)
Autoregressive RNN, $h_t = "ReLU"(W_(h x) x_t + W_(h h) h_(t-1))$, all biases $0$, $h_0 = bold(0)$, with
$ W_(h x) = mat(1,1,0; 0,-1,-1), quad W_(h h) = mat(1,0; 0,1) = I $
Sentence "Awful plot and actors. The movie is" with embeddings $x_"Awful"=[0,-5,0]^top$, $x_"plot"=[1,0,0]^top$, $x_"actors."=[-1,0,0]^top$, $x_"movie"=[-2,0,0]^top$, others $[0,0,0]^top$. Output row $w_"good"=[1,-1]$, $w_"bad"=[-1,1]$.

#psol[
Since $W_(h h)=I$, each step adds $W_(h x)x_t$ and re-applies ReLU:
$ h_1=[0,0], quad h_2=[0,5], quad h_3=[1,5], quad h_4=[1,5], quad h_5=[0,5], quad ... quad h_8=[0,5]. $
E.g. $h_2 = "ReLU"(W_(h x)[0,-5,0]^top + 0) = "ReLU"([-5,5]^top) = [0,5]^top$.
Logits at the "[Z]" slot: $"good" = w_"good" h_8 = -5$, $"bad" = w_"bad" h_8 = 5$. Since $"bad" > "good"$ ⇒ predict *negative* sentiment. (With *linear* activations, unrolling shows the classifier becomes *linear* in the embeddings.)
]

== R4 · Self-attention, last-token shortcut  (2024, Prob 2, transformer)
Single self-attention head, $W_Q=W_K=W_V=mat(0.1,0; 0,-0.1; 0,0.1)$, scaled dot-product with $sqrt(d_k)=sqrt(2)$. You only need the *last* token's query $q_8=[0, 0.7]$ and all keys.

#psol[
$ s_8 = q_8^top K^top \/ sqrt(2) = 1/sqrt(2)[0, 0.42, 0.14, 0.21, 0.28, 0.35, 0.42, 0.49] $
$ p_8 = "softmax"(s_8) = [0.101, 0.136, 0.112, 0.118, 0.124, 0.130, 0.136, 0.143] $
$ z_8 = p_8^top V = [-0.028, 0.436] $
Logits: $"good"=[1,-1] z_8 = -0.464$, $"bad"=[-1,1] z_8 = 0.464$ ⇒ predict *negative*. *Trick:* for the last-token prediction you never need the full $P$ matrix — just its last row.
]

= ✍ Practice Problems — my own (exam-style, with worked solutions)

== M1 · CNN output sizes & parameter count
A CNN takes a $32 times 32 times 3$ image: `Conv1` (16 filters $3 times 3$, stride 1, pad 1) → `MaxPool` ($2 times 2$, stride 2) → `Conv2` (32 filters $3 times 3$, stride 1, pad 1) → `MaxPool` ($2 times 2$, stride 2) → `Flatten` → `FC` (10 outputs). Give each layer's output size and trainable-parameter count.

#psol[
Output size $M' = floor.l (M - k + 2P)/S floor.r + 1$; conv params $(k dot k dot C_"in"+1)C_"out"$.
#table(columns: (auto, auto, auto), inset: 3pt, stroke: 0.35pt + gray, align: (left, center, right),
  [*Layer*], [*Output*], [*Params*],
  [Conv1], [$32 times 32 times 16$], [$(3 dot 3 dot 3+1) dot 16 = 448$],
  [MaxPool], [$16 times 16 times 16$], [$0$],
  [Conv2], [$16 times 16 times 32$], [$(3 dot 3 dot 16+1) dot 32 = 4640$],
  [MaxPool], [$8 times 8 times 32$], [$0$],
  [Flatten], [$2048$], [$0$],
  [FC], [$10$], [$(2048+1) dot 10 = 20 490$],
)
*Total $= 448 + 4640 + 20 490 = 25 578$ parameters.* (Conv sizes unchanged by "same" padding $P=1,k=3$; pooling halves each spatial dim.)
]

== M2 · Scaled dot-product self-attention (last token)
Three tokens, $d_k=2$, identity projections ($Q=K=V=X$), $sqrt(d_k)=sqrt(2)$:
$ X = mat(2,0; 0,2; 2,1) quad (x_1, x_2, x_3 "as rows}"). $
Compute the context vector $z_3$ for the last token, and predict $"good"$ vs $"bad"$ with $w_"good"=[1,-1]$, $w_"bad"=[-1,1]$.

#psol[
Query $q_3=[2,1]$. Raw scores $q_3 dot k_i$: $k_1{:}\, 2 dot 2 + 1 dot 0 = 4$, $k_2{:}\, 0 + 2 = 2$, $k_3{:}\, 4 + 1 = 5$; scaled by $1\/sqrt(2)$: $[2.828, 1.414, 3.536]$.
Softmax = exponentiate and normalize: $e^(2.828){=}16.92$, $e^(1.414){=}4.11$, $e^(3.536){=}34.33$; sum $= 55.36$:
$ p_3 = [16.92, 4.11, 34.33]\/55.36 = [0.306, 0.074, 0.620]. $
$ z_3 = 0.306[2,0] + 0.074[0,2] + 0.620[2,1] = [0.611{+}1.240, 0.148{+}0.620] = [1.851, 0.769]. $
Logits: $"good" = [1,-1] dot z_3 = 1.851 - 0.769 = 1.082 > "bad" = -1.082$ ⇒ predict *good*. (The last token attends most to itself, score $5$.)
]

== M3 · Backpropagation through a 2-layer MLP
$x=[1,2]$, hidden $h = "ReLU"(W_1 x)$, output $hat(y) = w_2^top h$, loss $L = 1/2 (hat(y)-y)^2$, target $y=1$, with
$ W_1 = mat(1,-1; 0,1), quad w_2 = [1,1]. $
Compute all gradients and one SGD step ($eta = 0.1$).

#psol[
*Forward:* $z_1 = W_1 x = [-1, 2]$, $h = "ReLU"(z_1) = [0,2]$, $hat(y) = 1 dot 0 + 1 dot 2 = 2$, $L = 1/2(2-1)^2 = 0.5$.
*Backward:* $partial L\/partial hat(y) = hat(y)-y = 1$.
$ partial L\/partial w_2 = (hat(y)-y) h = [0, 2]; quad partial L\/partial h = (hat(y)-y) w_2 = [1,1]. $
$ partial L\/partial z_1 = partial L\/partial h dot.o "ReLU"'(z_1) = [1,1] dot.o [0,1] = [0,1]. $
$ partial L\/partial W_1 = (partial L\/partial z_1) x^top = mat(0,0; 1,2). $
*SGD:* $w_2 <- [1, 0.8]$, $quad W_1 <- mat(1,-1; -0.1, 0.8)$.
]

== M4 · Logistic-regression SGD update
Binary logistic regression, $P(y{=}1|x)=sigma(w^top x + b)$, one step of SGD on the NLL loss with $eta=1$, from $w=[0,0], b=0$, on $(x,y) = ([1,2], 1)$.

#psol[
$z = w^top x + b = 0$, $sigma(0) = 0.5$. Gradient $(sigma - y)x = (0.5-1)[1,2] = [-0.5, -1]$, and $(sigma-y) = -0.5$ for $b$.
Update: $w <- [0,0] - 1 dot [-0.5,-1] = [0.5, 1]$, $quad b <- 0 - 1 dot(-0.5) = 0.5$. The boundary moves toward classifying $x$ as positive.
]

== M5 · Multi-head attention — effect of the projection matrices  (recurring MCQ + Prob 2)
Head 1 has $W_Q^((1)), W_K^((1)), W_V^((1))$. Relate $P, Z$ of a second head in two cases: *(a)* negated $W_Q^((2)){=}{-}W_Q^((1))$, $W_K^((2)){=}{-}W_K^((1))$, $W_V^((2)){=}{-}W_V^((1))$; *(b)* scaled $W_Q^((2)){=}2 W_Q^((1))$, $W_K^((2)){=}W_K^((1))$, $W_V^((2)){=}1/2 W_V^((1))$.

#psol[
*(a)* $Q^((2)) K^((2)top) = (-Q^((1)))(-K^((1)))^top = Q^((1)) K^((1)top)$ ⇒ $P^((2)) = P^((1))$ (signs cancel *inside* softmax). But $Z^((2)) = P^((2)) V^((2)) = P^((1))(-V^((1))) = -Z^((1))$. So $P^((1)){=}P^((2))$, $Z^((1)){=}{-}Z^((2))$.
*(b)* Scores scale by $2$: $P^((2)) = "softmax"(2 Q^((1)) K^((1)top)\/sqrt(d)) != P^((1))$ (softmax is *not* scale-invariant). And $Z^((2)) = P^((2))(1/2 V^((1))) != Z^((1))$. So *both* differ. \
*Rule:* a common factor on *both* $Q,K$ that cancels (like $-1 dot -1$) leaves $P$ unchanged; any *net* scaling of $Q K^top$ changes $P$.
]

== M6 · Causal (masked) self-attention matrix
Raw scaled scores $S = mat(2,4,1; 1,3,5; 0,2,4)$ (row $i$ = query $i$). Apply a *causal* mask (token $i$ attends only to $j <= i$) then row-softmax to get $P$.

#psol[
Set upper triangle to $-infinity$ *before* softmax ($e^(-infinity){=}0$), then softmax each row over the allowed entries.
- Row 1: only $j{=}1$ allowed → $[1,0,0]$.
- Row 2: $"softmax"([1,3])$: $e^1{=}2.72$, $e^3{=}20.09$, sum $22.81$ → $[2.72, 20.09]\/22.81 = [0.119, 0.881]$.
- Row 3: $"softmax"([0,2,4])$: $e^0{=}1$, $e^2{=}7.39$, $e^4{=}54.60$, sum $62.99$ → $[0.016, 0.117, 0.867]$.
$ P = mat(1, 0, 0; 0.119, 0.881, 0; 0.016, 0.117, 0.867) $
$P$ is lower-triangular ⇒ no token sees the future (autoregressive/decoder).
]

== M7 · BPTT — vanishing vs exploding gradient
Scalar RNN, linear activation, $h_t = w_(h h) h_(t-1) + w_(h x) x_t$. How does $partial h_t \/ partial h_(t-k)$ behave over $k$ steps for $w_(h h) = 0.5$ vs $w_(h h) = 1.5$?

#psol[
$partial h_t \/ partial h_(t-1) = w_(h h)$, so over $k$ steps $partial h_t \/ partial h_(t-k) = w_(h h)^k$.
$w_(h h){=}0.5$: $0.5^(10) approx 0.001$ ⇒ *vanishes*. $w_(h h){=}1.5$: $1.5^(10) approx 57.7$ ⇒ *explodes*.
General (vector): factor is $||W_(h h)^top "diag"(g')||$ per step; $<1$ vanish, $>1$ explode. LSTM/GRU's additive cell state $approx 1$ per step avoids this; clipping caps explosion.
]

== M8 · VAE — ELBO with KL between Gaussians
Encoder outputs $q(z|x) = cal(N)(mu, sigma^2)$ with $mu = [1, 0]$, $sigma^2 = [0.25, 1]$; prior $cal(N)(0, I)$. Given reconstruction term $bb(E)_q[log p(x|z)] = -1.5$, compute the KL term and the ELBO.

#psol[
$ "KL" = 1/2 sum_i (sigma_i^2 + mu_i^2 - 1 - ln sigma_i^2) $
Dim 1: $1/2(0.25 + 1 - 1 - ln 0.25) = 1/2(0.25 + 1.386) = 0.818$.
Dim 2: $1/2(1 + 0 - 1 - ln 1) = 0$.  So $"KL" = 0.818$.
$ "ELBO" = underbrace(-1.5, "recon") - underbrace(0.818, "KL") = -2.318. $
Maximizing ELBO = better reconstruction + latent close to the prior. Sample via $z = mu + sigma dot.o epsilon$, $epsilon ~ cal(N)(0,I)$.
]

== M9 · Computation-graph chain rule (MCQ style)
Given $y = sigma(a b + c)$ with $a{=}1, b{=}2, c{=}0$, which is $partial y \/ partial a$?  (a) $sigma'(a b + c)$  (b) $y(1-y) b$  (c) $y(1-y) a$  (d) $y(1-y)$

#psol[
*Answer (b).* Chain rule: $partial y\/partial a = sigma'(z) dot partial z\/partial a = y(1-y) dot b$ (since $z = a b + c$, $partial z\/partial a = b$). Numerically $z{=}2$, $y = sigma(2) = 0.881$, so $partial y\/partial a = 0.881 dot 0.119 dot 2 = 0.210$.
]

== M10 · Softmax temperature
Logits $z = [2, 1, 0]$. Give $"softmax"(z\/T)$ for $T = 1, 0.5, 2$ and describe the effect.

#psol[
#table(columns: (auto, auto, auto, auto), inset: 3pt, stroke: 0.35pt + gray, align: (center, left, left, left),
  [*$T$*], [*$z\/T$*], [*$e^(z\/T)$ / sum*], [*softmax*],
  [$1$], [$[2,1,0]$], [$[7.39, 2.72, 1]\/11.11$], [$[0.665, 0.245, 0.090]$],
  [$0.5$], [$[4,2,0]$], [$[54.6, 7.39, 1]\/62.99$], [$[0.867, 0.117, 0.016]$],
  [$2$], [$[1, 0.5, 0]$], [$[2.72, 1.65, 1]\/5.37$], [$[0.506, 0.307, 0.186]$],
)
Lower $T$ → *sharper* (→ argmax); higher $T$ → *flatter* (→ uniform). $T -> 0$: one-hot; $T -> infinity$: uniform. Adding a constant to $z$ changes *nothing* (shift-invariance); scaling by $1\/T$ *does*.
]

== M11 · Perceptron — counting mistakes
Run the (no-bias) perceptron, $w{=}[0,0]$, on $x_1{=}[1,0]\,(+1)$, $x_2{=}[0,1]\,(+1)$, $x_3{=}[-1,-1]\,(-1)$, in order. How many mistakes to converge?

#psol[
$x_1$: $w dot x_1 = 0 <= 0$ → mistake, $w <- w + x_1 = [1,0]$.
$x_2$: $w dot x_2 = 0 <= 0$ → mistake, $w <- w + x_2 = [1,1]$.
$x_3$: $w dot x_3 = -2$, $y_3(w dot x_3) = (-1)(-2) = 2 > 0$ → correct.
Re-check $x_1, x_2$: both $> 0$ ✓. Converged after *2 mistakes*. (Guaranteed finite because the data is linearly separable — Novikoff.)
]

#pagebreak()

= ✅ Multiple-Choice Bank — exam-style themes (with answers)
_Cover the ✓ Ans line and try each first. These mirror the recurring Part-1 MCQ themes._

#columns(2, gutter: 12pt)[

*Q1 (Softmax invariance).* For $g(s,t) = "softmax"(v s + bold(1) t)$ with $v != 0$ and scalars $s, t$: (a) constant in both $s, t$; (b) constant in $t$, not $s$; (c) constant in $s$, not $t$; (d) constant in neither.
#ans[(b). Adding $bold(1)t$ shifts *all* logits equally ⇒ softmax unchanged in $t$ (shift-invariance). Scaling by $s$ = temperature ⇒ output *does* change.]

*Q2 (Perceptron on scaled data).* A separable set $D$ and $D' = {(2x, y)}$ (inputs doubled). No bias, $w_0 = 0$. On which does the perceptron make more mistakes?
#ans[Same for both. Updates on $D'$ are $w{+}2y x$ = exactly twice those on $D$; the sign of $w dot x$ at each step is identical, so the mistake sequence matches.]

*Q3 (ResNet).* The main purpose of skip connections in ResNets is:
(a) connect input directly to output; (b) skip layers to cut compute; (c) randomly drop layers like dropout; (d) let the gradient flow directly, mitigating vanishing gradients.
#ans[(d). $y = cal(F)(x)+x$ gives $partial y\/partial x = cal(F)'(x)+1$, so gradient always has a path back.]

*Q4 (Multi-head, negated).* Head 2 uses $W_Q^((2)){=}{-}W_Q^((1))$, $W_K^((2)){=}{-}W_K^((1))$, $W_V^((2)){=}{-}W_V^((1))$. Relation of $P, Z$?
#ans[$P^((1)){=}P^((2))$ and $Z^((1)){=}{-}Z^((2))$. Signs cancel in $Q K^top$ (so $P$ equal) but survive in $P V$ (so $Z$ negated).]

*Q5 (Masked attention — pick the FALSE).*
(a) Decoder masking lets each token attend only to previous tokens; (b) masking is applied in the *encoder* to stop attending to later *input* tokens; (c) future tokens are hidden during training; (d) it is crucial for language modeling.
#ans[(b) is FALSE. Causal masking lives in the *decoder* and refers to the *output* sequence; the encoder attends bidirectionally.]

*Q6 (VAE vs AE).* What distinguishes a variational autoencoder?
(a) different hidden activation; (b) encodes to fixed points; (c) introduces randomness — a *distribution* over the latent space per input; (d) only does dimensionality reduction.
#ans[(c). The encoder outputs $mu, sigma$ defining $q(z|x){=}cal(N)(mu,sigma^2)$; sampling + the KL term regularize the latent space.]

*Q7 (Softplus derivative).* $d/(d t) log(1+e^t) = $
(a) $sigma(t)(1{-}sigma(t))$; (b) $sigma(t)$; (c) $sigma(-t)$; (d) $"softplus"(t)(1{-}"softplus"(t))$.
#ans[(b) $sigma(t) = 1\/(1+e^(-t))$. Differentiate: $e^t\/(1+e^t) = sigma(t)$.]

*Q8 (Commuting ops).* Which pair *always* gives the same result when swapped?
(a) conv & max-pool; (b) conv & ReLU; (c) ReLU & max-pool; (d) none.
#ans[(c). ReLU is monotonic non-decreasing, so $max("ReLU"(dot)) = "ReLU"(max(dot))$. Conv is linear and does *not* commute with either nonlinear op.]

*Q9 (Which layer?).* Which computes its output via dot products between input patches and filters?
(a) convolution; (b) max-pool; (c) fully-connected; (d) none.
#ans[(a). That *is* the definition of a convolutional layer (cross-correlation of a filter over patches).]

*Q10 (Pretrained — pick the FALSE).*
(a) BERT is encoder-only, trained with masked LM; (b) GPT is decoder-only, causal LM; (c) GPT-3 is encoder-only with masked LM; (d) T5 is encoder-decoder.
#ans[(c) is FALSE. GPT models are *decoder-only, autoregressive* — not encoder-only MLM.]

*Q11 (Gradient clipping — pick the FALSE).*
(a) clipping bounds the gradient norm; (b) it helps with exploding gradients in RNNs; (c) it prevents *vanishing* gradients; (d) it is common in BPTT.
#ans[(c) is FALSE. Clipping caps *large* gradients (exploding); vanishing needs LSTM/GRU/init/residuals.]

*Q12 (Autoregressive RNN — pick the FALSE).*
(a) it conditions on the full history via $h_t$; (b) it makes a Markov (fixed-window) assumption; (c) it can model long dependencies in principle; (d) it is trained by BPTT.
#ans[(b) is FALSE. The recurrent state summarizes *all* past tokens — no fixed Markov window.]

*Q13 (Parallelism).* Which is TRUE about training a Transformer vs an RNN?
(a) both are sequential over positions; (b) self-attention can be computed in parallel across positions; (c) RNNs are $O(L^2)$; (d) attention is cheaper than $O(L^2)$.
#ans[(b). Self-attention has no recurrence, so all positions compute together ($O(L^2 d)$, parallel); the RNN must go step by step.]

*Q14 (Conv output size).* Input $28 times 28$, filter $5 times 5$, stride $1$, padding $0$. Output spatial size?
(a) $28$; (b) $26$; (c) $24$; (d) $23$.
#ans[(c) $24$. $M' = floor.l (28-5+0)/1 floor.r + 1 = 24$.]

*Q15 (Parameter count).* A conv layer: input $32 times 32 times 3$, $10$ filters $5 times 5$. Trainable params?
(a) $250$; (b) $750$; (c) $760$; (d) $7680$.
#ans[(c) $760 = (5 dot 5 dot 3 + 1) dot 10$ (don't forget the $+1$ bias per filter).]

*Q16 (Diagnosis).* A high-degree polynomial fits the training points perfectly but tests poorly. This is:
(a) underfitting, add complexity; (b) overfitting, use $ell_2$ / more data; (c) underfitting, use dropout; (d) fine, no action.
#ans[(b). Low train + high test error = *overfitting* (high variance) → regularize, get more data, or reduce capacity.]

*Q17 (Zero init — pick the FALSE).*
(a) zero-init makes all neurons in a layer receive identical gradients; (b) this is the symmetry problem; (c) it prevents the network from learning distinct features; (d) it is the recommended default init.
#ans[(d) is FALSE. Zero init is *bad* (symmetry); use Xavier/Glorot or He.]

*Q18 (Attention scaling).* Why divide the dot-product scores by $sqrt(d_k)$?
(a) to normalize to a probability; (b) to keep score variance $approx 1$ so softmax gradients don't vanish; (c) to make it permutation-invariant; (d) to reduce parameters.
#ans[(b). For large $d_k$, unscaled dot products grow, pushing softmax into saturated regions with tiny gradients.]

*Q19 (Separability — pick the FALSE).*
(a) the perceptron converges in finite steps if data is separable; (b) gradient descent for logistic regression is guaranteed to find a separator in finite steps; (c) deep nets can have more params than samples and still generalize; (d) conv layers have fewer params than equivalent FC layers.
#ans[(b) is FALSE. Logistic-regression GD has *no* finite-step separation guarantee (loss minimized only as $||w|| -> infinity$).]

*Q20 (Linear autoencoder).* A single-hidden-layer autoencoder with *linear* activations and squared-error loss learns:
(a) the same subspace as PCA; (b) a strictly better code than PCA; (c) a random projection; (d) nothing useful.
#ans[(a). Its optimum spans the top principal subspace — equivalent to PCA.]

]

#pagebreak()

= 📄 Past-Exam MCQs — 2024-25 & 2025-26 (with answers & solutions)
_All 32 Part-1 multiple-choice questions from the last four finals, with the correct option and a worked reason. Cover the ✓ Ans line and try each first._

#columns(2, gutter: 12pt)[

== 2024-25 · 1st sitting

*Q1.* Temperature softmax $"temp-softmax"_T (z) = "softmax"(z\/T)$, $T > 0$. Which statement is *false*?
- [ ] a. For all $z$, there is no finite $T$ for which $"temp-softmax"_T (z)$ is uniform.
- [ ] b. For any $z$, as $T -> infinity$, it approaches a uniform distribution.
- [ ] c. For $z = (4,3,2,1)$, as $T -> 0$, it approaches $(1,0,0,0)$.
- [ ] d. For any $z$ and any $T > 0$, it has no zero components.
#ans[*(a)*. False: if all logits are equal, $z=(a,a,a,a)$, then $"temp-softmax"_T (z) = (1\/4,1\/4,1\/4,1\/4)$ for every finite $T$ — a uniform distribution is attainable at finite $T$, contradicting (a).]

*Q2.* Linearly separable data, labels in ${-1,+1}$, logistic regression $P(y=+1|x)=1\/(1+exp{-w^top x})$, cross-entropy loss $L(w)=sum_(y_i=1) log(1+exp{-w^top x_i}) + sum_(y_i=-1) log(1+exp{w^top x_i})$. The loss has
- [ ] a. one global minimum.
- [ ] b. no global minima.
- [ ] c. several global minima.
- [ ] d. several local minima.
#ans[*(b)*. $L(w) > 0$ everywhere (each $log(1+exp(dot)) > 0$), but along a separating direction $L(alpha w^*)$ decreases strictly with $lim_(alpha->infinity) L(alpha w^*)=0$. The infimum $0$ is never attained, so there is no global minimum.]

*Q3.* Logistic regression, $y in {0,1}$, loss $L(w)=-y_i log P - (1-y_i) log(1-P) + (lambda\/2)||w||_infinity^2$, with $||s||_infinity = max{|s_1|,dots.h,|s_D|}$. The SGD update is
- [ ] a. $w <- w - eta( (P-y)x + lambda w dot.o e_(op("argmax")(|w_1|,dots.h,|w_D|)) )$
- [ ] b. $w <- w - eta( (P-y)x + lambda w dot.o e_(op("argmax")(|x_1|,dots.h,|x_D|)) )$
- [ ] c. $w <- w - eta( (P-y)x + lambda e_(op("argmax")(|w_1|,dots.h,|w_D|)) )$
- [ ] d. $w <- w - eta( (P-y)x + lambda e_(op("argmax")(|x_1|,dots.h,|x_D|)) )$
#ans[*(a)*. CE gradient is $(P-y)x$. Since $||w||_infinity^2 = max_i w_i^2$, only the argmax coordinate has nonzero derivative $2 w_i$, so $partial/(partial w) ||w||_infinity^2 = 2 w dot.o e_(op("argmax")(|w_1|,dots.h,|w_D|))$. Combining gives (a).]

*Q4.* Conv layer, $3 times 5$ filter, stride $1$ in dim 1, stride $3$ in dim 2, no padding, input $128 times 128$. Output size?
- [ ] a. $126 times 124$
- [ ] b. $126 times 42$
- [ ] c. $42 times 126$
- [ ] d. $42 times 42$
#ans[*(b)*. Use $(N-F)\/S + 1$ per dimension. Rows: $(128-3)\/1 + 1 = 126$. Cols: $(128-5)\/3 + 1 = 42$.]

*Q5.* Which is most accurate about the latent space of a dense auto-encoder?
- [ ] a. High-dimensional space, usually more dimensions than the input.
- [ ] b. A space making the input linearly separable for classification.
- [ ] c. A lower-dimensional space capturing the most salient features (compressed representation).
- [ ] d. A space where data is perturbed with noise for robustness.
#ans[*(c)*. The latent space is the bottleneck: a lower-dimensional representation capturing the most salient features, enabling compression.]

*Q6.* RNN: $h_0=0$, $z_t = w h_(t-1) + a x_t$, $h_t = g(z_t)$, $hat(y)=h_T$, squared loss on $x=(x_1,x_2)$. SGD update for $w$?
- [ ] a. $w <- w - eta (h_2-y) g'(z_2)(x_2 + w g'(z_1) x_1)$
- [ ] b. $w <- w - eta (h_2-y) g'(z_2) h_1$
- [ ] c. $w <- w - eta (h_2-y) g'(z_2) g'(z_1)$
- [ ] d. $w <- w - eta (h_2-y) g'(z_2) w$
#ans[*(b)*. $hat(y)=g(w g(a x_1)+a x_2)$. Chain rule in $w$: $(h_2-y) dot g'(z_2) dot h_1$, since $partial z_2 / partial w = h_1 = g(a x_1)$ (treating $h_1$ as the incoming state).]

*Q7.* Which statement is true?
- [ ] a. Training forward pass parallelizable in an encoder-decoder RNN.
- [ ] b. Test forward pass parallelizable in an encoder-decoder RNN.
- [ ] c. Training forward pass parallelizable in a decoder transformer.
- [ ] d. Test forward pass parallelizable in a decoder transformer.
#ans[*(c)*. With teacher forcing + causal masking, all positions of a transformer decoder are computed in parallel during training. RNNs are inherently sequential, and at test time any decoder must generate autoregressively.]

*Q8.* Which is *not* a typical characteristic of multi-head attention?
- [ ] a. Multiple sets of query, key, value projections to attend to different aspects.
- [ ] b. Scales scores by $sqrt(d_k)$ to stabilize training.
- [ ] c. Computes a single weighted average of the value vectors.
- [ ] d. Concatenates the heads, followed by a linear transformation.
#ans[*(c)*. Multi-head attention computes several weighted averages (one per head) with different Q/K/V projections and concatenates them — not a single average. The others are all standard.]

== 2024-25 · 2nd sitting

*Q1.* Graph $y = log a + log b - x$, $a = 1\/(1+e^(-x))$, $b = x^2$. What is $(d y)/(d x)$?
- [ ] a. $-1$
- [ ] b. $x^2 - a$
- [ ] c. $a - x^2$
- [ ] d. $x^2 - a - 1$
#ans[*(b)*. Total derivative: $-1 + (1\/a) dot a(1-a) + (1\/b) dot 2x = -1 + (1-a) + 2\/x = -a + 2\/x$, which matches the intended answer (b).]

*Q2.* $s=[0.1+a, 0.2+a, 0.3+a, 0.4+a]$, $p="softmax"(s)$. Then $p_1$
- [ ] a. is strictly increasing in $a$.
- [ ] b. is strictly decreasing in $a$.
- [ ] c. is non-monotonic (non-constant) in $a$.
- [ ] d. does not depend on $a$.
#ans[*(d)*. Softmax is invariant to adding a constant $a$ to all logits: the shared $e^a$ cancels, so every $p_i$ is unchanged.]

*Q3.* Main purpose of a dropout layer?
- [ ] a. Reduce computational cost of training.
- [ ] b. Increase model capacity.
- [ ] c. Prevent overfitting by randomly deactivating neurons during training.
- [ ] d. Allow faster convergence.
#ans[*(c)*. Dropout randomly deactivates units during training, breaking co-adaptation and acting as regularization against overfitting.]

*Q4.* Input $128 times 128 times 3$; Conv: 64 filters, $5 times 5$, stride 2, no padding; then MaxPool $2 times 2$, stride 2. Output dims?
- [ ] a. $31 times 31 times 64$
- [ ] b. $32 times 32 times 64$
- [ ] c. $62 times 62 times 64$
- [ ] d. $64 times 64 times 64$
#ans[*(a)*. Conv: $floor((128-5)\/2)+1 = 62 -> 62 times 62 times 64$. Pool: $floor((62-2)\/2)+1 = 31 -> 31 times 31 times 64$.]

*Q5.* CNN: $2 times 2$ filter $W$, no padding, stride 1, then average pooling, then sigmoid; BCE loss, lr $eta$. SGD update for $w_(12)$?
- [ ] a. $w_(12) <- w_(12) - (1)/(N^2)(P-y)(sum_(i=1)^(N-1) sum_(j=2)^N x_(i j))$
- [ ] b. $w_(12) <- w_(12) - (eta)/((N-1)^2)(y-P)(sum_(i=1)^(N-1) sum_(j=2)^N x_(i j))$
- [ ] c. $w_(12) <- w_(12) - (eta)/(N^2)(P-y)(sum_(i=1)^N sum_(j=1)^N x_(i j))$
- [ ] d. $w_(12) <- w_(12) - (eta)/((N-1)^2)(P-y)(sum_(i=1)^(N-1) sum_(j=2)^N x_(i j))$
#ans[*(d)*. $partial L/partial z = (P-y)$. Average pooling of the $(N-1) times (N-1)$ map gives $partial z/partial w_(12) = (1)/((N-1)^2) sum a_(i j) $ derivatives, and $partial a_(i j)/partial w_(12)=x_(i,j+1)$, so the sum runs over $j=2..N$. Combining yields (d).]

*Q6.* Which statement about autoregressive RNN language models is true?
- [ ] a. RNNs make a Markov assumption; only remember the most recent $n$ words.
- [ ] b. RNNs handle variable-length sequences via a hidden state capturing past elements.
- [ ] c. RNNs have unbounded memory; remember initial words as accurately as recent ones.
- [ ] d. Being non-feedforward, RNNs cannot be learned by gradient descent.
#ans[*(b)*. RNNs carry a recurrent hidden state summarizing the past, allowing variable-length sequences. They are not exactly Markov, memory is limited in practice (not perfect), and they train via backpropagation through time.]

*Q7.* Primary purpose of the scaling factor in self-attention?
- [ ] a. Keep dot products of queries/keys from growing too large, avoiding unstable gradients.
- [ ] b. Normalize embeddings to zero mean, unit variance.
- [ ] c. Introduce non-linearity.
- [ ] d. Reduce computational complexity.
#ans[*(a)*. Dividing by $sqrt(d_k)$ keeps the dot-product magnitudes bounded, preventing saturated softmax and unstable/vanishing gradients.]

*Q8.* 10 transformer blocks, each with 4 heads; Q/K/V map $bb(R)^8 -> bb(R)^4$, output projection maps to $bb(R)^8$. Total trainable parameters?
- [ ] a. 2880
- [ ] b. 3360
- [ ] c. 4096
- [ ] d. None of the above.
#ans[*(d)*. Per block: $4 times 3 times (8 times 4) + (16 times 8) = 384 + 128 = 512$ (output projection maps concatenated $bb(R)^16 -> bb(R)^8$). Total $= 10 times 512 = 5120$, not listed.]

== 2025-26 · 1st sitting

*Q1.* A perceptron (unit-norm inputs) made 100 mistakes over 1,000,000 examples. About the margin?
- [ ] a. If separable, the margin must be smaller than 0.1.
- [ ] b. If separable, the margin must be larger than 0.1.
- [ ] c. The data is not linearly separable.
- [ ] d. We cannot conclude any of the above.
#ans[*(a)*. Perceptron mistake bound is $R^2\/gamma^2$ with $R=1$ (unit norm). Having made 100 mistakes forces $1\/gamma^2 >= 100$, i.e. $gamma <= 0.1$.]

*Q2.* In BPE, why merge the most frequent adjacent pair into a new subword?
- [ ] a. To reduce training epochs to convergence.
- [ ] b. To handle arbitrarily large vocabularies by representing rare words as sequences of frequent subwords.
- [ ] c. To ensure embeddings have a fixed dimension.
- [ ] d. To eliminate the need for positional encodings.
#ans[*(b)*. BPE builds a subword vocabulary so rare/unseen words are decomposed into frequent subword units, giving open-vocabulary coverage.]

*Q3.* How does standard self-attention scale with sequence length $L$?
- [ ] a. Linear, $O(L)$.
- [ ] b. Quadratic, $O(L^2)$.
- [ ] c. Log-linear, $O(L log L)$.
- [ ] d. Constant, $O(1)$.
#ans[*(b)*. All $L times L$ pairwise query-key interactions are computed, giving $O(L^2)$.]

*Q4.* Primary advantage of contextualized embeddings (BERT) over static ones (word2vec/GloVe)?
- [ ] a. Fewer dimensions for the same semantic information.
- [ ] b. A representation per word based on surrounding context, handling polysemy.
- [ ] c. Faster since they need no transformer.
- [ ] d. Allow small fixed vocabularies without subword tokenization.
#ans[*(b)*. Contextual embeddings depend on the surrounding tokens, so the same word gets different vectors in different senses (handles polysemy).]

*Q5.* $a = x^2 + tanh(x)$, $b = x - e^(-2x) - 1$, $y = a b + x$. Derivative $(d y)/(d x)$?
- [ ] a. $1$
- [ ] b. $b(2x + 1 - tanh^2(x)) + a(1 + 2 e^(-2x)) + 1$
- [ ] c. $b(2x + 1 - tanh^2(x)) + a(1 + 2 e^(-2x))$
- [ ] d. Cannot be computed; the graph is not a DAG.
#ans[*(b)*. $partial y/partial a = b$, $partial y/partial b = a$, plus the direct $+x$ term. With $a'=2x+1-tanh^2(x)$ and $b'=1+2e^(-2x)$: $y' = b(2x+1-tanh^2(x)) + a(1+2e^(-2x)) + 1$.]

*Q6.* $z=[1,2,3,4]^top$, scalars $alpha != 0$, $beta$. Most probable label under $"softmax"(alpha z + beta bold(1))$?
- [ ] a. Always the last label.
- [ ] b. Either the first or last label, depending on $alpha$.
- [ ] c. Any label, depending on $alpha$ and $beta$.
- [ ] d. Any label, but $beta$ does not influence the result.
#ans[*(b)*. $beta$ adds a constant (irrelevant to softmax). $alpha > 0$ keeps the ordering so the max entry (last) wins; $alpha < 0$ flips it so the min entry (first) wins. Hence first or last, per sign of $alpha$.]

*Q7.* MLP: $D=10$ input, one hidden layer $K=5$, output $C=2$, fully connected with biases. Total parameters?
- [ ] a. $60$
- [ ] b. $62$
- [ ] c. $67$
- [ ] d. $72$
#ans[*(c)*. Input$->$hidden: $10 times 5 + 5 = 55$. Hidden$->$output: $5 times 2 + 2 = 12$. Total $= 67$.]

*Q8.* RNN with all-zero recurrence matrix. Which statement is *false*? (If first three all false, pick last.)
- [ ] a. For sequence tagging, such RNN is permutation equivariant.
- [ ] b. As an autoregressive LM, it is first-order Markov, $p(y_t|y_(<t))=p(y_t|y_(t-1))$.
- [ ] c. Hidden states are necessarily all-zeros.
- [ ] d. All statements are false.
#ans[*(c)*. With zero recurrence the net becomes an element-wise feedforward map (so a and b hold), but the input contribution still produces nonzero hidden states — so c is false.]

== 2025-26 · 2nd sitting

*Q1.* Self-attention, $L=10$, $d_k=64$. Dimensions of $Q K^top$ (before softmax)?
- [ ] a. $10 times 64$
- [ ] b. $64 times 64$
- [ ] c. $10 times 10$
- [ ] d. $64 times 10$
#ans[*(c)*. $Q$ is $L times d_k = 10 times 64$ and $K^top$ is $64 times 10$, so $Q K^top$ is $L times L = 10 times 10$.]

*Q2.* Which statement is true?
- [ ] a. Teacher forcing is used to mitigate exposure bias.
- [ ] b. Local representations are generally more efficient than distributed representations.
- [ ] c. The first two claims are false.
- [ ] d. The first two claims are true.
#ans[*(c)*. Teacher forcing *causes* exposure bias, and distributed representations are *more* efficient than local ones (encode exponentially more configurations). Both (a) and (b) are false.]

*Q3.* Primary role of a pooling layer in a CNN?
- [ ] a. Perform convolution with learnable filters.
- [ ] b. Increase the spatial dimensions.
- [ ] c. Introduce non-linearity.
- [ ] d. Reduce spatial resolution (downsampling) and provide translation invariance.
#ans[*(d)*. Pooling downsamples the feature map and adds translation invariance; it has no learnable filters and adds no non-linearity.]

*Q4.* ResNet skip connection $H(x)=F(x)+x$. Primary motivation?
- [ ] a. Reduce the number of parameters.
- [ ] b. Mitigate the vanishing gradient problem in very deep networks.
- [ ] c. Perform downsampling without pooling.
- [ ] d. Interpret the network as an ensemble of shallow networks.
#ans[*(b)*. The identity path lets gradients flow directly, mitigating vanishing gradients and enabling training of very deep networks.]

*Q5.* From a regularization view, why randomly drop units during training?
- [ ] a. Increase training speed by lightening each forward pass.
- [ ] b. Prevent units from co-adapting too strongly, forcing robust features useful in many contexts.
- [ ] c. Handle missing/corrupted input features at inference.
- [ ] d. Simulate biological neuron regeneration to increase capacity.
#ans[*(b)*. Dropout breaks co-adaptation so each unit must be useful with random subsets of others, yielding more robust features (also an implicit ensemble).]

*Q6.* Trade-off in choosing the SGD mini-batch size?
- [ ] a. Smaller batches allow faster matrix-matrix GPU computations than larger ones.
- [ ] b. Increasing batch size introduces bias, preventing convergence.
- [ ] c. Larger batches reduce gradient variance and allow GPU speed-ups, but need more memory.
- [ ] d. Batch size only affects memory, not gradient noise or stability.
#ans[*(c)*. Larger batches average over more samples (lower-variance gradient) and exploit larger GPU matrix ops, at higher memory cost. They add no bias, and batch size does affect noise/stability.]

*Q7.* Corpus: "ban" 6, "can" 4, "cans" 3, "banned" 2. After first BPE merge "a"+"n"$->$"an", frequency of pair "(b, an)" and is it next?
- [ ] a. Freq 8. Yes, highest among remaining pairs.
- [ ] b. Freq 6. No, "(c, an)" is higher.
- [ ] c. Freq 15. No, will merge "an" with "s".
- [ ] d. Freq 8. No, tie with "(c, an)", chosen randomly.
#ans[*(a)*. "(b, an)" occurs in "ban" (6) and "banned" (2) $=> 8$. "(c, an)" occurs in "can" (4) and "cans" (3) $=> 7$. Since $8 > 7$, "(b, an)" is the next merge.]

*Q8.* BERT masked language modelling vs GPT-style LM?
- [ ] a. BERT predicts the next word from only previous words (left-to-right).
- [ ] b. BERT predicts masked words using both left and right context (bidirectional).
- [ ] c. BERT translates via an encoder-decoder.
- [ ] d. BERT predicts a label (e.g. sentiment) for the whole sentence.
#ans[*(b)*. Masked LM predicts masked tokens from bidirectional context, whereas GPT predicts the next token from left context only.]

]

#pagebreak()

= 🧩 Worked Part-3 Problems — last 4 exams (full solutions)
_The 25-point problems from 2024-25 & 2025-26 (both sittings), worked end-to-end. Problem 1 = CNN, Problem 2 = sequence/attention._
== 2024-25 · 1st sitting — Problem 1 (CNN, SVHN)

Grayscale input $x$ is $32 times 32 times 1$. Architecture: $x ->$ Conv-1 $->$ "ReLU" $->$ AvgPool-1 $-> (z_1) ->$ Conv-2 $->$ "ReLU" $->$ AvgPool-2 $-> (z_2) ->$ Flatten $->$ Linear $-> (z_3) ->$ "softmax" $-> hat(y)$.

- *Conv-1*: 8 filters, kernel $5 times 5$, stride 1, padding 0.
- *AvgPool-1*: kernel $4 times 4$, stride 4, padding 0.
- *Conv-2*: 16 filters, kernel $4 times 4$, stride 1, padding 1.
- *AvgPool-2*: kernel $2 times 2$, stride 2, padding 0.
- *Linear*: flattened activations $-> 10$ logits.

#psol[
*1. Advantage of pooling; max vs. average.* Pooling gives the network invariance, making it more robust to small variations in the input, and increases the receptive field of neurons in deeper layers. Max pooling selects the maximum value for each patch (the most prominent feature), while average pooling computes the average of all values within the patch (smoothing out all the feature values).

*2. Trainable weights per layer.* Formula: num. weights $=$ num. filters $times$ ((kernel width $times$ kernel height $times$ num. channels) $+ 1$ bias).
- *Conv-1*: $8 times ((5 times 5 times 1) + 1) = 8 times 26 = 208$.
- *Conv-2*: $16 times ((4 times 4 times 8) + 1) = 16 times 129 = 2064$.
- *AvgPool-1* and *AvgPool-2*: no trainable weights ($0$).

*3. Fix the PyTorch code.* Track the spatial size with $("input dim" - "kernel size")/"stride" + 1$:
- *Conv-1* output: $28 times 28 times 8$, since $(32 - 5)/1 + 1 = 28$ (8 = number of filters).
- *AvgPool-1* output: $7 times 7 times 8$, since $(28 - 4)/4 + 1 = 7$.
- *Conv-2* output: $6 times 6 times 16$. Padding of 1 makes the input $9 times 9 times 8$; then $(9 - 4)/1 + 1 = 6$ (16 = number of filters).
- *AvgPool-2* output: $3 times 3 times 16$, since $(6 - 2)/2 + 1 = 3$.

After flattening the vector has dimension $3 times 3 times 16 = 144$. Hence the linear layer must be `nn.Linear(144, 10)` (not `120`). Also, the two `nn.AvgPool2d(...)` lines in the original code are missing the trailing commas (a syntax error); those commas must be added. Corrected block:

```python
nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5,
               stride=1, padding=0),
    nn.AvgPool2d(kernel_size=4, stride=4, padding=0),
    nn.ReLU(),
    nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4,
               stride=1, padding=1),
    nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(144, 10)
)
```

*4. Replace Flatten + Linear with a convolution.* The output of AvgPool-2 is $3 times 3 times 16$, so the convolution's input has 16 channels.
- *v1:* Apply a kernel of size $3 times 3$ with 10 output channels (filters). This yields output $1 times 1 times 10$, directly giving $z_3 in bb(R)^10$.
- *v2:* Apply a $1 times 1$ kernel with 10 output channels, giving $3 times 3 times 10$; then apply a third pooling layer AvgPool-3 of size $3 times 3$ to reduce it to $1 times 1 times 10$.
]

== 2024-25 · 1st sitting — Problem 2 (Transformer: dedup + sort)

Decoder-only, single self-attention layer, single head, no feedforward/residuals. Parameters:
$ W_Q = W_K = W_V = mat(0, 3; 1, 2; 2, 1; 3, 0), quad W_O = mat(0, 1, 0, 1; 1, 0, 1, 0). $
Vocabulary ${"A", "B", "C", "D", chevron.l "start" chevron.r, chevron.l "stop" chevron.r}$, embedding matrix (rows A, B, C, D, start, stop):
$ E = mat(1,0,0,0; 0,1,0,0; 0,0,1,0; 0,0,0,1; 0,0,0,0; 0,0,0,0). $
Task: given a sequence of symbols, output the deduplicated symbols sorted lexicographically (e.g. `D A A` $->$ `A D`).

#psol[
*1. No positional encodings — wise?* Yes. The output is invariant to the order of the inputs — e.g. `D A A`, `A D A`, `A A D` should all map to `A D`. Without positional encodings the transformer treats the input as a "bag of symbols", ignoring sequential structure, which is a good inductive bias for this problem.

*2. Attention matrix for* `D A A` $chevron.l "start" chevron.r$ `A D` $chevron.l "stop" chevron.r$*.* The input embeddings (rows D, A, A, start, A, D, stop) are
$ X = mat(0,0,0,1; 1,0,0,0; 1,0,0,0; 0,0,0,0; 1,0,0,0; 0,0,0,1; 0,0,0,0). $
Then
$ Q = K = V = X W_Q = mat(3,0; 0,3; 0,3; 0,0; 0,3; 3,0; 0,0), quad Q K^top = mat(9,0,0,0,0,9,0; 0,9,9,0,9,0,0; 0,9,9,0,9,0,0; 0,0,0,0,0,0,0; 0,9,9,0,9,0,0; 9,0,0,0,0,9,0; 0,0,0,0,0,0,0). $
Apply causal masking (lower-triangular) and then the zero-temperature softmax (argmax, ties split uniformly) row by row. In each row only the allowed positions with the maximal score receive equal weight; row 4 (start) has all-zero scores, so its four allowed positions split evenly, and likewise row 7 which is all zeros. This gives
$ P = mat(1,0,0,0,0,0,0; 0,1,0,0,0,0,0; 0,1/2,1/2,0,0,0,0; 1/4,1/4,1/4,1/4,0,0,0; 0,1/3,1/3,0,1/3,0,0; 1/2,0,0,0,0,1/2,0; 1/7,1/7,1/7,1/7,1/7,1/7,1/7). $

*3. Cross-entropy loss (loss only on outputs* `A D` $chevron.l "stop" chevron.r$*).* Using the given $P$, the self-attention output is
$ Z = P V = mat(3,0; 0,3; 0,3; 3/4,3/2; 0,3; 3,0; 6/7,9/7). $
Multiplying by $W_O$:
$ R = Z W_O = mat(0,3,0,3; 3,0,3,0; 3,0,3,0; 3/2,3/4,3/2,3/4; 3,0,3,0; 0,3,0,3; 9/7,6/7,9/7,6/7). $
Only the bottom three rows (positions 5, 6, 7 — predicting `A`, `D`, $chevron.l "stop" chevron.r$) matter. Multiplying by $E^top$ gives the 6-dim logits:
$ z_5^top = [3,0,3,0] E^top = [3,0,3,0,0,0], $
$ z_6^top = [0,3,0,3] E^top = [0,3,0,3,0,0], $
$ z_7^top = [9/7,6/7,9/7,6/7] E^top = [9/7,6/7,9/7,6/7,0,0]. $
Relevant softmax probabilities:
$ P(y_5 = "A") = exp(3)/(2 exp(3) + 4) = 0.4547, $
$ P(y_6 = "D") = exp(3)/(2 exp(3) + 4) = 0.4547, $
$ P(y_7 = chevron.l "stop" chevron.r) = 1/(2 exp(9/7) + 2 exp(6/7) + 2) = 0.0717. $
Loss:
$ L = -log P(y_5 = "A") - log P(y_6 = "D") - log P(y_7 = chevron.l "stop" chevron.r) = 4.2114. $

*4. Second identical head, $W_O = I_(4 times 4)$.* Adding an identical second head concatenates two copies of $Z$; with $W_O$ the $4 times 4$ identity, $[Z | Z]$ times the identity reproduces the same $R$ as before. Hence the logits and loss are identical: $L = 4.2114$.

*5. Model not learning; $W_Q, W_K, W_V$ never change.* The argmax ("zero temperature") attention has zero gradient with respect to the scores, so no gradient flows back into $W_Q, W_K, W_V$ and they are never updated. The problem is fixed by switching to the usual softmax attention with temperature $T = sqrt(d)$, since softmax is differentiable and produces non-zero gradients.
]

== 2024-25 · 2nd sitting — Problem 1 (Inception blocks)

Input $128 times 28 times 28$ (128 channels, spatial $28 times 28$). Blocks:
- $B_1$: *C1* $1 times 1$, 32 filters, stride 1.
- $B_2$: *C2* $1 times 1$, 24 filters $->$ *C3* $3 times 3$, 32 filters, stride 1.
- $B_3$: *C4* $1 times 1$, 8 filters $->$ *C5* $5 times 5$, 16 filters, stride 1.
- $B_4$: *M1* $3 times 3$ max pool, stride 1 $->$ *C6* $1 times 1$, 16 filters, stride 1.

Filter concatenation joins all block outputs along the channel dimension (no parameters).

#psol[
*1. Total trainable parameters.* Parameters $= (K_h times K_w times C_"in" times C_"out") + C_"out"$ (one bias per filter).
- *C1* ($1 times 1$, 32, in 128): $(1 dot 1 dot 128 dot 32) + 32 = 4096 + 32 = 4128$.
- *C2* ($1 times 1$, 24, in 128): $(128 dot 24) + 24 = 3072 + 24 = 3096$.
- *C3* ($3 times 3$, 32, in 24): $(9 dot 24 dot 32) + 32 = 6912 + 32 = 6944$.
- *C4* ($1 times 1$, 8, in 128): $(128 dot 8) + 8 = 1024 + 8 = 1032$.
- *C5* ($5 times 5$, 16, in 8): $(25 dot 8 dot 16) + 16 = 3200 + 16 = 3216$.
- *M1*: max pooling, no parameters.
- *C6* ($1 times 1$, 16, in 128): $(128 dot 16) + 16 = 2048 + 16 = 2064$.

*Total* $= 4128 + 3096 + 6944 + 1032 + 3216 + 2064 = 20480$.

*2. Missing padding values.* All concatenated outputs must share spatial size. Output-size formula $W_"out" = (W_"in" - K + 2P)/S + 1$. For $B_1$ (C1, $1 times 1$, stride 1, no padding): $W_"out" = (28 - 1 + 0)/1 + 1 = 28$, so every block must preserve $28 times 28$. Solving for $P$:
- *C3* ($3 times 3$, stride 1): $28 = (28 - 3 + 2P)/1 + 1 => 28 = 26 + 2P => P = 1$.
- *C5* ($5 times 5$, stride 1): $28 = (28 - 5 + 2P)/1 + 1 => 28 = 24 + 2P => P = 2$.
- *M1* ($3 times 3$ max pool, stride 1): $28 = (28 - 3 + 2P)/1 + 1 => 28 = 26 + 2P => P = 1$.

(C6 is $1 times 1$ stride 1, so it already preserves $28$ with no padding.)

#table(
  columns: 2,
  align: center,
  table.header([*layer*], [*padding*]),
  [C3], [1],
  [C5], [2],
  [M1], [1],
)

*3. Factorized convolutions.* Only the factorized conv layers change; the $1 times 1$ layers (C1, C2, C4, C6) and M1 are unchanged.
- *C3.1* ($3 times 1$, 28, in 24): $(3 dot 1 dot 24 dot 28) + 28 = 2016 + 28 = 2044$.
- *C3.2* ($1 times 3$, 32, in 28): $(1 dot 3 dot 28 dot 32) + 32 = 2688 + 32 = 2720$.
- *C5.1* ($5 times 1$, 12, in 8): $(5 dot 1 dot 8 dot 12) + 12 = 480 + 12 = 492$.
- *C5.2* ($1 times 5$, 16, in 12): $(1 dot 5 dot 12 dot 16) + 16 = 960 + 16 = 976$.

Unchanged: C1 $= 4128$, C2 $= 3096$, C4 $= 1032$, C6 $= 2064$.

*New total* $= 4128 + 3096 + 2044 + 2720 + 1032 + 492 + 976 + 2064 = 16552$ (about 20% fewer than the 20480 of the original block). Factorized convolutions give fewer parameters and fewer operations, reducing computational cost without compromising expressive power.
]

== 2024-25 · 2nd sitting — Problem 2 (Transformer computes determinants)

Input: entries of $A$ fed row by row. Each element $x_t$ with $t = n(i-1) + j$ (entry $(i,j)$) is $x_t = [a_(i j), e_i + e_j]^top in bb(R)^(n+1)$. Sequence encoded as $X in bb(R)^(n^2 times (n+1))$; single self-attention layer gives $Z$; sum-pooling gives $h in bb(R)^(n+1)$; then $hat(y) = w^top h + b$, aiming for $hat(y) approx det(A)$.

#psol[
*1. Invariance to transpose.* The positional encoding sums the one-hot row and column vectors: $e_i + e_j$. This sum is unchanged if rows and columns are switched (transposing $A$ maps entry $(i,j)$ to $(j,i)$ but $e_i + e_j = e_j + e_i$). Hence $Z$ is equivariant to transposing $X$, and since we then apply sum-pooling, the final output $hat(y)$ is invariant to the transpose. This is a good design decision because $det(A) = det(A^top)$.

*2. Attention matrix $P$ and output $Z$ for $n = 2$.* Projection matrices:
$ W_Q = W_K = W_V = mat(1, 0; 0, 1; -1, -1), quad W_O = mat(1, 0, 0; 0, 1, 0). $
With $A = mat(0, 1; 2, 3)$ read row-by-row and augmented with $e_i + e_j$:
$ X = mat(0,2,0; 1,1,1; 2,1,1; 3,0,2), quad Q = K = V = X W_Q = mat(0,2; 0,0; 1,0; 1,-2). $
Then
$ P = "softmax"((Q K^top)/sqrt(2)) = "softmax"(1/sqrt(2) mat(4,0,0,-4; 0,0,0,0; 0,0,1,1; -4,0,1,5)) = mat(0.892, 0.053, 0.053, 0.003; 0.25, 0.25, 0.25, 0.25; 0.165, 0.165, 0.335, 0.335; 0.002, 0.027, 0.054, 0.917). $
And
$ Z = P V W_O = mat(0.892, 0.053, 0.053, 0.003; 0.25, 0.25, 0.25, 0.25; 0.165, 0.165, 0.335, 0.335; 0.002, 0.027, 0.054, 0.917) mat(0,2; 0,0; 1,0; 1,-2) mat(1,0,0; 0,1,0) = mat(0.056, 1.777, 0; 0.5, 0, 0; 0.670, -0.340, 0; 0.972, -1.832, 0). $

*3. Prediction $hat(y)$ and squared loss.* $det(A) = 0 times 3 - 1 times 2 = -2$. (There was a typo: it should be $w = [-1, 0, 1]^top$ with pooling over the rows of $Z$; column-pooling with $w = [0, -1, 0, 1]^top$ was also accepted.)

*Intended — pooling over rows of $Z$ with $w = [-1, 0, 1]^top$:*
$ h^top = bold(1)^top Z = [1,1,1,1] mat(0.056, 1.777, 0; 0.5, 0, 0; 0.670, -0.340, 0; 0.972, -1.832, 0) = [2.197, -0.395, 0]. $
$ hat(y) = w^top h + b = [-1, 0, 1] vec(2.197, -0.395, 0) = -2.197. $
$ L = 1/2 (hat(y) - det(A))^2 = 0.5 times (-2.197 + 2)^2 = 0.019. $

*Also accepted — pooling over columns of $Z$ with $w = [0, -1, 0, 1]^top$:*
$ h = Z bold(1) = mat(0.056, 1.777, 0; 0.5, 0, 0; 0.670, -0.340, 0; 0.972, -1.832, 0) vec(1, 1, 1) = vec(1.833, 0.5, 0.330, -0.860). $
$ hat(y) = w^top h + b = [0, -1, 0, 1] vec(1.833, 0.5, 0.330, -0.860) = -1.360. $
$ L = 0.5 times (-1.360 + 2)^2 = 0.205. $
]

== 2025-26 · 1st sitting — Problem 1 (CNN for scene attribute classification)

Input $x in bb(R)^(32 times 32 times 3)$, output one of five classes via softmax + cross-entropy. Architecture:
- Conv-1: 16 filters, $3 times 3$, stride 1, pad 1, ReLU
- MaxPool-1: $2 times 2$, stride 2
- Conv-2: 32 filters, $3 times 3$, stride 1, pad 1, ReLU
- MaxPool-2: $2 times 2$, stride 2
- Residual block (main branch): Conv-3 ($32, 3 times 3$, s1, p1) $->$ ReLU $->$ Conv-4 ($32, 3 times 3$, s1, p1); skip = identity; merge $Z = Z_4 + Z_2$
- Flatten $->$ Linear to 5 logits

#psol[
*1. Total trainable parameters.* Conv parameter count is $"filters" times (k_h dot k_w dot "in_channels") + "filters"$ (bias).
- Conv-1: $16(3 dot 3 dot 3) + 16 = 448$.
- Conv-2: $32(3 dot 3 dot 16) + 32 = 4640$.
- Conv-3: $32(3 dot 3 dot 32) + 32 = 9248$.
- Conv-4: same as Conv-3 $= 9248$.
- Spatial size: input $32 times 32$, two $2 times 2$ max-pools (stride 2) give $32 -> 16 -> 8$, so an $8 times 8$ grid with 32 channels. Flatten size $= 8 dot 8 dot 32 = 2048$.
- Linear: $5 dot 2048 + 5 = 10245$.

Total: $448 + 4640 + 9248 + 9248 + 10245 = bold(33829)$.

*2. Replacing the head with a CBAPM block + GAP + linear.* The CBAPM operations are

$ macron(Z)_(i j) = 1/32 sum_(c=1)^(32) Z_(i j c), quad A_(i j) = sigma(macron(Z)_(i j)), quad H_(i j c) = A_(i j) Z_(i j c), quad s_c = 1/64 sum_(i=1)^8 sum_(j=1)^8 H_(i j c), quad hat(y) = W s + b. $

The gating (mean + sigmoid + elementwise multiply) and global average pooling have *no trainable parameters*. Only the final linear layer counts.
- Old head (flatten + linear): $5 dot 2048 + 5 = 10245$.
- New head (GAP $-> s in bb(R)^(32)$ + linear): $5 dot 32 + 5 = 165$.

The head shrinks by $10245 - 165 = 10080$ parameters. The convolutional backbone is $33829 - 10245 = 23584$, so the new network has

$ 23584 + 165 = bold(23749) quad "parameters." $

*3. GAP vs flatten.* Flatten preserves the full $8 times 8$ spatial grid as a 2048-D vector feeding a fully connected layer, so it needs many parameters and ties the model to a fixed input size and absolute spatial position. Global average pooling collapses each channel to one scalar (a 32-D vector here), drastically reducing head parameters, and makes the model more robust to translations and to input-size changes (it always outputs a length-$C$ vector).

*4. Bug in the forward pass.* The offending line is the "spatial gating" mean:

```python
m = Z.mean(dim=(2, 3))     # (B,32) -> this is CHANNEL attention, wrong
```

Averaging over dims $(2,3)$ pools over the spatial axes and yields a per-*channel* vector, i.e. channel attention. For the *spatial* gate of question 2 we must average over the channel axis instead:

```python
m = Z.mean(dim=1)          # (B,8,8)  average over channels
A = torch.sigmoid(m)       # (B,8,8)
H = A[:, None, :, :] * Z   # broadcast to (B,32,8,8)
```
]

== 2025-26 · 1st sitting — Problem 2 (Transformers and reasoning chains)

Decoder-only transformer, single-head causal self-attention, no biases, no positional encoding, no output projection. Sequence: Ultimate, Question, then "wait" repeated $N$ times. Embeddings (one row per vocab token):

$ E = mat(2, 0; 0, 2; 1, 1; 1, 2; 2, 1; -1, 0) mat("# Ultimate"; "# Question"; "# wait"; "# 42"; "# Yes"; "# No") $

$ W_Q = mat(1, 0; 1, 1), quad W_K = mat(1, 1; 0, 1), quad W_V = mat(1, 0; 0, 2), quad W_"vocab" = E^top. $

#psol[
*1. Q, K, V for $N = 3$.* The input rows are Ultimate, Question, wait, wait, wait:

$ X = mat(2, 0; 0, 2; 1, 1; 1, 1; 1, 1). $

$ Q = X W_Q = mat(2, 0; 0, 2; 1, 1; 1, 1; 1, 1) mat(1, 0; 1, 1) = mat(2, 0; 2, 2; 2, 1; 2, 1; 2, 1) $

$ K = X W_K = mat(2, 0; 0, 2; 1, 1; 1, 1; 1, 1) mat(1, 1; 0, 1) = mat(2, 2; 0, 2; 1, 2; 1, 2; 1, 2) $

$ V = X W_V = mat(2, 0; 0, 2; 1, 1; 1, 1; 1, 1) mat(1, 0; 0, 2) = mat(2, 0; 0, 4; 1, 2; 1, 2; 1, 2) $

*2. Attention for the last token ($N = 3$).* Its query is $q = [2, 1]^top$. Scores $= q^top k \/ sqrt(2)$ (here $d_k = 2$):
- "Ultimate": $([2,1][2,2]^top)/sqrt(2) = 6/sqrt(2) approx 4.243$.
- "Question": $([2,1][0,2]^top)/sqrt(2) = 2/sqrt(2) approx 1.414$.
- "wait": $([2,1][1,2]^top)/sqrt(2) = 4/sqrt(2) approx 2.828$ (same for all three occurrences).

Exponentials: $e^(4.243) approx 69.591$, $e^(1.414) approx 4.113$, $e^(2.828) approx 16.918$.

Normalizer: $Z = 69.591 + 4.113 + 3 times 16.918 = 124.461$.

Attention probabilities: $p_"Ultimate" approx 0.559$, $p_"Question" approx 0.033$, $p_"wait" approx 0.136$ (each of the three).

Attention output $c = V^top p$:

$ c = mat(2, 0, 1, 1, 1; 0, 4, 2, 2, 2) vec(0.559, 0.033, 0.136, 0.136, 0.136) = mat(2, 0, 1; 0, 4, 2) vec(0.559, 0.033, 3 times 0.136) = vec(1.526, 0.948). $

*3. Logits and next-token distribution ($N = 3$).* Logits $z = W_"vocab" c = E c$:

$ z = mat(2, 0; 0, 2; 1, 1; 1, 2; 2, 1; -1, 0) vec(1.526, 0.948) = vec(3.052, 1.896, 2.474, 3.422, 4.000, -1.526). $

Exponentials: $e^(3.052) approx 21.162$, $e^(1.896) approx 6.657$, $e^(2.474) approx 11.869$, $e^(3.422) approx 30.622$, $e^(4) approx 54.598$, $e^(-1.526) approx 0.217$.

Normalizer $Z = 21.162 + 6.657 + 11.869 + 30.622 + 54.598 + 0.217 = 125.125$. Probabilities:

$ P("Ultimate") approx 0.169, quad P("Question") approx 0.053, quad P("wait") approx 0.095, $
$ P("42") approx 0.245, quad P("Yes") approx 0.436, quad P("No") approx 0.002. $

Most probable next token: *"Yes"*.

*4. The case $N -> infinity$.* The normalizer $Z = 69.591 + 4.113 + N times 16.918 -> infinity$, so $p_"Ultimate" -> 0$, $p_"Question" -> 0$, and each "wait" gets $p_"wait" -> 1\/N$ — together the $N$ "wait" tokens carry all the mass. The attention output becomes

$ c = mat(2, 0, 1; 0, 4, 2) vec(0, 0, N times 1/N) = vec(1, 2). $

Logits $z = E c$:

$ z = mat(2, 0; 0, 2; 1, 1; 1, 2; 2, 1; -1, 0) vec(1, 2) = vec(2, 4, 3, 5, 4, -1). $

Exponentials: $e^2 approx 7.389$, $e^4 approx 54.598$, $e^3 approx 20.086$, $e^5 approx 148.413$, $e^4 approx 54.598$, $e^(-1) approx 0.368$.

Normalizer $Z = 7.389 + 54.598 + 20.086 + 148.413 + 54.598 + 0.368 = 285.452$. Probabilities:

$ P("Ultimate") approx 0.026, quad P("Question") approx 0.191, quad P("wait") approx 0.070, $
$ P("42") approx 0.520, quad P("Yes") approx 0.191, quad P("No") approx 0.001. $

Most probable next token: *"42"*.
]

== 2025-26 · 2nd sitting — Problem 1 (Adversarial learning and CNNs / GANs)

A convolutional GAN generates $64 times 64$ RGB faces. Discriminator: Conv-1 (16, $3 times 3$, s1, p1) $->$ LeakyReLU $->$ Conv-2 (32, $3 times 3$, s2, p1) $->$ LeakyReLU $->$ Flatten $->$ Linear-1 (128 units) $->$ Linear-2 (2 units). Generator: input $z in bb(R)^(100)$ $->$ Linear-3 (8192 units) $->$ reshape $8 times 8 times 128$ $->$ Conv-3 (ConvTranspose2D, 64, $4 times 4$, s2, p1) $->$ ReLU $->$ Conv-4 (ConvTranspose2D, 3, $4 times 4$, s2, p1) $->$ Tanh.

#psol[
*1. GAN objective and procedure.*
(a) The discriminator $D$ estimates the probability that an input sample is *real* (drawn from the true data distribution) rather than generated: $D(x) = P("real" | x)$. It is trained on both real and generated samples to tell them apart.
(b) The generator $G$ is trained to produce samples that $D$ classifies as real, i.e. to increase $D(G(z))$ for noise $z$ — minimizing $log(1 - D(G(z)))$, or in practice maximizing $log D(G(z))$. $G$ learns via gradients backpropagated through $D$.

*2. Trainable parameters.* Conv: $("filters" times "in_ch" times k_h times k_w) + "bias"$; Linear: $("in" times "out") + "out"$.

*Discriminator:*
- Conv-1: $16 times 3 times 3 times 3 + 16 = 432 + 16 = 448$.
- Conv-2: $32 times 16 times 3 times 3 + 32 = 4608 + 32 = 4640$.
- Conv-2 has stride 2, so $64 times 64 -> 32 times 32$ with 32 channels $= 32 dot 32 dot 32 = 32768$ features. Linear-1: $32768 times 128 + 128 = 4194304 + 128 = 4194432$.
- Linear-2: $128 times 2 + 2 = 256 + 2 = 258$.
- *Total ($D$)* $= 448 + 4640 + 4194432 + 258 = bold(4199778)$.

*Generator:*
- Linear-3: $100 times 8192 + 8192 = 819200 + 8192 = 827392$.
- Conv-3 (transposed): $128 times 64 times 4 times 4 + 64 = 131072 + 64 = 131136$.
- Conv-4 (transposed): $64 times 3 times 4 times 4 + 3 = 3072 + 3 = 3075$.
- *Total ($G$)* $= 827392 + 131136 + 3075 = bold(961603)$.

*3. Receptive field before Flatten.* Use $r_ell = r_(ell-1) + (k-1) j_(ell-1)$, $j_ell = j_(ell-1) s$, with $r_0 = 1$, $j_0 = 1$.
- Layer 1 ($k=3, s=1$): $r_1 = 1 + (3-1) dot 1 = 3$, $j_1 = 1 dot 1 = 1$.
- Layer 2 ($k=3, s=2$): $r_2 = 3 + (3-1) dot 1 = 5$, $j_2 = 1 dot 2 = 2$.

So one activation just before Flatten depends on a $5 times 5$ region of the input, i.e. *25 pixels*. (Intuitively: layer 1 gives a $3$-pixel span; layer 2's $3 times 3$ kernel adds $(3-1) = 2$ more input pixels, $3 + 2 = 5$.)

*4. Debugging the generator training.* Two errors in the generator part:

```python
loss_G = criterion(fake_output, real_labels)  # was: criterion(real_output, real_labels)
loss_G.backward()                             # was: missing
```

The generator loss must compare `fake_output = D(fake_images)` against `real_labels` (so $G$ is pushed to fool $D$ into labelling fakes as real), and `loss_G.backward()` must run before `optimizer_G.step()` so gradients are actually computed.
]

== 2025-26 · 2nd sitting — Problem 2 (Language model for recipe generation)

An RNN with ReLU activation, zero biases, $h_0 = 0$:

$ W_(h x) = mat(1, 0, 0, 0, 0; 0, 1, 0, 0, 0; 0, 0, 1, 0, 0; 0, 0, 0, 1, 0), quad W_(h h) = mat(1, 0, 0, 0; 0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1). $

Word embeddings (one row per word):

$ E = mat(1, -1, 0, 0, 0; 0, 1, -1, 0, 0; 0, 0, 1, -1, 0; 0, 0, 0, 0, 2; 0, 0, 0, 0, -1; 0, 0, 0, 0, 1) mat("# Main"; "# course"; "# is"; "# codfish"; "# pasta"; "# deep_learning"). $

#psol[
*1. Last hidden vector after "Main course is".* Embeddings: $x_"Main" = [1,-1,0,0,0]^top$, $x_"course" = [0,1,-1,0,0]^top$, $x_"is" = [0,0,1,-1,0]^top$. With $h_0 = 0$ and ReLU:

$ z_1 = W_(h x) x_"Main" + W_(h h) h_0 = vec(1, -1, 0, 0), quad h_1 = "ReLU"(z_1) = vec(1, 0, 0, 0). $

$ z_2 = W_(h x) x_"course" + W_(h h) h_1 = vec(0, 1, -1, 0) + vec(1, 0, 0, 0) = vec(1, 1, -1, 0), quad h_2 = "ReLU"(z_2) = vec(1, 1, 0, 0). $

$ z_3 = W_(h x) x_"is" + W_(h h) h_2 = vec(0, 0, 1, -1) + vec(1, 1, 0, 0) = vec(1, 1, 1, -1), quad h_3 = "ReLU"(z_3) = vec(1, 1, 1, 0). $

Last hidden vector: $h_3 = [1, 1, 1, 0]^top$.

*2. Transformer Q, K, V.* Switching to a transformer with the same embeddings,

$ W_Q = W_K = W_V = mat(1, 0, 0, 0; 0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 1, 0, 1), quad W_O = W_Q^top. $

Position encodings $p_i = [0,0,0,0,i]$ for tokens Main, course, is at $i = 0, 1, 2$. Adding them to the embeddings:

$ X = mat(1, -1, 0, 0, 0; 0, 1, -1, 0, 0; 0, 0, 1, -1, 0) + mat(0, 0, 0, 0, 0; 0, 0, 0, 0, 1; 0, 0, 0, 0, 2) = mat(1, -1, 0, 0, 0; 0, 1, -1, 0, 1; 0, 0, 1, -1, 2). $

Since $W_Q = W_K = W_V$:

$ Q = K = V = X W_Q = mat(1, -1, 0, 0, 0; 0, 1, -1, 0, 1; 0, 0, 1, -1, 2) mat(1, 0, 0, 0; 0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 1, 0, 1) = mat(1, -1, 0, 0; 0, 2, -1, 1; 0, 2, 1, 1). $

*3. Normalized-ReLU causal attention.* With

$ Q = K = V = mat(1, -1, 0, 0; 0, 2, -1, 1; 0, 2, 1, 1), quad d_k = 4, quad sqrt(d_k) = 2, $

the masked scaled scores are

$ S = "masked"(1/2 Q K^top) = mat(1, -infinity, -infinity; -1, 3, -infinity; -1, 2, 3). $

Applying normalized ReLU $"ReLU"(s_i) \/ sum_j "ReLU"(s_j)$ row-wise:
- Row 1: only $"ReLU"(1) = 1$ survives $-> [1, 0, 0]$.
- Row 2: $"ReLU"(-1) = 0$, $"ReLU"(3) = 3$ $-> [0, 1, 0]$.
- Row 3: $"ReLU"(-1)=0$, $"ReLU"(2)=2$, $"ReLU"(3)=3$; normalize by $5$ $-> [0, 0.4, 0.6]$.

$ P = mat(1, 0, 0; 0, 1, 0; 0, 0.4, 0.6). $

Then $P V$:

$ P V = mat(1, 0, 0; 0, 1, 0; 0, 0.4, 0.6) mat(1, -1, 0, 0; 0, 2, -1, 1; 0, 2, 1, 1) = mat(1, -1, 0, 0; 0, 2, -1, 1; 0, 2, 0.2, 1). $

Finally $Z = (P V) W_O$ with $W_O = W_Q^top$ (a $4 times 5$ matrix):

$ Z = mat(1, -1, 0, 0, -1; 0, 2, -1, 1, 3; 0, 2, 0.2, 1, 3). $

*4. Most probable next token after "Main course is".* With $W_"vocab" = E^top$ and last hidden state $c = [0, 2, 0.2, 1, 3]^top$, the logits are

$ z = W_"vocab" c = E c = mat(1, -1, 0, 0, 0; 0, 1, -1, 0, 0; 0, 0, 1, -1, 0; 0, 0, 0, 0, 2; 0, 0, 0, 0, -1; 0, 0, 0, 0, 1) vec(0, 2, 0.2, 1, 3) = vec(-2, 1.8, -0.8, 6, -3, 3). $

The largest logit is $6$ (4th entry), so the most probable next word is *"codfish"*.

*5. NaN after normalized-ReLU attention.* If all pre-ReLU scores in a row are negative, the normalized ReLU gives $0\/0$ (division by zero) $->$ NaN. Fixes: fall back to fixed (e.g. uniform) attention probabilities in that case, add an $epsilon$ to the denominator, replace normalized ReLU with softmax, or use the relumax transformation from the practicals.
]

