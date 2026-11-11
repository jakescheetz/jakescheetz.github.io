# Reading notes

One section per source. Key = the BibTeX key in `sources.bib`. `[full]` = full text read; `[abs]` = abstract/record only.
Quotes are verbatim. Anything marked *inference* is ours, not the source's.

---

## Game track (primary)

## pleines2025pokemon [full]
- **Claim:** A PPO agent with a CNN plus game-state input and dense shaped rewards plays Pokémon Red through Cerulean City. Ablations "reveal vulnerabilities in reward shaping."
- **Evidence:** Reward is $R = R_{event} + R_{nav} + R_{heal} + R_{lvl}$ (sec. II-F). The documented exploits (sec. IV-B):
  - *Heal-reward loop:* with Bulbasaur, the agent "converges to a strategy of battling wild Zubats in Mt. Moon, using Bulbasaur's Leech Seed to restore health while Zubat heals with Leech Life … nearly endless battles, providing frequent heal rewards."
  - *Time-bias starter choice:* it picks Charmander "simply because its reward is triggered sooner", even though Squirtle is better against Brock.
  - Navigation reward ×10 → the agent "rarely defeats" Brock.
  - It "frequently skips Misty's badge" even though the badge is mandatory.
  - Removing the level reward gives the *best* ablation.
  - No configuration reaches Vermilion's gym (the Cut tree).
- **Compute:** about 9.4k env steps/s with random actions on an 8-core Ryzen 2700X. 32 workers, 2048-step horizons. "a single training run lasting approximately 36 hours"; GRU runs on CPU only took 24 days. Evaluation takes as long as training. Results are highly unstable across seeds.
- **Limits:** The analysis is qualitative and anecdotal. There are no proxy-vs-true curves, and "exploit" is judged by eye. The true objective (story progress) is partly *inside* the proxy (event rewards), so proxy and truth are not cleanly separated.
- **Relevance:** This is the anchor prior work for G1. It establishes that Pokémon reward shaping produces nameable hacks; the gap is measurement. Whidden is a co-author, and the paper is the follow-up to his repo.

## mudireddy2026pokerl [full, grepped]
- **Claim:** In early-game Pokémon Red (house exit → first rival battle), vanilla PPO degenerates into "action loops, menu spam, or unproductive wandering". A loop-aware wrapper with anti-loop/anti-spam penalties fixes most of it.
- **Evidence:** Five failure modes; the first is "infinite action loops … pacing between two tiles that gave small positive movement rewards without progress." Loop episodes fell from 41.2% to 4.7% with the anti-loop system. Runs were "several hundred thousand timesteps on a single CPU & GPU." They "settled on" the reward after "several iterations".
- **Limits:** A small-scale student-style paper. Loops are treated as bugs to engineer away, not as data. There is no held-out true metric for checking whether the patches overfit.
- **Relevance:** (a) Early-game hacks show up within ~10^5 steps, so laptop-scale G1/G2 is plausible. (b) This is an *informal G2*: patch → retrain → patch, with the dynamics never measured.

## whidden2023pokemonred [README]
- **Claim:** Open code for RL on Pokémon Red. V2 "replaces the frame KNN with a coordinate based exploration reward" and reaches Cerulean. The user must supply a legally obtained ROM (it checks a SHA-1).
- **Limits:** Behavior anecdotes (e.g. avoiding Pokémon Centers after a blackout) come from press coverage of the video, which I did **not** watch: unverified, don't cite.
- **Relevance:** A G1 baseline codebase. The KNN→coordinate change is itself a "patch" of an exploration proxy.

## pyboy [README]
- Open-source Game Boy emulator in Python, used by all the above. ROM not included (good for our ROM rule).

## hafner2022crafter [abs]
- **Claim:** Crafter is an open-source 2D survival game that evaluates agents "by semantically meaningful achievements" (22 per the paper; I did not verify the count) separately from the reward signal.
- **Relevance:** *Inference:* it is a strong substitute for Pokémon on the "proxy vs true" question. The true metric (achievements) is built in and separate from any proxy, it's open (no ROM issue), and it is designed to be fast on CPU. Developer should check throughput.

## pan2022effects [full, grepped]
- **Claim:** Across four RL environments (traffic, pandemic, glucose, Atari) with nine misspecified proxies, "more capable agents often exploit reward misspecifications, achieving higher proxy reward and lower true reward". There are "phase transitions: capability thresholds at which the agent's behavior qualitatively shifts."
- **Evidence:** Optimization power is varied via model size, training time, action-space resolution and observation noise (sec. 4). They also propose anomaly detection for aberrant policies.
- **Limits:** Hand-designed proxies, small environments, no learned RM.
- **Relevance:** This is the closest methodological template for G1: proxy vs true reward *as a function of optimization power*. G1's novelty is the setting (a long-horizon game with natural progress flags) plus a hack catalogue, not the method.

## karwowski2023goodhart [full, grepped]
- **Claim:** Goodhart drops are "common (albeit not universal)" in RL. They propose an optimal early-stopping rule, with geometric intuition in occupancy-measure space.
- **Evidence:** 30,400 sampled (MDP, true, proxy, optimization pressure) data points over gridworlds, Cliff and random MDPs; a Goodhart drop appears in 19.3% of them.
- **Limits:** Tabular MDPs; random proxies are not like designed proxies.
- **Relevance:** Directly sets the bar for G4. A notebook "proxy diverges with optimization" result is **not** novel by itself. Their early-stopping rule is a ready-made baseline defense for G2/G4.

## leike2017ai [full, grepped]
- **Claim:** A suite of gridworlds for safety problems. The "reward gaming" ones are **Boat race**, a CoastRunners analogue where the agent can loop on one arrow tile, and **Tomato watering**: "By putting a bucket over its head, it can't see any dried out tomatoes", i.e. observation tampering. Each env has a hidden performance function separate from the reward.
- **Relevance:** G4's "CoastRunners loop" and "tampering cell" already exist as standard environments. Reuse them; don't claim them as new.

## ibarz2018reward [full, grepped]
- **Claim:** In Atari with a reward model learned from preferences (plus demonstrations), a *frozen* RM gets exploited: "the perceived performance (according to the reward model) increases, while the actual performance (according to the game score) plummets" (fig. 4) in Hero, Montezuma and Private Eye. Online labeling fixes loopholes "similar to adversarial training."
- **Relevance:** G3 has already been done in Atari, which lowers G3's novelty. It also provides the key G3 control (frozen vs online RM) and the G2 framing (online labels = patches).

## christiano2017deep [full]
- **Claim:** RL from human preferences on segment comparisons, with the RM trained online and asynchronously. In the ablation, offline RM training leads to "bizarre behavior", e.g. a Pong agent that avoids losing points but never scores (sec. 3.3).
- **Relevance:** The original learned-RM hack in a game. G3 baseline.

## booth2023perils [Crossref abstract]
- **Claim:** Trial-and-error dense reward design overfits rewards "to learning algorithms and their hyperparameters." Shown computationally and in a controlled study of expert designers. Myopic per-state-action reward design yields "invalid task specifications."
- **Relevance:** The key prior work for G2. Patching against observed behavior is itself an optimization loop (by the designer) that can overfit. G2 therefore needs a held-out true metric and held-out conditions (seeds, start states, algorithm) that the patcher never sees.

## laidlaw2025correlated [abs + intro]
- **Claim:** Defines reward hacking as the breakdown, under optimization, of the proxy–true correlation measured on a reference policy's states. Argues χ² occupancy-measure regularization beats action-KL, and shows it in four realistic settings.
- **Relevance:** A theory-backed defense to compare against "patching" in G2/G4. Its definition gives G1 an operational hack detector: track the proxy–true correlation on states the current policy visits.

## skalse2022defining [abs]
- Formal definition of an unhackable proxy; for all stochastic policies, unhackability essentially requires a constant reward. Supports "patching can't converge to unhackable in general". *Inference*: this is a theoretical prediction G2 can probe empirically in a finite setting.

## zhuang2021consequences [abs]
- A proxy with support on J < L attributes under resource constraints → unbounded utility loss under optimization (also cited by Gao sec. 4.2.2). The theory behind "the proxy omits a feature" hacks (e.g. Pokémon's "skip Misty").

## everitt2021reward [abs] · farquhar2025mona [abs]
- Everitt: causal-influence-diagram account of reward-function and RF-input tampering, with design principles that remove the incentive. MONA: myopic optimization with far-sighted approval prevents multi-step reward hacks, including in "longer-horizon gridworld environments representing sensor tampering."
- **Relevance:** The tampering cell in G4 has theory (Everitt) and a recent mitigation (MONA) to test against. A MONA-vs-patching comparison in a gridworld is a cheap, possibly novel add-on (*not verified as novel*).

## hadfieldmenell2017inverse [abs] · langosco2022goal [abs]
- IRD: proxy rewards are designed for training scenarios and fail in new ones (the "lava" example in the paper; I didn't verify the details beyond the abstract). Goal misgeneralization: capable agent, wrong goal out of distribution.
- **Relevance:** G4's "proxy diverges off-distribution" cell is IRD's setting. Langosco is the needed contrast: off-distribution failure can be *goal misgeneralization* rather than proxy overoptimization. These must not be conflated in G1 write-ups.

## clark2016faulty [full] · krakovna2020specification [opened] · lehman2018surprising [abs]
- Anecdote catalogues: CoastRunners (lagoon, 20% above human score), DeepMind's specification-gaming post and list, and the digital-evolution anecdotes.
- **Relevance:** The baseline for "catalogue the hacks". A catalogue alone isn't novel. A catalogue *indexed by optimization strength and patch round* is.

## gao2023scaling [full] · manheim2018categorizing [full] · beirami2025theoretical [full]
- Covered in detail in `research/reward-model-post-rewrite/edit-notes.md` (Substance). Key points for here: proxy-vs-gold curves against √KL; regressional (α) vs extremal (β) Goodhart; iterated RLHF helps β but not α (sec. 4.3, a model-based argument); the BoN KL formula is an upper bound (Beirami).

---

## LLM track (secondary)

## wolf2025reward [abs]
- The first systematic study of overoptimisation *in iterated RLHF* (AlpacaFarm). Overoptimisation falls over iterations; gains diminish (plateau after a few rounds per the search summary; abstract says "gains diminish over time"). Base-policy re-init is robust; other inits "often fail to recover from early overoptimisation".
- **Relevance:** This substantially pre-empts LLM candidate 1. What remains open is a *label-budget-matched* comparison against a bigger RM or an ensemble.

## coste2024reward [abs] · eisenstein2024helping [abs]
- Ensembles: conservative objectives "practically eliminate" BoN overoptimization in Gao's synthetic setup (Coste). They mitigate but don't eliminate, because members share errors (Eisenstein).
- Together with Wolf, these cover most of candidate 1's arms separately. What has not been run is the matched comparison of all of them.

## bai2022training [full sec. 4.5] · singhal2024long [abs] · wen2024language [abs]
- Bai: PMs are less calibrated at high scores, which motivates iterated online RLHF. Singhal: RLHF gains are largely length. Wen: RLHF makes models more convincing to time-limited humans without making them more correct.

## zheng2025cheating · zhao2025onetoken · zhou2026convincing · liu2026examining [all abs]
- Judges are gamed by a constant "null" response (Zheng), by "master key" tokens (Zhao), and by plausibility under self-play, where "the errors transfer across judge families" (Zhou 2026). With a gold judge, non-reasoning judges get hacked during training "regardless of judge size" (Liu 2026).
- **Relevance:** LLM candidate 2 (judge exploitability and transfer) is largely covered. Its residual novelty is a feature decomposition at small scale.

## zhong2025impossiblebench · gabor2025evilgenie · wang2026catch · baker2025monitoring [all abs]
- Test-gaming in code is now a crowded benchmark area. EvilGenie finds an LLM judge "highly effective" at detecting unambiguous hacks. CATCH (2026) is a controllable coding-RL testbed with gold labels. Baker shows CoT monitors work but that optimizing against them yields obfuscation.
- **Relevance:** LLM candidate 3 has low novelty and high compute.

## rando2024universal · baumgartner2024bestofvenom · wu2024preference · souly2025poisoning [all abs]
- Preference poisoning works with small fractions of data (Wu: in the extreme, 100% success with 0.3% poisoned, per the search summary; I did not confirm the number in the paper). The universal trigger backdoor survives RLHF (Rando). In pretraining, ~250 documents suffice regardless of model size (Souly).
- **Relevance:** LLM candidate 4 is well covered. "Do iterated-RM defenses remove a planted preference?" may still be open; not verified.

## ng1999policy [full, Thm 1 + Cor. 2]
- **Claim:** a shaping reward $F$ added to $R$ preserves the set of optimal policies, for every transition function and every $R$, **if and only if** $F(s,a,s') = \gamma\Phi(s') - \Phi(s)$ for some potential $\Phi$ over states (Theorem 1). With $\Phi = 0$ at the absorbing state, $Q_{M'} = Q_M - \Phi$ (Corollary 2).
- **Limits:** the result is about *optimal* policies in a fully observed MDP. For undiscounted MDPs it assumes an absorbing state. It says nothing about which suboptimal policy a learner ends up with under a finite budget, and that is exactly where shaping helps or hurts in practice. *Our inference:* E1 is partially observed (tiles plus a few RAM features) and truncates episodes at a time limit, so invariance holds only for the underlying MDP, not for the learner we actually run.
- **Relevance:** this justifies densifying A2's oracle with a potential built only from route distance to the next milestone. The densified oracle keeps T's optimal policies and adds no new way to earn reward that T wouldn't pay (§4, A2′).

## wiewiora2003potential [full, Thm 1-2]
- **Claim:** a Q-learner given potential-based shaping rewards makes the same updates throughout learning as an unshaped learner whose Q-table starts at Φ (Thm 1). Under advantage-based policies, the two behave identically (Thm 2). The note says this extends to Sarsa and other TD methods.
- **Limits:** the proof is tabular, with no function approximation and no actor-critic.
- **Relevance (our derivation, not in the paper):** the same identity holds for PPO with GAE when the shaping γ equals the PPO γ. With a critic V′ trained on shaped returns, $\delta'_t = r_t + \gamma\Phi_{t+1} - \Phi_t + \gamma V'_{t+1} - V'_t = r_t + \gamma\tilde V_{t+1} - \tilde V_t$ where $\tilde V = V' + \Phi$. So shaped PPO is unshaped PPO with a critic offset by Φ. As V′ fits the shaped returns, V′ → V − Φ and $\tilde V$ → V, which is the sparse-reward value. The shaping's guidance therefore lasts only while the critic is wrong about it. This explains why A2′ had no lasting signal (experiment-e1.md §9, A2′ pilot).
