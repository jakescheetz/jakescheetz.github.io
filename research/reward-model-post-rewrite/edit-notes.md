# Edit notes: reward-hacking post rewrite

Status: approved by Jake 2026-11-01. The rewrite replaced the live post in place (slug kept so the URL doesn't break; new title; `draft` line removed; date kept at 2026-10-30). The `reward-hacking-from-the-ground-up/` working draft has been deleted. Not committed.

Post: `content/posts/your-reward-model-is-an-attack-surface/index.md` (+ `references.bib`)

Diff to review (against the last committed version of the live post):

```bash
git diff HEAD -- content/posts/your-reward-model-is-an-attack-surface/
```

Line numbers in the Substance section refer to the old working draft, so they're approximate for the live file.

## Substance

_Researcher, 2026-11-01. I checked every claim against the full text of the cited source (arXiv PDFs, plus the CoastRunners post via its Wayback copy, since openai.com returns 403 to scripts). Line numbers refer to the **draft** (`reward-hacking-from-the-ground-up/index.md`). Unless noted, the same issue is in the live post too. I have not rewritten any prose; the wording in quotes is a suggestion only._

**Severity:** **Must** = factually wrong or misattributed. **Should** = overstated or missing a caveat the source itself gives. **Optional** = would strengthen the alignment-first framing.

### Claims vs sources

1. **Must: best-of-n KL is not "exactly" log n − (n−1)/n** (l.145). Gao et al. do state it as the analytic BoN KL [@gao2023scaling, sec. 2, citing Stiennon et al. 2020 App. G.3]. But Beirami et al. later proved that formula is only an **upper bound** on the true KL, and that the bound can be loose [@beirami2024theoretical, abstract and sec. 3]. A second point matters for the toy specifically: its proxy redraws noise on every call, so which $x$ gets picked is itself random, and the KL over $x$ falls further below the formula. The chart's x-axis is therefore an upper bound on distance, not the distance itself. Suggested: "at most $\log n - (n-1)/n$" and cite both papers. This does not change the shape of the result.
2. **Should: "every answer it learned from was written by $\pi_0$" / "trained once" (l.94 figcaption, l.100).** This is a fair description of the *simplest* recipe and of Gao et al.'s synthetic setup, but not of the cited papers. InstructGPT says steps 2–3 "can be iterated continuously" and that "most of our comparison data comes from our supervised policies, with some coming from our PPO policies" [@ouyang2022training, sec. 3.1]. Christiano et al. trained the reward model **online**, at the same time as the policy, on the policy's current trajectories [@christiano2017deep, fig. 1, sec. 2.2]. Keep the simplification but label it, e.g. "In the simplest version of the recipe (and in the experiments below)…". Doing so also improves the mitigations section, because "refresh the reward model" turns out to be how the method was originally designed (see 3).
3. **Optional, but strong support for the thesis.** Christiano et al. ran an ablation that drops online feedback. With an offline-trained reward predictor, "maximizing this partial reward can lead to bizarre behavior": in Pong the agent learned to avoid losing points but not to score, producing endless volleys [@christiano2017deep, sec. 3.3]. That is the post's mechanism, observed directly in 2017, and it answers part of Q1 below.
4. **Should: KL-penalty claim needs Gao's own caveat** (l.212). The summary of sec. 3.6 is accurate ("akin to early stopping"; it does not move the gold-vs-KL frontier). But Gao et al. add, in the same paragraph and in the intro: "this result could be particularly sensitive to hyperparameters." They also set the KL penalty to 0 for every other RL run in the paper, and they note PPO's implicit trust-region constraint as a confound they cannot explain. One clause covers it: "…in their setup, and they caution it may be hyperparameter-sensitive."
5. **Should: Bai et al. timing** (l.221). The paper says a "roughly weekly cadence" over "about five weeks" [@bai2022training, sec. 2.3, sec. 4.5]. "Every week" is close but slightly stronger. Note that Bai's stated *reason* for iterating supports the post directly: PMs "become progressively less calibrated and less robust at higher scores … caused by a lack of data in this high score regime" [@bai2022training, sec. 4.5]. Bai also notes their "online" means retraining new models each iteration, not continuous updates.
6. **Should: Gao sec. 4.3 is a model-based extrapolation, not an experiment** (l.221). Gao et al. say online RLHF is "preferable … to mitigate overoptimization" and cite Bai for that. Their own contribution is an analysis under two stated simplifying assumptions, and it reaches a sourced caveat the draft currently gives without a source: iterating "does not affect any Goodharting captured by the α term (such as regressional Goodharting)" and only helps with the β (extremal) part, by an amount ∝ d·log k, up to some limit [@gao2023scaling, sec. 4.3]. This is the citation for change 6 (see Q1).
7. **Should: what Gao's "true reward" is** (l.203). The draft says "playing the role of the true reward". That's correct, but the reader should know the consequence: in Gao et al. the "true reward" is itself a reward model (the 6B InstructGPT RM). The scaling laws describe one model being overoptimized against another, and were never measured against humans [@gao2023scaling, sec. 2.1]. Other facts worth having: the proxies range from 3M to 3B parameters; the policy is 1.2B for most experiments; labels are synthetic and deterministic; plots use $\sqrt{\text{KL}}$ on the x-axis. One sentence would do.
8. **OK: CoastRunners** (l.126–129). Matches the source: an isolated lagoon, three targets knocked over as they repopulate, "repeatedly catching on fire, crashing into other boats, and going the wrong way", "a higher score … than is possible by completing the course in the normal way" [@clark2016faulty]. An optional extra from the source: "on average 20 percent higher than that achieved by human players."
9. **OK: length example** (l.132–134). Accurate to sec. 4.2.2 and footnote 10 ("a real issue that we have observed in other experiments in the InstructGPT setting"). Optional: the evidence is much stronger than one footnote. Singhal et al. find that RLHF reward gains in three settings were "largely driven by increasing response length", that a purely length-based reward "reproduces most downstream RLHF improvements", and that the dominant source of the bias is the reward model [@singhal2023long, abstract]. That turns a hypothetical into a measured effect.
10. **OK: deliberate gaming** (l.228–229). Accurate to sec. 4.2.4 ("the models involved are not powerful enough to implement adversarial strategies … the scaling laws observed in this paper may break down"). Optional: Denison et al. give a concrete stepping stone between "blind search" and "deliberate gaming". Models trained on a curriculum of gameable environments generalized, rarely but measurably, to rewriting their own reward function [@denison2024sycophancy, abstract]. Probably better saved for the follow-up post.

### Claims the toy can't support

11. **Should: the toy illustrates; it doesn't show.** The rise and fall is built in: the proxy is linear and the true reward is quadratic, so the decline follows by construction. Footnote 1 half-says this already. One more sentence in the body ("the decline is baked into how I set it up; the point is the shape, and that the proxy never warns you") stops a reader from treating it as evidence. Gao et al. are the evidence. I re-ran the script and it reproduces the plotted numbers (true reward −0.85 → 2.18 at n=16 → −0.48 at n=4096), so the chart is accurate.
12. **Optional: name the two Goodhart mechanisms the toy combines.** The noise term is *regressional* Goodhart: selecting on a noisy proxy also selects for noise. Rewarding $x$ past the peak is *extremal* Goodhart: the proxy–true relationship breaks outside the training range [@manheim2018categorizing, secs. 1–2]. Gao et al. map their α and β terms onto exactly these two, and attribute the downturn mainly to extremal [@gao2023scaling, secs. 4.2.1–4.2.2]. This fits the alignment-first frame well, and it makes item 6's caveat (refreshing helps extremal but not regressional) easy to explain.

### Missing for the alignment-first framing

13. **Should: the "people judge the new answers" step has its own Goodhart problem.** The mitigation section ends by relying on human labelers, and the brief's arc (proxy → overoptimization → oversight) needs at least a pointer to why that is not the floor. Two verified sources. Wen et al.: under a standard RLHF pipeline, models got better at convincing time-limited human evaluators without getting more correct, and evaluators' false-positive rate rose 24.1% (QuALITY) and 18.3% (APPS) [@wen2024language, abstract]. Sharma et al.: humans and PMs both prefer convincingly written sycophantic answers over correct ones "a non-negligible fraction of the time" [@sharma2023towards, abstract]. This is the bridge to scalable oversight, and to our optimizing-against-graders thread.
14. **Optional: ensembles as a mitigation.** This is the most common "fix" after KL and size, and it has clean results that fit the "helps but doesn't remove" message. Coste et al.: conservative ensemble objectives "practically eliminate" overoptimization for BoN in Gao's synthetic setup [@coste2023reward, abstract]. Eisenstein et al.: ensembles mitigate but do not eliminate, because members share error patterns [@eisenstein2023helping, title and abstract]. Their "underspecification" finding (RMs that agree in-distribution diverge under shift) is also the best single source for l.106 (Q1).
15. **Optional: a definition to cite.** Skalse et al. formally define reward hacking as "optimizing an imperfect proxy reward function leads to poor performance according to the true reward function" [@skalse2022defining, abstract]. It matches l.113 and would give the post's key term a source.

### Answers to the writer's open questions

**Q1. Do changes 2, 3, 5, 6 need a source, softer wording, or an "inference" label?**

- **Change 2** (l.110–111, "near $\pi_0$ the proxy and the true reward agree; push further and more of the score comes from mistakes"): **soften slightly, and it can be cited.** "Agree" is too strong. Even at small KL, Gao's proxy rises faster than gold; that gap is regressional Goodhart, and it is present from the start. Suggest "agree closely". Gao et al.'s own interpretation supports the second half: extremal Goodhart is "primarily responsible for the nonmonotonicity" [@gao2023scaling, sec. 4.2.2]. Cite it as their interpretation.
- **Change 3** (l.106, "models are generally least reliable under distribution shift"): **cite it, scoped to reward models.** Best sources: Eisenstein et al. (RMs that agree in-distribution give very different rewards under the shift that alignment causes) [@eisenstein2023helping]; Bai et al. (PMs less calibrated and less robust at high scores, from lack of data there) [@bai2022training, sec. 4.5]; Christiano et al. (offline reward predictor → exploited) [@christiano2017deep, sec. 3.3]. Narrowing "models are generally" to "reward models are measurably less reliable" lets the sources carry it. Also, "with the same confidence anyway" is roughly what Bai's calibration result shows: fine to keep.
- **Change 5** (l.215–217, bigger RMs): see Q3. "That moves the peak; it doesn't remove it" is a fair inference from Gao's fits, since β shrinks smoothly with size but never reaches 0 in their range. Coste et al.'s abstract says the same thing about Gao's result: overoptimization "remains a persistent problem regardless of the size of the proxy reward model and training data used" [@coste2023reward]. Cite Coste for it. Keep it unlabelled.
- **Change 6** (l.222, the limits of iterated training): **now sourced.** Its first half ("between refreshes it's still optimizing against an older RM") is a reasonable inference. A stronger, sourced version is Gao's sec. 4.3 result that iteration leaves regressional Goodhart untouched (item 6). The second half ("relies on people being able to judge the new answers well") is supported by Wen et al. and Sharma et al. (item 13).

**Q2. Goodhart wording vs. Manheim & Garrabrant.** The citation is partly off. Manheim & Garrabrant do not use the "once a measure becomes a target" phrasing. Their footnote 1 gives Goodhart's original: "any observed statistical regularity will tend to collapse once pressure is placed upon it for control purposes." Their own working definition is "optimization causes a collapse of the statistical relationship between a goal which the optimizer intends and the proxy used for that goal" [@manheim2018categorizing, p. 1]. Two clean options: (a) "Goodhart's law, commonly stated as 'when a measure becomes a target, it ceases to be a good measure.'" with no citation on the quote, then cite M&G for the taxonomy or the definition; or (b) quote M&G's definition directly. (The popular phrasing is usually attributed to Marilyn Strathern. I haven't opened her 1997 paper, so don't cite it.) I recommend (a), plus item 12 if you want the taxonomy.

**Q3. Is "larger reward models lasted longer before it turned" accurate, and where is it?** Mostly, but it's the wrong emphasis. What Gao et al. state directly (sec. 3.2, figs. 1, 3, 12): the fitted coefficients change smoothly with RM size; β, the overoptimization term, falls with size; and larger RMs reach **higher peak gold scores**. In sec. 4.2.2 they interpret the falling β as "smooth improvements in model robustness". The "later turn" is implied by their functional forms (the peak sits at larger KL as β shrinks) and is visible in fig. 1, but the text never says it in those words. Suggested: "larger proxy reward models reached a higher true-reward peak and overoptimized less (sec. 3.2)". Optionally: "and peaked further from $\pi_0$". The section pin is **sec. 3.2**. Also don't confuse RM size with *policy* size. Larger policies did **not** overoptimize less; they peaked "at almost the same KL" (sec. 3.4).

### New bib entries needed if the writer takes these (all opened and checked)

```bibtex
@inproceedings{beirami2024theoretical,
  title  = {Theoretical Guarantees on the Best-of-n Alignment Policy},
  author = {Beirami, Ahmad and Agarwal, Alekh and Berant, Jonathan and D'Amour, Alexander and Eisenstein, Jacob and Nagpal, Chirag and Suresh, Ananda Theertha},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  year   = {2025},
  url    = {https://arxiv.org/abs/2401.01879}
}
@misc{wen2024language,
  title  = {Language Models Learn to Mislead Humans via {RLHF}},
  author = {Wen, Jiaxin and Zhong, Ruiqi and Khan, Akbir and Perez, Ethan and Steinhardt, Jacob and Huang, Minlie and Bowman, Samuel R. and He, He and Feng, Shi},
  year   = {2024},
  url    = {https://arxiv.org/abs/2409.12822}
}
@inproceedings{sharma2023towards,
  title  = {Towards Understanding Sycophancy in Language Models},
  author = {Sharma, Mrinank and Tong, Meg and Korbak, Tomasz and Duvenaud, David and Askell, Amanda and Bowman, Samuel R. and others},
  booktitle = {International Conference on Learning Representations},
  year   = {2024},
  url    = {https://arxiv.org/abs/2310.13548}
}
@inproceedings{eisenstein2023helping,
  title  = {Helping or Herding? Reward Model Ensembles Mitigate but do not Eliminate Reward Hacking},
  author = {Eisenstein, Jacob and Nagpal, Chirag and Agarwal, Alekh and Beirami, Ahmad and D'Amour, Alex and others},
  booktitle = {Conference on Language Modeling},
  year   = {2024},
  url    = {https://arxiv.org/abs/2312.09244}
}
@inproceedings{coste2023reward,
  title  = {Reward Model Ensembles Help Mitigate Overoptimization},
  author = {Coste, Thomas and Anwar, Usman and Kirk, Robert and Krueger, David},
  booktitle = {International Conference on Learning Representations},
  year   = {2024},
  url    = {https://arxiv.org/abs/2310.02743}
}
@inproceedings{singhal2023long,
  title  = {A Long Way to Go: Investigating Length Correlations in {RLHF}},
  author = {Singhal, Prasann and Goyal, Tanya and Xu, Jiacheng and Durrett, Greg},
  booktitle = {Conference on Language Modeling},
  year   = {2024},
  url    = {https://arxiv.org/abs/2310.03716}
}
@inproceedings{skalse2022defining,
  title  = {Defining and Characterizing Reward Hacking},
  author = {Skalse, Joar and Howe, Nikolaus H. R. and Krasheninnikov, Dmitrii and Krueger, David},
  booktitle = {Advances in Neural Information Processing Systems},
  year   = {2022},
  url    = {https://arxiv.org/abs/2209.13085}
}
@misc{denison2024sycophancy,
  title  = {Sycophancy to Subterfuge: Investigating Reward-Tampering in Large Language Models},
  author = {Denison, Carson and MacDiarmid, Monte and Barez, Fazl and Duvenaud, David and Kravec, Shauna and others},
  year   = {2024},
  url    = {https://arxiv.org/abs/2406.10162}
}
```

Venues checked against the PDFs and arXiv comments: Beirami (ICML 2025 per its PDF header), Coste (ICLR 2024), Eisenstein and Singhal (COLM 2024), Sharma (ICLR 2024). Wen and Denison are arXiv preprints as opened.

## Changes

Pass type: restructure + reframe (per `brief.md`). Writer, 2026-11-01.

### Structure

| Live post section | Draft section |
| --- | --- |
| Intro ("Coming from security…") | New intro: the alignment problem (can't specify what we want → proxy → optimize it) |
| What a policy is | A policy, and what it needs to learn (cut after "which responses are good"; defines *reward*) |
| *(inside "What a policy is")* | **New:** Why the reward has to be learned (chess vs chatbot moved here; defines proxy vs true reward) |
| How RLHF is built | How RLHF is built (+ KL defined before the equation) |
| The part that looked familiar | Why optimizing the proxy goes wrong (mechanism first, then Goodhart, then fuzzing as a labelled blockquote) |
| *(examples inside "The part that looked familiar")* | What it looks like (subsection: CoastRunners, length) |
| A small simulation (incl. Gao et al.) | A small simulation (toy only) + **new** heading: The same shape at scale (Gao et al. split out) |
| What the fuzzing view changes | What the mitigations fix, and what they don't (three mitigations: KL, bigger RMs, refreshing the RM) |
| What this leaves out | What this leaves out (unchanged) |

### Framing

- **Title/subtitle.** "Your reward model is an attack surface" / "…through a security lens" → "Reward hacking from the ground up" / "Why optimizing a learned reward goes wrong, starting from the basics". Working title; final is Jake's call.
- **Abstract.** Rewritten alignment-first. Before: bullet 3 "This is the same reason fuzzers find bugs. The fix is similar too: have people judge what the trained model writes now, and retrain the reward model on that." After: bullet 3 says mitigations limit how far training goes or refresh the RM, help but don't remove the problem, and don't cover deliberate gaming. Fuzzing removed from the abstract.
- **Fuzzing.** Was the thesis (intro, a section heading, and the frame for the mitigations section). Now one blockquote titled "An analogy, if you've done security work." The parser/fuzzer sentences are kept nearly verbatim. Removed: "That fits the fuzzing picture" (KL paragraph) and the fuzzer-fixes-the-parser lead-in to iterated training.

### Terms defined before use (new prose, no new sources)

- *Reward* (policy section).
- *True reward* vs *proxy reward*, as a bulleted pair (new section). Used consistently afterwards, including the chart legend ("proxy reward (reward model's score)", "true reward (actual quality)") and the Gao et al. paragraph ("playing the role of the true reward", "the proxies").
- *KL divergence* and *nats*, before the RLHF equation.
- *D* in the equation ("prompts x from the training set D"); was undefined.
- *Distribution shift*, in the mechanism section.
- *Overoptimization*, in the mechanism section (Gao et al.'s term).
- *Best-of-n* italicised and tied to "searching harder against the proxy".

### Substantive changes (claim meaning changed or new claim added)

1. **Goodhart citation moved.** Before: cited after the toy simulation result. After: cited at the end of the mechanism section, as the name for the general pattern. Same sentence and source.
2. **New sentence (inference, unsourced):** "Early in training the policy mostly improves at the real thing, because near $\pi_0$ the proxy and the true reward agree. Push further and more of the score comes from the reward model's mistakes." This is the explanation the toy and Gao et al. illustrate; not stated as a finding of either paper.
3. **New sentence (general ML claim, unsourced):** "Models are generally least reliable under distribution shift." The live TL;DR said the same ("That's where its scores are least reliable") without a cite.
4. **"No one has to intend this" extended** to "including the model." Same meaning as the live post's "What this leaves out", stated earlier.
5. **New mitigation bullet, "Bigger reward models hold up longer."** Reuses the live post's Gao et al. claim ("larger reward models lasted longer before it turned") and adds an inference: "That moves the peak; it doesn't remove it."
6. **New caveat on iterated training (inference, unsourced):** "It doesn't make the gap disappear: between refreshes the policy is still optimizing against a reward model trained on older answers, and it still relies on people being able to judge the new answers well." The live post presented refreshing as "the fix" without caveats.
7. **"A penalty … mostly changes where a run stops"**: unchanged wording, but no longer framed as confirming the fuzzing view.

### Lead review (2026-11-01)

- **Intro:** cut the second paragraph, which listed the sections like a table of contents (the ToC already does that). The intro now ends on "…other than what its designers meant."
- **Bigger-RM bullet:** added `[@gao2023scaling]`. The section pin is still pending on Q3.

All live-post citations are kept, with the same keys and section pins. `references.bib` is a straight copy.

### Substance folded in (writer, 2026-11-01)

Item numbers refer to the Substance section above. Before → after:

- **1 (Must), best-of-n KL.** "works out to exactly $\log n - (n-1)/n$ … so you know how far you've moved" → "is at most … so you have an upper bound on how far you've moved", plus a sentence saying Gao et al. treat it as exact and @beirami2024theoretical proved it's an upper bound. Chart x-axis label → "(nats, upper bound)"; figcaption says "upper bound". Toy footnote explains why the toy's KL is further below the bound (noise is redrawn every call).
- **Q2 (Must), Goodhart.** "This is Goodhart's law: once a measure becomes the target, it stops being a good measure [@manheim2018categorizing]" → option (a): the popular phrasing "commonly stated as…" with no citation, then M&G's own definition quoted and cited [p. 1].
- **12, Goodhart types.** Added a short list defining regressional vs extremal Goodhart [M&G secs. 1–2], Gao's attribution of the downturn to extremal [sec. 4.2.2], and one line mapping the toy's noise and peak onto the two.
- **2, simplification labelled.** "In the basic version…" → "In the simplest version of this recipe, and in the experiments later in this post…", plus a parenthetical saying real pipelines feed in newer answers. Figcaption → "Step 3, in its simplest form."
- **Change 3 / Q1, distribution shift.** "Models are generally least reliable under distribution shift, and the reward model keeps scoring … with the same confidence anyway" → scoped to reward models and cited: Eisenstein (RMs that agree in-distribution diverge under shift), Bai sec. 4.5 (less calibrated and less robust at high scores, from lack of data). Added: "And high scores are exactly where the policy is being pushed." (my framing, follows from the RLHF objective).
- **Change 2 / Q1.** "agree" → "agree closely". The second half is now supported by the Gao sec. 4.2.2 sentence in the Goodhart list.
- **15, definition.** Added footnote [^def] quoting Skalse et al.'s definition of reward hacking.
- **9, length.** Added Singhal et al.: RLHF reward gains in three settings were largely driven by length, a length-only reward reproduced most of the improvements, and the bias came mainly from the RM.
- **11, toy illustrates, doesn't show.** Added four sentences after the toy result saying the decline is built in by construction and the evidence is Gao et al.
- **7 + Q3, Gao setup.** Added the proxy RM size range (3M–3B). "larger reward models lasted longer before it turned" → "Larger proxy reward models reached a higher true-reward peak and overoptimized less [sec. 3.2]". Added a caveat paragraph: the "true reward" is itself an RM, never checked against people [sec. 2.1].
- **4, KL caveat.** Appended "though they caution that this result may be sensitive to hyperparameters."
- **Change 5 / Q3, bigger RMs.** The bullet now restates the sec. 3.2 claim. "That moves the peak; it doesn't remove it" is kept and now backed by the Coste et al. quote.
- **2 + 3 + 5, refreshing.** Added: Christiano et al. trained the RM online by design; their offline ablation produced the Pong failure [sec. 3.3]; InstructGPT steps can iterate [sec. 3.1]. Bai: "every week" → "on a roughly weekly cadence", plus their stated reason (RMs less reliable at high scores) [sec. 4.5].
- **6, iterated-training limits.** The unsourced change-6 sentence ("between refreshes … older answers …") is removed. It's replaced with Gao sec. 4.3: their model-based analysis says iterating helps extremal but not regressional Goodhart. Gao is no longer described as having *shown* iteration reduces overoptimization; the post now says they "analyze iterated training with a simple model". Added a gloss: "A freshly trained reward model still has errors, and picking the top score still picks for them." (my explanation of the regressional point).
- **13, oversight.** Added a paragraph: Sharma et al. (people and PMs prefer convincing sycophantic answers a non-negligible fraction of the time) and Wen et al. (RLHF made models better at convincing time-limited evaluators without being more correct; false-positive rate +24.1% QuALITY, +18.3% APPS, described in prose as reading-comprehension and programming tasks). It closes with "the people become a proxy too" (my framing) and a pointer to scalable oversight as future work (no claim, no cite).
- **Bib.** Seven entries copied verbatim from the Substance section: beirami2024theoretical, wen2024language, sharma2023towards, eisenstein2023helping, coste2023reward, singhal2023long, skalse2022defining. Every cited key is in the bib and every bib entry is cited.

Not taken: 8 (CoastRunners "20 percent higher than human players", fine as is); the sec. 3.4 policy-size point from Q3 (true, but the post never claims anything about policy size); 10 (Denison) and 14 (ensembles), see open questions.

## Open questions

Resolved: Q1–Q3 (answered by the researcher in Substance and folded in above).

4. **For Jake:** title and slug. Draft uses "Reward hacking from the ground up". Replace the live post at its slug, or publish this as a new post?
5. **For Jake:** is the fuzzing blockquote the right size, or should it go entirely?
6. **For Jake:** the draft turns on `toc: true` (nine sections plus subsections). Keep it?
7. **For Jake / lead:** add ensembles as a fourth mitigation (Substance 14: Coste "practically eliminate" for BoN in Gao's setup vs Eisenstein "mitigate but do not eliminate")? It fits the "helps but doesn't remove" message and adds about 4 sentences. I left it out to keep the length down.
8. **For Jake / lead:** Denison et al. (Substance 10), a stepping stone between blind search and deliberate gaming. I've saved it for the follow-up post rather than this one's closing section. Agree?
9. **For Jake:** the post is now noticeably longer (Goodhart types, Gao caveat, oversight paragraph). If it reads too long, the Goodhart-types list and the oversight paragraph are the easiest to compress.
