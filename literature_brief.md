# Literature brief (compiled 2026-10-23 by a research agent from papers/repos; numbers "as reported")

NOTE (orchestrator correction): the brief calls the 0.772 non-VLA P2N baseline "DINOv3, end to end". That is wrong: the 0.772 checkpoint is Past2NextSelfPastPolicy with a robomimic ResNet-18 (SpatialSoftmax, 76x76 random crops) trained from scratch, a 256-d 4-layer cross-attention AR decoder, n_obs_steps=2, deploy temperature 1.0/top-k 10, scored by the legacy LiberoRunner (not the official protocol) and selected as the best epoch.

## Report

DESIGN-REFERENCE BRIEF: how current VLAs attach action heads, and what it implies for a discrete AR Past2Next (P2N) head
(Compiled 2026-10-23 from paper full texts / official repos; numbers were extracted by an automated reader from arXiv HTML and code, so treat second-decimal figures as "as reported", and figure-only results are flagged.)

0. The design under review
P2N-VLA = PaliGemma (SigLIP + Gemma-2B, LoRA r16) + Gemma-300M autoregressive head that emits the 8 FSQ tokens of a frozen OAT tokenizer (16x7 chunk -> 8 ordered tokens), conditioned on the 7 previously executed commands, reading the VLM's per-layer K/V through a stop-gradient ("knowledge insulation", KI). The VLM is trained only by a FAST/OAT token-prediction CE loss. LIBERO-10 (450 demos): 0.446 (LoRA r16, SigLIP frozen), 0.706 (SigLIP trainable), vs 0.772 for the non-VLA P2N with a DINOv3 encoder trained end to end (HF card SeanLiu0272/past2next-libero10). Structurally this is exactly the pi0/pi0.5 "mixture-of-transformers" coupling plus the KI gradient rule, but with (i) a discrete AR head instead of a flow expert, (ii) LoRA-r16/frozen-SigLIP instead of the full-backbone training every KI/pi0 experiment used, and (iii) past-action conditioning that no production VLA uses.

1. Taxonomy of head couplings found in the literature
(a) In-VLM AR tokens (no separate head): RT-2, OpenVLA (256 bins/dim, per timestep), pi0-FAST (DCT+BPE tokens decoded by the full 2B LM), UniVLA (4 latent-action tokens/second), MolmoAct, CoT-VLA. Pro: zero new parameters, VLM representation is shaped directly by the action loss. Con: slow decoding (pi0-FAST ~750 ms/chunk vs 100 ms for pi0's flow expert), and per-timestep binning collapses at high control rates because consecutive tokens are nearly deterministic copies ("marginal information approaches zero", FAST paper); OpenVLA 53.7 and pi0-FAST 60.2 on LIBERO-10 vs 85-97 for continuous heads.
(b) Separate expert in the SAME attention sequence, reading per-layer K/V (pi0, pi0.5, KI; X-VLA is a single joint-self-attention stack): expert tokens (state + noisy actions) attend to the prefix in every layer with a blockwise-causal mask; information flows one way VLM->expert. In openpi there is no stop_gradient anywhere: the flow loss updates SigLIP, Gemma-2B and the expert (pi0.py, config.py: no freeze_filter in pi0_libero / pi05_libero). KI adds sg() on the K/V seen by expert tokens plus a FAST CE loss on the backbone.
(c) Cross-attention head to a CHOSEN VLM layer: GR00T N1/N1.5 (DiT cross-attends to the 12th LLM layer: "middle-layer instead of final-layer LLM embeddings resulted in both faster inference speed and higher downstream policy success rate"), FLOWER (intermediate-layer fusion: LIBERO-Long early 33.4 / intermediate 93.4 / late 69.2), SmolVLA (features from the first L/2 layers, interleaved cross/self attention: CA-only 79.0, SA-only 74.5, interleaved 85.5), Fast-ThinkAct (early-layer KV cache 89.7 > late-layer KV 88.3 > output hidden states 87.1 on LIBERO).
(d) Per-layer bridge from ALL layers: VLA-Adapter Bridge Attention (each policy layer cross-attends to the same-depth VLM layer's raw tokens, gated by tanh(g), g=0 init, plus 64 learnable "ActionQuery" tokens appended to the VLM input). Its LIBERO-Long ablation is the cleanest coupling comparison available: last-layer raw (RoboVLMs-style) 85.8; last-layer action queries (OFT-style) 90.2; one intermediate layer (GR00T-style) 88.4; all-layer raw (pi0-style) 90.6; all-layer action queries 92.6; raw + queries 95.0. VLANeXt's "soft" layer-wise cross-attention with a learnable query buffer (91.8) beat tight (90.0) and loose (90.0) coupling.
(e) Pooled / readout embedding: Octo (readout tokens -> diffusion MLP), CogACT (one learnable "cognition token" conditions a DiT), OpenVLA-OFT (final hidden states at K parallel action-token positions -> L1 MLP head), UniVLA (12.6M decoder attends over the VLM's latent-action embeddings). Simple, but FLOWER/VLA-Adapter/Fast-ThinkAct all show that top-layer-only features are the weakest conditioning signal.
Takeaways for P2N: per-layer K/V reading is a sound choice (it is the pi0 pattern), but (1) the layers that matter most for action are early/middle, so a shallow head should read early/mid-layer K/V rather than the top; (2) appending learnable action-query tokens to the PaliGemma prefix (VLA-Adapter, VLANeXt, Octo readouts) is the single cheapest documented gain (+4.4 pts over all-layer raw features) and it forces the VLM to compute action-oriented features even when the head is gradient-insulated; (3) proprio injected at the VLM level beat head-level injection in VLANeXt (98.0 vs 96.2 original; 87.7 vs 83.4 LIBERO-plus).

2. Gradient-flow choices and what the action loss updates
Full fine-tuning of everything: pi0 (pretrain and fine-tune), pi0.5 post-training and the official openpi pi05_libero recipe (flow-only loss, batch 256, peak LR 5e-5, 30k steps, EMA 0.999, nothing frozen -> 92.4 LIBERO-10), OpenVLA pretraining ("fine-tuning the vision encoder during VLA training [is] crucial"), CogACT, RT-2 (co-fine-tuned with web data to avoid forgetting), FLOWER (separate LR for VLM and head), X-VLA (reduced LR for VL modules), RoboVLMs.
LoRA on all linear layers including the vision tower: OpenVLA / OpenVLA-OFT (r32, alpha 16, target_modules="all-linear" applied to the whole VLA via get_peft_model(vla); OFT trains the L1 head and proprio projector fully), VLA-Adapter (LoRA scheme + 97M policy from scratch), UniVLA (~123M trainable). X-VLA shows the cost of parameter-efficient adaptation on a small suite: LoRA (1% params) 84.2 vs full FT 97.6 on LIBERO-Long.
Frozen LLM, trainable vision: GR00T N1 ("we keep the language component of the VL backbone frozen and fine-tune the rest"; vision encoder unfrozen).
Fully frozen VLM: GR00T N1.5 (frozen in pretraining AND finetuning; language following on real GR-1 46.6% -> 93.3%, RoboCasa 30-demo 17.4 -> 47.5, but the win comes with a VLM re-tuned for grounding, FLARE, DreamGen data and a large pretrained DiT), SmolVLA ("We train only the action expert module, keeping the VLM frozen"; LIBERO-Long 77 with 10 trials/task), Fast-ThinkAct, HAMLET on N1.5, HPT (trunk frozen at transfer).
Stop-gradient + token loss (KI): "The pre-trained model backbone and action expert only interact via the attention layers ... we propose to stop the gradient flow from the action expert to the pre-trained weights"; the backbone is trained by FAST next-token CE plus VLM co-training (captioning/VQA/localization) because "Freezing the backbone is not a viable option for knowledge insulation, since the representations in the pre-trained model are not sufficient for robotics". Motivation: a randomly initialised expert's gradients degrade the VLM (language following drops) and flow training converges slowly (pi0 needed 7.5x the steps of the KI recipe; KI trains as fast as pi0-FAST, +20% compute). KI's own LIBERO-10 numbers: specialist trained from the VLM init with full backbone training 84.8 (pi0 85.2, pi0-FAST 60.2); fine-tuned from the KI generalist 85.8. The paper says nothing about small or single-domain fine-tuning, and it does not state whether sg() stays on during fine-tuning; the pi0.5 paper never mentions a stop-gradient (KI "formalizes the approach of pi0.5" after the fact), and Physical Intelligence's released LIBERO fine-tune does not use KI at all.
Takeaway for P2N: KI was designed to protect web knowledge during large-scale pretraining and to speed up flow training. In the user's setting the backbone's only learning signal is CE on 8 OAT tokens routed through LoRA r16 (and in one variant a frozen SigLIP), which is a far weaker representation-learning signal than KI's full-backbone FAST CE + VLM co-training. Every ablation that isolates "how much the backbone/vision tower is allowed to adapt" on single-domain data points the same way (section 3 of small_data_evidence), and the 0.446 -> 0.706 jump from unfreezing SigLIP is the OpenVLA frozen-vision result (47.0 -> 69.7) reproduced.

3. Discrete vs continuous heads (same backbone, same data)
OpenVLA-OFT: AR discrete 76.5 avg / 53.7 Long -> parallel-decoded discrete + chunking 90.2 / 86.5 -> continuous L1 95.3 / 90.7 -> diffusion 95.4 / 91.1 ("continuous action variants further improve success rates by 5% (absolute)"; L1 and diffusion "comparable", L1 is 26x faster than 50-step diffusion). Octo (WidowX): diffusion 83%, MSE 35%, discretised 18% (discrete "lacked precision and often missed the grasp"). RoboVLMs (CALVIN ABCD->D avg len): KosMos discrete 0.55 vs continuous 4.09, PaliGemma 0.44 vs 4.04. VLANeXt (LIBERO original): bin classification 74.6, VQ-VAE classification 58.8, regression 85.4, DDIM 80.0, flow 80.0. FLOWER (CALVIN ABC): discrete tokens 1.12 vs flow 4.44. pi0-FAST vs pi0 on LIBERO-10: 60.2 vs 85.2 (the FAST paper reports parity on large real datasets and 5x less training compute; its Fig. 9 says "on small datasets (Libero, T-Shirt Folding), both perform comparably", figure only). Discrete heads that do well use LEARNED CHUNK tokenizers plus a decoder: UniVLA 92.0 Long (4 latent tokens/sec, 12.6M decoder, LoRA), MiniVLA (VQ: 10 actions -> 7 tokens from a 256-codebook; LIBERO-90 62% -> 82%), OAT (ordered FSQ tokens, 1000 codes, prefix-decodable; beats DP 56.3 vs 36.6 and FAST 23.0 in its from-scratch protocol). The emerging compromise is KI / OAT-v2 / the BEHAVIOR-1K winner: discrete tokens as an auxiliary representation loss on the VLM, a continuous expert for control.
Takeaway for P2N: an 8-token OAT head avoids the per-timestep-copy failure of binning and the invalid-sequence problem of FAST, and 8 AR steps in a 300M head are cheap. But the LIBERO-10 evidence says the discrete head itself costs roughly 5-10 points versus L1/flow at equal backbone adaptation, and nothing in the literature shows an AR head recovering a backbone that is under-adapted. Options in increasing order of change: (i) parallel-decode the 8 tokens with bidirectional masks (OFT's PD gave +33 pts over AR on Long for binned tokens; OAT's ordering makes tokens non-independent, so AR within the head may still be preferred but should be benchmarked); (ii) add a continuous auxiliary head (flow/L1) co-trained with the OAT CE; (iii) keep OAT CE on the VLM as the representation objective and make the executed action come from a flow expert (KI/OAT-v2 pattern).

4. Action-history conditioning in VLAs
Default is no history: pi0, pi0.5, pi0-FAST, OpenVLA(-OFT), GR00T N1/N1.5 (observation_indices=[0], action_indices=range(16)), SmolVLA (n_obs_steps=1), VLA-Adapter, FLOWER, X-VLA, CogACT (only an inference-time ensemble over past predictions). Where history exists it is observation history fused in the HEAD: Octo (2 frames; longer gave no gain), RoboVLMs ("policy-head" history > interleaving history into the VLM, 4.49 vs 4.12; interleaving costs far more compute), HAMLET (memory transformer over 4 timesteps of "moment tokens", concatenated into the action expert; N1.5 LIBERO 95.6 -> 97.7, Long 87.8 -> 92.2; VLM frozen), HiF-VLA (motion-vector hindsight tokens via AdaLN into the expert; LIBERO-Long 91.0 -> 94.4; explicitly rejects VLM-input injection: "Injecting extra history motion into the VLM risks disrupting the established alignment between visual and language modalities"), while VLANeXt found that feeding past frames into the VLM slightly degrades (LIBERO-Spatial 91.8 -> 85.0). Past ACTIONS as inputs are rare: UniVLA appends one previous latent action (4 tokens) to the prompt, LIBERO-Long 88.1 -> 92.0; HPT reports that "adding previous actions as inputs can help with HPT pre-training performance" (Fig. 15); ARP feeds a causal sequence of past actions plus empty tokens into a Chunking Causal Transformer (teacher forcing at train time, its own actions at test time; Push-T 87.1 vs DP 78.8) but never ablates removing them. The imitation-learning literature warns about exactly this input: the copycat problem (Wen et al., NeurIPS 2020: the imitator predicts the expert's previous action instead of the next one), causal confusion, and Past-Token Prediction (Torne et al. 2025): long histories "introduce features that spuriously correlate with actions"; "adding past actions in the observation doesn't improve performance" without the auxiliary past-token objective, which raised history-critical task success to 80% where baselines failed.
Takeaway for P2N: keep past commands in the head, not in the VLM prefix (RoboVLMs, HiF-VLA, HAMLET, VLANeXt all agree); the "self-past" variant (conditioning on the policy's own executed actions) is the right instrument against exposure bias; consider (a) PTP-style auxiliary prediction of the past tokens, (b) dropout/noise on the past-action channel, (c) measuring the predictability ratio of next vs previous actions, and (d) checking whether one previous chunk suffices (UniVLA used one).

5. LIBERO-10 (LIBERO-Long) reference numbers and protocol
Protocol used by OpenVLA(-OFT)/openpi and inherited by most papers: 10 tasks x 50 rollouts = 500 trials per suite; max 520 env steps for LIBERO-10 (longest demo 505; 220/280/300/400 for Spatial/Object/Goal/90) plus 10 dummy no-op steps for objects to settle; 256x256 render resized to 224 (openpi also rotates both cameras 180 deg and re-plans every 5 actions of a 10-step chunk); training on the "modified" LIBERO data with no-op and failed demos removed; OFT reports the best checkpoint among evaluations every 50k steps; SmolVLA used only 10 trials/task. Reported LIBERO-10 success (%): OpenVLA 53.7 | TraceVLA 54.1 | SpatialVLA 55.5 (both from the OFT/SpatialVLA tables) | pi0-FAST 60.2 | Diffusion Policy (scratch) 50.5 | Octo 51.1 | DiT Policy 63.8 | CoT-VLA 69.0 and MolmoAct 77.2 (third-party table in VLA-Adapter; MolmoAct avg 86.6 is official) | SmolVLA-2.25B 77 (10 trials) | KI specialist 84.8 / KI generalist-finetuned 85.8 | pi0 85.2 (openpi; SmolVLA's reproduction 73) | OpenVLA + parallel-decoded discrete tokens 86.5 | GR00T N1.5 fine-tuned 87.8 (HAMLET's baseline; another paper reports 78.4) | GR00T N1 90.6 (third-party, source unclear) | OFT third-person-only 90.7 | GR00T N1.7 91.2 (456/500, PhyAI) | UniVLA 92.0 | pi0.5 92.4 (openpi pi05_libero @30k; one reproduction 84.9) | HAMLET 92.2 | OFT + wrist + proprio 94.5 | HiF-VLA 94.4 (third view) / 96.4 (multi-view) | VLANeXt 94.8 | FLOWER 94.9 | VLA-Adapter 95.0 (Pro 96.4) | X-VLA 97.6 (LoRA 84.2). Caveats: LIBERO-Plus/LIBERO-PRO show these scores largely measure memorised layouts (OFT 97.1 -> 70.0 under perturbations; OpenVLA 76.5 -> 17.3; pi0 94.2 -> 54.6; blanking the instruction barely changes OFT's object-suite score), so LIBERO-10 is a fit benchmark, not a generalisation benchmark; KI-style knowledge preservation is not what it rewards.

6. Concrete integration recommendations for a discrete AR P2N head, ranked by evidence
R1. Unblock backbone adaptation before touching the head: raise LoRA to r>=32 on ALL linear layers including SigLIP (OpenVLA r32 = r64 = full FT within noise; the 2026 pi0 LoRA study saturates at r=32 and finds that freezing the VLM or restricting the vision encoder to LoRA "significantly degrades performance"), or train SigLIP fully + LoRA r32 on Gemma (their recommended configuration, 10.8 GiB static VRAM). Expect the 0.706 variant to move most.
R2. Decide what KI is for here. Published evidence supports KI for protecting language/web knowledge at pretraining scale, not for raising single-domain success; the official pi0.5 LIBERO fine-tune lets the action loss flow into the VLM. Two consistent choices: (A) drop the stop-gradient and let the OAT-token CE from the head update the (LoRA/full) backbone, keeping an optional FAST/OAT CE on the VLM as auxiliary (mask the two token streams from each other as pi0.5/KI do); or (B) keep sg() but give the backbone a KI-strength learning signal: full-parameter or high-rank LoRA training of PaliGemma incl. SigLIP on the token CE, plus VLM-data co-training, plus learnable action-query tokens in the prefix. Do not keep sg() together with r16/frozen SigLIP; that combination has no analogue that works.
R3. Add learnable ActionQuery tokens to the PaliGemma prefix and let the head read them per layer (VLA-Adapter +4.4 over all-layer raw; +9.2 over last-layer raw). This is compatible with either gradient regime and with discrete decoding.
R4. If the head is shallower than the VLM, read K/V from early/mid layers (GR00T 12th layer, SmolVLA L/2, Fast-ThinkAct early KV) and skip the top layers, which also cuts latency.
R5. Benchmark the decoding objective: AR-8 (current) vs parallel/bidirectional 8-token decoding vs a continuous auxiliary head (L1 or flow) co-trained with OAT CE. Across OFT, VLANeXt, FLOWER, Octo and RoboVLMs the continuous objective is worth 5-10 points on LIBERO-10 at matched backbones; UniVLA shows a discrete latent token + small decoder can reach 92.0.
R6. Keep past-action conditioning in the head, self-past/own-rollout sourced, with PTP-style past-token prediction and past-action dropout as regularisers; ablate 7 vs 1 previous chunks (UniVLA used one).
R7. Keep the eval protocol identical to OFT/openpi (500 trials, 520 + 10 steps, best-of-checkpoints vs last-checkpoint stated explicitly) so the 0.772 non-VLA baseline and any VLA numbers are comparable to the table above.

Sources (all consulted directly): pi0 https://arxiv.org/abs/2410.24164 ; pi0.5 https://arxiv.org/abs/2504.16054 ; Knowledge Insulation https://arxiv.org/abs/2505.23705 ; FAST https://arxiv.org/abs/2501.09747 ; openpi code/README https://github.com/Physical-Intelligence/openpi (src/openpi/models/pi0.py, src/openpi/training/config.py, examples/libero/README.md, examples/libero/main.py) ; OpenVLA https://arxiv.org/abs/2406.09246 ; OpenVLA-OFT https://arxiv.org/abs/2502.19645 and https://github.com/moojink/openvla-oft (vla-scripts/finetune.py, experiments/robot/libero/run_libero_eval.py) ; GR00T N1 https://arxiv.org/abs/2503.14734 ; GR00T N1.5 https://research.nvidia.com/labs/gear/gr00t-n1_5/ and https://github.com/NVIDIA/Isaac-GR00T/tree/n1d5 ; SmolVLA https://arxiv.org/abs/2506.01844 and https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/smolvla/configuration_smolvla.py ; VLA-Adapter https://arxiv.org/abs/2509.09372 and https://github.com/OpenHelix-Team/VLA-Adapter ; FLOWER https://arxiv.org/abs/2509.04996 ; X-VLA https://arxiv.org/abs/2510.10274 ; UniVLA https://arxiv.org/abs/2505.06111 ; CogACT https://arxiv.org/abs/2411.19650 ; Octo https://arxiv.org/abs/2405.12213 ; RT-2 https://arxiv.org/abs/2307.15818 ; RoboVLMs https://arxiv.org/abs/2412.14058 ; HPT https://arxiv.org/abs/2409.20537 ; HAMLET https://arxiv.org/abs/2510.00695 ; HiF-VLA https://arxiv.org/abs/2512.09928 ; VLANeXt https://arxiv.org/abs/2602.18532 ; Fast-ThinkAct https://arxiv.org/abs/2601.09708 ; OAT https://arxiv.org/abs/2602.04215 and https://arxiv.org/abs/2607.21670 ; ARP https://arxiv.org/abs/2410.03132 ; Past-Token Prediction https://arxiv.org/abs/2505.09561 ; Copycat https://proceedings.neurips.cc/paper/2020/hash/1b113258af3968aaf3969ca67e744ff8-Abstract.html ; MiniVLA https://ai.stanford.edu/blog/minivla ; MolmoAct https://arxiv.org/abs/2508.07917 ; CoT-VLA https://arxiv.org/abs/2503.22020 ; LIBERO-PRO https://arxiv.org/abs/2510.03827 ; LIBERO-Plus https://arxiv.org/abs/2510.13626 ; ReVLA https://arxiv.org/abs/2409.15250 ; How Do VLAs Inherit from VLMs https://arxiv.org/abs/2511.06619 ; LoRA efficiency for pi0 https://arxiv.org/abs/2607.10172 ; PhyAI (GR00T N1.7 LIBERO-10) https://arxiv.org/abs/2608.03682 ; Past2Next checkpoints https://huggingface.co/SeanLiu0272/past2next-libero10.

## Small-data evidence

EVIDENCE ON FROZEN / INSULATED vs END-TO-END ADAPTATION WHEN FINE-TUNING DATA IS SMALL (LIBERO-scale: ~400-500 demos per suite; Franka ablations: 10-150 demos/task)

(a) Frozen vs fine-tuned vision tower / backbone
- OpenVLA (Franka tabletop, 10-150 demos/task, 33 rollouts): full FT 69.7 +/- 7.2; LoRA r32 68.2; LoRA r64 68.2; sandwich (vision + projector + last layer) 62.1; frozen vision encoder 47.0; last layer only 30.3. 'We find that only fine-tuning the network's last layer or freezing the vision encoder leads to poor performance, suggesting that further adaptation of the visual features to the target scene is crucial.' Pretraining section: 'fine-tuning the vision encoder during VLA training [is] crucial for good VLA performance' (opposite of VLM practice). https://arxiv.org/abs/2406.09246
- OpenVLA-OFT applies LoRA r32 to all linear layers of the entire VLA (vision backbone included) and still gets 94.5 on LIBERO-10; its from-scratch (no OpenVLA pretraining) OFT recipe loses only 5.2 avg (91.9 vs 97.1; Long 86.5 vs 94.5), i.e. the recipe (parallel decoding + chunking + L1 + adapting the whole stack) matters more than the robotics pretraining. https://arxiv.org/abs/2502.19645
- LoRA efficiency study on pi0 (4 UR5e precision-assembly tasks, 2026): no statistically significant advantage of full FT over LoRA; performance saturates at r=32 spread uniformly over VLM backbone and action expert; 'freezing the VLM, or limiting the vision encoder to LoRA only, significantly hurts performance'; recommended configuration LoRA r=32 + full vision-encoder fine-tuning (static VRAM 36.2 -> 10.8 GiB). Directly relevant to the r16 / frozen-SigLIP variants. https://arxiv.org/abs/2607.10172
- VLA-Adapter (LIBERO-Long): LoRA-adapted backbone 95.0 vs frozen backbone 86.4 (-8.6); frozen OpenVLA-OFT backbone 0.0; SmolVLA (frozen VLM) 77.0. https://arxiv.org/abs/2509.09372
- FLOWER (CALVIN ABC avg len): full 4.44 vs 'No VLM training' 2.65 vs no VLM at all 3.42, i.e. a frozen VLM was worse than removing it. https://arxiv.org/abs/2509.04996
- X-VLA: LoRA (1% of params) 84.2 vs full FT 97.6 on LIBERO-Long. https://arxiv.org/abs/2510.10274
- Octo: full-model fine-tuning 'outperformed those that freeze subsets of the pretrained parameters' (~100 trajectories/task). https://arxiv.org/abs/2405.12213
- HPT freezes its trunk at transfer, but it is a from-scratch robotics trunk with re-trained vision/proprio stems, not a VLM. https://arxiv.org/abs/2409.20537
- Counter-evidence (knowledge preservation): 'How Do VLAs Effectively Inherit from VLMs?' (pi0-style PaliGemma VLA): full FT 'performs well on narrow tasks but results in catastrophic forgetting of the pre-trained priors' (recognition on held-out emojis 45.2% vs execution 93.6%); LoRA gives slightly higher recognition; but 'only fine-tuning action head does not fit well even in in-distribution scenarios'. https://arxiv.org/abs/2511.06619 . ReVLA: fine-tuning OpenVLA's DINOv2/SigLIP destroyed depth/visual generalisation; gradually reverting the vision backbone restored visual OOD performance (+77% grasping) while keeping in-domain accuracy. https://arxiv.org/abs/2409.15250 . GR00T N1.5 froze the VLM in both stages and improved language following (46.6 -> 93.3 real GR-1) and 30-demo RoboCasa (17.4 -> 47.5), but with a grounding-tuned Eagle 2.5, FLARE and a large pretrained DiT, i.e. not a controlled freeze ablation. https://research.nvidia.com/labs/gear/gr00t-n1_5/
- Net: for in-distribution success on a few hundred demos, every controlled comparison favours letting the vision tower (and, via LoRA >= r32 or full FT, the LLM) adapt; freezing or low-rank-only vision costs 9-23 points; the only benefits of freezing are language following/visual OOD robustness, which LIBERO-10 does not measure (LIBERO-Plus shows these models mostly ignore language anyway: OFT 97.1 -> 70.0, pi0 94.2 -> 54.6 under perturbations). https://arxiv.org/abs/2510.13626

(b) Discrete-token heads vs continuous (flow / diffusion / L1) heads
- OpenVLA-OFT ablation on the same 7B backbone and LIBERO data: AR discrete 76.5 avg / 53.7 Long; parallel-decoded discrete tokens + 8-step chunk 90.2 / 86.5; continuous L1 95.3 / 90.7; diffusion 95.4 / 91.1 ('continuous action variants further improve success rates by 5% (absolute)'; L1 and diffusion 'comparable'; L1 109.7 Hz vs diffusion 4.2 Hz). https://arxiv.org/abs/2502.19645
- pi0-FAST vs pi0, same PaliGemma, same LIBERO fine-tune: Long 60.2 vs 85.2 (avg 85.5 vs 94.2); Goal 88.6 vs 95.8. FAST's own figure caption says both are 'comparable' on small datasets, and FAST is 5x cheaper to train, but the official numbers favour the flow expert on LIBERO-10 by 25 points. https://arxiv.org/abs/2501.09747 ; https://github.com/Physical-Intelligence/openpi
- Octo (WidowX): diffusion 83% vs MSE 35% vs discretised 18% (discrete policies 'lacked precision and often missed the grasp'). https://arxiv.org/abs/2405.12213
- RoboVLMs (CALVIN): continuous beats AR discrete for every backbone (KosMos 4.09 vs 0.55; PaliGemma 4.04 vs 0.44), attributed to compounding errors on long horizons. https://arxiv.org/abs/2412.14058
- VLANeXt (LIBERO): bin classification 74.6, VQ-VAE classification 58.8, regression 85.4, DDIM 80.0, flow 80.0. https://arxiv.org/abs/2602.18532 ; FLOWER: discrete tokens 1.12 vs flow 4.44 on CALVIN. https://arxiv.org/abs/2509.04996
- Discrete heads that are competitive use a learned chunk tokenizer + decoder: UniVLA 92.0 LIBERO-Long (16-code latent actions, 12.6M decoder, LoRA; +3.9 from one previous latent action token) https://arxiv.org/abs/2505.06111 ; MiniVLA VQ chunk tokens LIBERO-90 62 -> 82 https://ai.stanford.edu/blog/minivla ; OAT's ordered FSQ tokens beat diffusion policy and FAST in a from-scratch regime (56.3 vs 36.6 vs 23.0) and fix FAST's non-decodable sequences https://arxiv.org/abs/2602.04215 . KI and OAT-v2 converge on using discrete tokens as a representation-learning loss on the VLM while a continuous expert produces the executed action. https://arxiv.org/abs/2505.23705 ; https://arxiv.org/abs/2607.21670
- Net: at matched backbone adaptation a discrete AR head costs roughly 5-10 points on LIBERO-10 versus L1/flow (25 for pi0-FAST), with a good chunk tokenizer narrowing but not closing the gap; discrete tokens are most valuable as an auxiliary objective for the backbone.

(c) Stop-gradient / knowledge insulation vs joint training when data is small
- What KI shows: at pretraining scale, stopping gradients from a randomly initialised flow expert into the backbone, while training the backbone with FAST CE + VLM co-training, improves language following and trains 7.5x faster than pi0's joint flow training; 'Freezing the backbone is not a viable option ... since the representations in the pre-trained model are not sufficient for robotics'. Its LIBERO-10 specialist trained this way from the VLM init reaches 84.8 (pi0 85.2) with FULL backbone training; the paper gives no recommendation for small or single-domain fine-tuning and does not say whether sg() stays active in fine-tuning. https://arxiv.org/abs/2505.23705
- What pi0.5 and its official fine-tuning do: post-training combines next-token CE and flow loss with no stop-gradient mentioned; the released openpi pi05_libero fine-tune (the 92.4 LIBERO-10 checkpoint) uses the flow loss only, full-parameter updates (no freeze_filter), gradients flowing through the shared attention into PaliGemma and SigLIP, batch 256, peak LR 5e-5, 30k steps. No FAST loss, no KI. https://arxiv.org/abs/2504.16054 ; https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py ; https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py
- Frozen-VLM head-only training on small data: SmolVLA (frozen VLM) 77 Long with 10 trials/task https://arxiv.org/abs/2506.01844 ; VLA-Adapter frozen backbone 86.4 vs 95.0 https://arxiv.org/abs/2509.09372 ; 'only tuning the action head results in inadequate learning for robotic execution' https://arxiv.org/abs/2511.06619 ; GR00T N1 kept the LLM frozen but the vision encoder unfrozen https://arxiv.org/abs/2503.14734 ; GR00T N1.5's fully frozen VLM works because the DiT is large and robotics-pretrained and the VLM was re-tuned for grounding https://research.nvidia.com/labs/gear/gr00t-n1_5/ ; a pi0.5 LIBERO-Goal expert-only (frozen VLM) checkpoint exists for paraphrase-robustness studies but publishes no success number https://huggingface.co/HAI-Lab/pi05-libero_goal-expert_only
- Where to inject new conditioning signals under insulation: evidence consistently favours the head over the VLM input for history (RoboVLMs policy-head 4.49 vs interleaved 4.12; HiF-VLA expert-side AdaLN > VLM-input; HAMLET memory at the expert with VLM frozen +4.4 Long; VLANeXt past frames into the VLM -6.8 Spatial), but for proprio the VLM level was better (VLANeXt 98.0 vs 96.2). https://arxiv.org/abs/2412.14058 ; https://arxiv.org/abs/2512.09928 ; https://arxiv.org/abs/2510.00695 ; https://arxiv.org/abs/2602.18532
- Net for the P2N-VLA numbers (0.446 / 0.706 vs 0.772): the literature predicts the observed ordering. KI + LoRA r16 + frozen SigLIP leaves the backbone with the weakest adaptation signal of any configuration in the table (no ablation anywhere shows this combination working; KI itself trained the full backbone and still needed FAST CE + web co-training). Unfreezing SigLIP recovered 26 points, matching OpenVLA's frozen-vision gap (47.0 -> 69.7). The remaining gap to end-to-end DINOv3 (0.772) and to flow/L1 VLAs (92-97) is consistent with (i) r16 < the r32 saturation point, (ii) sg() preventing the OAT-token loss from shaping PaliGemma features, and (iii) the 5-10 point discrete-head penalty. No published result supports KI as a success-rate improvement for single-domain fine-tuning on a few hundred demos; its documented benefits are language following, OOD generalisation and training speed at pretraining scale.

## Designs

### pi0 (Physical Intelligence, 2024)
- head: Continuous flow-matching action expert (~300M Gemma-shaped weights, width 1024), 50-step action chunk, 10 Euler steps; proprio state as its own token block
- coupling: Shared sequence / per-layer K/V: expert tokens (state + noisy actions) sit in the same transformer forward as the PaliGemma image+language prefix with a blockwise-causal mask; expert has its own Q/K/V/MLP weights (mixture-of-transformers) and attends to the VLM's keys/values in every layer; information flows only VLM -> expert
- gradient flow: Full fine-tuning: flow loss updates SigLIP, Gemma-2B and the expert; openpi pi0.py has no stop_gradient and the pi0_libero TrainConfig sets no freeze_filter (nothing frozen); LoRA variants (gemma_2b_lora + gemma_300m_lora) freeze all non-LoRA params and disable EMA; pre-trained for 700k steps then fine-tuned
- action history: None: o_t = images, language, joint state; chunks executed open-loop; temporal ensembling hurt
- LIBERO: openpi/OFT Table I (one policy for all suites, 500 trials/suite): Spatial 96.8 / Object 98.8 / Goal 95.8 / LIBERO-10 85.2, avg 94.2. SmolVLA's reproduction (10 trials/task): 90/86/95/73, avg 86.0; PaliGemma init without robot pretraining: 87/63/89/48
- source: https://arxiv.org/abs/2410.24164 ; https://github.com/Physical-Intelligence/openpi (src/openpi/models/pi0.py, src/openpi/training/config.py)

### pi0.5 (Physical Intelligence, 2025)
- head: Hybrid: FAST discrete action tokens predicted by the VLM (pre-training, alpha=0, 280k steps; kept in post-training for text/subtask prediction) + flow-matching action expert (~300M, randomly initialised at post-training, alpha=10, 80k steps); adaRMSNorm time conditioning; proprio discretised as text tokens
- coupling: Same shared-sequence per-layer attention as pi0; image/text/continuous-action tokens bidirectional; discrete FAST tokens and continuous action tokens are masked from attending to each other; one-way VLM -> expert
- gradient flow: Paper: post-training jointly minimises next-token CE (to preserve text prediction) and flow loss; no stop-gradient is mentioned (the KI paper later 'formalizes the approach of pi0.5'). Official openpi pi05_libero fine-tune: flow-only loss, no FAST loss, no stop-gradient, no freeze_filter (full-parameter), batch 256, peak LR 5e-5, 30k steps, EMA 0.999
- action history: None (current observation + state; ~1 s chunk, 10 denoising steps)
- LIBERO: openpi pi05_libero @30k: Spatial 98.8 / Object 98.2 / Goal 98.0 / LIBERO-10 92.4, avg 96.85 (LIBERO-PRO lists 0.92 on Long; Libra-VLA's reproduction 84.9 Long)
- source: https://arxiv.org/abs/2504.16054 ; https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/README.md ; https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py

### Knowledge Insulation recipe (Driess et al., 2025; single-stage pi0.5-style)
- head: Flow-matching action expert (2B backbone + ~300M expert, width 1024, mlp 4096, 50-step horizon) for inference; FAST discrete tokens predicted by the backbone as a training-only representation objective
- coupling: Per-layer K/V: expert tokens attend to backbone keys/values in every layer with sg() applied to those K/V ('The pre-trained model backbone and action expert only interact via the attention layers'); unidirectional VLM -> expert; FAST tokens attend to prefix and earlier FAST tokens only
- gradient flow: Stop-gradient from expert into backbone (Eq. 5-6); backbone updated by FAST next-token CE + co-training with VLM data (captioning/VQA/localisation); expert updated by flow loss. 'Freezing the backbone is not a viable option ... representations in the pre-trained model are not sufficient for robotics.' Joint training without sg degrades language following; pi0 needs 7.5x the steps; +20% compute; no LoRA, no guidance on small/single-domain fine-tuning; unspecified whether sg stays on during LIBERO fine-tuning
- action history: None
- LIBERO: Specialist trained from the VLM init with the KI recipe: Spatial 96.6 / Object 97.2 / Goal 94.6 / LIBERO-10 84.8 / LIBERO-90 92.7; fine-tuned from the KI generalist: 98.0 / 97.8 / 95.6 / 85.8 / 96.0 (pi0 85.2, pi0-FAST 60.2 on LIBERO-10 in the same table); DROID 0.55 vs pi0 0.49 vs pi0-FAST 0.45
- source: https://arxiv.org/abs/2505.23705

### pi0-FAST (Pertsch et al., 2025)
- head: Discrete autoregressive FAST tokens (per-dimension DCT, rounding scale 10, BPE vocab 1024) over 1-s chunks, decoded by the full 2B PaliGemma LM (no expert); overwrites least-used vocabulary tokens; openpi LIBERO config uses action_horizon 10
- coupling: In-VLM: actions are language-model outputs in the shared sequence (same weights)
- gradient flow: Full fine-tuning with cross-entropy (openpi pi0_fast_libero: no freeze_filter; LoRA variant freezes non-LoRA params); 5x fewer GPU-hours than pi0 and 3x faster convergence on table bussing
- action history: None
- LIBERO: openpi/OFT Table I: 96.4 / 96.8 / 88.6 / LIBERO-10 60.2, avg 85.5; FAST paper (figure only): 'on small datasets (Libero, T-Shirt Folding), both perform comparably'; inference ~750 ms/chunk vs ~100 ms for pi0; naive binning 'unable to make progress' at 20-50 Hz
- source: https://arxiv.org/abs/2501.09747 ; https://github.com/Physical-Intelligence/openpi

### OpenVLA (Kim et al., 2024)
- head: Discrete AR tokens: each of 7 dims binned into 256 bins (1st-99th quantile), 7 tokens per timestep, overwriting the 256 least-used Llama-2 tokens; CE on action tokens only
- coupling: In-VLM: Llama-2 7B decodes action tokens over DINOv2+SigLIP fused patch tokens via an MLP projector (same sequence, same weights)
- gradient flow: Pretraining: full fine-tuning incl. vision encoder ('crucial for good VLA performance', contradicting VLM practice). Fine-tuning ablation (Franka tabletop, 10-150 demos/task): full FT 69.7, LoRA r32 68.2, LoRA r64 68.2, sandwich (vision+projector+last layer) 62.1, frozen vision 47.0, last layer only 30.3; LoRA applied to all linear layers of the whole VLA
- action history: None; single image; observation history listed as future work
- LIBERO: LoRA fine-tuned per suite on filtered demos: Spatial 84.7 / Object 88.4 / Goal 79.2 / LIBERO-10 53.7, avg 76.5 (4.2 Hz AR decoding)
- source: https://arxiv.org/abs/2406.09246

### OpenVLA-OFT (Kim et al., 2025)
- head: Continuous L1-regression MLP head over K=8 action chunk via parallel decoding (empty action tokens, bidirectional mask); diffusion variant matches (95.4 vs 95.3 avg) at 4.2 Hz vs 109.7 Hz; discrete parallel variant 90.2 avg
- coupling: Head reads the LLM's final hidden states at the K parallel action-token positions (same sequence); optional FiLM of language into the vision backbone (ALOHA only); proprio projector token
- gradient flow: LoRA r32 (alpha 16, dropout 0, target_modules='all-linear', get_peft_model over the whole VLA incl. vision backbone and projector; 111M adapter params); action head and proprio projector trained fully; LR 5e-4 -> 5e-5 after 100k; 50-150k steps, batch 64-128 on 8 A100/H100; from-scratch OFT (no OpenVLA pretraining) 91.9 avg / 86.5 Long
- action history: None; third-person image (+ wrist + proprio in the 97.1 variant); chunk executed open-loop
- LIBERO: Third-person only: 96.2 / 98.3 / 96.2 / Long 90.7 (avg 95.3); + wrist + proprio: 97.6 / 98.4 / 97.9 / Long 94.5 (avg 97.1); ablation chain on Long: AR 53.7 -> +PD&AC discrete 86.5 -> L1 90.7 -> diffusion 91.1. Protocol: 500 trials/suite, 520 max steps + 10 no-op steps on LIBERO-10, best checkpoint of evaluations every 50k steps
- source: https://arxiv.org/abs/2502.19645 ; https://github.com/moojink/openvla-oft (vla-scripts/finetune.py, experiments/robot/libero/run_libero_eval.py)

### GR00T N1 (NVIDIA, 2025)
- head: Continuous flow-matching DiT (System 1), action horizon H=16, Beta timestep sampling, 4 Euler steps; embodiment-specific MLP action/state encoders and decoders
- coupling: Cross-attention from DiT blocks to vision-language token embeddings taken from the 12th LLM layer of Eagle-2 ('middle-layer instead of final-layer LLM embeddings resulted in both faster inference speed and higher downstream policy success rate'); self-attention over noised action tokens + state embedding
- gradient flow: Table 6: vision encoder unfrozen, DiT unfrozen, text component frozen; post-training: 'As in pretraining, we keep the language component of the VL backbone frozen and fine-tune the rest of the model'; AdamW 1e-4 cosine
- action history: Current state q_t only; no past actions (data configs: observation_indices=[0], action_indices=range(16))
- LIBERO: Not reported in the paper. Third-party fine-tune listed in VLA-Adapter Table 5: 94.4 / 97.6 / 93.0 / Long 90.6, avg 93.9 (source of that run unclear). Low-data: RoboCasa 30/100/300 demos 17.4/32.1/49.6 vs DP 14.7/25.6/43.2
- source: https://arxiv.org/abs/2503.14734 ; https://github.com/NVIDIA/Isaac-GR00T

### GR00T N1.5 (NVIDIA, 2025)
- head: Same flow-matching DiT as N1 plus FLARE future-latent-representation alignment loss (coef 0.2) and DreamGen synthetic data; Eagle 2.5 VLM re-tuned for grounding
- coupling: Cross-attention to mid-layer VLM features; adapter MLP between vision encoder and LLM simplified with layer norm on visual and text token embeddings
- gradient flow: 'The VLM model is frozen during both pretraining and finetuning' (N1 froze only the language part); only DiT/adapter trained; 'greatly improved language following and generalization': real GR-1 language following 46.6% -> 93.3%, success 43.3 -> 83.0; RoboCasa 30 demos/task 17.4 -> 47.5; LoRA r64/alpha128 supported but 'we recommend using the full model finetuning'
- action history: None by default (observation_indices=[0]); HAMLET adds an observation memory at the action expert
- LIBERO: Fine-tuned N1.5 as HAMLET's baseline: 98.2 / 99.4 / 97.2 / Long 87.8, avg 95.6; another 2026 paper reports 93.4 / 85.0 / 89.4 / 78.4 (86.6); GR00T N1.7 on LIBERO-10: 91.2% (456/500, PhyAI)
- source: https://research.nvidia.com/labs/gear/gr00t-n1_5/ ; https://github.com/NVIDIA/Isaac-GR00T/tree/n1d5 ; https://arxiv.org/abs/2510.00695 ; https://arxiv.org/abs/2608.03682

### SmolVLA (Shukor et al., 2025)
- head: Flow-matching action expert (~100M, width 0.75x VLM), chunk 50 (ablation: n=10 84.0 > n=50 80.3 > n=100 74.5)
- coupling: Interleaved cross-attention (to VLM K/V from the first N=L/2 layers; upper layers skipped) and causal self-attention over action tokens; CA-only 79.0, SA-only 74.5, interleaved 85.5 (LIBERO avg)
- gradient flow: 'We train only the action expert module, keeping the VLM frozen' in pretraining, fine-tuning and ablations; lerobot defaults freeze_vision_encoder=True, train_expert_only=True, train_state_proj=True
- action history: None (n_obs_steps=1); asynchronous inference over chunks
- LIBERO: Multi-task, 10 trials/task: 2.25B 93 / 94 / 91 / Long 77 (avg 88.75); 0.45B 90 / 96 / 92 / 71 (87.3); 0.24B Long 63. VLA-Adapter Table 3 lists SmolVLA (frozen backbone) at 77.0 Long
- source: https://arxiv.org/abs/2506.01844 ; https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/smolvla/configuration_smolvla.py

### VLA-Adapter (Wang et al., 2025)
- head: Continuous L1-regression Policy (97M, same depth as the 24-layer Qwen2.5-0.5B backbone), H=8 chunk refined from an all-zero initial action; DiT alternative 91.6 vs L1 95.0 on Long
- coupling: Per-layer 'Bridge Attention': each policy layer cross-attends to the same-depth VLM layer's raw tokens (tanh(g)-gated, g=0 init) and to 64 learnable ActionQuery tokens appended to the VLM input (+ proprio), then self-attends over action latents. Long ablation: last-layer raw 85.8; last-layer AQ (OFT-style) 90.2; one intermediate layer (GR00T-style) 88.4; all-layer raw (pi0-style) 90.6; all-layer AQ 92.6; raw+AQ 95.0
- gradient flow: End-to-end; backbone adapted with LoRA ('AdamW optimizer and LoRA scheme'; README --use_lora True), Policy and ActionQuery from scratch; LR 1e-4 cosine, batch 16, 150k steps, 4xH100 (~half a day for Long). Frozen-backbone variant 86.4 Long vs 95.0; frozen OpenVLA-OFT 0.0
- action history: None
- LIBERO: 97.8 / 99.2 / 97.2 / Long 95.0, avg 97.3; VLA-Adapter-Pro 99.6 / 99.6 / 98.2 / 96.4 (98.5)
- source: https://arxiv.org/abs/2509.09372 ; https://github.com/OpenHelix-Team/VLA-Adapter

### FLOWER (Reuss et al., 2025)
- head: Rectified-flow transformer (18 layers, 339M) with Action-Space Global-AdaLN-Zero, 4 denoising steps (8 dual-arm)
- coupling: Intermediate-layer fusion: hidden states from an intermediate layer of a pruned Florence-2-L (decoder removed / 30% of layers dropped) -> linear + RMSNorm -> cross-attention in the flow transformer. Fusion ablation (LIBERO-Long): early 33.4, intermediate 93.4, late 69.2
- gradient flow: VLM fine-tuned with its own LR schedule; frozen-VLM ablation on CALVIN ABC 2.65 vs 4.44 full; discrete-token variant 1.12; 947M params total
- action history: None
- LIBERO: 97.5 / 99.1 / 96.1 / Long 94.9 (avg 96.9), LIBERO-90 94.7; CALVIN ABC->D 4.53
- source: https://arxiv.org/abs/2509.04996

### X-VLA (2025)
- head: Flow matching (OT path) in a 0.9B standard 24-layer transformer; domain-specific action input/output projections (0.04% unshared params)
- coupling: Single joint bidirectional self-attention stack over Florence-Large vision-language tokens, proprio, noisy action tokens, flow-time embeddings and per-dataset soft-prompt tokens
- gradient flow: Pretraining: backbone + soft prompts jointly, reduced LR for VL modules; adaptation: prompt warm-up with frozen weights then joint training; LoRA PEFT (1% params) 93 avg / 84.2 Long vs full FT 97.6 Long
- action history: None
- LIBERO: Full fine-tune: 98.2 / 98.6 / 97.8 / Long 97.6, avg 98.1; LoRA: 95.4 / 96.6 / 96.0 / 84.2
- source: https://arxiv.org/abs/2510.10274

### UniVLA (Bu et al., 2025)
- head: Discrete AR latent-action tokens (codebook |C|=16, N=4 tokens per ~1 s latent action) predicted by the Prismatic-7B LLM, decoded to continuous chunks (12 steps real-world) by a 12.6M action decoder
- coupling: Pooled-embedding style head: pooled visual features (learned query) attend over the VLM's latent-action token embeddings, then linear projection to the robot action space
- gradient flow: LoRA fine-tuning (~123M trainable) with CE on latent tokens plus decoder regression, 'trained end-to-end'; 960 A100-h pretraining vs 21.5k for OpenVLA
- action history: Yes: one previous latent action (4 tokens) appended to the prompt each step; LIBERO-Long 88.1 -> 92.0
- LIBERO: 96.5 / 96.8 / 95.6 / Long 92.0, avg 95.2 (Bridge-only pretraining 87.5 Long; LIBERO-PRO reproduction 0.79 Long)
- source: https://arxiv.org/abs/2505.06111

### CogACT (Li et al., 2024)
- head: Diffusion DiT action module (DiT-S/B/L 13M/89M/308M; avg success 58.5/62.5/64.8), 16-step chunk, adaptive action ensemble at inference
- coupling: Pooled embedding: one learnable 'cognition token' appended to vision+language tokens; its final-layer output feature conditions the DiT (plus noisy actions as tokens)
- gradient flow: Vision encoders, LLaMA-2 and action module 'all trained end-to-end' through the diffusion loss (full FT, init from OpenVLA weights)
- action history: No past actions as inputs; ensemble over cached predictions from past observations
- LIBERO: Not reported (SIMPLER Google-robot visual matching 74.8 vs OpenVLA 34.3, RT-2-X 46.3)
- source: https://arxiv.org/abs/2411.19650

### Octo (Octo Model Team, 2024)
- head: Diffusion head (3-layer MLP, 20 steps) predicting action chunks; head ablation: diffusion 83%, MSE 35%, discretised 18% (WidowX)
- coupling: Learned readout tokens attend to observation/task tokens in a shared transformer; head reads the pooled readout embedding
- gradient flow: Full-model fine-tuning 'outperformed those that freeze subsets of the pretrained parameters'; T5 language encoder frozen
- action history: 2 frames of observation history (1 frame of history beats none; longer gave no benefit); no past actions
- LIBERO: 78.9 / 85.7 / 84.6 / Long 51.1, avg 75.1 (OFT Table I)
- source: https://arxiv.org/abs/2405.12213

### RT-2 (Google DeepMind, 2023)
- head: Discrete AR tokens: 256 uniform bins per dimension emitted as integer/text tokens by PaLI-X / PaLM-E
- coupling: In-VLM (shared sequence, VQA-format prompt)
- gradient flow: Full co-fine-tuning with web data (robot data ~50-66% of the mixture) 'to not forget its previous concepts'; co-FT vs robot-only FT on unseen categories: 63 vs 52 (55B), 44 vs 42 (5B)
- action history: None stated (single camera image + instruction)
- LIBERO: n/a
- source: https://arxiv.org/abs/2307.15818

### RoboVLMs (Li et al., 2024)
- head: Continuous policy head (RNN/transformer/diffusion) vs discrete tokens; continuous wins every pairing (CALVIN avg len KosMos 4.09 vs 0.55, PaliGemma 4.04 vs 0.44); chunk 10
- coupling: 'Policy-head' design: per-step last-layer VLM features feed a separate head that fuses history; beats interleaving history into the VLM (KosMos 4.49 vs 4.12; PaliGemma 4.42 vs 4.25) at much lower compute
- gradient flow: VLM fully fine-tuned (hyperparameters inherited from VLM training; LoRA not used)
- action history: Observation history window 16 (CALVIN) / 8 (real) fused in the head; 'models incorporating history observations as input consistently outperform one-step models'; past actions not described
- LIBERO: Not reported (CALVIN, SimplerEnv, real robot)
- source: https://arxiv.org/abs/2412.14058

### HPT (Wang et al., 2024)
- head: MLP head (sim) or diffusion head (real), action horizon 8
- coupling: Embodiment-specific stems (cross-attention onto 16 tokens for proprio and vision) -> shared trunk transformer -> task head on pooled trunk features
- gradient flow: Transfer: stems and heads re-initialised and trained, trunk frozen (also compares full fine-tuning)
- action history: Observation horizon 4 in pretraining, 2 (real) / 1 (sim) at transfer; 'adding previous actions as inputs can help with HPT pre-training performance' (Fig. 15)
- LIBERO: n/a
- source: https://arxiv.org/abs/2409.20537

### HAMLET (history-aware add-on on GR00T N1.5 / CogACT, 2025)
- head: Inherits the base head (N1.5 flow DiT / CogACT DiT)
- coupling: 4 learnable 'moment tokens' per timestep appended to the VLM input (time-contrastive init); a 2-layer causal memory transformer over the last 4 timesteps' moment tokens; output concatenated with the current VLM representation and fed to the action expert (history injected at the head, not into the VLM)
- gradient flow: N1.5 variant: VLM and moment tokens frozen during fine-tuning, memory module + head trained (60k steps, lr 1e-5); CogACT variant trainable
- action history: Past observations only; proprio current only; no past actions
- LIBERO: 99.0 / 100.0 / 99.2 / Long 92.2, avg 97.7 (base N1.5: 98.2 / 99.4 / 97.2 / 87.8, 95.6); real history-dependent tasks 29.2 -> 76.4
- source: https://arxiv.org/abs/2510.00695

### HiF-VLA (hindsight/foresight motion on OpenVLA-OFT, 2025)
- head: OFT L1 head plus a foresight motion stream
- coupling: Hindsight tokens (MPEG-4 motion vectors -> 4-layer ViT) injected via AdaLN into every expert block; VLM-input injection rejected because it 'risks disrupting the established alignment between visual and language modalities'; expert-side injection 'consistently achieves higher success rates'
- gradient flow: As OFT (LoRA) plus the new encoders
- action history: Past motion history (length 8 best), not past actions
- LIBERO: 98.8 / 99.4 / 97.4 / Long 96.4, avg 98.0 (Long third-view 94.4 vs OFT 91.0)
- source: https://arxiv.org/abs/2512.09928

### VLANeXt (2026, design study)
- head: 12-layer flow-matching policy with 16 learnable queries and adaLN time conditioning; objective ablation (LIBERO original / LIBERO-plus): bin classification 74.6/43.4, VQ-VAE classification 58.8/36.5, regression 85.4/48.4, DDIM 80.0/48.3, flow 80.0/45.0
- coupling: Layer-wise 'soft' cross-attention conditioning with a learnable query buffer between VLM and policy (soft 91.8 vs tight 90.0 vs loose 90.0); proprio at the VLM level best (98.0 vs 96.2 at the policy)
- gradient flow: Training mode not specified (LR 1e-4 for <3B backbones); backbone capacity matters (PaliGemma-3B 18.9 vs Qwen3-VL-4B 64.8 on LIBERO-plus)
- action history: Temporal observation history into the VLM 'slightly degrades performance' (Spatial 91.8 -> 85.0; LIBERO-plus 56.2 -> 50.2)
- LIBERO: 99.0 / 99.2 / 96.6 / Long 94.8, avg 97.4; LIBERO-plus 83.9
- source: https://arxiv.org/abs/2602.18532

### Fast-ThinkAct (2026, KV-conditioning evidence)
- head: Diffusion-transformer action model (RDT init for LIBERO)
- coupling: Action model conditioned on the KV cache of spatial tokens from EARLY VLM layers: early-layer KV 89.7 > late-layer KV 88.3 > output hidden states 87.1 (LIBERO)
- gradient flow: VLM and state encoder frozen; only the action model (and latent projector) trained
- action history: None
- LIBERO: 89.7 LIBERO (ablation metric)
- source: https://arxiv.org/abs/2601.09708

### OAT: Ordered Action Tokenization (Liu et al., RSS 2026) - the frozen tokenizer used by P2N
- head: Discrete AR transformer decoder (4 layers, d=256) over 8 ordered FSQ register tokens (levels [8,5,5,5] = 1000 codes, nested dropout, any prefix decodes to a valid chunk); paper chunk 32x7 -> 8 tokens (P2N: 16x7 -> 8)
- coupling: Not a VLA: from-scratch policy conditioned on the past 2 observations; v2 paper adds 'token co-training policies that use token losses to shape the vision-language model context' feeding a flow-based action expert (KI-style)
- gradient flow: n/a (tokenizer frozen; policy trained with teacher forcing); notes the BEHAVIOR-1K winner 'employs FAST as an auxiliary discrete action representation alongside continuous flow-based experts'
- action history: Observation history 2; no past actions
- LIBERO: Own protocol (not OFT's): LIBERO 56.3 vs DP 36.6, QueST 48.2, FAST 23.0, Bin 14.4; FAST 'suffers from invalid or non-decodable token sequences'
- source: https://arxiv.org/abs/2602.04215 ; https://arxiv.org/abs/2607.21670

### ARP: Autoregressive Policy (Zhang et al., 2024) - past-action conditioning
- head: Discrete/continuous chunked autoregression in a Chunking Causal Transformer (causal over input actions, bidirectional among empty future tokens)
- coupling: From-scratch policy (no VLM)
- gradient flow: Teacher forcing on ground-truth past actions; conditions on its own sampled actions at inference
- action history: Explicit past-action sequence as input; no ablation removing it
- LIBERO: n/a (Push-T 87.1 vs DP 78.8; ALOHA insertion 24.8 / transfer 94; RLBench 81.6)
- source: https://arxiv.org/abs/2410.03132

### Past-Token Prediction (Torne et al., 2025) and the copycat problem (Wen et al., 2020) - history-conditioning evidence
- head: Long-context diffusion policy with an auxiliary head predicting past action tokens
- coupling: n/a (visual encoder pretrained on short context, frozen, embeddings cached; long-context head trained on cached features)
- gradient flow: Multistage: encoder first, then head on cached embeddings (>5x faster)
- action history: Long observation/action history; 'longer observation histories often introduce features that spuriously correlate with actions'; 'adding past actions in the observation doesn't improve performance' without PTP; PTP enables 80% on history-critical tasks where baselines fail; copycat: the imitator 'ends up predicting the expert's previous action rather than the next one'
- LIBERO: n/a
- source: https://arxiv.org/abs/2505.09561 ; https://proceedings.neurips.cc/paper/2020/hash/1b113258af3968aaf3969ca67e744ff8-Abstract.html

### MiniVLA (Stanford, 2024) - small discrete chunk-token VLA
- head: Discrete VQ action-chunk tokens (10 actions -> 7 tokens, 256-entry codebook) decoded autoregressively by Qwen2.5-0.5B
- coupling: In-VLM (OpenVLA-style Prismatic stack with the OpenVLA vision encoder, multi-image input)
- gradient flow: OpenVLA-style training
- action history: None
- LIBERO: LIBERO-90: 62% -> 82% from VQ chunking + multi-image (blog); a third-party re-evaluation got 72.2%
- source: https://ai.stanford.edu/blog/minivla

### MolmoAct (AI2, 2025) and CoT-VLA (NVIDIA, 2025) - reasoning-token AR VLAs
- head: Discrete AR tokens: depth-aware perception tokens + trajectory trace + action tokens (MolmoAct); subgoal image tokens then action tokens (CoT-VLA, 7B)
- coupling: In-VLM
- gradient flow: Full VLM training with mid-training robot data (MolmoAct dataset, 10k trajectories)
- action history: None
- LIBERO: MolmoAct-7B-D avg 86.6 (official; Long 77.2 per VLA-Adapter's table); CoT-VLA 87.5 / 91.6 / 87.6 / Long 69.0, avg 81.1 (per VLA-Adapter's table)
- source: https://arxiv.org/abs/2508.07917 ; https://arxiv.org/abs/2503.22020

### Past2Next non-VLA baseline (user's own, for reference)
- head: Discrete AR causal transformer predicting the next chunk's 8 OAT tokens from observations and the policy's own earlier actions ('self-past'), receding-horizon execution
- coupling: DINOv3 vision encoder trained end to end (no VLM)
- gradient flow: End-to-end
- action history: 7 previously executed commands (own actions during training in the self-past variant)
- LIBERO: 0.772 on LIBERO-10 (500 demos; HF card)
- source: https://huggingface.co/SeanLiu0272/past2next-libero10

