## 1. Verdict on the current P2N-VLA design

The implementation is correct. Most of the head is worth keeping:
- **Verified mechanics.** Masks, RoPE offsets, KI rows hidden from the expert, cached decoding equal to teacher forcing (9.5e-7), gradient reachability and the pi05 fold all check out on tiny models (findings §1).
- **History in the head.** Past commands sit in the head as static RAWDIFF memory. The 0.772 non-VLA P2N and the history literature do the same (literature: RoboVLMs 4.49 vs 4.12, HiF-VLA, HAMLET, VLANeXt −6.8), and C6 was refuted as the cause of the gap.
- **Token head.** The 8 ordered OAT tokens avoid the per-timestep binning failure.
- **Coupling.** The same-depth per-layer K/V read is the pi0 coupling, i.e. VLA-Adapter's "all-layer raw" arm (90.6, vs 88.4 for one intermediate layer and 85.8 for the last layer).

What limits it is the training regime around the head:
- **The P2N loss never reaches the VLM.** Knowledge insulation in a 450-demo single-domain fine-tune keeps it out entirely (C2: 0/39 SigLIP tensors get AR gradient).
- **The VLM's only teacher is a history-free OAT cross-entropy.** It barely learns through LoRA r16 at 5e-5 (KI train accuracy 0.24 at 30k) and memorises once SigLIP trains (vision_action KI train/val accuracy 0.975/0.05).
- **λ_ki is only an on/off switch** (C1).
- **The one large lever so far is trainable vision**, even when trained only through KI: +20.6 pp at matched 30k (0.446 vs 0.652), concentrated on K3/K8/S1/L6 (empirical s4). D4 still confounds this with LR annealing.
- **No control was ever run** (D3). Neither 0.446 nor 0.706 can be credited to the P2N head, KI, self-past or the pi05 init.
- **Every reference is on a different protocol.** 0.706 is a best-of-test 5-env number (D6, D8). 0.772 is a legacy-runner best epoch at T=1 (D7). The repo's own records (COMPARISON_015_043_046.md, explicitly unverified) list 85.8–89.2 % official for fine-tuned 8×256 ResNet-18 P2N policies.

In short: a faithful pi0.5-KI implementation, run in a regime KI was not designed for, and the head itself has never been priced.

## 2. Recommended integration design

### 2.0 Pick, resolved disagreements, corrected errors

**Primary design: P2's integration, built on P1's code path with P1's guards, sized to P4's measured 2-GPU footprint, plus P3's diagnostics.** P2's integration means three things: the stop-gradient on the per-layer K/V is replaced by a scheduled gradient scale α(t); KI stays as a λ=1 auxiliary; and a matched α≡0 control arm is run.

Judge scores:
- J1 (evidence/attribution): P2 8 > P1 7 > P4 6.5 > P3 5.
- J2 (feasibility/gain): P1 7.4 > P2 6.6 > P4 6.1 > P3 4.8.

I did not average them. P2 is the only proposal whose decisive experiment isolates the gradient path: α=1 vs α=0 at identical scope, both arms trained fresh. P1's own E1 changes three things at once against the old VA run (stop-gradient, KI removed, LoRA trainable). It also drops KI, the only VLM-side teacher with in-repo evidence. P1's code path is still the cheapest correct implementation: J2 verified every line reference, and P1 found both λ=0 traps. Both judges' graft lists converge on "P2's A/B on P1's chassis".

I did not take P2's recipe drift: r64, SigLIP 2e-5, batch 64 on 4 GPUs, 16k updates, global clip and sharded EMA. None of these has in-repo evidence. Each moves the control away from the one anchored run (VA), and the sharded EMA can deadlock.

| Question | P1 | P2 | P3 | P4 | Decision | Evidence |
|---|---|---|---|---|---|---|
| Executor head | pi05 MoT expert | same | 9 M 6×256 decoder over 32 readouts | MoT | **MoT expert.** The native head becomes rung H1 | Only head with VLA evidence (0.706). Per-layer read = VLA-Adapter all-layer arm. P3's cost argument rests on a wrong memory bound (correction 2) |
| Stop-gradient | removed | GradScale α: 0 until 2k, ramp to 1 by 4k, plus α=0 control | removed | removed | **P2's ramp plus a matched α=0 arm** | tok_emb/RAWDIFF are freshly initialised, so early AR gradients come from a random head (KI paper's motivation). The ramp is free |
| KI auxiliary | off | λ=1 | none | λ=1 | **λ=1 in both E1 arms; λ=0 is rung 7** | Only in-repo VLM teacher. λ only matters on shared parameters (C1). Gives a history-free check against copycat behaviour. Anchors the I arm to VA |
| LoRA | r32 @5e-5 | r64 @1e-4 | r32 @1e-4 | r32 @1e-4 | **r32/α32 @1e-4**, via a policy flag that moves LoRA into the 'new' group | pi0 LoRA study saturates at r=32. OpenVLA: r32 = r64 = full fine-tune. r16 @5e-5 learned little (KI accuracy 0.24) |
| SigLIP LR | 5e-5 | 2e-5 | 2e-5 | 2e-5 | **5e-5**. 2e-5 is rung A5, triggered by the dev curve | Only setting with in-repo positive evidence. VA's memorisation coexisted with rising SR |
| Clipping | per group | global | global | per side | **Keep the existing per-group clip** | Under VA's scope, 'ki' = SigLIP + projector + LoRA + KI table and 'ar' = expert + new, so it is already per side. No workspace edit. No evidence links clip mode to SR |
| EMA | none | 4-way sharded on GPU | on GPU | pinned CPU, every 10 updates | **Pinned-CPU EMA.** EMA weights are pre-registered; live weights are a paired secondary | fp32 EMA on GPU OOMs at micro-batch 4 (P1, P4). Sharded gather risks deadlock (correction 6) |
| Batch / GPUs | 32 = 4×2×4 | 64 on 4 GPUs | 32 = 8×2×2 | 32 = 8×2×2 | **32 on one NUMA-local pair: micro-batch 4 × accum 4.** Micro-batch 8 × accum 2 only if `--probe` passes | Micro-batch 8 is 20.5 GiB, only 0.46 GB inside the 22.5 GB gate before DDP/NCCL buffers |
| Schedule | VA curve, stop at 40k | 16k cosine | 30k cosine | 30k cosine | **VA curve** (warmup 1k, cosine horizon 100k, floor 0.1), hard stop at 40k | VA rose to 40k, then plateaued (0.706/0.674/0.704/0.678, all n.s.), so 40k is a fair endpoint for the control. Exact LR anchor. Resume extension stays legal |
| Self-past temperature | T=1/top-10 | same | same | T=0 | **T=1/top-10. T=0 is rung A2** | Keeps the A/B single-factor. Greedy self-past on a memorising model risks curriculum collapse |
| Action-query tokens | – | – | readouts | AQ16 | **None in the main design. AQ16 is rung A3** | +4.4 in VLA-Adapter with a continuous head; unproven for AR. Needs coupling to get any gradient |
| Split | VA's | VA's, hash-pinned | re-split aligned to tokenizer | content-addressed re-split | **VA's zarr and split, with per-episode hashes recorded** | Keeps the VA anchor valid. D2 affects offline metrics only |
| Selection | official schedule (R0b, early stop) | dev | official | official | **Dev schedule for every choice. Official + test-B only at the pre-registered snapshot** | D8 |

**Judge-found errors that change the design or its numbers:**

1. **λ=0 with un-detached K/V does not run as P2 (E2) and P3 (E5a/E5c) wrote it.** I re-ran the judges' tiny-model check (`scratchpad/design-judge-feasibility/gradflow_traps.py`):
   - Leaving the `torch.no_grad()` at p2n_vla_common.py:560 in place gives 67 trainables with grad=None (39 SigLIP + 28 LoRA).
   - Removing it still leaves 10 dead tensors: the layer-17 q/o/gate/up/down LoRA. `_check_gradient_coverage` (train_p2n_vla.py:937-941) then aborts at step 1.
   - P1's claim that this fails "silently" is wrong in the vision scopes; the failure is loud. It is silent only when no VLM parameter is trainable. P3's literal E5a is that case: λ=0 freezes LoRA at lines 202-205 and SigLIP is frozen, so it trains a frozen-VLM MoT.
   - With λ=1 (the main arms), no tensor is dead.
2. **P3's MoT cost bound is wrong.** It claimed micro-batch 1 only and 2.7× the native head. Three measurements contradict it: P1 18.22 GiB at micro-batch 4, P2 19.89 at micro-batch 4 (r64 + 25 % EMA), P4 20.53 at micro-batch 8. The real ratio is about 1.25–1.7×.
3. **The flow control needs two edits P4 missed.** pi05_ki_flow.py has its own no_grad gate (:60) and calls `_detached_kv` (:62, :83). Renaming the method as P4 proposed breaks the flow control or leaves it insulated. It also needs the vision/LoRA scope through a mixin.
4. **Global clipping does not make λ meaningful on disjoint parameter sets** (contrary to P3's fix-first). Adam's per-parameter scale invariance makes λ inert under any clip mode; only shared parameters change that. Whether clipping is even active depends on scope: P1's LoRA-only E2E probe showed a VLM-group pre-clip norm of 0.33.
5. **P1's and P2's timings ran next to other jobs.** This affects P1's 3.68 samples/s/GPU at micro-batch 4, its "micro-batch 6 gives no gain", and P2's single-GPU timings. P1's harness under-read the logged LoRA run (5.19 vs 6.15). P4's harness is the best calibrated (8.06 vs 8.15 logged). J1's view that micro-batch 8 at 20.5 GiB is safe is optimistic.
6. **P2's sharded EMA would gather inside main-process-only save code** (train_p2n_vla.py:1025-1026, 1686-1690): deadlock risk. **P1's early-stop rule** (pre-clip 'ki' norm > 100) would have fired on VA, which had 10 harmless spikes above 50 (max 226.9).
7. **Evidence hygiene:**
   - The vision effect is 0.446 vs 0.652 at matched 30k (+20.6 pp), not 0.446 → 0.706.
   - Held-out token accuracy is 0.09–0.12 in every run regardless of SR (D5). P4's no-vision proxy therefore supports only its first-token claim (CE 2.94 with past vs 5.15 task-only).
   - P3's "ji0nzb1p 0.836 official", "seed SD 2.2 pp" and shuffle retentions are not in the evidence base.
   - The in-repo 0.772 config uses `self_past_ramp_steps: 0` (train_past2next_self_past.yaml:73).
8. **Fewer registration edits are needed than P2–P4 listed**, provided variant names are kept and the LoRA LR moves through a policy flag. Untouched: workspace `VARIANTS` (:92), group names (:93-94), `OPTIMIZER_KEYS` (:95, :549; scripts/train_p2n_vla.py:52, :198), scripts/train_p2n_vla.py:38. P4's `vision_lr` key would be rejected. P4's `history_dropout_p` would be rejected by the non-gate `history_*` check in scripts/train_p2n_vla.py `validate_config`.
9. **Stale facts:**
   - Disk now has 213 GB free.
   - Local `output/` was archived to HF SeanLiu0272/ysk at 18:53 UTC, so every re-evaluation must first download snapshots and pi05_base (absent from /workspace/.hf_home).
   - `/workspace/hf_upload` (yaml:41 and the docs' 0.772 path) does not exist.
   - All 8 GPUs are idle now (checked).

### 2.1 Architecture

```
TRAIN (deploy drops the KI block and the losses)
agentview 128px -openpi aug (crop .95, rot ±5°, colour)-> 224 ┐
wrist     128px -colour jitter--------------------------> 224 ┴-> SigLIP So400m/14 + projector -> 2x256 IMG  [train, fp32 master, 5e-5]
"Task: <instr>, State: <8 x 256-bin>;\nAction: " (≤96, right-pad) -> frozen embed x sqrt(2048) -> PROMPT
[KI_BOS, z0..z6] -> ki_table (5001x2048) x sqrt(2048) -> KI rows (training only)                 [train, 1e-4]

PREFIX  Gemma-2B, 18 L, frozen bf16 base + LoRA r32/α32 on q,k,v,o,gate,up,down of every layer    [train, 1e-4]
  rows [IMG 512 | PROMPT 96 | KI 8]
  IMG/PROMPT -> valid IMG/PROMPT keys (prefix-LM), never KI
  KI -> valid IMG/PROMPT + causal KI
  RoPE cumsum(valid)-1; KI at pos0+i                                    (layout.py:51-87, unchanged)
  out 1: h_top[KI] @ ki_table^T -> L_KI
  out 2: layer l = 1..18, post-RoPE (K_l, V_l) of the 608 IMG/PROMPT rows -> GradScale(α(t)) -> expert layer l

SUFFIX  pi05 Gemma-300M expert, 18 L, w1024, 8q/1kv, hd256, adaRMS folded at t0=0.6 per segment   [train, 5e-5]
  [RAWDIFF 9 | AR 8]
  RAWDIFF: raw_proj(a_{t-7..t-1}) + lag emb + type0 (7); acc (1); jerk (1); self-only rows; invalid zeroed + masked
  AR: tok_emb([BOS, z0..z6]) x 32; row i -> valid same-layer prefix K/V + valid RAWDIFF + AR j ≤ i
  positions: RAWDIFF pos0+0..8, AR pos0+10+i                             (layout.py:117-185, unchanged)
  head: h_AR @ tok_emb^T (tied, BOS masked) -> L_AR
L = L_AR + λ·L_KI with λ = 1
α(t) = 0 for t < 2k, linear to 1 at 4k        (control arm: α ≡ 0, plain detach)

DEPLOY: 608-row prefix -> prefill [RAWDIFF, BOS] -> 7 cached greedy steps
        -> frozen OAT FSQ [8,5,5,5,5] -> 16x7 -> execute 8 -> acknowledge
```

**What the head reads.** In each of the 18 Gemma layers it reads the post-RoPE K/V of all valid image (512) and prompt (≤96) rows. It never reads KI rows (gemma_joint.py:259 slices to `nonki_len`). The pairing stays same-depth because the pretrained expert was trained on it (C3) and `check_joint_compatible` enforces it. The main design adds no tokens to layout.py and truncates no layers; AQ16 and the readout head are separate rungs.

**Information flow is one way, VLM → expert.** Prefix rows never read suffix rows, and RAWDIFF rows never read the prefix, so history cannot send gradient into the VLM.

### 2.2 Gradient flow

| Parameters | L_AR (α-scaled into the VLM) | λ·L_KI | Optimizer group, peak LR | Clip group |
|---|---|---|---|---|
| Expert, 18 layers + folded segment modulations | ✓ | – | pretrained, 5e-5 | ar |
| tok_emb (tied), RAWDIFF projections, type/lag embeddings | ✓ | – | new, 1e-4 | ar |
| KI table | – | ✓ | new, 1e-4 | ki |
| LoRA r32 (7 projections × 18 layers, 39.2 M) | ✓ ×α | ✓ | new, 1e-4 (flag) | ki |
| SigLIP + projector (414.8 M) | ✓ ×α | ✓ | pretrained, 5e-5 | ki |
| Gemma base, embed_tokens, norms, OAT tokenizer | frozen | frozen | – | – |

**Stop-gradients that stay:**
- Self-past regeneration runs under no_grad in eval mode.
- OAT targets and the detokenizer are frozen.
- The one-way masks described above.

**Making λ meaningful.** On the shared VLM parameters θ_V the gradient is g_V = α·∇L_AR + λ·∇L_KI. Per-group clipping (VA: 'ki' pre-clip p50 9.5, clipped on 100 % of updates) rescales g_V to norm 1, then AdamW normalises per coordinate. So λ/α sets the mixture direction on θ_V and nothing else. On disjoint sets (KI table, expert) λ stays inert, as C1 established.

The mixture is measured, not assumed: every 500 updates the probe logs ‖α g_AR‖, ‖λ g_KI‖, their cosine and the action share ‖α g_AR‖ / (‖α g_AR‖ + ‖λ g_KI‖). Expected share is 0.15–0.35: at random init the AR-into-VLM norm was 1.5–3 (P1's probe) against VA's KI norm of 6–11.

λ is bracketed experimentally: λ=1 in E1, λ=0 in rung 7. It is tuned further (A1, λ_eq that equalises the measured shares at 5k) only if the bracket shows a ≥4 pp difference or E1 comes back null with share < 0.15.

**α in the KI-off arm.** When KI is off (rung 7), the ramp degenerates to a hard switch at 2k, because of Adam's scale invariance. That is harmless.

**Clipping and LR.** The existing per-group clip is kept at 1.0 per group, which under this scope is per side; both norms are logged. AdamW uses betas (0.9, 0.95), eps 1e-8, weight decay 1e-10, fused. The LR follows VA's curve: linear warmup over 1k updates from peak/1001, cosine horizon 100k, floor 0.1, hard stop at 40k. At the stop, 'pretrained' is at 3.5e-5 and 'new' at 7.0e-5.

### 2.3 History injection and self-past

- **Past commands.** At deployment these are the 7 acknowledged executed commands a_{t−7..t−1} (`record_executed_actions`). In training they come from the dataset or from self-generation. They are normalised by the dataset action normaliser and enter as 9 RAWDIFF tokens:
  - 7 raw commands through a shared Linear, plus a lag embedding and type 0;
  - the first difference and the second difference through their own Linears, plus types 1 and 2.
  
  Invalid slots are zeroed and masked (layout.py:171-174). These are static self-only rows, read by every AR row in all 18 expert layers. This is unchanged and is the 0.772 model's mechanism (C6). The difference tokens stay: masking them cost the modern non-VLA P2N −15.8 pp at ep100. Left out: "enriched" history (≈ raw past, −0.019 n.s., and +0.21 nats worse held-out NLL) and the state gate (C4).
- **History never enters the VLM.** Proprio enters only as pi05's 256-bin prompt text (VLANeXt: proprio at VLM level 98.0 vs 96.2).
- **Self-past (unchanged).**
  - Applied to p = 0.5 of rows that have a valid t−8 window, after a 1k warmup and a 4k linear ramp on successful updates.
  - Depth-1 regeneration by the live model in eval mode (no augmentation, no grad), in chunks of 4. The history becomes the last 7 of [prev_past, pred[:8]].
  - Under coupling, the generator is the whole adapted stack.
- **Deploy vs self-past temperature.** Deployment is greedy (T=0, BOS masked, k=8). Self-past stays at T=1/top-10 in every phase-A/B arm. The C5 mismatch is deliberate:
  - It keeps E1 single-factor; both trained runs used it.
  - The modern non-VLA P2N lost 7.4 pp at ep100 when deployed at T=1.
  - The unverified 043/046 records used exactly this pairing (T=1 self-past, greedy deploy) and logged 88.0/88.2 % official.
  - Greedy self-past on a model that reaches 0.95 train accuracy would regenerate near-expert histories, i.e. curriculum collapse.
  
  Deploy temperature is probed on the dev schedule (R1). Greedy self-past is rung A2, monitored with `self_past_mae`.
- **New diagnostics:**
  - `train/self_past_mae`: normalised |generated − dataset| over the selected valid entries. A collapse towards the OAT round-trip floor raises an alarm.
  - Offline shuffle-past vs shuffle-obs first-token retention, as a copycat check.
  - Zero-history closed-loop evaluation on dev. Mid-episode zero history looks like an episode start (closed-loop findings: zero-history contexts are extra sharp), so this measures reliance on history, not robustness.

### 2.4 Decoding

- **Head.** Autoregressive over the 8 ordered OAT tokens with KV cache: prefill [RAWDIFF, BOS], then 7 single-token steps (p2n_vla_common.py:458-483). Cached decoding matches teacher forcing to 9.5e-7. Measured latency at batch 1 is 184 ms per chunk (prefix 39.7 ms, decode 141.6 ms), well inside a 400 ms 8-step chunk at 20 Hz.
- **Deployment.** Greedy, k=8, receding horizon (execute 8 of 16). `--use-k-tokens` is restricted to {1, 2, 4, 8}: the frozen OAT decoder was trained with pow2 nested dropout, and docs/P2N_VLA.md lists the same set. k=4 remains a dev-only diagnostic: executed-half xyz RMSE is 0.058 at k=4 vs 0.047 at k=8 (design-coupling/tok_analysis.json).
- **No parallel or bidirectional decoding.** In P4's no-vision proxy, conditioning on z_<j removes 7.7 nats per chunk of held-out CE (design-coupling/ar_dependency.json). OFT's +33 pt gain from parallel decoding was for independent per-dimension bins combined with chunking.

### 2.5 Training recipe

- **Data.**
  - zarr: /workspace/past_action/data/libero/libero10_N500.zarr (identity 05dcceae…, the one VA trained on).
  - Split: seed-42 positional, 450 train / 50 val (VA's ids). Per-episode action hashes are recorded in a new `dataset_split.json` field, not inside `identity`, which resume compares by dict equality at train_p2n_vla.py:1532.
  - `allow_dataset_change=false` (D1).
  - Tokenizer: data/pretrained/p2n_vla/oat_from_upd030000.ckpt (sha256 8dae1b3a…, VA's).
  - The D2 tokenizer leak (about 45/50 val demos were in the tokenizer's training data) is documented; it affects offline metrics only.
- **Batch.** 32 = micro-batch 4 × 2 GPUs (one NUMA node) × accumulation 4, i.e. 100 updates per "epoch". 40k updates = 1.28 M samples = 10.3 passes over 124,106 windows. The data order is identical across arms (same sampler seed and per-update sample sets; this also matches VA's 1×16 arrangement).
- **Optimiser, LR, clip and coupling** as in 2.2.
- **EMA.** Decay 0.999 per update, applied as 0.999¹⁰ every 10 updates on pinned host shadows. Snapshots (every 5k) store bf16 EMA. `keep_resume_checkpoint=true` keeps live and EMA weights at 40k for paired evaluation.
- **Precision and checkpointing.** bf16 autocast; fp32 expert attention and heads; fp32 masters for SigLIP, LoRA and the expert. Activation checkpointing in Gemma (per layer) and SigLIP (HF).
- **Augmentation.** openpi: agentview 95 % crop + U(−5°, 5°) rotation; colour jitter on both cameras. None in self-past, validation or evaluation.
- **Validation.** A fixed 3,200 windows (400 × 4 × 2) every 10k updates, in expert and generated history modes, on EMA weights. Monitoring only: Spearman(SR, val CE) was +0.75 in VA (D5).
- **Evaluation.** `lazy_eval=true` (rank 0 cannot host 5.5 GB of renderers). Dev-200 curves at 10k/20k/30k/40k run on dedicated GPUs. The pre-registered endpoint is in §3.
- **Seed.** 42 for every phase-A/B arm.

### 2.6 Memory and compute (4090, ≈23.5 GiB usable; repo gate 22.5 GB reserved and ≥2 GB headroom)

| Configuration | Micro-batch | Peak reserved (GiB) | Samples/s/GPU (self-past / none) | Source |
|---|---|---|---|---|
| LoRA run as trained (r16, frozen SigLIP, GPU EMA) | 8 | 14.71 logged; 14.70 / 15.21 emulated | 6.15 / 8.15 logged | empirical s1; P2 calib.json; P4 mem_probe A |
| VA as trained | 1 | 16.7–18.7 logged; 16.81 emulated | 2.3 logged | empirical s1; P1 out_va.json |
| VA scope | 2 | 17.37 | – | P1 |
| E2E MoT, r32 + full SigLIP, KI off, no EMA | 4 | 18.22 | 3.7 (co-tenant contaminated) | P1 out_tput2.json |
| E2E MoT, r32 + full SigLIP, KI on (+AQ16), no GPU EMA | 8 | 20.53 | 5.05 / 6.33 | P4 mem_probe B_b8_noema |
| Insulated, same scope | 8 | 20.47 | 5.40 / 6.96 | P4 C_insulated |
| E2E + fp32 EMA on GPU | 2 / 4 | 20.46 / OOM | – | P1; P4 OOM at 4 and 8 |
| E2E r64 + 25 % EMA | 4 / 8 | 19.89 / 21.10 | – | P2 e2e_r64_adam_ema025.json |
| P3 native head, SigLIP trained, GPU EMA | 8 | 16.13 | 4.9 raw (6.3 calibrated) | P3 measurements.json |
| **Recommended (r32, KI on, CPU EMA)** | 4 | **≈18.5** (derived) | **≈4.3 planned (range 3.7–5.0)** | derived (see below) |

- **How ≈18.5 GiB was derived.** P1's 18.22 plus the KI block (~0.2–0.3). Cross-check: P2's 19.89, minus the 25 % EMA (0.78), minus r64→r32 (0.58).
- **Planning throughput.** P4's 5.05 at micro-batch 8 × ~0.85 for micro-batch 4. Per 2-GPU run, with ≤5 % all-reduce (3.1 GB fp32 gradients over PCIe without P2P, mostly overlapped; the 1.4 GB LoRA run showed no measurable overhead) that gives ≈8.2 samples/s.
- **Per-run cost.**
  - U arm: 1.28 M samples ≈ 43.5 h ≈ 87 GPU-h, +3 GPU-h evals → **~90 GPU-h**.
  - I arm: **~85 GPU-h** (P4's insulated arm is 7 % faster).
  - Arms without self-past (C1, C2, C3): **~72 GPU-h**.
  - Micro-batch 8, if it passes the probe: U ≈ 73 GPU-h.
- **Host.** The CPU EMA needs ~6 GB of host RAM per run (shadow plus pinned staging); the host has 503 GB. Disk: ~26 GB per run (12.5 GB resume checkpoint + 8 × 1.6 GB snapshots) plus 14.5 GB pi05_base, against 213 GB free.
- **Evaluation.** 500 episodes take 37–40 min (0.67 GPU-h) with 10 envs; policy 8.8–9.6 GB plus renderers 5.5 GB. A dev-200 run takes about 16 min.

## 3. Experiment ladder

**Schedules.** `--n-parallel-envs 10` everywhere (D6). EGL renders on the evaluation GPU; for evaluate_candidate.py set `MUJOCO_EGL_DEVICE_ID` to the driver minor, which is reversed against nvidia-smi on this host.
- **official:** `--protocol official --n-test 500 --seed 44 --episode-start-seed 3000`. Used only at the pre-registered snapshot.
- **test-B:** `--protocol corrected --n-test 500 --seed 46 --episode-start-seed 5000` (the repo's "corrected final" schedule). Pre-registered secondary, used only at the pre-registered snapshot.
- **dev:** `--protocol corrected --seed 45 --episode-start-seed 4000`; 200 episodes for curves, 500 for decoding decisions. The only schedule ever used for choices.
- **Pre-registered endpoint per arm:** EMA weights at update 40k, greedy, k=8.

**Power.** Backing discordance out of the logged McNemar results gives 30–41 % between snapshots and runs (VA 40k vs 70k: Δ 2.8 pp, p = 0.34 → ~186/500 discordant; LoRA vs VA at 30k: 199/500). Exact McNemar (scratchpad/design-synthesis/mcnemar_power.py):

| Pairs | Smallest significant observed Δ | Δ for 80 % power | Power for a true +5 pp |
|---|---|---|---|
| 500 | 5.2–6.0 pp | 7.2–8.2 pp | 40–50 % |
| 1000 (official + test-B) | 3.6–4.2 pp | 5.0–5.7 pp | – |

So decisions use the pooled 1000 pairs, the official-only result is reported alongside, and 2–4 pp effects are replicated with a second training seed. Training-seed variance is not captured by McNemar.

**Phases.** A = steps 4–6, about 45 h wall on GPU pairs {0,1}, {2,3}, {4,5}, with GPUs 6–7 for dev curves and probes. B = steps 7–9 or 10, another ~45 h. Through phase B: ≈ 500 GPU-h, +175 if a replication is triggered.

| # | Step | Change | Purpose | Control | GPU-h | Decision rule |
|---|---|---|---|---|---|---|
| 0 | F0: fixes, guards, tests | B1 (yaml:39/:41), B2, B4; coupling flag off by default; λ=0 guards; CPU EMA; `lr_cosine_horizon`; attribution probe; eval flags; tiny-model tests | Make every later number trustworthy | Today's code: bitwise logits and gradients on the default path; the 247-test suite | 0 (CPU) | Proceed only when 247/247 plus the new tests pass and defaults are bitwise identical |
| 1 | R0: re-baseline (eval only) | Official 500 for VA upd-10k/20k/30k/40k/70k (live), LoRA upd-020000_ema (last pre-D1 snapshot), the 0.772 checkpoint at T=1/top-10 and at greedy, and 046 ep4 if recoverable. VA@40k also on test-B, plus dev-200 at 10k–40k | One protocol for every reference (D4/D6/D7). Fix the bar before E1 | Logged in-training numbers | ~7 | Re-scored numbers replace logged ones. Bar = best verified non-VLA official score; the 0.858–0.892 records stay a stretch target until re-scored. If VA@40k moves >5 pp from 0.706, D6 is material and no comparison may mix env counts |
| 2 | R1: decode probe (dev only) | VA@40k on dev 500: T=1/top-10; k=4; zero history | Freeze deployment decoding before any test evaluation (C5) | VA@40k greedy, k=8, same dev 500 | ~2 | Change decoding only if the alternative wins by ≥5 pp with p < 0.05. Otherwise greedy k=8 is pre-registered. Record VA's zero-history drop |
| 3 | P0: probe | `--probe` for the U config on a NUMA-local pair at micro-batch 4 and 8 (p=0.5 plus worst-case p=1 pass); 3.1 GB NCCL all-reduce benchmark | Choose micro-batch; pin GPU-h | P1 18.22 GiB @4; P4 20.53 GiB @8 | ~1 | Micro-batch 8 × accum 2 only if ≤22.5 GB reserved and ≥2 GB headroom on both ranks; else 4 × 4. bf16 comm hook if the all-reduce adds >0.5 s per update |
| 4 | E1-U (phase A) | Coupled K/V, α 0→1 over updates 2k–4k; λ=1; scope `vision_lora_action`; recipe 2.5 | Does letting the P2N loss into the VLM raise SR? | E1-I, paired | ~90 | E1 rule below |
| 5 | E1-I (phase A, concurrent) | Identical, with `kv_coupling_max=0` (plain detach) | Matched insulated control; anchor to VA | VA re-scored (live weights, same LR curve) | ~85 | If I-live@40k < VA@40k − 6 pp (p < 0.05, 1000 pairs), the new code or recipe regressed: stop and debug before reading E1 |
| 6 | C1: flow head (phase A, concurrent) | `PI05FlowVisionActionPolicy`: pi05 flow expert (unfolded adaRMS, 10 Euler steps, horizon 16, execute 8). Same scope, coupling, λ=1 and recipe; no history, no self-past | Should P2N be the executor at all? (D3) | E1-U | ~72 | Flow ≥ U + 4 pp (p < 0.05): the executor should be continuous; OAT/P2N becomes a history-conditioned token auxiliary and history moves into the flow expert. U ≥ flow − 2 pp: the P2N head is justified. In between: decide with C2 and cost. If E1 adopts insulation, rerun flow insulated (C1′, ~72) before deciding |
| 7 | E2 = U0 (phase B) | Coupled stack with λ_ki=0: no KI block, table frozen, layer-17 q/o/MLP LoRA frozen, VLM pass with grad | Is KI still needed once the action loss reaches the VLM? (P1 vs P2/P4) | E1-U | ~88 | Drop KI if U0 ≥ U − 2 pp and U0's zero-history and prompt-swap probes are not worse by >5 / >20 pp. Keep KI if U − U0 ≥ 4 pp (p < 0.05). If E1 was null with action share < 0.15, U0 vs I decides whether coupling helps at all |
| 8 | C2: no past (phase B) | Adopted stack; `use_past=false`, `self_past_p=0` | Does past-command conditioning still pay inside a VLA? (non-VLA: +14.6 pp paired) | Adopted arm | ~72 | Keep history if adopted − C2 ≥ 4 pp (p < 0.05). If ≤2 pp, history is unsupported in the VLA: drop RAWDIFF and self-past, and the case for P2N rests on C1 |
| 9 | C3: ground-truth past (phase B) | Adopted stack; `self_past_p=0` | Is self-past worth its +25–29 % compute? | Adopted arm | ~72 | Keep if adopted − C3 ≥ 3 pp; drop if ≤1 pp; otherwise keep as the default |
| 10 | S43: replication (conditional) | Seed 43 for both arms of any decision with +2 ≤ Δ < +4 pp, or Δ ≥ +2 with p ≥ 0.05 | Training-seed variance (unmeasured for the VLA) | The seed-42 pair | ~175 per pair | Adopt only if the sign agrees in both seeds and the pooled 2000-pair exact McNemar has p < 0.05 |
| 11 | L1: language and copycat probes (eval only) | Final snapshots of U, I, U0, C1: prompt swap 30↔31 (both LIVING_ROOM_SCENE2; success on the env's own task under the other task's instruction = instruction-ignoring rate); zero history on dev 500; offline shuffle-past vs shuffle-obs retention | Costs that LIBERO-10 SR cannot show | Same snapshot, true prompt, full history | ~6 | Report. Disqualify an arm only if its instruction-ignoring rate is ≥20 pp above its control's with no SR gain |
| 12 | C4: random-init expert (phase C) | Adopted stack; `expert_init=random` | Price the pi05 expert prior (C3) | Adopted arm | ~90 | Within ±3 pp: the prior is unnecessary, run H1. ≤ −5 pp: keep the fold |
| 13 | H1: native head (phase C) | P3's 6×256 cross-attention decoder over 32 bidirectional readouts (taps 9/18), at the adopted gradient regime and KI choice, with the λ=0 guards | Head isolation at identical VLM scope; memorisation; cost | Adopted MoT arm | ~50 | Adopt native if ≥ MoT − 2 pp (about half the GPU-h, lower latency). Keep MoT if MoT ≥ native + 4 pp |
| 14 | A1–A6 (phase C, one factor each) | A1 λ_eq (only if E1 is null with share < 0.15 and U0 > I). A2 greedy self-past. A3 AQ16. A4 past dropout q=0.1 (`past_dropout_p`). A5 SigLIP 2e-5 (only if dev SR falls ≥5 pp from its peak). A6 extend to 70k by resume (only if the dev slope 30k→40k is ≥ +3 pp) | Single-factor refinements | Adopted arm | ~90 each | Adopt at ≥ +4 pp pooled with p < 0.05. A2 and A4 also need `self_past_mae` not to collapse |

**E1 rule (pre-registered).** Endpoint as above, on 1000 pairs (official + test-B), exact McNemar on matched (schedule, episode_index):
- **Adopt coupling** if Δ(U − I) ≥ +4 pp, p < 0.05, and the official-only Δ is positive.
- **Replicate (step 10)** if +2 ≤ Δ < +4, or Δ ≥ +2 with p ≥ 0.05.
- **Keep insulation** if Δ < +2. It is ~7 % cheaper and leaves the pretrained VLM pathway alone. Exception: if U0 later beats I by ≥4 pp, coupling helps and KI was masking it.
- **Abort U early only on sustained signals:** the 200-update mean of train L_KI rises by >0.5 nats within 2k updates of the ramp start, or dev-200 SR at 10k is more than 10 pp below I's. Then restart with α_max = 0.25 (~90 GPU-h). Never abort on single gradient-norm spikes.

**Expected outcomes (hypotheses, official protocol):**
- **R0.** VA@40k with 10 envs: 0.66–0.74 (D6 effects are zero-mean on K4/K6/K8). The 0.772 checkpoint: 0.68–0.80 (inflated by best-epoch selection, deflated by the legacy first-chunk swallow).
- **R1.** T=1 deploy: −8 to +2 pp (modern AR −7.4 pp; VA's held-out CE of 7.33 at accuracy 0.089 implies confident wrong codes). k=4: −3 to +1 pp.
- **E1-I.** 0.68–0.76, central 0.72 (VA 0.706, plus 0–3 from KI-trained LoRA r32 and 0–3 from EMA).
- **E1-U.** 0.70–0.82, central 0.75. Δ from −3 to +10, central +3, with ~40 % chance that |Δ| < 2.
  - For: every controlled small-data comparison favours adapting the backbone with the action loss (VLA-Adapter frozen −8.6; OpenVLA frozen vision −22.7; FLOWER 2.65 vs 4.44), and openpi's pi05_libero uses no stop-gradient.
  - Against: the KI specialist on LIBERO-10 is at parity with pi0 (84.8 vs 85.2); KI already trains the VLM on the same targets and dominates the VLM-side gradient by norm; and history explains much of the AR target (first-token CE 2.94 vs 5.15).
  - Gains should land on K3/K8/S1/L5/L6. K8 stays capped near 0.5–0.6 by the 550-step budget (demos average 416 steps; VA successes 426).
- **C1.** 0.70–0.88, central 0.78 (literature: a discrete head costs 5–10 points; openpi's 92.4 used batch 256, four suites and 520 steps).
- **U0.** 0.64–0.80, central 0.72.
- **C2.** 3–15 pp below the adopted arm, central −8.
- **C3.** 0–6 pp below the adopted arm, central −2.
- **Offline.** Held-out CE rises in every arm (D5) and is not read.

## 4. Code change map

| File:line | Change |
|---|---|
| oat/config/train_p2n_vla.yaml:39, :41 | B1: `spm_path` and `tokenizer_checkpoint` point to repo-local data/pretrained/p2n_vla/{paligemma_tokenizer.model, oat_from_upd030000.ckpt}. Same root fix in docs/P2N_VLA.md and the parity script. Turns the 15 red tests green |
| oat/env_runner/{byte_observation_policy, executed_action_common, state_history_common}.py | B4: commit together with the modified runners |
| oat/policy/p2n_vla_common.py:86-96, :133, :249-260 | Constructor args `kv_coupling_max` (default 0.0), `kv_coupling_start`, `kv_coupling_ramp`, `lora_lr_group` (default 'pretrained'); stored and recorded in `_construction` so export/restore round-trips them |
| p2n_vla_common.py:202-205 | λ=0 always freezes the KI table, but freezes LoRA only when insulated. New `_freeze_unreachable_vlm()` (sketch 2) |
| p2n_vla_common.py:431-433 | `_detached_kv` becomes an instance method applying α(t). The name is kept so pi05_ki_flow.py:62/83 inherit it. New `coupled` property and `kv_grad_scale()` |
| p2n_vla_common.py:503-536 | Log `self_past_mae` |
| p2n_vla_common.py:545 | B2: set `self._last_self_past_rows = 0` before the history branch |
| p2n_vla_common.py:557-561 | VLM pass runs with grad when `use_ki or self.coupled` |
| p2n_vla_common.py:566-581 | `return_losses=True` path for the probe; add `kv_grad_scale` to the logged components |
| p2n_vla_common.py:595-597 | `_is_pretrained_group` honours `lora_lr_group` (instance-level) |
| oat/policy/p2n_vla_vision_action.py:28-46 | Scope `vision_lora_action` (re-enable LoRA after L43, then `_freeze_unreachable_vlm()`); the λ check at 31-33 allows λ=0 when coupled |
| p2n_vla_vision_action.py:66-73, :85-92, after :100 | Instance-level group predicate. `_is_ki_clip` unchanged: under coupling, 'ki' = the VLM side. Metadata gains `language_trainable` and `vision_gradient_source` ∈ {ki, ki+action, action}. New class `PI05FlowVisionActionPolicy` |
| oat/policy/pi05_ki_flow.py:57-60 | Same VLM-grad gate as p2n_vla_common.py:560; lines 62/83 unchanged |
| oat/model/common/trainable_ema.py:40-66, :114-115, :133-146, :150-198 | Pinned host shadows; `update_every=k` with decay^k; `step` still counts every call, so the resume check at train_p2n_vla.py:1606 holds; `swap_in` stashes live values on the host |
| oat/workspace/train_p2n_vla.py:98, :618-620, :1565 | `lr_cosine_horizon`: added to `RESUME_TRAINING_KEYS`; when set, the `max_optimizer_steps` equality check is skipped so the stop can move on resume |
| train_p2n_vla.py:1553-1556 | `TrainableEMA(..., device=cfg.ema.device, update_every=cfg.ema.update_every)` |
| train_p2n_vla.py:1017-1040 | Gradient-attribution probe every `grad_probe_every` successful updates (main process); logs `attr/*` |
| train_p2n_vla.py:101-103 | Add the new configs to `SOURCE_FILES` (provenance hashes) |
| scripts/train_p2n_vla_vision_action.py:27-36, :46-52, :86, :207-209, :220 | E2E variant table (p2n_vla → train_p2n_vla_e2e; pi05_ki_flow → train_pi05_flow_e2e with `PI05FlowVisionActionPolicy`); accept the new scope and λ=0 when coupled; `--finetune e2e`; report the actual scope |
| oat/config/train_p2n_vla_e2e.yaml, train_pi05_flow_e2e.yaml (new) | Sketch 5. Arms are overrides: I = `policy.kv_coupling_max=0.0`; U0 = `policy.lambda_ki=0.0`; C2 = `policy.use_past=false policy.self_past_p=0.0 training.validate_generated_history=false`; C3 = `policy.self_past_p=0.0` |
| scripts/evaluate_p2n_vla.py:79, :107-108 | `--use-k-tokens` restricted to {1,2,4,8}; eval-only `--zero-history` and `--prompt-uid-override 30:31,31:30` |
| oat/workspace/train_p2n_new.py:163-166 | Per-episode action sha256 written to a new `dataset_split` field |
| tests/test_p2n_vla_e2e.py (new) | (a) Default path bitwise identical (logits and gradients) for p2n_vla, vision_action and pi05_ki_flow. (b) α=0: logits bitwise equal to detach, VLM AR-gradient exactly zero (not None), linear in α, never reaches the KI table. (c) Step-1 coverage for U, I, U0 and C1; λ=0 with insulation rejected in vision scopes. (d) CPU EMA with k=1 bitwise equals the GPU EMA; `swap_in` round trip is bitwise; EMA resume. (e) α schedule resumes from the self-past counter. (f) Launcher accepts the new configs |

No change needed: workspace `VARIANTS`, group names and `OPTIMIZER_KEYS`, and scripts/train_p2n_vla.py:38/52/198.

**Sketch 1: coupling (p2n_vla_common.py)**
```python
# __init__(..., kv_coupling_max=0.0, kv_coupling_start=2000, kv_coupling_ramp=2000, lora_lr_group="pretrained")
if not 0.0 <= kv_coupling_max <= 1.0 or min(kv_coupling_start, kv_coupling_ramp) < 0:
    raise ValueError("kv_coupling_max must lie in [0, 1]; start/ramp must be nonnegative")
if lora_lr_group not in ("pretrained", "new"):
    raise ValueError("lora_lr_group must be 'pretrained' or 'new'")
self.kv_coupling_max = float(kv_coupling_max)
self.kv_coupling_start, self.kv_coupling_ramp = int(kv_coupling_start), int(kv_coupling_ramp)
self.lora_lr_group = lora_lr_group          # all four also go into self._construction (L249-260)

@property
def coupled(self) -> bool:
    return self.kv_coupling_max > 0.0

def kv_grad_scale(self) -> float:
    """alpha(t), t = successful optimizer updates (the resume-checked counter, L232)."""
    if not self.coupled:
        return 0.0
    t = self.self_past_step - self.kv_coupling_start
    if t < 0:
        return 0.0
    return self.kv_coupling_max * (min(1.0, t / self.kv_coupling_ramp) if self.kv_coupling_ramp else 1.0)

def _detached_kv(self, prefix_out):         # was @staticmethod; name kept for pi05_ki_flow.py:62/83
    kv = prefix_out.kv
    if not self.coupled or not torch.is_grad_enabled():
        return tuple((k.detach(), v.detach()) for k, v in kv)
    a = self.kv_grad_scale()                # value bitwise k; d/dk = a; a == 0 gives zeros, never None
    return tuple((k.detach() + a * (k - k.detach()), v.detach() + a * (v - v.detach())) for k, v in kv)
```

**Sketch 2: λ=0 guards (p2n_vla_common.py:202-205, :557-561; pi05_ki_flow.py:60)**
```python
if self.lambda_ki == 0:
    self.ki_table.requires_grad_(False)                  # no KI block: the table is unreachable
    if not self.coupled:                                 # insulated, no KI: nothing reaches the VLM
        for p in lora_parameters(self.joint):
            p.requires_grad_(False)
self._freeze_unreachable_vlm()

def _freeze_unreachable_vlm(self):
    """Coupled, no KI: the expert reads layer-17 K/V only (10 LoRA tensors, verified on tiny)."""
    if self.coupled and self.lambda_ki == 0:
        top = self.joint.vlm.layers[-1]
        for m in (top.self_attn.q_proj, top.self_attn.o_proj,
                  top.mlp.gate_proj, top.mlp.up_proj, top.mlp.down_proj):
            m.lora_A.requires_grad_(False)
            m.lora_B.requires_grad_(False)

# forward (same edit in PI05KIFlowPolicy.forward)
with contextlib.nullcontext() if (use_ki or self.coupled) else torch.no_grad():
    prefix_out, layout, ki_logits = self.vlm_pass(prefix, targets if use_ki else None)
```

**Sketch 3: scope, LoRA LR group and flow control (p2n_vla_vision_action.py)**
```python
SCOPES = ("vision_action", "vision_lora_action")

def __init__(self, *args, training_scope="vision_action", vision_activation_checkpointing=True, **kwargs):
    if training_scope not in SCOPES:
        raise ValueError(f"training_scope must be one of {SCOPES}")
    lambda_ki = float(kwargs.get("lambda_ki", 1.0))
    coupled = float(kwargs.get("kv_coupling_max", 0.0)) > 0.0
    if not math.isfinite(lambda_ki) or lambda_ki < 0 or (lambda_ki == 0 and not coupled):
        raise ValueError("vision scopes need lambda_ki > 0 or coupled K/V: some loss must reach SigLIP")
    super().__init__(*args, **kwargs)
    ...                                                  # TrainableSiglipEncoder wrap (L39-40) unchanged
    self.joint.vlm.requires_grad_(False)                 # L43 unchanged
    if training_scope == "vision_lora_action":
        for p in lora_parameters(self.joint):
            p.requires_grad_(True)
        self._freeze_unreachable_vlm()

def _is_pretrained_group(self, name):                    # was @staticmethod; no new optimizer keys
    if ".lora_" in name:
        return self.lora_lr_group == "pretrained"        # 'new' -> new_module_lr = 1e-4
    return name.startswith("siglip.") or P2NVLACommonPolicy._is_pretrained_group(name)

class PI05FlowVisionActionPolicy(_VisionActionTraining, PI05KIFlowPolicy):
    """Same-stack flow control: pi05's flow expert under identical scope, coupling and KI."""
```

**Sketch 4: pinned-CPU EMA (trainable_ema.py)**
```python
def __init__(self, named_parameters, decay=0.999, warmup_power=None, device=None, update_every=1):
    ...
    self.host, self.update_every = device == "cpu", int(update_every)
    self.shadow = {n: (p.detach().float().cpu().pin_memory() if self.host
                       else p.detach().to(dtype=torch.float32, copy=True).contiguous())
                   for n, p in zip(names, parameters)}
    self._staging = {n: torch.empty_like(s).pin_memory() for n, s in self.shadow.items()} if self.host else None

@torch.no_grad()
def step(self, named_parameters):
    params = self._ordered(named_parameters)           # device equality check skipped when self.host
    decay = self.current_decay()
    self.updates += 1                                  # one call per update: resume check unchanged
    if self.updates % self.update_every:
        return decay
    decay = decay ** self.update_every
    if self.host:
        for n, p in zip(self._names, params):
            self._staging[n].copy_(p.detach(), non_blocking=True)
        torch.cuda.current_stream().synchronize()
        sources = [self._staging[n] for n in self._names]
    else:
        sources = [p.detach().float() for p in params]
    torch._foreach_lerp_([self.shadow[n] for n in self._names], sources, 1.0 - decay)
    return decay
# swap_in: when self.host, stash parameter.detach().to("cpu", copy=True) and parameter.copy_(shadow);
# restore with parameter.copy_(original). No 3.1 GB clone on the GPU.
```

**Sketch 5: workspace horizon, attribution probe and recipe**
```python
# train_p2n_vla.py:1565 — fixed LR curve independent of the stop (VA's curve; extendable by resume)
horizon = training.get("lr_cosine_horizon")
horizon = int(horizon) if horizon is not None else (
    int(max_steps) if max_steps is not None else schedule["planned_optimizer_updates"])

# in the train_step block (L1028-1037), main process only
every = int(cfg.training.get("grad_probe_every", 0) or 0)
if success and every and self.completed_optimizer_steps % every == 0:
    record.update(self._grad_attribution(run, batch))

def _grad_attribution(self, run, batch, rows=2):
    """||alpha g_action|| vs ||lambda g_KI|| on the VLM side; autograd.grad never fires DDP hooks."""
    policy = run.policy
    params = [p for n, p in run.named if n.startswith("siglip.") or ".lora_" in n]
    with torch.random.fork_rng(devices=[run.accelerator.device]):
        losses = policy(_take_rows(batch, rows), history_mode="expert", return_losses=True)
    g_a = torch.autograd.grad(losses["action"], params, retain_graph=losses["ki"] is not None, allow_unused=True)
    host = [torch.zeros(p.shape) if g is None else g.float().cpu() for g, p in zip(g_a, params)]
    del g_a
    n_a = float(sum((h * h).sum() for h in host)) ** 0.5
    out = {"attr/action_norm": n_a, "attr/kv_grad_scale": policy.kv_grad_scale()}
    if losses["ki"] is not None:
        g_k = torch.autograd.grad(policy.lambda_ki * losses["ki"], params, allow_unused=True)
        n_k2 = dot = 0.0
        for g, h in zip(g_k, host):
            if g is not None:
                c = g.float().cpu()
                n_k2 += float((c * c).sum())
                dot += float((c * h).sum())
        n_k = n_k2 ** 0.5
        out.update({"attr/ki_norm": n_k, "attr/cos": dot / (n_a * n_k + 1e-12),
                    "attr/action_share": n_a / (n_a + n_k + 1e-12)})
    return out
```
```yaml
# oat/config/train_p2n_vla_e2e.yaml
defaults: [train_p2n_vla_vision_action, _self_]
name: train_p2n_vla_e2e
policy:
  training_scope: vision_lora_action
  lora_rank: 32
  lora_alpha: 32.0
  lora_lr_group: new
  lambda_ki: 1.0                 # rung 7: 0.0
  kv_coupling_max: 1.0           # E1-I: 0.0
  kv_coupling_start: 2000
  kv_coupling_ramp: 2000
training:
  gradient_accumulate_every: 4   # max_train_steps resolves to 400 -> 100 updates/epoch
  num_epochs: 1000
  max_optimizer_steps: 40000
  lr_cosine_horizon: 100000
  use_ema: true
  val_every: 100
  max_val_steps: 400             # 3,200 fixed windows
  snapshot_every: 5000
  keep_resume_checkpoint: true
  grad_probe_every: 500
ema: {decay: 0.999, warmup_power: null, device: cpu, update_every: 10}
dataloader: {batch_size: 4}
val_dataloader: {batch_size: 4}
task: {policy: {lazy_eval: true}}
```

## 5. What not to do

- **Don't sweep λ_ki in an insulated design.** It is an on/off switch (C1: disjoint parameter sets, clip binding on 100 % of updates, AdamW scale invariance). Global clipping doesn't change that; only shared parameters do.
- **Don't run λ=0 with un-detached K/V without both guards.** Verified: 67 tensors get grad=None with the no_grad at :560 left in, and 10 dead layer-17 LoRA tensors without the freeze. With SigLIP and LoRA both frozen, it silently becomes a frozen-VLM MoT.
- **Don't judge coupling against the old VA run alone** (P1's E1). That comparison changes stop-gradient, KI and LoRA together. Use the fresh matched I arm.
- **Don't select snapshots, horizons or decoding on the official or test-B schedule** (D8), **or on held-out teacher-forced CE** (D5: Spearman with SR +0.75). Do not read held-out token accuracy as evidence about which loss teaches vision: it is 0.09–0.12 in every run.
- **Don't compare evaluations with different `n_parallel_envs`, or in-training against standalone** (D6: 147/500 episodes differ between 5- and 10-env runs on K4/K6/K8). Don't quote 0.772 or the 85.8–89.2 % records before they are re-scored (D7).
- **Don't resume across zarr identities or set `allow_dataset_change`** (D1: ~45/50 "validation" demos had been training demos).
- **Don't keep an fp32 EMA on the GPU at E2E scope** (OOM at micro-batch 4; P1, P4). Don't shard it with gathers inside main-process-only save code (train_p2n_vla.py:1025-1026, 1686-1690). Don't train at micro-batch 1: VA's 4.6 samples/s and 294 GPU-h were self-inflicted, since micro-batch 2 and 4 fit at 17.4 and 18.2 GiB.
- **Don't move past commands into the VLM prefix** (C6; RoboVLMs, HiF-VLA, HAMLET, VLANeXt −6.8). Don't add enriched history or the state gate (enriched ≈ raw, −0.019 n.s.; the gate saturates and was never trained, C4).
- **Don't parallel-decode the OAT tokens** (−7.7 nats per chunk of dependency in the proxy; OFT's gain was for independent bins). Don't decode with k ∉ {1, 2, 4, 8}.
- **Don't scale the AR head up.** The 155 M modern AR memorised (val CE minimum at ep26) and lost 12 pp paired from ep100 to ep300. The size sweep went 0.57 → 0.50 → 0.31, though confounded with block type.
- **Don't switch self-past to greedy or deployment to T=1 inside the decisive A/B** (C5 is low severity; T=1 deploy cost −7.4 pp).
- **Don't bundle AQ16, history dropout, SigLIP 2e-5, r64, batch 64 or full fine-tuning into E1.** None has in-repo evidence; each is its own rung. Don't go to FSDP2 full-parameter Gemma before LoRA E2E shows the backbone matters: OpenVLA r32 ≈ full, the pi0 LoRA study saturates at r32, and it is communication-bound on PCIe without P2P.
- **Don't co-locate jobs on training GPUs or span NUMA nodes.** Foreign renderers caused OOMs during the P1 and P3 probes.
- **Don't name the history-dropout key `history_*`.** The non-gate validator rejects it; use `past_dropout_p`.

## 6. Open risks and uncertainties

1. **E1 may come back null for structural reasons.** With λ=1, KI dominates the VLM-side gradient by norm (VA's KI norm 6–11 vs 1.5–3 for AR-into-VLM at init). The coupled arm may then be "mostly KI". Rung 7 and the attribution probe exist to tell a masked effect from a true null. Subjective chance of |Δ| < 2 pp: ~40 %.
2. **Statistical power.** Measured discordance of 30–41 % means a single 500-pair comparison detects only effects of ≥7–8 pp reliably; 1000 pairs gets to ~5–5.7 pp. Training-seed variance for the VLA is unmeasured, and real 3–5 pp effects need about 175 GPU-h of replication per pair.
3. **Coupling could damage or over-fit the VLM early.** The fresh head's gradients, plus faster memorisation (VA already hit 0.95/0.975 train accuracy), could hurt. The ramp, the KI canary and the dev curve detect this. The 40k endpoint is fixed in advance and a dev-curve peak is never "selected" on test.
4. **Memory margin.** ≈18.5 GiB at micro-batch 4 is derived, not measured with the KI block plus DDP. Micro-batch 8 sits 0.46 GB inside the gate. Throughput is uncertain by ±25 % and the all-reduce over PCIe without P2P is unmeasured; the probe settles both.
5. **C1 confounds head with history.** The flow arm has no past input. A flow win argues for a continuous executor; whether a flow expert with RAWDIFF history beats both is the next design question, not answered here.
6. **LIBERO-10 measures fit, not language or out-of-distribution behaviour** (LIBERO-Plus/PRO). KI-off or coupled arms could erode PaliGemma priors invisibly. The 30↔31 prompt-swap probe is coarse (100 episodes).
7. **The bar is uncertain.** If the 85.8–89.2 % non-VLA records verify, even the central E1 outcome (~0.75) leaves the VLA more than 10 pp behind a ResNet-18 P2N on 450 demos. The VLA's case would then rest on things LIBERO-10 does not measure, and the next lever is the head (C1/H1) or more data, not the coupling.
8. **Shared ceilings.** The 550-step budget caps K8. The OAT k=8 quantisation floor (xyz RMSE ≈ 0.047) is common to every AR arm.
9. **Untested knobs ride inside U.** The α ramp and LoRA at 1e-4 are untested. The LoRA LR is shared with I; the ramp is not. If U wins, whether the ramp was needed stays open (low priority).
10. **New code can fail silently.** CPU EMA, the probe and the new scope are new code paths. The bitwise default-path tests and the I-vs-VA anchor (−6 pp stop rule) are the defence.
11. **Known evaluation and offline biases remain.** D6 pairing restores comparability but not the "true" official K4/K6/K8 initial states. The D2 tokenizer leak biases offline metrics only.
12. **Asset logistics.** pi05_base (lerobot/pi05_base@b211f3d4, sha256 0eb11ca9…), the VA/LoRA snapshots (HF SeanLiu0272/ysk `outputs/training/...`), the 0.772 checkpoint and possibly the 046 checkpoint must be downloaded before R0. Without them nothing in the ladder can start.