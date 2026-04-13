# SAM 🔪 — Segment Anything Model

**Paper:** [Segment Anything](https://arxiv.org/abs/2304.02643) — Kirillov et al., Meta AI Research (FAIR), 2023

**Annotated PDF:** [SAM.pdf](https://drive.google.com/file/d/1yNiL3bdRW2va_h2EiVQgv9J3LYZMXIci/view?usp=sharing)

**Part of a series:** [DINO](./DINO.md) → [DINOv2](./DINOv2.md) *(in progress)* → SAM — building towards a competition pipeline for [PlantCLEF 2026 @ CVPR](https://www.kaggle.com/competitions/plantclef-2026).

---

Unlike DINO 🦕, SAM does exactly what it says on the tin 😅 — but what it says is remarkably ambitious: a **foundation model for Computer Vision**. Where DINO gave us emergent properties through self-supervised training, SAM takes the next step by introducing a novel data creation engine, a promptable segmentation task, and a simple yet powerful architecture that composes cleanly into larger systems.

---

## 👀 What is it?

SAM is a compilation of proven techniques combined with a novel model-in-the-loop data engine. At its core it is a **promptable segmentation model** — given any prompt (a point, a box, a mask, or text), it returns a valid segmentation mask. At the time of release this approach was deeply inspired by the generalisation achieved in NLP for unseen domains with pre-trained models (GPT, BERT) through prompt engineering. Its zero-shot transfer performance is often competitive with or superior to fully supervised methods on tasks it was never explicitly trained for.

---

## 🏗️ The Architecture

SAM follows a deliberately (& deceptively 😅) simple divide-and-conquer design, separating concerns across three components to enable ~50ms inference in a web browser.

### Image Encoder

A heavyweight MAE-pretrained ViT that generates rich high-dimensional image embeddings. The critical engineering insight: **it runs once per image and the embedding is cached**. Since the image is fixed while prompts vary, amortising this cost across multiple prompts is what makes real-time interactive use possible. A native fusion model would require re-encoding for every new prompt which is prohibitively expensive.

### Prompt Encoder

The prompt encoder has one job: turn multiple heterogeneous input types into a single uniform embedding space for the mask decoder. It handles three prompt categories differently:

**Sparse prompts (points and boxes):**
- Suffer from low native dimensionality — a coordinate pair is just two numbers. Passing raw coordinates to a DNN induces spectral bias: by default DNNs fit low-frequency functions, producing smooth, generic segmentation rather than the fine-grained boundaries SAM needs. The fix is **Fourier Feature encoding**:

$$f(v) = [\sin(2\pi Bv), \cos(2\pi Bv)]$$

- where `B ∈ ℝ^{d×2}` is a matrix of frequencies sampled from a Gaussian `B ~ N(0, σ²)`. This lifts the low-dimensional input into a high-dimensional space covering a wide spectrum of frequencies. `σ` controls the bandwidth — small `σ` gives smooth embeddings, large `σ` gives granular spatial detail. Crucially, `B` is **fixed at initialisation** and never resampled. This retains the embedding space keeping it consistent across inputs.

- After Fourier encoding, a **learned type embedding** (one per prompt type: foreground point, background point, box top-left, box bottom-right) is added to encode semantic identity. The sum of spatial encoding and type embedding forms the **sparse token** which is provided as input to the decoder.

**Dense prompts (masks):** 
- They are processed by a lightweight trainable CNN that generates feature maps matched to the spatial dimensions of the image embedding, which are then added element-wise.

**Text prompts:** 
- Utilise an off the shelf frozen CLIP text encoder. These are described in the paper as a proof-of-concept (future experimentation) when the paper was released. The model does not fully expose this pathway.

### Mask Decoder

The mask decoder is SAM's core contribution. It is a modified transformer decoder with **bidirectional attention** making two passes of cross-attention in both directions (prompt-to-image and image-to-prompt) interleaved with self-attention across all tokens. This lets the prompt tokens query spatial regions of the image embedding (what ?) while the image embedding simultaneously updates based on the prompt context (where ?).

Before entering the decoder, a **learned `<output>` token** is prepended to the prompt sequence. Its sole purpose is to aggregate all attention-driven interactions across both passes and produce the final logits. The output token's final state is passed to a lightweight MLP and then to the mask prediction head.

### Mask Prediction Head

A lightweight linear classifier applied to the `<output>` token's logits. It computes mask foreground probability at each image location, followed by bilinear upsampling to restore full image resolution.

**Resolving ambiguity:** A single prompt is often ambiguous (paper eg: a click on a shirt could mean the shirt or the person). SAM resolves this by predicting **three masks per prompt**: whole object, part, and subpart. During training only the minimum loss across the three masks is backpropagated (efficiency and noise reduction). Each mask is accompanied by a predicted IoU confidence score for ranking.

---

## 🧮 The Loss Functions

SAM supervises mask prediction with a linear combination of two losses that together provide a complete picture for dense spatial prediction:

**Focal Loss**: 
- A specialised extension of cross-entropy that exponentially reduces the contribution of easy predictions during training, forcing the model to focus on hard cases:

$$FL(p_t) = -(1 - p_t)^\gamma \log(p_t)$$

- where `p_t` is model confidence on the correct class, `(1-p_t)^γ` is the modulating factor (γ tunable, typically 2), and the loss operates at the pixel level.

**Dice Loss** 
- A region-based loss analogous to IoU, robust to class imbalance:

$$\text{Dice} = \frac{2|GT \cap Pred|}{|GT| + |Pred|}$$

Focal loss ensures pixel-level precision. Dice loss ensures region-level overlap. Neither alone is sufficient. Thus, together they cover both the local and global objectives of segmentation.

---

## ⚙️ The Data Engine — Model in the Loop

The revolutionary contribution behind SA-1B (1.1B masks, 11M images). Three stages:

**Stage 1 — Cold Start (Assisted Manual):** The MAE-pretrained ViT, bootstrapped on existing public segmentation datasets, works alongside professional annotators who use a browser-based tool powered by SAM to label masks. The model learns what segmentation means from a small supervised foundation.

**Stage 2 — Weakly Supervised + Self-Training:** The grunt work stage 💪. The model uses its Stage 1 params to generate masks automatically for a subset of objects. Human annotators focus on the remaining unlabelled regions and low-confidence edge cases flagged by the system. Model is thus retrained iteratively. The result: diversity of masks increases dramatically.

**Stage 3 — Fully Automatic:** The model now generates high-quality masks autonomously at scale. A 32×32 grid of foreground points is used to prompt SAM across each image, producing ~100 masks per image. Confident and stable masks are selected (stability = similar masks when thresholding at 0.5±δ), duplicates filtered via NMS. This stage produced the full SA-1B dataset.

*A note on Stage 3:* The fully automatic stage can be read as a form of self-distillation — the model generates its own training signal through its own predictions. This is the same philosophical thread running through DINO and MAE, now applied to annotation at scale.

---

## 🌍 Zero-Shot Transfer — The Real Result

SAM was trained on promptable segmentation. At inference time, via prompt engineering alone, it solves tasks it was never explicitly trained for:

- **Edge detection** — prompt with a dense grid, threshold low-confidence masks
- **Object proposal generation** — segment everything mode
- **Instance segmentation** — combine with an off-the-shelf detector as the box prompter
- **Text-to-mask** — proof-of-concept using CLIP image→text embedding alignment

This composability cements SAM as a reliable segmentation interface that other models can prompt — is its most lasting contribution. Much like CLIP enabled DALL-E, SAM enables any system that needs to go from "where should I look" to "here is the precise region."

---

*Currently integrating SAM into the pipeline for PlantCLEF 2026 @ CVPR — using SAM's region proposals as semantically-guided crops upstream of DINOv2 classification, replacing brute-force grid tiling. Next up: DINOv2 ⬆️*