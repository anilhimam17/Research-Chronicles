# DINO 🦕 — Self-Distillation with No Labels

**Paper:** [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294) — Caron et al., Facebook AI Research, 2021

**Annotated PDF:** [DINO.pdf](https://drive.google.com/file/d/1-wVGG8KBoI8JIauGuukhrhMe3jGR_yNY/view?usp=sharing)

**Part of a series:** DINO → [DINOv2](./DINOv2.md) *(in Progress)* → [SAM](./SAM.md) *(in progress)* — building towards a competition pipeline for [PlantCLEF 2026 @ CVPR](https://www.kaggle.com/competitions/plantclef-2026).

---

The fact that the authors turned their paper title into a dinosaur is legendary 😅 but the science behind it is anything but a joke.

## 👀 What is it?

DINO is a training technique that transforms Vision Transformers (ViTs) into models that are strong at segmentation, detection and classification without ever seeing a single labelled image during pretraining.

## The Marriage of Ideas

DINO doesn't invent from scratch. It brings together four existing ideas and makes them work in harmony:

- **Self-Supervised Learning:** The model creates its own training signal from the structure of the data itself — no human annotations needed.

- **Knowledge Distillation:** A teacher model guides a student model to emulate its representations, transferring knowledge without moving parameters.

- **BYOL:** Two identical networks; a student (trained via backprop) and a teacher (updated via EMA of the student's weights) — are trained with asymmetric objectives to prevent representational collapse. Unlike BYOL however, DINO removes the need for an extra predictor head in the student. Collapse prevention is handled more cleanly through centering and sharpening instead, which is discussed below.

- **Self-Distillation:** The teacher is simply a slow, stable version of the student — the model learns by distilling knowledge from itself. Hence the name: Self-Distillation with No Labels, or according to the authors 🦕

## 🤖 How does it actually work?

Both networks share identical backbone architecture. The student is updated aggressively via backprop — it's reactive and optimised for learning rich representations. The teacher is updated slowly via Exponential Moving Average (EMA) of the student's weights — it's stable and exists solely to provide consistent targets.

The key trick preventing collapse is a centering and sharpening tradeoff applied to the teacher's outputs:

- **Centering** c pulls the teacher's output towards a uniform distribution, preventing it from collapsing to a single dominant dimension. It's tracked via a running EMA over batch statistics — no learned parameters.<br>
$$c \leftarrow \lambda \times c + (1 - \lambda) \times \frac{1}{B} \sum_{i=1}^{B} g_{t}(x_{i})$$

- **Sharpening** via temperature $\tau$ scaling pulls the outputs towards a peaky distribution, giving the student a crisp, confident target to chase. A lower teacher temperature = sharper distribution = stronger learning signal.<br>
$$softmax\left( \frac{g_t(x_{i})}{\tau} \right)$$

They are in a constant tug of war where centering prevents overconfidence, sharpening prevents laziness. Together they keep training stable without any labels or contrastive pairs.

## 🧮 The Loss Function (the elegant bit)

DINO minimises the cross-entropy between the teacher and student output distributions:
$$L = -\sum p_t \times \log(p_s)$$

Here, `p_t` is the probability distribution output from the teacher model and `p_s` is the distribution output from the student model.

This has a beautiful property. When the student's prediction `p_s` perfectly matches the teacher's `p_t`, `log(p_s) → 0` and the loss vanishes. When they diverge, `loss → ∞`. The student is mathematically forced to genuinely emulate the teacher and not find shortcuts.

To make this work across scale, DINO uses multi-crop training (a computer vision 🪄 trick):

- A few **global crops** (>50% of image) go to both teacher and student.
- Many **local crops** (<50%) go to the student only.

The student must reconcile local texture with global structure — this is what builds rich, spatially coherent representations. The teacher sees only global views, ensuring its targets remain stable regardless of crop noise.

Cross-view consistency is enforced for a given image by swapping: the loss is computed as `H(g_t(v1), g_s(v2))` and `H(g_t(v2), g_s(v1))`, forcing the student to ignore view-specific noise. Where `g_t` is the teacher model and `g_s` is the student model.

## The Results — and why they're surprising

Here's what makes DINO remarkable: the model was never told what segmentation is. Yet the local-to-global generalisation enforced during training causes semantic segmentation to emerge spontaneously in the CLS token's attention maps.

**The model that never saw an annotation learned to segment. That's the real result!**

Beyond that:

- **Zero-shot image retrieval** matching KNN and linear probe performance on frozen embeddings.
- **Strong copy detection** even on distorted, printed or masked images.
- **Strong object detection** even on occluded images, also driven by local-to-global generalisation.
- **ImageNet top-1 accuracy** on par with or better than supervised methods — with a frozen backbone.

---

*Currently working through the DINO paper family as part of building a competition pipeline for PlantCLEF 2026 at CVPR.<br>Next up: DINOv2 ⬆️ and SAM.*