# 📚 Research Chronicles

A running log of research paper breakdowns — written for people who want to understand the *why*, not just the *what*.

Each note pairs with a hand-annotated PDF of the original paper. The writing aims to be precise enough for practitioners and readable enough for anyone adjacent to the field. For the full depth, the annotated paper is always linked.

Started as a birthday resolution. Target: **6 ~ 8 papers a month**.

---

## 🌱 Current Focus

Building a computer vision pipeline for [PlantCLEF 2026 @ CVPR](https://www.kaggle.com/competitions/plantclef-2026) — a multi-label plant species identification challenge with a hard domain shift between training and test data. The papers below are being reviewed in direct service of that goal, working towards a novel solution combining self-supervised pretraining, semantic region proposals, and taxonomic multi-head classification.

The broader aim: specialise in **CV and multimodal AI**, with a longer horizon towards embodied and robotic systems.

---

## 🦕 The Papers

### Self-Supervised Vision

| Paper | Notes | Annotated PDF | Status |
|---|---|---|---|
| DINO — Self-Distillation with No Labels | [DINO.md](./DINO.md) | [PDF](https://drive.google.com/file/d/1-wVGG8KBoI8JIauGuukhrhMe3jGR_yNY/view?usp=sharing) | ✅ Done |
| DINOv2 — Learning Robust Visual Features without Supervision | [DINOv2.md](./DINOv2.md) | — | 🔄 In Progress |
| MAE — Masked Autoencoders Are Scalable Vision Learners | [MAE.md](./MAE.md) | — | 🔜 Up Next |

### Segmentation & Detection

| Paper | Notes | Annotated PDF | Status |
|---|---|---|---|
| SAM — Segment Anything | [SAM.md](./SAM.md) | [PDF](https://drive.google.com/file/d/1yNiL3bdRW2va_h2EiVQgv9J3LYZMXIci/view?usp=sharing) | ✅ Done |
| Grounding DINO | [GroundingDINO.md](./GroundingDINO.md) | — | ⏳ Queued |
| DETR — End-to-End Object Detection with Transformers | [DETR.md](./DETR.md) | — | ⏳ Queued |

### Multimodal Systems

| Paper | Notes | Annotated PDF | Status |
|---|---|---|---|
| CLIP — Learning Transferable Visual Models from Natural Language | [CLIP.md](./CLIP.md) | [PDF](https://drive.google.com/file/d/1N51JLIitDam4qNPLqiHJ7jJI1TU42N-V/view?usp=sharing) | ✅ Done |
| BLIP-2 | [BLIP2.md](./BLIP2.md) | — | ⏳ Queued |

---

## 🗂️ How Each Note Is Structured

Every markdown file follows the same loose template:

- **What is it** — one paragraph, no jargon, honest about what the paper actually does
- **The ideas it builds on** — prerequisites made explicit rather than assumed
- **How it works** — the mechanism, including the maths where it earns its place
- **The results and why they're surprising** — what actually came out of this
- **Annotated PDF** — where the real detail lives, including margin notes on implementation, open questions, and things worth revisiting

---

## 🔗 Related Work

Competition notebook and pipeline code → *coming soon*

Writeups → [LinkedIn](https://www.linkedin.com/in/godugu-anil-himam-040158170/)

---

*Reviews are written while building, not after. Expect the understanding to be honest about what's clear and what isn't.*
