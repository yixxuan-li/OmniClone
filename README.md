# OmniClone

**OmniClone** is a robust, all-rounder whole-body humanoid teleoperation system designed to enable affordable, reproducible, and versatile robot control across diverse operator body shapes and complex task scenarios.

[[Paper](https://omniclone.github.io/resources/OmniClone.pdf)] [[Project Page](https://omniclone.github.io/)]

---

![Teaser](teaser.png)

---

## Overview

OmniClone takes a systematic perspective to develop an affordable, robust, and versatile whole-body teleoperation system. The system provides:

- **Bias mitigation** — identification and correction of distortions arising from retargeting and hardware fluctuations
- **Transformer-based whole-body control** — robust, affordable humanoid control with integrated optimizations
- **Multi-operator support** — stable operation across operator heights ranging from 147 cm to 194 cm
- **High-fidelity dexterous manipulation** — long-horizon task execution across diverse domestic environments
- **Autonomy policy training** — expert trajectory generation enabling downstream robot learning
- **Zero-shot motion execution** — diverse motion generation without task-specific training

---

## Release Timeline

| Date | Release |
|------|---------|
| TBD | Full teleoperation codebase and checkpoint |
| TBD | Full teleoperation codebase on-board version|
| TBD | Pre-collected demonstration datasets |
| TBD | Training code |

---

## Related Projects
- [Sim2Anything](https://github.com/Yutang-Lin/Sim2Everything): Seamlessly transfer motion from simulation to real humanoid robots. Bridge the Sim-to-Real gap with a single parameter change.
- [GMR](https://github.com/YanjieZe/GMR): Retarget human motions into diverse humanoid robots in real time on CPU.
- [Gen2Humanoid](https://github.com/RavenLeeANU/Gen2Humanoid): A complete Text-to-Motion pipeline for humanoid robots powered by HY-Motion-1.0.
- [Clone](https://humanoid-clone.github.io/): A related humanoid teleoperation system.
- [COLA](https://yushi-du.github.io/COLA/): A proprioception-only learning approach that enable human-humanoid collaborative object carrying.
- [LessMimic](https://lessmimic.github.io/): Long-horizon humanoid-scene interaction with unified distance field representations
---

## Citation

```bibtex
@article{omniclone2026,
  title   = {OmniClone: Engineering a Robust, All-Rounder Whole-Body Humanoid Teleoperation System},
  author  = {Yixuan Li and Le Ma and Yutang Lin and Yushi Du and Mengya Liu and Kaizhe Hu and Jieming Cui and Yixin Zhu and Wei Liang and Baoxiong Jia and Siyuan Huang},
  journal = {arXiv preprint},
  year    = {2026}
}
```

---

## Acknowledgements

We extend our sincere gratitude to Peiyang Li, Zimeng Yuan, Zhen Chen, Chengcheng Zhang, Yang Zhang, Nian Liu, Zhidan Liu, Zihui Liu and Mulin Sui, for their invaluable help in filming this demo.

This work is supported in part by the National Key Research and Development Program of China (2025YFE0218200), the National Natural Science Foundation of China (62172043 to W.L., 62376009 to Y.Z.), the PKU-BingJi Joint Laboratory for Artificial Intelligence, the Wuhan Major Scientific and Technological Special Program (2025060902020304), the Hubei Embodied Intelligence Foundation Model Research and Development Program, and the National Comprehensive Experimental Base for Governance of Intelligent Society, Wuhan East Lake High-Tech Development Zone.
