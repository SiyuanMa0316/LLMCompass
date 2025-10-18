# Proteus: Achieving High-Performance Processing-Using-DRAM with Dynamic Bit-Precision, Adaptive Data Representation, and Flexible Arithmetic
[![Academic Code](https://img.shields.io/badge/Origin-Academic%20Code-C1ACA0.svg?style=flat)]() [![Language Badge](https://img.shields.io/badge/Made%20with-C/C++-blue.svg)](https://isocpp.org/std/the-standard) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![contributions welcome](https://img.shields.io/badge/Contributions-welcome-lightgray.svg?style=flat)]() [![Preprint: arXiv](https://img.shields.io/badge/cs.AR-2501.17466-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/pdf/2501.17466.pdf) 

## Overview

Proteus is the first hardware framework that addresses the high execution latency of bulk bitwise Processing-Using-DRAM (PUD) operations by implementing a **data-aware runtime engine** for PUD. Proteus reduces PUD latency in three complementary ways:

1. **Dynamic bit-precision:** Proteus automatically detects and exploits *narrow values* (e.g., many leading zeros/ones) to **lower the precision** of PUD operations on the fly.  
2. **Array-level concurrency:** Proteus **concurrently executes** independent in-DRAM primitives that belong to a single PUD instruction **across multiple DRAM arrays**.  
3. **Adaptive representation & arithmetic:** Proteus **chooses the most appropriate data representation and arithmetic uProgram** per PUD instruction—**transparently** to the programmer.

This repository contains the **simulation infrastructure** used to evaluate Proteus.

---

## What’s in this repository?

We **instrument 12 real-world applications** from multiple benchmark suites to use an **automated analytical model** (see `util/bbop_manager.*`).  
The analytical model is driven by our implementations using a **cycle-level, gem5-based simulator** from our prior works **SIMDRAM** and **MIMDRAM** (MIMDRAM repo: https://github.com/CMU-SAFARI/MIMDRAM).

The model automatically:
- **Identifies the target bit-precision** for each **bbop** instruction (a PUD instruction embedded in the application), and
- Selects the **best-performing, most energy-efficient uProgram** implementation for the required arithmetic by consulting **pre-computed cost models**.

> **Note:** Of the 12 instrumented applications, **11 are publicly included** here. We **could not publish the SPEC 2017 `x264` workload** due to copyright restrictions.

---

## Citation

Geraldo F. Oliveira, Mayank Kabra, Yuxin Guo, Kangqi Chen, A. Giray Yaglikci, Melina Soysal, Mohammad Sadrosadati, Joaquin Olivares, Saugata Ghose, Juan Gomez-Luna, and Onur Mutlu,  "[Proteus: Achieving High-Performance Processing-Using-DRAM via Dynamic Precision Bit-Serial Arithmetic](https://arxiv.org/pdf/2501.17466).”  
*Proceedings of the 37th ACM International Conference on Supercomputing (ICS),* Salt Lake City, UT, USA, **June 2025**.  

```bibtex
@inproceedings{proteus-ics25,
  author    = {Oliveira, Geraldo F. and Kabra, Mayank and Guo, Yuxin and Chen, Kangqi and Yaglikci, A. Giray and Soysal, Melina and Sadrosadati, Mohammad and Olivares, Joaquin and Ghose, Saugata and Gomez-Luna, Juan and Mutlu, Onur},
  title     = {Proteus: Achieving High-Performance Processing-Using-DRAM via Dynamic Precision Bit-Serial Arithmetic},
  booktitle = {ICS},
  year      = {2025}
}
```

---

## Repository Structure and Installation

We point out next to the repository structure and some important folders and files.


```
.
+-- README.md
+-- 2mm/
+-- 3mm/
+-- backprop/
+-- covariance/
+-- doitgen/
+-- fdtd-apml/
+-- gramschmidt/
+-- heartwall/
+-- kmeans/
+-- pca/
+-- util/
|   +-- bbop_manager.c  
|   +-- bbop_manager.h 
```

- Each application directory contains its **own README** with build and run instructions.  
- **All applications were compiled using `gcc-15`** in our evaluation (please check each workload README for any additional dependencies or platform notes).  
- **PIM versions** write a `bbop_statistics.csv` summarizing **execution time** and **energy consumption** per PUD (bbop) operation for **SIMDRAM** and **Proteus**.

> **Note on SPEC:** We could not publish the **x264** workload from SPEC 2017 since it is copyrighted.

---

## Getting Started (Quick Guide)

1. **Pick a workload** (e.g., `gemm/`, `kmeans/`, `backprop/`, etc.).  
2. **Read its README** for dataset generation (when applicable), build, and run commands.  
3. **Build** (typical):  
   ```bash
   make            # or: make CC=gcc-15
   ```
4. **Run** as described in each workload’s README.  
5. For PIM builds, inspect **`bbop_statistics.csv`** for per-operation time/energy.

> On macOS with Apple Clang, use Homebrew GCC or LLVM+`libomp` for OpenMP; see per-workload README notes.

---

## Getting Help
If you have any suggestions for improvement, please contact geraldo dot deoliveira at safari dot ethz dot ch.
If you find any bugs or have further questions or requests, please post an issue at the [issue page](https://github.com/CMU-SAFARI/proteus/issues).

## Acknowledgments
We acknowledge the generous gifts from our industrial partners; including in part by the ETH Future Computing Laboratory (EFCL), Huawei ZRC Storage Team, Semiconductor Research Corporation, AI Chip Center for Emerging Smart Systems (ACCESS), sponsored by InnoHK funding, Hong Kong SAR, and European Union’s Horizon programme for research and innovation [101047160 - BioPIM].
