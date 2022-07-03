# Enhanced Instant Neural Graphics Primitives

This repository tries to add more function supports to [instant-ngp](https://github.com/NVlabs/instant-ngp), but it also removes some unconcerned parts (e.g. we have carefully removed all the related primitives of SDF, Image and Volume).

This repository is still written in CUDA C++ so as to keep its efficiency.

**[Under Active Development]** Don't use it unless you are familiar with CUDA programing as well.

# Performance Overview

**LEGO setting:**
- train on trainset with 100 imgs
- test on testset with 200 imgs
- view-independent color (no dir encoding)
- on a single 3090

average training speed: `157+ steps/sec`

| TrainSteps | 200    | 1000   | 2000   | 10000  | 50000  |
| :--------: | :----: | :----: | :----: | :----: | :----: |
| time(sec.) | 1.5    | 6.2    | 12.3   | 61.3   | 318.1  |
| PSNR       | 25.568 | 30.909 | 32.122 | 33.565 | 33.970 |

# What's Changed Compared to the Official Implementation?

**Only Support/Allow:**
- only NeRF is available
- no GUI
- build on Linux with Python
- no DLSS, no OptiX

**What is New?**
- better support envmap
- support two densifying losses
- simplify the network arch (remove directional view encoding)

# Installation

```bash
git clone git@github.com:Karbo123/instant_nerf.git --depth=1
cd instant_nerf && git submodule update --init --recursive
source build.sh
```
