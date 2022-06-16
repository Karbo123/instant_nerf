# PyTorch Implementation of Instant Neural Graphics Primitives (unofficial)

This repository tries to exactly reproduce the performance of instant-ngp in terms of both accuracy and speed, but it may only involve the **Neural Radiance Field (NeRF)** part, because NeRF is becoming increasingly popular among the research community.

**Work in Progress**

# What's Changed Compared to the Official Implementation?

**Only Support/Allow:**
- only NeRF (intentionally remove other primitives)
- no GUI (for fast experiment)
- build on Linux with Python (because most people use Linux's Python)
- no DLSS, no OptiX (useless, so remove them)

TODO
- [ ] ray marching (point location sampling + composite point info)
- [ ] use tiny-cuda-nn's encoding + network, and enable learning using torch's optimizer 
- [ ] remove internal c++ file loading; loading from torch.Tensor instead
- [ ] unified coordinate format (e.g. Blender only)

# Installation

```bash
git clone git@github.com:Karbo123/pytorch_instant_ngp.git --depth=1
cd pytorch_instant_ngp
git submodule update --init --recursive
ln -s `pwd`/third_party/instant-ngp/dependencies `pwd`/model
cd model && source build.sh
```

