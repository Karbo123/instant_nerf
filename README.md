# PyTorch Implementation of Instant Neural Graphics Primitives (unofficial)

This repository tries to exactly reproduce the performance of instant-ngp in terms of both accuracy and speed, but it may only involve the **Neural Radiance Field (NeRF)** part, because NeRF is becoming increasingly popular among the research community.

**Work in Progress**

# What's Changed Compared to the Official Implementation?

TODO
- [x] remove `GUI`
- [ ] only support `Blender` dataset loading

# Installation

```bash
git clone git@github.com:Karbo123/pytorch_instant_ngp.git --depth=1
cd pytorch_instant_ngp
git submodule update --init --recursive
ln -s `pwd`/third_party/instant-ngp/dependencies `pwd`/model
cd model && source build.sh
```

