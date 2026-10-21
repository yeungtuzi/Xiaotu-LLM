# MUSA / S4000 目标 OS

## 结论

```text
Ubuntu Server 22.04.4 LTS x86_64
Kernel 5.15.0-105-generic
Secure Boot: OFF
Python: 3.12 venv
```

不要使用 Ubuntu 24.04 / Debian 12/13 / 滚动发行版。

## 依据

- 摩尔线程 S4000 环境文档：
  https://docs.mthreads.com/vllm-musa/vllm-musa-doc-online/S4000/environment_setup/
  - Ubuntu 22.04.x LTS
  - kernel 5.15.0-105-generic
- vllm-musa Docker：
  https://github.com/MooreThreads/vllm-musa/blob/v0.28.0-dev/docker/musa.Dockerfile
  - `BASE_IMAGE=ubuntu:22.04`
  - MUSA apt source `ubuntu2204`
- MUSA 5.2 提供 cp312 wheel：
  `torch` / `torch_musa` / `triton` cp312 x86_64。

## 计算路径开关

```bash
GPU_DEVICE=MUSA   # 走 MUSA 线
GPU_DEVICE=CUDA   # 或不设，走 CUDA 线
```

`GPU_DEVICE=MUSA` 会自动启用 MUSA correctness 默认值；显式
`XTU_MUSA_CORRECTNESS=1` 仍然优先。

## Python 版本

- 我们的插件：`requires-python >=3.12`；
- 官方 vllm-musa Docker 默认 Python 3.10；
- 不要用官方 3.10 镜像直接装我们的 wheel；
- 在 Ubuntu 22.04 上另建 Python 3.12 venv，构建 vllm-musa + 我们的插件。

## 安装检查

```bash
lsb_release -a
uname -r
dpkg -s musa mtml sgpu-dkms mt-container-toolkit | grep -E 'Package|Version'
mthreads-gmi
python3 --version
```

目标：

```text
Ubuntu 22.04.x
5.15.0-105-generic
Python 3.12.x
```

## 内核锁定

```bash
sudo apt-mark hold linux-image-5.15.0-105-generic \
  linux-headers-5.15.0-105-generic \
  linux-image-generic linux-headers-generic
```

## BIOS

- Above 4G Decoding: On
- Resizable BAR: On
- IOMMU: On（多卡建议）
- Secure Boot: Off
