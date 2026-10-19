# v0.2.9 —— 针对 Qwen3.8-Flash-Next 的优化：专家层常驻

> 主题：**针对 Qwen3.8-Flash-Next(qfn) 进行优化，增加专家层常驻**。

## 一、新增能力

### 1. 专家层常驻(qfn)
* 把**整层的全部专家**一次性搬上 GPU：该层之后**不再有 D2H/CPU/H2D 往返**，decode 与 prefill 都直接用 ✓
* 常驻层由 `XIAOTU_MOE_GPU_RESIDENT_LAYERS` 在**加载期**选层；支持 `0-4,10` 语法 ✓
* 默认配置：**12 层精选** `15,9,11,16,31,3,39,25,7,23,24,13`
  （按"该层 CPU 时间 ÷ 活跃专家数"排序选出，见 `docs/EXPERIMENTS.md` B1053/B1055）✓
* 显存账：每层每 rank **1.41 GiB**；40GB 卡上 12 层是实测上限（再叠热专家池会挤爆 KV）✓

### 2. decode 专用 MoE 内核(qfn)
* 旧路径复用 **prefill 内核**(grid = `E×2I/128` ≈ 6,144 block/层)，decode M=1 时 64 倍算力空转 ✗
* 新增 **pair 中心**内核(`decode_moe_fp8.py`)：grid = `T×K`，gate/up 与 down 再按列/行块拆分 + 原子累加 ✓
* 与 CPU 反量化参考对拍 `rms_rel = 5.6e-7`；同等输入下比旧内核快 **2.7×**(T=1)✓

### 3. 引擎:零权重 pair 跳过
* `forward_many_m` 建映射时跳过 `weight==0` 的 pair，并按每 token 有效数推进 ✓
  这是"热专家命中能减 CPU 工作量"的前提（此前命中项仍被完整计算 ✗）

## 二、实测(qfn,统一口径:15K prompt / MBT 8192 / MTP 关 / TP=2 / GPU 预填开)

| 并发 | prefill (tok/s) | decode (tok/s) |
|---|---|---|
| 1 | **1,517** | **46.3** |
| 2 | **1,538** | **92.9** |
| 4 | **1,529** | **127.3** |

* decode 随并发 **2.75×** 扩展（步时 21.6 / 21.5 / 31.4 ms）✓
* 口径：`docs/BENCH_PROTOCOL.md` §6/§6.1（稳态解码窗口）/§6.2（必须充分预热）✓

## 三、修复

### 3.1 CUDA Graph 捕获崩溃(复发类问题，本轮定位到根因)

**现象**:GLM 在启动捕获阶段随机崩溃(3 次),签名 `cudaStreamCaptureInvalidated`,
报错表面落在 `result = out.to(hidden_states.dtype)`。

**根因**:CPU 预填心跳日志里有 `int(ids_i32[0,0])` —— 对 GPU 张量取 `int()` 就是 **`.item()`**,
会**同步设备**;一旦落在 CUDA capture 窗口内,捕获立即作废;
而错误**延迟到下一个无关 CUDA 调用**才抛出(本例 `out.to(dtype)`)⇒ 极易误判成"改配置改坏了" ✗

**修复**:
* 两处心跳**只输出 host 侧数据**(时间 / 层名 / qlen / 计数 / `memory_allocated()`),
  **不再读取任何 device 值** ✓
* 心跳外层加 `_in_capture_window()` 守卫(`is_current_stream_capturing()` 不可靠)✓
* 该心跳位于**共享 CPU 预填路径** ⇒ 修复对 **qfn / ds41f / GLM / MiMo 全部生效** ✓

**对"未来模型"是否可靠?** 分两层看:
* ⭐ **这一类**(捕获窗口内做 device→host 读取)已**从根上消除**:唯一在默认路径上的
  device 读取点被删掉,且写入了 `IRON_RULES.md` **R65 ④**(捕获窗口禁止 `.item()`/`int(t)`/
  `.cpu()`/`bool(t.all())`/`synchronize`,统计打印也只许 host 侧数据)✓
* ⚠️ **不能保证"永远不再犯"**:① 其它 env 打开的调试分支(计时/verify/pin 统计)里仍有
  `.item()`,开启时同样会作废捕获;② 捕获无效还可由别的非法操作引起(`cudaHostAlloc`、
  在捕获中创建 Stream/Event、`cudaMemcpyBatchAsync` 等 —— 最后一项本轮已实测不可用)✓
  ⇒ 交付了机械判据:`scripts/check_capture_log.py <log> --expect-capture`
  (捕获窗口内出现插件打印 ⇒ 必须回炉)✓
* **`-mavx512vbmi` 编译**：`_vbmi_on` 原只在 `#if __AVX512VBMI__` 内定义、却在宏外使用 ⇒ 修正作用域 ✓
* 记录：`cudaMemcpyBatchAsync` **非 capture-safe**(`cudaErrorStreamCaptureUnjoined`)，图模式不可用 ✓

## 四、口径与文档

* `docs/BENCH_PROTOCOL.md`：新增 §6(用户统一口径)、§6.1(稳态解码窗口 / 输出过短判样本不足 /
  扩展性必须用纯解码探针)、§6.2(必须充分预热)✓
* `README`：qfn 性能行按上述口径更新；GLM/ds41f/MiMo 行保持原值
  （GLM 同配置充分预热复测为**持平**，在 MTP 接受率噪声内，故不改）✓

## 五、已知限制

* 12 层常驻是 40GB 卡的上限；"全 GPU 速度"需 **80GB 级显存**或把 KV 挪走 ✓
* qfn 的 decode 并发扩展受 CPU MoE 引擎串行约束；进一步扩展依赖 **EP(0.3)** ✓
* 热专家池在 C=1 仍为净负（每层握手 + GPU partial 内核），当前默认关闭 ✓
