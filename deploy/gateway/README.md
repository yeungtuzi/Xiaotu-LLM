# vLLM 前置网关（:4000）—— 纯透传 + 虚拟 key + 按 key 计量

> 2026-11-17 建立。约束来自用户：「代理不要修改 vLLM 行为，就是做一个权限管理和统计」。

## 1. 它做什么 / 不做什么

| 做 | 不做 |
|---|---|
| 校验 `Authorization: Bearer sk-...`（只存 sha256，可吊销）| **不改写**请求体、**不注入**参数 |
| 把虚拟 key 换成上游 `VLLM_API_KEY` 转发 | **不改写**响应体、不重组 JSON |
| 读响应里的 `usage` 落 SQLite（按 key）| 不做计费/限流/重试/缓存 |
| 管理面：建 key / 列表 / 吊销 / 统计 | 不缓存、不排队 |

**已验证**：非流式与流式响应，经网关与直连**逐字节一致**（仅 `id`/`created` 这类服务端生成字段不同）。

## 2. 拓扑

```
客户端 ──(sk-xxx)──▶ gateway :4000 ──(VLLM_API_KEY)──▶ vLLM :8700
                        └── SQLite: /home/user/lvllm/gateway/gateway.db
```

## 3. 起停

```bash
bash /home/user/lvllm/gateway/start_gateway.sh start|stop|status
```

* 密钥文件（0600，不进仓库）：`<DIR>/upstream.key`（上游 key）、`<DIR>/master.key`（管理面 key）
* 端口：`GATEWAY_PORT`（默认 4000）；纯透传：`GATEWAY_INJECT_STREAM_USAGE=0`（默认）

## 4. 管理面（需 master key）

```bash
MK=$(cat /home/user/lvllm/gateway/master.key)
GW=http://127.0.0.1:4000

# 建 key（密钥只显示一次）
curl -s -H "Authorization: Bearer $MK" -H 'Content-Type: application/json' \
     -d '{"alias":"team-a"}' $GW/admin/keys

# 列出 / 吊销
curl -s -H "Authorization: Bearer $MK" $GW/admin/keys
curl -s -X DELETE -H "Authorization: Bearer $MK" $GW/admin/keys/3

# 按 key 统计（请求数、prompt/completion/cached tokens、usage 缺失数）
curl -s -H "Authorization: Bearer $MK" $GW/admin/stats

# 某 key 的明细（最近 N 条）
curl -s -H "Authorization: Bearer $MK" "$GW/admin/usage?key_id=3&limit=20"
```

## 5. 计量口径

| 列 | 含义 |
|---|---|
| `requests` | 请求数（含流式）|
| `prompt_tokens` / `completion_tokens` | 来自响应 `usage`（精确，不是估算）|
| `cached_tokens` | 前缀缓存命中的 prompt token（计费/算力口径自行决定是否扣除）|
| `usage_missing` | 该请求没拿到 `usage` 的计数（正常应为 0）|

**流式为什么也有 usage**：vLLM 默认流式不返回 usage；本部署给 **vLLM 加了服务端参数**
`--enable-force-include-usage`（`serve_ds41.sh` 的 `FORCE_INCLUDE_USAGE=1`），
因此流式一定带 usage。

⚠️ **不要**改成让网关注入 `stream_options`（`GATEWAY_INJECT_STREAM_USAGE=1`）——
那会改动请求/响应，违反"不改 vLLM 行为"的前提。

## 6. 坑

1. **`/health`、`/metrics` 不鉴权**（与 vLLM 一致，方便探活/抓取）。
2. **`/metrics` 不要走网关**：Prometheus 直连 8700。
3. **SQLite 单机够用**；多副本要换 Postgres（本网关未实现，需改 `db()`）。
4. **不要经网关做基准**：多一跳会引入 1–3 ms 与流式缓冲风险；测速直连 8700。
5. 网关挂了不影响 vLLM，但客户端会 502；用 `start_gateway.sh status` 探活。

## 7. 管理界面（浏览器）

网关自带一个管理页，**不需要装任何东西**：

```
http://<gpu 主机>:4000/admin/ui
```

打开后把 **master key** 粘进去（`cat /home/user/lvllm/gateway/master.key`），点「保存并连接」。
master key 只保存在该标签页的 `sessionStorage`，关掉标签页即失效；关浏览器后重新输入即可。

页面能力：

| 区域 | 能做什么 |
|---|---|
| 新建密钥 | 填别名 → 「生成」→ 密钥只显示一次，可一键复制 |
| 密钥与用量 | 表格列出每把 key 的 请求数 / prompt / completion / cached / usage缺失 |
| 操作列 | 「明细」看最近 100 条请求；「吊销」立即生效（之后请求 401）|
| 右上角 | 显示连接状态；「清除 master key」退出 |

### 访问不到时（两种情况）

1. **浏览器不在同一台机器**：网关默认监听 `0.0.0.0:4000`，直接用 `http://<gpu 主机 IP>:4000/admin/ui`；
   若网络策略只允许本机，用 SSH 隧道：
   ```bash
   ssh -L 4000:127.0.0.1:4000 <gpu 主机>
   # 然后浏览器打开 http://127.0.0.1:4000/admin/ui
   ```
2. **想只允许本机**：起网关时加 `GATEWAY_HOST=127.0.0.1`。

### 安全边界（务必知道）

* `/admin/ui` 页面本身**不含任何密钥**，只是个表单；
* **所有** `/admin/**` 接口都要 `Authorization: Bearer <master key>`，没带一律 401；
* `/v1/**` 要业务 key；`/health`、`/metrics` 不鉴权（与 vLLM 一致）；
* 管理页**同源**，不引入 CDN / 外部脚本，无跨域配置。
