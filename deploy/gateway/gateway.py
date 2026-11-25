#!/usr/bin/env python3
"""gateway.py —— vLLM 前置网关：**纯透传** + 虚拟 key 鉴权 + 按 key 计量。

设计约束（用户 2026-11-17）：「代理不要修改 vLLM 行为，就是做一个权限管理和统计」
  ⇒ 本网关**不改写**请求体、**不注入参数**、**不改写响应体**，字节级转发；
  ⇒ 唯一的改动是鉴权：客户端的虚拟 key 换成上游 `VLLM_API_KEY`。

组成
  * 鉴权：`Authorization: Bearer sk-...`，只存 sha256，可吊销；
  * 计量：从响应里读 `usage`（非流式读完整 JSON；流式读 SSE 末尾帧），落 SQLite；
  * 管理面：`/admin/keys`（建/列/吊销）、`/admin/stats`、`/admin/usage`，需 master key。

不转发的内容：`/admin/**`（本进程处理）。
透传但不鉴权的：`/health`、`/metrics`、`/docs`、`/openapi.json`（与 vLLM 自身一致）。

环境变量
  VLLM_API_KEY        上游密钥（必填）
  GATEWAY_MASTER_KEY  管理面密钥（必填）
  UPSTREAM_BASE       默认 http://127.0.0.1:8700
  GATEWAY_DB          默认 /home/user/lvllm/gateway/gateway.db
  INJECT_STREAM_USAGE 默认 0；=1 时对 stream 请求补 `stream_options.include_usage`
                      （会多一个尾帧，属"改行为"，默认关）
  GATEWAY_HOST/PORT   默认 127.0.0.1 / 4000
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import sqlite3
import time
from contextlib import closing
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

UPSTREAM = os.environ.get("UPSTREAM_BASE", "http://127.0.0.1:8700").rstrip("/")
UPSTREAM_KEY = os.environ.get("VLLM_API_KEY", "")
MASTER_KEY = os.environ.get("GATEWAY_MASTER_KEY", "")
DB_PATH = os.environ.get("GATEWAY_DB", "/home/user/lvllm/gateway/gateway.db")
INJECT_USAGE = os.environ.get("INJECT_STREAM_USAGE", "0") == "1"
OPEN_PATHS = ("/health", "/metrics", "/docs", "/openapi.json", "/redoc")

# 逐跳头：不能透传
HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
    "accept-encoding",  # 见下：统一 identity，避免"上游 gzip / 我们已解压"错位
}

app = FastAPI(title="vllm-gateway", docs_url=None, redoc_url=None, openapi_url=None)
_client: httpx.AsyncClient | None = None


# ── 存储 ────────────────────────────────────────────────────────────────
def db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    c = sqlite3.connect(DB_PATH, timeout=10)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=NORMAL")
    return c


def init_db() -> None:
    with closing(db()) as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS keys(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              key_hash TEXT UNIQUE NOT NULL,
              alias TEXT NOT NULL DEFAULT '',
              created_at REAL NOT NULL,
              revoked INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS usage(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              key_id INTEGER NOT NULL,
              ts REAL NOT NULL,
              path TEXT NOT NULL,
              model TEXT,
              prompt_tokens INTEGER NOT NULL DEFAULT 0,
              completion_tokens INTEGER NOT NULL DEFAULT 0,
              cached_tokens INTEGER NOT NULL DEFAULT 0,
              status INTEGER NOT NULL DEFAULT 0,
              latency_ms INTEGER NOT NULL DEFAULT 0,
              usage_missing INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_usage_key_ts ON usage(key_id, ts);
            """
        )
        c.commit()


def h(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def key_id_for(token: str) -> int | None:
    with closing(db()) as c:
        row = c.execute(
            "SELECT id FROM keys WHERE key_hash=? AND revoked=0", (h(token),)
        ).fetchone()
    return int(row[0]) if row else None


def record(key_id: int, path: str, model: str | None, u: dict | None,
           status: int, ms: int) -> None:
    pt = int((u or {}).get("prompt_tokens") or 0)
    ct = int((u or {}).get("completion_tokens") or 0)
    cached = 0
    try:
        cached = int(((u or {}).get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    except Exception:  # noqa: BLE001
        cached = 0
    with closing(db()) as c:
        c.execute(
            "INSERT INTO usage(key_id,ts,path,model,prompt_tokens,completion_tokens,"
            "cached_tokens,status,latency_ms,usage_missing) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (key_id, time.time(), path, model, pt, ct, cached, status, ms, 0 if u else 1),
        )
        c.commit()


# ── 鉴权 ────────────────────────────────────────────────────────────────
def bearer(request: Request) -> str:
    a = request.headers.get("authorization") or ""
    return a[7:].strip() if a.lower().startswith("bearer ") else ""


async def body_bytes(request: Request) -> bytes:
    return await request.body()


def upstream_headers(request: Request, with_key: bool) -> dict[str, str]:
    hd: dict[str, str] = {}
    for k, v in request.headers.items():
        if k.lower() in HOP or k.lower() == "authorization":
            continue
        hd[k] = v
    hd["Accept-Encoding"] = "identity"
    if with_key and UPSTREAM_KEY:
        hd["Authorization"] = f"Bearer {UPSTREAM_KEY}"
    return hd


def err(status: int, msg: str, code: str = "invalid_api_key") -> JSONResponse:
    return JSONResponse(
        {"error": {"message": msg, "type": code, "code": code}}, status_code=status
    )


# ── usage 抽取 ──────────────────────────────────────────────────────────
def parse_usage(payload: bytes) -> dict | None:
    r"""从响应体里取 usage（只读，不改字节）。

    两种形态：
      * 非流式：整包 JSON，直接取 `usage`；
      * 流式（SSE）：逐行看 `data: {...}`，取**最后一个**带 `usage` 的帧
        （vLLM 在 `--enable-force-include-usage` 下会把 usage 放在末帧）。
    ⚠️ 不能用 `\{"usage":\{...\}\}` 这类正则：usage 里有嵌套对象
       （`prompt_tokens_details` / `completion_tokens_details`），正则会漏。
    """
    if not payload:
        return None
    try:
        d = json.loads(payload)
        u = d.get("usage")
        if isinstance(u, dict):
            return u
    except Exception:  # noqa: BLE001
        pass
    found = None
    for line in payload.split(b"\n"):
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            continue
        try:
            j = json.loads(data)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(j.get("usage"), dict):
            found = j["usage"]
    return found


# ── 生命周期 ────────────────────────────────────────────────────────────
@app.on_event("startup")
async def _startup() -> None:
    global _client
    if not UPSTREAM_KEY:
        raise RuntimeError("VLLM_API_KEY 未设置")
    if not MASTER_KEY:
        raise RuntimeError("GATEWAY_MASTER_KEY 未设置")
    init_db()
    _client = httpx.AsyncClient(
        base_url=UPSTREAM, timeout=httpx.Timeout(None, connect=10.0), follow_redirects=False
    )


@app.on_event("shutdown")
async def _shutdown() -> None:
    if _client:
        await _client.aclose()



# ── 内置管理页（同源，零依赖） ────────────────────────────────────────────
_UI_HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>vLLM 网关 · 密钥管理</title>
<style>
 :root{--bg:#0f1115;--card:#171a21;--line:#262b36;--fg:#e6e9ef;--dim:#9aa4b2;--acc:#4c8dff;--danger:#ff5d5d;--ok:#3ddc84}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.55 -apple-system,Segoe UI,Roboto,"Helvetica Neue",Arial,"PingFang SC","Microsoft YaHei",sans-serif}
 header{padding:18px 24px;border-bottom:1px solid var(--line);display:flex;gap:12px;align-items:center;flex-wrap:wrap}
 h1{font-size:16px;margin:0;font-weight:600} .dim{color:var(--dim)}
 main{padding:20px 24px;max-width:1100px;margin:0 auto}
 .card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:16px}
 label{display:block;font-size:12px;color:var(--dim);margin-bottom:6px}
 input{background:#0d1016;border:1px solid var(--line);color:var(--fg);border-radius:8px;padding:9px 11px;font:inherit;width:100%}
 .row{display:flex;gap:10px;flex-wrap:wrap;align-items:end} .row>div{flex:1;min-width:220px}
 button{background:var(--acc);color:#fff;border:0;border-radius:8px;padding:9px 16px;font:inherit;cursor:pointer}
 button.ghost{background:transparent;border:1px solid var(--line);color:var(--fg)}
 button.danger{background:transparent;border:1px solid var(--danger);color:var(--danger);padding:5px 10px;font-size:12px}
 button:disabled{opacity:.45;cursor:not-allowed}
 table{width:100%;border-collapse:collapse;font-size:13px} th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
 th{color:var(--dim);font-weight:500;font-size:12px} tr:hover td{background:#1c2028}
 code{background:#0d1016;border:1px solid var(--line);border-radius:6px;padding:2px 6px;font-size:12px}
 .badge{font-size:11px;padding:2px 8px;border-radius:999px;border:1px solid var(--line);color:var(--dim)}
 .badge.on{color:var(--ok);border-color:#1f4d36} .badge.off{color:var(--danger);border-color:#4d1f1f}
 .msg{margin-top:10px;font-size:13px} .msg.err{color:var(--danger)} .msg.ok{color:var(--ok)}
 .keybox{display:flex;gap:8px;align-items:center;margin-top:10px} .keybox code{flex:1;overflow:auto;padding:8px 10px}
 .hint{font-size:12px;color:var(--dim);margin-top:6px}
</style></head><body>
<header>
  <h1>vLLM 网关 · 密钥管理</h1>
  <span class="badge" id="conn">未验证</span>
  <span class="dim" id="gwinfo"></span>
  <span style="flex:1"></span>
  <button class="ghost" id="logout">清除 master key</button>
</header>
<main>
  <div class="card">
    <label>管理面密钥（Master Key，只存本标签页 sessionStorage）</label>
    <div class="row">
      <div><input id="mk" type="password" placeholder="粘贴 master.key 内容" autocomplete="off"></div>
      <div style="flex:0"><button id="save">保存并连接</button></div>
    </div>
    <div class="msg" id="msg1"></div>
  </div>

  <div class="card">
    <label>新建密钥</label>
    <div class="row">
      <div><input id="alias" placeholder="别名，例如 team-a"></div>
      <div style="flex:0"><button id="create">生成</button></div>
    </div>
    <div id="newkey"></div>
    <div class="hint">密钥只显示这一次；服务端只保存 sha256，无法反查。</div>
  </div>

  <div class="card">
    <div class="row" style="margin-bottom:10px">
      <div><strong>密钥与用量</strong></div>
      <div style="flex:0"><button class="ghost" id="refresh">刷新</button></div>
    </div>
    <table><thead><tr>
      <th>ID</th><th>别名</th><th>状态</th><th>请求数</th><th>prompt</th><th>completion</th><th>cached</th><th>usage缺失</th><th>操作</th>
    </tr></thead><tbody id="rows"><tr><td colspan="9" class="dim">（先保存 master key）</td></tr></tbody></table>
    <div class="msg" id="msg2"></div>
  </div>

  <div class="card" id="detailCard" style="display:none">
    <div class="row" style="margin-bottom:10px">
      <div><strong>明细 · key <span id="detailId"></span></strong></div>
      <div style="flex:0"><button class="ghost" id="closeDetail">关闭</button></div>
    </div>
    <table><thead><tr><th>时间</th><th>路径</th><th>prompt</th><th>completion</th><th>cached</th><th>状态</th><th>耗时</th></tr></thead>
    <tbody id="detailRows"></tbody></table>
  </div>
</main>
<script>
const $ = id => document.getElementById(id);
const store = sessionStorage;
let MK = store.getItem("gw_master") || "";

function msg(el, text, cls){ const e=$(el); e.textContent = text || ""; e.className = "msg " + (cls||""); }
function headers(){ return {"Authorization":"Bearer " + MK, "Content-Type":"application/json"}; }

async function api(path, opt){
  const r = await fetch(path, Object.assign({headers: headers()}, opt||{}));
  let body = null;
  try { body = await r.json(); } catch(e){}
  if(!r.ok) throw new Error((body && body.error && body.error.message) || (body && body.detail) || ("HTTP " + r.status));
  return body;
}

async function connect(){
  if(!MK){ msg("msg1","请先填入 master key","err"); return; }
  store.setItem("gw_master", MK);
  try {
    const s = await api("/admin/stats");
    $("conn").textContent = "已连接"; $("conn").className = "badge on";
    msg("msg1","验证通过，共 " + s.stats.length + " 把密钥","ok");
    await refresh();
  } catch(e){
    $("conn").textContent = "验证失败"; $("conn").className = "badge off";
    msg("msg1", e.message, "err");
  }
}

async function refresh(){
  try{
    const [ks, st] = await Promise.all([api("/admin/keys"), api("/admin/stats")]);
    const m = new Map(st.stats.map(x => [x.id, x]));
    const tb = $("rows"); tb.innerHTML = "";
    if(!ks.keys.length){ tb.innerHTML = '<tr><td colspan="9" class="dim">还没有密钥</td></tr>'; return; }
    for(const k of ks.keys){
      const s = m.get(k.id) || {requests:0,prompt_tokens:0,completion_tokens:0,cached_tokens:0,usage_missing:0};
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${k.id}</td><td>${esc(k.alias||"—")}</td>
        <td><span class="badge ${k.revoked?"off":"on"}">${k.revoked?"已吊销":"有效"}</span></td>
        <td>${s.requests}</td><td>${s.prompt_tokens}</td><td>${s.completion_tokens}</td>
        <td>${s.cached_tokens}</td><td>${s.usage_missing}</td>
        <td><button class="ghost" data-act="detail" data-id="${k.id}" style="padding:5px 10px;font-size:12px">明细</button>
            ${k.revoked?"":`<button class="danger" data-act="revoke" data-id="${k.id}">吊销</button>`}</td>`;
      tb.appendChild(tr);
    }
    msg("msg2","","");
  }catch(e){ msg("msg2", e.message, "err"); }
}

function esc(s){ return String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c])); }

async function create(){
  const alias = $("alias").value.trim();
  try{
    const d = await api("/admin/keys", {method:"POST", body: JSON.stringify({alias})});
    $("newkey").innerHTML = `<div class="keybox"><code id="nk">${esc(d.key)}</code>
      <button class="ghost" id="copy">复制</button></div>
      <div class="hint">别名 ${esc(alias||"—")} · id=${d.id} · 只显示这一次</div>`;
    $("copy").onclick = () => { navigator.clipboard.writeText(d.key); msg("msg2","已复制","ok"); };
    $("alias").value = "";
    await refresh();
  }catch(e){ msg("msg2", e.message, "err"); }
}

async function revoke(id){
  if(!confirm("确认吊销 key id=" + id + "？吊销后立即 401。")) return;
  try{ await api("/admin/keys/" + id, {method:"DELETE"}); msg("msg2","已吊销 id=" + id,"ok"); await refresh(); }
  catch(e){ msg("msg2", e.message, "err"); }
}

async function detail(id){
  try{
    const d = await api("/admin/usage?key_id=" + id + "&limit=100");
    $("detailCard").style.display = "block"; $("detailId").textContent = id;
    const tb = $("detailRows"); tb.innerHTML = "";
    if(!d.usage.length){ tb.innerHTML = '<tr><td colspan="7" class="dim">暂无记录</td></tr>'; return; }
    for(const u of d.usage){
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${new Date(u.ts*1000).toLocaleString()}</td><td>${esc(u.path)}</td>
        <td>${u.prompt_tokens}</td><td>${u.completion_tokens}</td><td>${u.cached_tokens}</td>
        <td>${u.status}</td><td>${u.latency_ms} ms</td>`;
      tb.appendChild(tr);
    }
  }catch(e){ msg("msg2", e.message, "err"); }
}

$("save").onclick = () => { MK = $("mk").value.trim(); connect(); };
$("mk").addEventListener("keydown", e => { if(e.key === "Enter"){ MK = $("mk").value.trim(); connect(); } });
$("create").onclick = create;
$("alias").addEventListener("keydown", e => { if(e.key === "Enter") create(); });
$("refresh").onclick = refresh;
$("closeDetail").onclick = () => { $("detailCard").style.display = "none"; };
$("logout").onclick = () => { store.removeItem("gw_master"); MK=""; $("mk").value=""; $("conn").textContent="未验证"; $("conn").className="badge"; $("rows").innerHTML='<tr><td colspan="9" class="dim">（先保存 master key）</td></tr>'; };
$("rows").addEventListener("click", e => {
  const b = e.target.closest("button[data-act]"); if(!b) return;
  const id = Number(b.dataset.id);
  if(b.dataset.act === "revoke") revoke(id); else detail(id);
});

(async () => {
  try{
    const r = await fetch("/admin/stats");
    if(r.status === 401) $("gwinfo").textContent = "网关在线（需要 master key）";
  }catch(e){}
  if(MK){ $("mk").value = MK; connect(); }
})();
</script></body></html>
"""


@app.get("/admin/ui", response_class=HTMLResponse)
async def admin_ui() -> HTMLResponse:
    """内置管理页：建 key / 吊销 / 看用量。页面本身不含密钥，鉴权靠 master key（存在 sessionStorage）。"""
    return HTMLResponse(_UI_HTML)


# ── 管理面 ──────────────────────────────────────────────────────────────
def need_master(request: Request) -> None:
    if bearer(request) != MASTER_KEY:
        raise HTTPException(status_code=401, detail="master key required")


@app.post("/admin/keys")
async def create_key(request: Request) -> dict[str, Any]:
    need_master(request)
    body = {}
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    alias = str(body.get("alias") or "")
    token = "sk-" + secrets.token_hex(24)
    with closing(db()) as c:
        cur = c.execute(
            "INSERT INTO keys(key_hash,alias,created_at) VALUES(?,?,?)",
            (h(token), alias, time.time()),
        )
        c.commit()
        kid = cur.lastrowid
    return {"id": kid, "alias": alias, "key": token,
            "note": "密钥只显示这一次；服务端只存 sha256"}


@app.get("/admin/keys")
async def list_keys(request: Request) -> dict[str, Any]:
    need_master(request)
    with closing(db()) as c:
        rows = c.execute(
            "SELECT id,alias,created_at,revoked FROM keys ORDER BY id"
        ).fetchall()
    return {"keys": [
        {"id": r[0], "alias": r[1], "created_at": r[2], "revoked": bool(r[3])}
        for r in rows]}


@app.delete("/admin/keys/{kid}")
async def revoke_key(kid: int, request: Request) -> dict[str, Any]:
    need_master(request)
    with closing(db()) as c:
        c.execute("UPDATE keys SET revoked=1 WHERE id=?", (kid,))
        c.commit()
    return {"id": kid, "revoked": True}


@app.get("/admin/stats")
async def stats(request: Request) -> dict[str, Any]:
    need_master(request)
    with closing(db()) as c:
        rows = c.execute(
            """
            SELECT k.id, k.alias, k.revoked,
                   COUNT(u.id), COALESCE(SUM(u.prompt_tokens),0),
                   COALESCE(SUM(u.completion_tokens),0), COALESCE(SUM(u.cached_tokens),0),
                   COALESCE(SUM(u.usage_missing),0), MAX(u.ts)
            FROM keys k LEFT JOIN usage u ON u.key_id=k.id GROUP BY k.id ORDER BY k.id
            """
        ).fetchall()
    return {"stats": [
        {"id": r[0], "alias": r[1], "revoked": bool(r[2]), "requests": r[3],
         "prompt_tokens": r[4], "completion_tokens": r[5], "cached_tokens": r[6],
         "usage_missing": r[7], "last_ts": r[8]}
        for r in rows]}


@app.get("/admin/usage")
async def usage(key_id: int, limit: int = 50, request: Request = None) -> dict[str, Any]:  # type: ignore[assignment]
    need_master(request)  # type: ignore[arg-type]
    limit = max(1, min(limit, 1000))
    with closing(db()) as c:
        rows = c.execute(
            "SELECT ts,path,model,prompt_tokens,completion_tokens,cached_tokens,status,latency_ms"
            " FROM usage WHERE key_id=? ORDER BY id DESC LIMIT ?", (key_id, limit)
        ).fetchall()
    return {"usage": [
        {"ts": r[0], "path": r[1], "model": r[2], "prompt_tokens": r[3],
         "completion_tokens": r[4], "cached_tokens": r[5], "status": r[6], "latency_ms": r[7]}
        for r in rows]}


# ── 透传 ────────────────────────────────────────────────────────────────
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def proxy(path: str, request: Request) -> Response:
    full = "/" + path
    if full.startswith("/admin"):
        raise HTTPException(status_code=404, detail="not found")

    open_path = any(full == p or full.startswith(p + "/") for p in OPEN_PATHS)
    key_id = None
    if not open_path:
        tok = bearer(request)
        if not tok:
            return err(401, "缺少 Authorization: Bearer <key>")
        key_id = key_id_for(tok)
        if key_id is None:
            return err(401, "无效或已吊销的 key")

    raw = await body_bytes(request)
    if INJECT_USAGE and raw and full.endswith("/chat/completions"):
        try:
            d = json.loads(raw)
            if d.get("stream") and "stream_options" not in d:
                d["stream_options"] = {"include_usage": True}
                raw = json.dumps(d).encode()
        except Exception:  # noqa: BLE001
            pass

    url = full + (("?" + request.url.query) if request.url.query else "")
    hdrs = upstream_headers(request, with_key=True)
    t0 = time.time()
    try:
        req = _client.build_request(request.method, url, content=raw, headers=hdrs)
        up = await _client.send(req, stream=True)
    except Exception as exc:  # noqa: BLE001
        return err(502, f"上游不可达：{type(exc).__name__}: {exc}", "bad_gateway")

    out_headers = {
        k: v for k, v in up.headers.items() if k.lower() not in HOP
    }
    ctype = up.headers.get("content-type", "")

    # 非流式（含 JSON 错误）：读完 → 原样返回 → 记账
    if "text/event-stream" not in ctype:
        try:
            payload = await up.aread()
        finally:
            await up.aclose()
        ms = int((time.time() - t0) * 1000)
        if key_id is not None:
            u = parse_usage(payload)
            model = None
            try:
                model = json.loads(payload).get("model")
            except Exception:  # noqa: BLE001
                pass
            record(key_id, full, model, u, up.status_code, ms)
        return Response(content=payload, status_code=up.status_code, headers=out_headers)

    # 流式：边转发边在尾部找 usage；不缓冲、不改字节
    tail = bytearray()

    async def gen():
        try:
            async for chunk in up.aiter_raw():
                if chunk:
                    tail.extend(chunk)
                    if len(tail) > 65536:
                        del tail[: len(tail) - 65536]   # 只留最后 64KB（usage 在末帧）
                    yield chunk
        finally:
            await up.aclose()
            if key_id is not None:
                record(key_id, full, None, parse_usage(bytes(tail)),
                       up.status_code, int((time.time() - t0) * 1000))

    return StreamingResponse(gen(), status_code=up.status_code, headers=out_headers)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("GATEWAY_HOST", "127.0.0.1"),
        port=int(os.environ.get("GATEWAY_PORT", "4000")),
        log_level=os.environ.get("GATEWAY_LOG_LEVEL", "warning"),
        access_log=False,
    )
