# DeepSeek-V4.1-Flash on RTX 5090 Laptop (24 GB) + 64 GB RAM — Strata 部署与调优实录

在单张 RTX 5090 Laptop（24 GB）+ 64 GB 内存的笔记本上，用 [Strata](https://github.com/Niko1221/Strata) 引擎运行 1T 级 MoE 模型 **DeepSeek-V4.1-Flash-1017**（ISTA-DASLab IQ2_XXS / 2.00 bpw 量化）的完整过程：部署踩坑记录、显存/内存/SSD 三层分配实测、MTP 投机解码与参数调优数据、以及最终推荐配置。

**TL;DR**

| 指标 | 实测值 |
|---|---|
| 解码速度（同题缓存全热） | **26.2 tok/s** |
| 解码速度（换题真实场景） | **13–16 tok/s**（`--experts 5`） |
| 冷启动首个回答 | 6.8 tok/s（缓存预热后翻 4 倍） |
| 长文档 prefill（28,219 tokens） | 67.8 s（**≈416 tok/s**） |
| 上下文 | 8K → **64K 实测可用**（KV 为 fp4，显存几乎无增量） |
| 显存驻留 | 1,651 / 15,360 个 experts（14.57 GB）+ 5.86 GB dense 权重 |
| 官方参照 | RTX 5070 12 GB + 64 GB RAM ≈ 4.7 tok/s |

## 硬件与软件

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 Laptop 24 GB（sm_120，PCIe 5.0 x16） |
| CPU | Intel Core Ultra 9 275HX（24 核，AVX2） |
| RAM | 64 GB |
| 引擎 | Strata 0.2.6（control 0.1.51）+ strata-dsv41 0.1.51 cuda13 |
| 模型 | ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF · IQ2_XXS（2.00 bpw，KL 0.170，ppl +15.5%） |

## 模型文件构成（共 257 GB）

| 文件 | 大小 | 角色 |
|---|---:|---|
| IQ2_XXS shard ×4 | 138 GB | 模型本体（40 层 × 384 experts，每 token 激活 6 个） |
| DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf | 111 GB | n-gram 查找表，常驻 SSD 按 token 读行 |
| dspark-...-Q8_0.gguf | 8 GB | 投机解码 drafter（DSpark） |
| UD-Q8_K_XL-00001-of-00009.gguf | 5.8 MB | 仅元数据，pack 时用 |

注意 IQ2_XXS 的 4 个 shard 只是模型本体，Engram + drafter 还要 119 GB。全放 **NVMe SSD**（Engram 和缓存 miss 的 experts 都按需读盘，慢盘直接拖慢每个字）。

## 三层资源分配（实测）

```
GPU 24 GB  ─ dense/attention 权重 5.86 GB + 热点 experts 14.57 GB（1,651 个）+ KV(fp4) ≈ 0.01 GB
RAM 64 GB  ─ expert RAM store 37.6 GB（引擎自动定量，预读最热 26.5 GB 仅需 12.8 s）+ 系统
SSD NVMe   ─ 模型文件 138 GB 常驻（13,709 个 experts 原位流式读）+ Engram 111 GB
```

- 显存多 1 GB ≈ 多驻 ~110 个 experts = CPU 少算 110 个，**显存比 GPU 算力更值钱**。
- 每个字：GPU 算 dense + 缓存命中的 experts，CPU（22 线程 AVX2）并行算未命中的，Engram 行从盘读。

## 部署踩坑记录（重点）

1. **HF 仓库文件更新导致校验失败**：仓库在 2026-10-25 更新过 IQ2_XXS shard 1 的 GGUF header（+11,104 字节，仅 metadata，权重不变），Strata 0.2.6 的 pinned 尺寸/SHA 还是旧版，`install` 直接拒绝本地文件。修复：把 `control/setup.py` 中该文件的 pinned 条目更新为 HF 官方 LFS 当前指纹（size + sha256），校验流程不变，随后 39 GB 文件全量哈希通过。**文件本身没有任何问题，下载与复制均可排除。**
2. **内置下载器代理 502**：Strata 的 Python 下载器走本地代理访问 huggingface.co 稳定 502（同代理 curl 正常）。绕过：从镜像站按 pinned revision 手动下载 DSpark（8 GB）与 UD-Q8_K_XL 元数据 shard（5.8 MB），放入 `--gguf-dir` 目录，Strata 校验 SHA 后识别为 "already here"。
3. **引擎单独分发**：DeepSeek 引擎 `strata-dsv41` 与主引擎独立。离线/网络差时从 release 页下载 `strata-dsv41-<ver>-windows-x86_64-cuda-cuda13.tar.gz`，用 `--dsv41-package` 参数交给 setup，签名校验通过后自动安装。
4. **上下文与 thinking**：默认 8192 上下文；thinking 默认开启，API 关闭用 `reasoning_effort: "none"`（测速时必关，否则输出全是思考 token）。

## 调优实验（全部读服务端 /metrics 的引擎 decode 口径，固定题 3 连发）

### MTP / 投机解码（DSpark）

| 配置 | 换题均值 | 热态峰值 | draft 接受率 | 结论 |
|---|---:|---:|---:|---|
| `--dspark auto`（判定 off） | 11.5 | **26.2** | — | ✅ **最快，保持默认** |
| `--dspark on` | 9.8 | 17.5 | 84% | 慢 33% |
| `--dspark on --dspark-block 3` | 12.1 | 20.8 | 81% | 慢 21% |

drafter 接受率虽有 84%，但它是负优化：drafter 驻留显存挤占 experts 名额，且 MoE 的 decode 瓶颈在 expert 权重读取而非算力——投机解码省算力不省读取。**引擎的 auto 判定是准的。**

### 参数扫描（600-token 换题长文场景）

| 配置 | 三次 tok/s | 均值 | 备注 |
|---|---|---:|---|
| 默认（6 experts） | 8.2 / 14.6 / 11.7 | 11.5 | 基线 |
| `--ssd-cache 46000`（RAM store 46 GB） | 4.6 / 11.6 / 14.5 | 10.2 | ❌ 内存压力负优化，37.6 GB 自动档就是平衡点 |
| **`--experts 5`**（每 token 算 5/6 专家） | 11.0 / 16.5 / 13.4 | **13.6** | ✅ +18%，质量损失官方称通常无感 |

### 上下文扩展（64K 实测）

- KV 为 fp4，8K 上下文仅 ~0.01 GB 显存，开 64K 几乎无增量。
- 28,219 tokens 文档 prefill 67.8 s（416 tok/s），回答正常。
- 长上下文 decode 掉到 ~6.7 tok/s：长 prompt 把 RAM store 的热 experts 全部轮转污染，属正常现象，短对话不受影响。

## 最终配置

引擎参数（`strata-dsv41.exe`）：

```
--pack <pack 目录> --max-context 65536 --eos 1 --dspark auto --dspark-conf 0.65 --experts 5
```

日常使用：双击 `run-deepseek-v4.1-flash-ist-das-2.00bpw.bat` → `http://127.0.0.1:8080/`。加载 1–3 分钟（期间机器卡是正常的，RAM store 预热 34–55 GB），关窗即停。

OpenAI 兼容 API：`http://127.0.0.1:8080/v1/chat/completions`；Anthropic 兼容：`/v1/messages`。API key 任意，模型名任意。单并发（此版本不支持并行请求）。

`configs/` 里有各次实验的完整 server 配置；`data/` 是每组实验的原始 `/metrics` 快照（含逐请求 decode_ms、draft 接受率、KV 状态）。

## 复现要点

1. 模型文件放 NVMe，`--gguf-dir` 指向含 4 shards + Engram + DSpark 的目录。
2. 先 `strata launch --model-path <目录> --model ist-das-2.00bpw --plan` 看本机资源规划（只读）。
3. 显存 <16 GB 的卡注意：dense 权重放不下会导致引擎启动 OOM。
4. 内存 <40 GB 的机器官方标注未测试且慢；本机 64 GB 自动档 RAM store 37.6 GB，不建议手动加大（实测负优化）。
5. 投机解码在"显存换 experts"的机器上大概率是负优化，让 `auto` 自己判。

## 数据与方法说明

- 速度一律取引擎 decode 时间（`/metrics` 的 `decode_tok_s`），非客户端墙钟时间。
- 冷→热：同一 prompt 第 1 次为冷（experts 未缓存），第 3 次为全热；换题测试让 experts 真实轮转。
- 每配置 3 次运行取峰值/均值；温度 0、`reasoning_effort: none`、max_tokens 300–600。
- 未跑通官方 `standard-v1` 全套件（模型加载 + 12 项协议超出单次会话时限），数据来自 `/metrics` 快照，方法与原始数据见 `data/`。

## 相关链接

- [Strata](https://github.com/Niko1221/Strata)（MIT）· [模型仓库](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF) · [Strata DeepSeek 文档](https://github.com/Niko1221/Strata/blob/main/docs/DEEPSEEK41.md)

模型与引擎各自许可适用；本仓库只含配置、数据与文字，不含模型文件。
