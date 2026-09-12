# TradeTool — 币安 BTC/ETH/SOL 永续合约 AI 交易系统

提取自 [TradingAgents](https://github.com/TauricResearch/TradingAgents) 的多 Agent 行情分析逻辑，实现为一个独立的、
可部署至服务器的币安 USDT-M 永续合约交易系统。

## 核心特性

- **多周期技术分析**（沿用 TradingAgents 指标体系）：SMA50/200、EMA10、MACD(12,26,9)、
  RSI(14)、Bollinger(20,2)、ATR(14)、VWMA(20)、MFI(14)，覆盖 15m/1h/4h/1d 周期
- **新闻面分析**（提取自 TradingAgents 的 yfinance 新闻接口）：标的新闻（BTC-USD 等）
  与宏观新闻（`yf.Search`）注入市场分析师报告
- **市场分析师 Agent**：基于指标快照 + 新闻输出结构化市场分析报告
  （趋势/动量/波动率/量能/新闻催化剂/关键位/方向偏置）
- **决策者 Agent（Portfolio Manager）**：接收「市场分析报告 + 新闻 + 持仓假设检查 +
  当前账户现状」，按指定 JSON 格式输出**具体交易举措**——开多/开空（市价或限价挂单）、
  平多/平空、清仓、持有，并给出数量、杠杆、止损、止盈与理由
- **账户现状注入**：每一轮**最先**获取账户权益、可用余额、未实现盈亏与各币种持仓
  （方向/数量/开仓均价/强平价），注入决策模型上下文
- **按余额交易**：仓位必须基于当前账户余额计算（提示词给出 `quantity = 目标保证金 × 杠杆 / 价格`），
  风控强制单笔保证金不超过可用余额
- **激进策略指导**：策略风格激进，开仓杠杆要求 **≥15x**（上限受风控硬约束），
  唯一底线是保证金充足、到止损价不爆仓
- **强制止损 + 保证金下限**：所有开仓动作**必须带 stop_loss**（Schema 与风控双层强制），
  止盈可选；单笔开仓保证金（quantity×price/leverage）**≥ MIN_MARGIN（默认 2.5 USDT）**
- **持仓假设记录与偏离检查**：开仓/调仓后把完整理由写入假设存储；下一轮决策时
  将「原始假设 vs 当前行情」注入模型，检查行情是否偏离了原本的假设
  （顺向则持有/收紧止损，触发止损或假设证伪则平仓）；账户无持仓则从零分析
- **自我反省 + 自主经验库**：每轮执行后反思者复盘决策与账户结果；模型可自主
  向经验库（`state/experience_library.json`）**写入**严重亏损教训 / 策略改进 / 风控失误等经验，
  也**有权修改或删除**已有条目；后续决策轮自动参考经验库内容
- **仓位自主决定**：仓位大小（quantity/保证金占用）由模型基于信号强度、账户余额、
  杠杆与止损距离自主决定，风控不做比例拦截；系统仅保留硬底线
- **硬风控底线**：杠杆上限、最小下单价值、最小保证金、保证金不超过可用余额、
  止损强制与方向校验，不满足的指令在执行前拦截
- **DRY_RUN 模式**：默认开启，只输出决策不真实下单，安全演练
- **测试网支持**：可一键切换币安测试网

## 目录结构

```
trade_tool/
├── main.py                  # 入口（--once / 循环模式）
├── config.py                # 配置（.env）
├── agents/
│   ├── llm.py               # OpenAI 兼容 LLM 客户端
│   ├── schemas.py           # 结构化输出 Schema（交易举措 + 反思/经验库操作）
│   ├── market_analyst.py    # 市场分析师（技术 + 新闻）
│   ├── decision_maker.py    # 决策者（激进策略 + 假设检查 + 账户现状注入）
│   └── confirmer.py         # 确认者（执行前复核/修正 + 轮次反思与经验库维护）
└── trading/
    ├── binance_client.py    # 币安 USDT-M 合约 API 客户端
    ├── indicators.py        # 技术指标计算（TradingAgents 指标体系）
    ├── market.py            # 行情数据服务
    ├── news.py              # 新闻服务（yfinance，提取自 TradingAgents）
    ├── hypothesis.py        # 操作理由列表存储（ThesisStore：增删改查 + 父子层级 + COMPLETE 级联清理）
    ├── experience.py        # 自主经验库（写入/修改/删除/参考）
    ├── risk.py              # 风控校验（含止损强制、保证金下限）
    ├── executor.py          # 订单执行器
    └── types.py             # 数据类型
```

## 快速开始

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 配置环境变量：

```bash
cp .env.example .env
# 编辑 .env：填入 LLM_API_KEY / LLM_MODEL（OpenAI 兼容接口，支持 deepseek/qwen/glm 等）
# 以及币安 API Key（测试网或主网）
```

3. 运行：

```bash
# 只执行一轮（推荐先 DRY_RUN 演练）
python main.py --once

# 循环模式（默认 60 分钟一轮）
python main.py

# 指定间隔
python main.py --interval 30
```

## 模型输出的指定格式

决策者模型必须输出如下 JSON（`agents/schemas.py` 定义）：

```json
{
  "market_assessment": "整体偏多/偏空/震荡的简短判断...",
  "instructions": [
    {
      "symbol": "BTCUSDT",
      "action": "OPEN_LONG",
      "order_type": "MARKET",
      "price": null,
      "quantity": 0.05,
      "leverage": 15,
      "stop_loss": 94000,
      "take_profit": 100000,
      "reason": "MACD金叉且价格站上50SMA，趋势偏多，新闻面无利空"
    }
  ],
  "risk_notes": "风险提示..."
}
```

`action` 可选值：

| action | 含义 | 备注 |
|---|---|---|
| `OPEN_LONG` | 开多 | **必须**带 quantity、leverage(≥15x 激进要求)、stop_loss |
| `OPEN_SHORT` | 开空 | **必须**带 quantity、leverage(≥15x 激进要求)、stop_loss |
| `CLOSE_LONG` | 平多 | 数量缺省=全部 |
| `CLOSE_SHORT` | 平空 | 数量缺省=全部 |
| `FLATTEN` | 清仓该币种 | 全部持仓平掉 |
| `CANCEL_ORDERS` | 撤销挂单 | 撤销该币种**全部未成交限价单**；只给 symbol + reason |
| `REPLACE_LIMIT` | 更改挂单 | 先撤销全部挂单再重挂：**必须**给 side=BUY/SELL、quantity、price（新价） |
| `SET_SL_TP` | 调整止盈止损 | 调整**已有持仓**的止损/止盈：至少给 stop_loss 或 take_profit 一个；系统先撤旧保护单再按新价重挂 |
| `HOLD` | 持有不动 | 不产生订单 |

> 挂单管理说明：未成交的 LIMIT 单不占保证金、不影响持仓；`CANCEL_ORDERS` / `REPLACE_LIMIT`
> 不会触碰已有仓位，也无需带止损。模型在下 LIMIT 开仓单后，若价格迟迟未触及可随时撤销或改价重挂。
>
> 止盈止损调整说明：`SET_SL_TP` 只作用于已有持仓（无持仓会被风控拦截），可随时上移/下移
> 止损与止盈（行情走好后锁定利润、行情反转时收紧风险）；多仓要求 止损<现价<止盈，空仓相反，
> 方向不合理会被风控拦截。

## 执行前逐条确认（Confirmer）

决策者输出完整 JSON 后，**每一条指令实际下单前**都会再调用确认者复核一次，
以便及时发现并修正参数错误（价格/数量/方向/止损距离等）：

1. 决策者输出 N 条指令 → 风控校验 → 逐条进入确认环节；
2. 每条指令执行前，确认者收到：该指令全文 + 该币种最新价格 + **最新账户现状**（前几条
   已执行可能已改变持仓/余额）+ 本轮已执行结果摘要；
3. 确认者返回三选一：
   - `PROCEED`：指令合理，原样执行；
   - `SKIP`：指令有问题且无法可靠修正，跳过本指令；
   - `REPLACE`：**输出修正后的完整新指令**替换执行（新指令会**重新过一遍风控**，
     不通过则跳过）；
4. 确认期间 LLM 全部不可用（主备均失效）同样触发紧急平仓；单次确认失败则保守跳过该指令。

确认者使用快速模型（quick），只做轻量复核；完整决策仍由决策者（deep）负责。
每轮确认结果记录在 `result["confirmations"]`，执行结果在 `result["execution"]`。

## 条件唤醒（Watch Triggers）

系统默认按固定间隔循环分析；模型可以在每轮决策中设定**唤醒条件**，让系统在
正常循环外**提前唤醒**自己（例如价格触及关键位时），避免错过行情：

1. 决策者输出 `watch_conditions`（与 instructions 平级）：
   `[{"symbol": "BTCUSDT", "condition": "price_above", "value": 102000, "note": "突破追多"}]`
2. `condition` 支持：`price_above`（价格≥value 唤醒）、`price_below`（价格≤value 唤醒）；
3. 空列表=清除所有唤醒条件；非空列表=全量替换上一轮条件；
4. 循环等待期间，系统按 `WATCH_CHECK_INTERVAL`（默认 30 秒）轮询价格，
   任一条件满足立即提前执行一轮完整分析；
5. 条件为**一次性**：触发后自动清除，过期（`WATCH_MAX_AGE_HOURS`，默认 24h）自动失效；
   下一轮决策会重新设定。

配置项：`WATCH_ENABLED`（开关，默认 true）、`WATCH_CHECK_INTERVAL`、`WATCH_MAX_AGE_HOURS`。
条件持久化在 `state/watch_triggers.json`，结果记录在 `result["watch_conditions"]`。

模型同时收到**账户现状**与**持仓假设检查**：

```
# 当前账户现状
- 账户权益: 1234.56 USDT
- 可用余额: 1000.00 USDT
- 未实现盈亏: +23.45 USDT
## 当前持仓
| 币种 | 方向 | 数量 | 开仓均价 | 标记价格 | 未实现盈亏 | 杠杆 | 强平价 |
| BTCUSDT | 多 | 0.1 | 60000 | 61000 | +100 | 15 | 55000 |

## 未成交挂单（open orders）
### BTCUSDT
- orderId=888001 BUY LIMIT price=95000.00 stopPrice=0 qty=0.050 filled=0 reduceOnly=False status=NEW
- orderId=888002 SELL STOP_MARKET price=0 stopPrice=94000.00 qty=0.050 filled=0 reduceOnly=True status=NEW

# 持仓假设检查（上次开仓/调仓理由 vs 当前行情）
## BTCUSDT
- 开仓时间: 2026-08-15T10:00:00
- 方向: LONG  开仓均价: 95000  止损: 92000  止盈: 100000
- 原始开仓理由: MACD金叉趋势偏多
- 当前持仓: 数量 0.05  标记价 98000  未实现盈亏 +150.0000
- 相对开仓价偏离: +3.16%（顺向）
```

账户上下文包含模型决策与挂单管理所需的**全部信息**：权益/可用余额/未实现盈亏、
持仓（方向/数量/开仓均价/标记价/强平价）、以及**所有未成交挂单**（含 `orderId`、类型、
方向、价格、触发价、数量、已成交量、reduceOnly 标志、状态）——模型可据此决定
`CANCEL_ORDERS` / `REPLACE_LIMIT` / `SET_SL_TP`；确认者（Confirmer）在每条指令执行前
也会收到同样的最新账户与挂单快照。

## 持仓模式（Hedge Mode）

系统启动时自动检查账户持仓模式：

1. 查询 `GET /fapi/v1/positionSide/dual`；
2. 若为**单向持仓（One-way）**，自动尝试切换为**双向持仓（Hedge Mode）**
   （`POST dualSidePosition=true`），以便同时持有多空两个方向；
3. **切换失败**（币安要求切换前账户无持仓、无挂单）→ 记录错误并按单向模式继续运行，
   不会阻塞启动；等持仓清零后重启即可切换；
4. 双向模式下所有下单自动携带 `positionSide`（LONG/SHORT），单向模式则不携带。

> 若长期使用双向模式，建议在币安 App/网页端手动将账户持仓模式设为"双向持仓"，
> 避免程序每次启动时尝试切换。

## 服务器部署

### Docker

```bash
cp .env.example .env   # 填写真实配置
docker compose up -d --build
```

容器默认以循环模式运行，`restart: unless-stopped` 保证崩溃自动重启。

### 直接部署（systemd / supervisor）

```bash
python main.py --interval 60
```

建议配合 supervisor/systemd 托管，日志输出到 stdout 由进程管理器收集。

## 安全说明

- 默认 `DRY_RUN=true`、`BINANCE_TESTNET=false`，**首次运行请务必保持 DRY_RUN 并在测试网演练**
- 确认无误后：将 `DRY_RUN` 改为 `false` 再切换主网 API Key
- 主网 Key 只授予合约交易 + 交易权限，不要开放提币权限
- 币安 API Key 存在服务器本机 .env，注意文件权限与密钥保管
- 高杠杆（≥15x）会显著放大盈亏，止损价务必合理，确保极端行情下到止损价不会爆仓

## 与 TradingAgents 的逻辑映射

| TradingAgents | 本系统 |
|---|---|
| market_analyst（get_stock_data/get_indicators） | `MarketDataService` + `MarketAnalyst` |
| news_analyst（get_news / get_global_news，yfinance） | `NewsService`（yfinance，注入市场分析师） |
| stockstats 指标（SMA/MACD/RSI/BOLL/ATR/VWMA/MFI） | `indicators.py`（纯 pandas 实现，同名指标） |
| TraderProposal（action/reasoning/entry/stop_loss/sizing） | `TradeInstruction`（动作/理由/价格/数量/杠杆/止损止盈，开仓强制止损） |
| Portfolio Manager + 账户上下文 | `DecisionMaker`（注入账户现状 + 假设检查后结构化决策） |
| TradingMemoryLog（决策复盘记忆） | `ThesisStore`（操作理由列表 + 父子层级编号 + COMPLETE 级联删除 + 自动清理） |
| Reflector（回测结果反思） | `Confirmer.reflect`（反思职责已并入确认者）+ `ExperienceStore`（每轮复盘 + 自主经验库） |
| 风控约束 | `RiskManager` 硬校验层 |

## 两档模型分工 + 三档推理强度（仿照 TradingAgents）

TradingAgents 用 `deep_think_llm` / `quick_think_llm` 两个模型按任务复杂度分工，
并用 `reasoning_effort`（low/medium/high）控制深度推理强度。本系统同样实现：

| 配置项 | 用途 | 默认 |
|---|---|---|
| `LLM_MODEL` | **quick_think_llm**：市场分析师、确认者/反思（快速任务） | 必填 |
| `LLM_DEEP_MODEL` | **deep_think_llm**：决策者/研究经理（复杂推理）；留空则回退 `LLM_MODEL` | 空 |
| `LLM_REASONING_EFFORT` | 推理强度三档 `low` / `medium` / `high`，随请求发送；留空则不发送 | `medium` |

分工与 TradingAgents 对齐：`DecisionMaker`（对应 Research Manager / Portfolio Manager）
使用深度模型；`MarketAnalyst` 与 `Confirmer`（确认 + 反思）使用快速模型。

注意：部分 OpenAI 兼容接口（如某些 SenseNova/DeepSeek 网关）不接受
`reasoning_effort` 参数，系统检测到接口报错后会自动移除该参数降级重试，不会中断运行。

## 备用 LLM 与紧急平仓

主 LLM API 故障时系统自动降级，链路如下：

1. **主 LLM 调用失败**（含内部快速重试）→ 自动切换到**备用 LLM**（`LLM_BACKUP_*`），
   市场分析师 / 决策者 / 反思者所有环节共用同一个备用；
2. **备用也失败** → 进入**连通性确认**：按 `LLM_EMERGENCY_INTERVAL` 秒间隔、
   连续尝试 `LLM_EMERGENCY_ATTEMPTS` 次（每次先主后备用）；
3. 确认全部不可用 → **立即市价平掉当前所有持仓**（reduceOnly，不依赖 LLM），
   本轮结束，等待下一轮重新尝试。

| 配置项 | 用途 | 默认 |
|---|---|---|
| `LLM_BACKUP_API_KEY` | 备用 LLM Key；留空回退主 Key | 空 |
| `LLM_BACKUP_BASE_URL` | 备用 LLM 接口地址；留空回退主地址 | 空 |
| `LLM_BACKUP_MODEL` | 备用模型名；**留空则不启用备用** | 空 |
| `LLM_EMERGENCY_ATTEMPTS` | 主备全失效后的确认尝试次数 | 5 |
| `LLM_EMERGENCY_INTERVAL` | 每次确认尝试的间隔（秒） | 60 |

紧急平仓尊重 `DRY_RUN`：演练模式下只打印不平仓；真实模式用市价 reduceOnly 单，
平仓失败会记录错误并留待下一轮重试。
