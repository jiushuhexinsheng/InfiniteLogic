# InfiniteLogic Grafana 看板

## 导入步骤

1. 打开 Grafana → Dashboards → Import
2. 上传 `openbase.json` 或粘贴 JSON 内容
3. 选择 Prometheus 数据源
4. 点击 Import

## 面板说明

| 区块 | 面板 | 指标 |
|------|------|------|
| Turns | Turn Rate | `openbase_turns_total` |
| Tools | Tool Call Rate | `openbase_tool_calls_total` |
| Tokens & Cost | Token Usage | `openbase_tokens_total` |
| Tokens & Cost | Cost (USD) | `openbase_cost_usd_total` |
| Errors | LLM Errors | `openbase_llm_errors_total` |

## Prometheus 配置

```yaml
scrape_configs:
  - job_name: openbase
    static_configs:
      - targets: ['localhost:8000']
```
