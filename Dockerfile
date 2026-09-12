# 币安永续合约 AI 交易系统（TradingAgents 行情分析逻辑提取版）
FROM python:3.11-slim

WORKDIR /app

# 时区（可选）
ENV TZ=Asia/Shanghai

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 启动：默认循环模式；可通过 CMD 覆盖为 --once
CMD ["python", "main.py"]
