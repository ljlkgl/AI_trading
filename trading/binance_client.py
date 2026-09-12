"""币安 USDT-M 永续合约 API 客户端。

基于 requests 直连币安 fapi 接口（避免重依赖），支持主网/测试网切换。
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import time
import urllib.parse
from typing import Any, Optional

import requests

from trading.types import AccountInfo, Candle, Position, SymbolInfo

logger = logging.getLogger(__name__)

_MAINNET_BASE = "https://fapi.binance.com"
_TESTNET_BASE = "https://testnet.binancefuture.com"


class BinanceError(Exception):
    """币安 API 返回的业务错误。"""


class BinanceClient:
    """币安 USDT-M 合约客户端（单向持仓模式）。"""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = False,
        timeout: int = 20,
        proxies: Optional[dict] = None,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = _TESTNET_BASE if testnet else _MAINNET_BASE
        self.session = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": api_key})
        self.session.proxies = proxies or {}
        self.timeout = timeout
        self._symbols_cache: dict[str, SymbolInfo] = {}

    # ---------------- 基础请求 ----------------

    def _public_request(self, path: str, params: dict | None = None) -> Any:
        url = f"{self.base_url}{path}"
        resp = self.session.get(
            url, params=params, timeout=self.timeout
        )
        return self._handle(resp)

    def _signed_request(
        self, method: str, path: str, params: dict | None = None
    ) -> Any:
        params = dict(params or {})
        params["timestamp"] = int(time.time() * 1000)
        query = urllib.parse.urlencode(params)
        sig = hmac.new(
            self.api_secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()
        url = f"{self.base_url}{path}?{query}&signature={sig}"
        if method == "GET":
            resp = self.session.get(url, timeout=self.timeout)
        elif method == "POST":
            resp = self.session.post(url, timeout=self.timeout)
        elif method == "DELETE":
            resp = self.session.delete(url, timeout=self.timeout)
        else:
            raise ValueError(f"Unsupported method: {method}")
        return self._handle(resp)

    def _handle(self, resp: requests.Response) -> Any:
        if resp.status_code >= 400:
            try:
                body = resp.json()
                msg = body.get("msg", body)
            except Exception:
                msg = resp.text[:300]
            raise BinanceError(f"[{resp.status_code}] {msg}")
        if resp.status_code == 204:
            return {}
        return resp.json()

    # ---------------- 行情（公开接口） ----------------

    def get_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 500,
    ) -> list[Candle]:
        """获取K线，返回按时间升序排列的蜡烛。"""
        data = self._public_request(
            "/fapi/v1/klines",
            {"symbol": symbol, "interval": interval, "limit": limit},
        )
        candles = []
        for row in data:
            candles.append(
                Candle(
                    open_time=int(row[0]),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                    close_time=int(row[6]),
                    quote_volume=float(row[7]),
                )
            )
        return candles

    def get_ticker_price(self, symbol: str) -> float:
        data = self._public_request("/fapi/v1/ticker/price", {"symbol": symbol})
        return float(data["price"])

    def get_24hr_ticker(self, symbol: str) -> dict:
        return self._public_request("/fapi/v1/ticker/24hr", {"symbol": symbol})

    def get_funding_rate(self, symbol: str, limit: int = 20) -> list[dict]:
        return self._public_request(
            "/fapi/v1/fundingRate", {"symbol": symbol, "limit": limit}
        )

    def get_symbol_info(self, symbol: str) -> SymbolInfo:
        """从 exchangeInfo 解析精度信息并缓存。"""
        if symbol in self._symbols_cache:
            return self._symbols_cache[symbol]
        data = self._public_request("/fapi/v1/exchangeInfo", {"symbol": symbol})
        if not data.get("symbols"):
            raise BinanceError(f"Unknown symbol: {symbol}")
        info = data["symbols"][0]
        filters = {f["filterType"]: f for f in info.get("filters", [])}
        price_filter = filters.get("PRICE_FILTER", {})
        lot_filter = filters.get("LOT_SIZE", {})
        notional_filter = filters.get("MIN_NOTIONAL", {})

        def _precision(step: str) -> int:
            s = str(step)
            if "." in s:
                return len(s.split(".")[1].rstrip("0"))
            return 0

        sym = SymbolInfo(
            symbol=info["symbol"],
            price_precision=_precision(price_filter.get("tickSize", "0.01")),
            qty_precision=_precision(lot_filter.get("stepSize", "0.001")),
            min_qty=float(lot_filter.get("minQty", 0)),
            min_notional=float(notional_filter.get("notional", 0)),
            price_tick=float(price_filter.get("tickSize", 0.01)),
            qty_step=float(lot_filter.get("stepSize", 0.001)),
        )
        self._symbols_cache[symbol] = sym
        return sym

    def refresh_symbol_info(self, symbol: str) -> SymbolInfo:
        """强制刷新某币种精度缓存（清除旧缓存后重新拉取），返回新 SymbolInfo。"""
        self._symbols_cache.pop(symbol, None)
        return self.get_symbol_info(symbol)

    # ---------------- 账户（签名接口） ----------------

    def get_position_mode(self) -> bool:
        """查询持仓模式：True=双向持仓(Hedge Mode)，False=单向持仓(One-way)。"""
        data = self._signed_request("GET", "/fapi/v1/positionSide/dual")
        return bool(data.get("dualSidePosition", False))

    def set_position_mode(self, dual: bool = True) -> dict:
        """切换持仓模式（默认切到双向）。注意：切换前账户不能有持仓或未成交挂单。"""
        return self._signed_request(
            "POST", "/fapi/v1/positionSide/dual",
            {"dualSidePosition": "true" if dual else "false"},
        )

    def get_account(self) -> AccountInfo:
        """获取账户资金与持仓快照。"""
        data = self._signed_request("GET", "/fapi/v3/account")
        balances = data.get("assets", [])
        total_balance = sum(float(b.get("walletBalance", 0)) for b in balances)
        available_balance = sum(float(b.get("availableBalance", 0)) for b in balances)
        unrealized_pnl = sum(float(b.get("unrealizedProfit", 0)) for b in balances)
        margin_balance = total_balance + unrealized_pnl

        positions = self.get_positions()
        return AccountInfo(
            total_balance=total_balance,
            available_balance=available_balance,
            unrealized_pnl=unrealized_pnl,
            margin_balance=margin_balance,
            positions=positions,
        )

    def get_positions(self) -> list[Position]:
        data = self._signed_request("GET", "/fapi/v3/positionRisk")
        positions = []
        for row in data:
            amt = float(row.get("positionAmt", 0))
            if abs(amt) < 1e-12:
                continue
            positions.append(
                Position(
                    symbol=row["symbol"],
                    position_side=row.get("positionSide", "BOTH"),
                    position_amt=amt,
                    entry_price=float(row.get("entryPrice", 0)),
                    mark_price=float(row.get("markPrice", 0)),
                    unrealized_pnl=float(row.get("unRealizedProfit", 0)),
                    leverage=float(row.get("leverage", 0)),
                    isolated_margin=float(row.get("isolatedMargin", 0)),
                    liquidation_price=float(row.get("liquidationPrice", 0)),
                )
            )
        return positions

    # ---------------- 交易（签名接口） ----------------

    def set_leverage(self, symbol: str, leverage: int) -> dict:
        return self._signed_request(
            "POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage}
        )

    def set_margin_type(self, symbol: str, margin_type: str = "ISOLATED") -> dict:
        return self._signed_request(
            "POST",
            "/fapi/v1/marginType",
            {"symbol": symbol, "marginType": margin_type},
        )

    def place_order(
        self,
        symbol: str,
        side: str,                     # BUY / SELL
        order_type: str,               # MARKET / LIMIT（条件单请用 place_algo_order）
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        reduce_only: bool = False,
        time_in_force: str = "GTC",
        client_order_id: Optional[str] = None,
        position_side: Optional[str] = None,
    ) -> dict:
        """下单。quantity 为标的币数量（BTC 等），非张数。

        注意：2025-12-09 起 STOP_MARKET/TAKE_PROFIT_MARKET 条件单必须走
        place_algo_order()（/fapi/v1/algoOrder），本方法仅用于 MARKET/LIMIT。

        position_side：双向持仓模式(Hedge Mode)下必填 LONG/SHORT；单向模式为 None。
        双向模式下禁止传 reduceOnly（用 positionSide 指定方向）；因此仅在
        单向模式且 reduce_only=True 时才发送 reduceOnly=true。
        """
        params: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": quantity,
            "newOrderRespType": "RESULT",
        }
        if reduce_only and not position_side:
            params["reduceOnly"] = "true"
        if price is not None:
            params["price"] = price
        if stop_price is not None:
            params["stopPrice"] = stop_price
        # 只对限价单(LIMIT)发送 timeInForce；市价单(MARKET)一律不携带该参数，
        # 否则币安返回 400 "Parameter 'timeInForce' sent when not required"，
        # 曾导致市价减仓(close_position)全部被拒。
        if time_in_force and order_type == "LIMIT":
            params["timeInForce"] = time_in_force
        if client_order_id:
            params["newClientOrderId"] = client_order_id
        if position_side:
            params["positionSide"] = position_side
        return self._signed_request("POST", "/fapi/v1/order", params)

    def close_position(
        self, symbol: str, side: str, quantity: float, order_type: str = "MARKET",
        price: Optional[float] = None, stop_price: Optional[float] = None,
        position_side: Optional[str] = None,
    ) -> dict:
        """平仓（reduceOnly）。side 为平仓方向：多头→SELL，空头→BUY。

        position_side：双向持仓模式(Hedge Mode)下必填 LONG/SHORT；单向模式为 None。
        """
        return self.place_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            reduce_only=True,
            position_side=position_side,
        )

    def get_order(self, symbol: str, order_id: int) -> dict:
        return self._signed_request(
            "GET", "/fapi/v1/order", {"symbol": symbol, "orderId": order_id}
        )

    def cancel_order(self, symbol: str, order_id: int) -> dict:
        return self._signed_request(
            "DELETE", "/fapi/v1/order", {"symbol": symbol, "orderId": order_id}
        )

    def get_open_orders(self, symbol: str) -> list[dict]:
        return self._signed_request("GET", "/fapi/v1/openOrders", {"symbol": symbol})

    def cancel_all_orders(self, symbol: str) -> list[dict]:
        return self._signed_request(
            "DELETE", "/fapi/v1/allOpenOrders", {"symbol": symbol}
        )

    # ---------------- 条件单（Algo Order API，2025-12-09 迁移） ----------------
    # 2025-12-09 起 STOP_MARKET / TAKE_PROFIT_MARKET 必须走 /fapi/v1/algoOrder；
    # 参数 stopPrice 改为 triggerPrice，新增 algoType=CONDITIONAL；
    # 响应句柄为 algoId（与普通订单 orderId 不同）。

    def place_algo_order(
        self,
        symbol: str,
        side: str,                     # BUY / SELL（保护单方向）
        order_type: str,               # STOP_MARKET / TAKE_PROFIT_MARKET
        quantity: float,
        trigger_price: float,
        position_side: str = "BOTH",   # 双向模式 LONG/SHORT；单向模式 BOTH
        working_type: str = "CONTRACT_PRICE",
        client_algo_id: Optional[str] = None,
        reduce_only: Optional[bool] = None,
    ) -> dict:
        """挂条件单（止损/止盈）。返回的 algoId 为该条件单句柄。"""
        params: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "algoType": "CONDITIONAL",
            "triggerPrice": trigger_price,
            "quantity": quantity,
            "positionSide": position_side,
            "workingType": working_type,
        }
        if client_algo_id:
            params["clientAlgoId"] = client_algo_id[:36]
        if reduce_only is not None:
            params["reduceOnly"] = "true" if reduce_only else "false"
        return self._signed_request("POST", "/fapi/v1/algoOrder", params)

    def get_open_algo_orders(self, symbol: Optional[str] = None) -> list[dict]:
        """查询未成交条件单（STOP/TP 保护单）。端点返回 {"orders":[...]}，此处解包为 list。"""
        params: dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        res = self._signed_request("GET", "/fapi/v1/openAlgoOrders", params)
        if isinstance(res, dict):
            return res.get("orders", []) or []
        return res or []

    def cancel_algo_order(self, symbol: str, algo_id: Any) -> dict:
        """撤销单个条件单（按 algoId）。"""
        return self._signed_request(
            "DELETE", "/fapi/v1/algoOrder",
            {"symbol": symbol, "algoId": algo_id},
        )
