#!/usr/bin/env python3
"""
中国股票市场行情MCP服务
提供实时股票价格、历史数据、市场指数等功能
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import pytz

import akshare as ak
import pandas as pd
from mcp.server.fastmcp import FastMCP

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stock-mcp-server")

# 创建MCP服务器实例
mcp = FastMCP("中国股票市场行情服务")

class TimeService:
    """时间服务类"""
    
    @staticmethod
    def get_current_time(timezone: str = "Asia/Shanghai") -> Dict[str, Any]:
        """获取当前时间"""
        try:
            tz = pytz.timezone(timezone)
            current_time = datetime.now(tz)
            
            return {
                "当前时间": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "时区": timezone,
                "时间戳": int(current_time.timestamp()),
                "是否交易日": TimeService.is_trading_day(current_time),
                "是否交易时间": TimeService.is_trading_time(current_time),
                "下一个交易日": TimeService.get_next_trading_day(current_time).strftime("%Y-%m-%d")
            }
        except Exception as e:
            return {"error": f"获取时间失败: {str(e)}"}
    
    @staticmethod
    def is_trading_day(dt: datetime) -> bool:
        """判断是否为交易日（简单判断，不考虑节假日）"""
        # 周一到周五为交易日
        return dt.weekday() < 5
    
    @staticmethod
    def is_trading_time(dt: datetime) -> bool:
        """判断是否为交易时间"""
        if not TimeService.is_trading_day(dt):
            return False
        
        time = dt.time()
        # 上午交易时间：9:30-11:30
        morning_start = dt.replace(hour=9, minute=30, second=0).time()
        morning_end = dt.replace(hour=11, minute=30, second=0).time()
        
        # 下午交易时间：13:00-15:00
        afternoon_start = dt.replace(hour=13, minute=0, second=0).time()
        afternoon_end = dt.replace(hour=15, minute=0, second=0).time()
        
        return (morning_start <= time <= morning_end) or (afternoon_start <= time <= afternoon_end)
    
    @staticmethod
    def get_next_trading_day(dt: datetime) -> datetime:
        """获取下一个交易日"""
        next_day = dt + timedelta(days=1)
        while not TimeService.is_trading_day(next_day):
            next_day += timedelta(days=1)
        return next_day

# 创建时间服务实例
time_service = TimeService()

class StockDataService:
    """股票数据服务类"""
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 300  # 5分钟缓存
    
    def _safe_convert(self, value: Any, convert_type: type, default: Any = 0) -> Any:
        """安全转换数据类型，处理NaN值"""
        try:
            if pd.isna(value) or value is None:
                return default
            if convert_type == int:
                return int(float(value))
            elif convert_type == float:
                return float(value)
            else:
                return convert_type(value)
        except (ValueError, TypeError):
            return default

    async def get_stock_realtime(self, symbol: str) -> Dict[str, Any]:
        """获取股票实时行情"""
        try:
            # 获取当前时间信息
            time_info = time_service.get_current_time()
            
            # 获取实时行情数据
            df = ak.stock_zh_a_spot_em()
            stock_data = df[df['代码'] == symbol]
            
            if stock_data.empty:
                return {"error": f"未找到股票代码 {symbol} 的数据"}
            
            row = stock_data.iloc[0]
            result = {
                "代码": str(row['代码']),
                "名称": str(row['名称']),
                "最新价": self._safe_convert(row['最新价'], float),
                "涨跌幅": self._safe_convert(row['涨跌幅'], float),
                "涨跌额": self._safe_convert(row['涨跌额'], float),
                "成交量": self._safe_convert(row['成交量'], int),
                "成交额": self._safe_convert(row['成交额'], float),
                "振幅": self._safe_convert(row['振幅'], float),
                "最高": self._safe_convert(row['最高'], float),
                "最低": self._safe_convert(row['最低'], float),
                "今开": self._safe_convert(row['今开'], float),
                "昨收": self._safe_convert(row['昨收'], float),
                "更新时间": time_info.get("当前时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                "是否交易时间": time_info.get("是否交易时间", False),
                "数据说明": "实时数据" if time_info.get("是否交易时间") else "非交易时间数据"
            }
            
            return result
            
        except Exception as e:
            logger.error(f"获取股票实时数据失败: {e}")
            return {"error": f"获取数据失败: {str(e)}"}

# 创建服务实例
stock_service = StockDataService()

@mcp.tool()
async def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    """
    获取当前时间信息
    
    Args:
        timezone: 时区，默认为Asia/Shanghai（北京时间）
    
    Returns:
        当前时间信息的JSON字符串，包含时间、交易状态等
    """
    result = time_service.get_current_time(timezone)
    return json.dumps(result, ensure_ascii=False, indent=2)

@mcp.tool()
async def get_stock_realtime(symbol: str) -> str:
    """
    获取股票实时行情数据
    
    Args:
        symbol: 股票代码，如 '000001' 或 '600000'
    
    Returns:
        股票实时行情数据的JSON字符串
    """
    result = await stock_service.get_stock_realtime(symbol)
    return json.dumps(result, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # 运行MCP服务器
    mcp.run()