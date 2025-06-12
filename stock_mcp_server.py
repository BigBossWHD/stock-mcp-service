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

class SecurityConfig:
    """安全配置类"""
    MAX_HISTORY_DAYS = 365  # 历史数据最大天数限制
    MAX_TECHNICAL_DAYS = 120  # 技术指标最大天数限制
    CACHE_TIMEOUT = 300  # 缓存超时时间（秒）
    RATE_LIMIT_REQUESTS = 100  # 每分钟最大请求数
    
    @staticmethod
    def sanitize_error_message(error: Exception) -> str:
        """清理错误消息，避免泄露内部细节"""
        error_str = str(error)
        # 移除可能的敏感信息
        sensitive_patterns = [
            r'File ".*?"',  # 文件路径
            r'line \d+',    # 行号
            r'0x[0-9a-fA-F]+',  # 内存地址
        ]
        
        import re
        for pattern in sensitive_patterns:
            error_str = re.sub(pattern, '[REDACTED]', error_str)
        
        # 返回通用错误消息
        if "网络" in error_str or "连接" in error_str or "timeout" in error_str.lower():
            return "数据获取超时，请稍后重试"
        elif "权限" in error_str or "permission" in error_str.lower():
            return "数据访问受限"
        elif "格式" in error_str or "format" in error_str.lower():
            return "数据格式错误"
        else:
            return "服务暂时不可用，请稍后重试"

class TradingCalendarService:
    """交易日历服务类"""
    
    def __init__(self):
        self._trading_calendar_cache = {}
        self._cache_date = None
    
    def _get_trading_calendar(self, year: int) -> pd.DataFrame:
        """获取指定年份的交易日历"""
        cache_key = f"trading_calendar_{year}"
        
        # 检查缓存
        if (cache_key in self._trading_calendar_cache and 
            self._cache_date and 
            (datetime.now() - self._cache_date).days < 1):
            return self._trading_calendar_cache[cache_key]
        
        try:
            # 使用akshare获取交易日历
            calendar_df = ak.tool_trade_date_hist_sina()
            if not calendar_df.empty:
                # 过滤指定年份的数据
                calendar_df['trade_date'] = pd.to_datetime(calendar_df['trade_date'])
                year_calendar = calendar_df[calendar_df['trade_date'].dt.year == year]
                
                self._trading_calendar_cache[cache_key] = year_calendar
                self._cache_date = datetime.now()
                return year_calendar
        except Exception as e:
            logger.warning(f"获取交易日历失败，使用简单判断: {e}")
        
        return pd.DataFrame()
    
    def is_trading_day(self, dt: datetime) -> bool:
        """判断是否为交易日（使用真实交易日历）"""
        try:
            year = dt.year
            calendar_df = self._get_trading_calendar(year)
            
            if not calendar_df.empty:
                date_str = dt.strftime('%Y-%m-%d')
                trading_dates = calendar_df['trade_date'].dt.strftime('%Y-%m-%d').tolist()
                return date_str in trading_dates
            else:
                # 回退到简单判断：周一到周五，排除明显的节假日
                if dt.weekday() >= 5:  # 周末
                    return False
                
                # 简单的节假日判断（可以根据需要扩展）
                month, day = dt.month, dt.day
                
                # 元旦
                if month == 1 and day == 1:
                    return False
                # 春节期间（简化判断）
                if month == 1 and 20 <= day <= 30:
                    return False
                if month == 2 and 1 <= day <= 10:
                    return False
                # 清明节期间
                if month == 4 and 3 <= day <= 6:
                    return False
                # 劳动节
                if month == 5 and 1 <= day <= 3:
                    return False
                # 端午节期间
                if month == 6 and 20 <= day <= 25:
                    return False
                # 中秋节期间
                if month == 9 and 15 <= day <= 17:
                    return False
                # 国庆节
                if month == 10 and 1 <= day <= 7:
                    return False
                
                return True
                
        except Exception as e:
            logger.warning(f"交易日判断失败，使用简单判断: {e}")
            # 回退到最简单的判断
            return dt.weekday() < 5
    
    def get_trading_days_in_range(self, start_date: datetime, end_date: datetime) -> List[datetime]:
        """获取指定日期范围内的所有交易日"""
        trading_days = []
        current_date = start_date
        
        while current_date <= end_date:
            if self.is_trading_day(current_date):
                trading_days.append(current_date)
            current_date += timedelta(days=1)
        
        return trading_days
    
    def get_last_n_trading_days(self, n: int, end_date: Optional[datetime] = None) -> List[datetime]:
        """获取最近N个交易日"""
        if end_date is None:
            end_date = datetime.now()
        
        trading_days = []
        current_date = end_date
        days_checked = 0
        max_days_to_check = n * 3  # 防止无限循环
        
        while len(trading_days) < n and days_checked < max_days_to_check:
            if self.is_trading_day(current_date):
                trading_days.append(current_date)
            current_date -= timedelta(days=1)
            days_checked += 1
        
        return list(reversed(trading_days))
    
    def get_next_trading_day(self, dt: datetime) -> datetime:
        """获取下一个交易日"""
        next_day = dt + timedelta(days=1)
        days_checked = 0
        max_days_to_check = 10  # 防止无限循环
        
        while not self.is_trading_day(next_day) and days_checked < max_days_to_check:
            next_day += timedelta(days=1)
            days_checked += 1
        
        return next_day

class TimeService:
    """时间服务类"""
    
    def __init__(self):
        self.trading_calendar = TradingCalendarService()
    
    def get_current_time(self, timezone: str = "Asia/Shanghai") -> Dict[str, Any]:
        """获取当前时间"""
        try:
            tz = pytz.timezone(timezone)
            current_time = datetime.now(tz)
            
            return {
                "当前时间": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "时区": timezone,
                "时间戳": int(current_time.timestamp()),
                "是否交易日": self.trading_calendar.is_trading_day(current_time),
                "是否交易时间": self.is_trading_time(current_time),
                "下一个交易日": self.trading_calendar.get_next_trading_day(current_time).strftime("%Y-%m-%d")
            }
        except Exception as e:
            error_msg = SecurityConfig.sanitize_error_message(e)
            return {"error": f"获取时间失败: {error_msg}"}
    
    def is_trading_time(self, dt: datetime) -> bool:
        """判断是否为交易时间"""
        if not self.trading_calendar.is_trading_day(dt):
            return False
        
        time = dt.time()
        # 上午交易时间：9:30-11:30
        morning_start = dt.replace(hour=9, minute=30, second=0).time()
        morning_end = dt.replace(hour=11, minute=30, second=0).time()
        
        # 下午交易时间：13:00-15:00
        afternoon_start = dt.replace(hour=13, minute=0, second=0).time()
        afternoon_end = dt.replace(hour=15, minute=0, second=0).time()
        
        return (morning_start <= time <= morning_end) or (afternoon_start <= time <= afternoon_end)
    
    def get_date_range_for_days(self, days: int, end_date: Optional[datetime] = None) -> tuple:
        """获取指定天数的日期范围（基于自然日）"""
        if end_date is None:
            end_date = datetime.now()
        
        start_date = end_date - timedelta(days=days)
        return start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")
    
    def get_date_range_for_trading_days(self, trading_days: int, end_date: Optional[datetime] = None) -> tuple:
        """获取指定交易日数量的日期范围"""
        if end_date is None:
            end_date = datetime.now()
        
        # 限制交易日数量
        trading_days = min(trading_days, SecurityConfig.MAX_HISTORY_DAYS // 2)
        
        # 获取最近N个交易日
        trading_day_list = self.trading_calendar.get_last_n_trading_days(trading_days, end_date)
        
        if trading_day_list:
            start_date = trading_day_list[0]
            return start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")
        else:
            # 回退到自然日计算
            return self.get_date_range_for_days(trading_days * 2, end_date)

# 创建时间服务实例
time_service = TimeService()

class StockDataService:
    """股票数据服务类"""
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = SecurityConfig.CACHE_TIMEOUT
        self.request_count = {}
        self.last_reset_time = datetime.now()
    
    def _check_rate_limit(self, client_id: str = "default") -> bool:
        """检查请求频率限制"""
        current_time = datetime.now()
        
        # 每分钟重置计数器
        if (current_time - self.last_reset_time).seconds >= 60:
            self.request_count = {}
            self.last_reset_time = current_time
        
        # 检查请求数量
        count = self.request_count.get(client_id, 0)
        if count >= SecurityConfig.RATE_LIMIT_REQUESTS:
            return False
        
        self.request_count[client_id] = count + 1
        return True
    
    def _validate_days_parameter(self, days: int, max_days: int) -> int:
        """验证并限制days参数"""
        if days <= 0:
            return 30  # 默认值
        return min(days, max_days)
    
    def _is_cache_valid(self, key: str) -> bool:
        """检查缓存是否有效"""
        if key not in self.cache:
            return False
        return (datetime.now() - self.cache[key]['timestamp']).seconds < self.cache_timeout
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """从缓存获取数据"""
        if self._is_cache_valid(key):
            return self.cache[key]['data']
        return None
    
    def _set_cache(self, key: str, data: Any):
        """设置缓存"""
        self.cache[key] = {
            'data': data,
            'timestamp': datetime.now()
        }
    
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
        # 检查请求频率
        if not self._check_rate_limit():
            return {"error": "请求过于频繁，请稍后重试"}
        
        cache_key = f"realtime_{symbol}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        try:
            # 验证股票代码格式
            if not symbol or len(symbol) != 6 or not symbol.isdigit():
                return {"error": "股票代码格式错误，请输入6位数字代码"}
            
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
            
            self._set_cache(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"获取股票实时数据失败: {e}")
            error_msg = SecurityConfig.sanitize_error_message(e)
            return {"error": f"获取数据失败: {error_msg}"}
    
    async def get_stock_history(self, symbol: str, period: str = "daily", 
                               start_date: str = None, end_date: str = None,
                               days: int = None) -> Dict[str, Any]:
        """获取股票历史数据"""
        # 检查请求频率
        if not self._check_rate_limit():
            return {"error": "请求过于频繁，请稍后重试"}
        
        # 验证股票代码格式
        if not symbol or len(symbol) != 6 or not symbol.isdigit():
            return {"error": "股票代码格式错误，请输入6位数字代码"}
        
        # 验证并限制days参数
        if days is not None:
            days = self._validate_days_parameter(days, SecurityConfig.MAX_HISTORY_DAYS)
        
        cache_key = f"history_{symbol}_{period}_{start_date}_{end_date}_{days}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        try:
            # 智能日期范围处理
            if not end_date or not start_date:
                current_time = datetime.now()
                if not end_date:
                    end_date = current_time.strftime("%Y%m%d")
                if not start_date:
                    if days:
                        # 使用基于交易日的日期范围
                        start_date, end_date = time_service.get_date_range_for_trading_days(days, current_time)
                    else:
                        # 默认获取30个交易日的数据
                        start_date, end_date = time_service.get_date_range_for_trading_days(30, current_time)
            
            # 获取历史数据
            df = ak.stock_zh_a_hist(symbol=symbol, period=period, 
                                   start_date=start_date, end_date=end_date)
            
            if df.empty:
                return {"error": f"未找到股票代码 {symbol} 在指定时间范围内的历史数据"}
            
            # 转换数据格式
            history_data = []
            for _, row in df.iterrows():
                history_data.append({
                    "日期": str(row['日期']),
                    "开盘": self._safe_convert(row['开盘'], float),
                    "收盘": self._safe_convert(row['收盘'], float),
                    "最高": self._safe_convert(row['最高'], float),
                    "最低": self._safe_convert(row['最低'], float),
                    "成交量": self._safe_convert(row['成交量'], int),
                    "成交额": self._safe_convert(row['成交额'], float),
                    "振幅": self._safe_convert(row['振幅'], float),
                    "涨跌幅": self._safe_convert(row['涨跌幅'], float),
                    "涨跌额": self._safe_convert(row['涨跌额'], float),
                    "换手率": self._safe_convert(row['换手率'], float)
                })
            
            result = {
                "代码": symbol,
                "周期": period,
                "开始日期": start_date,
                "结束日期": end_date,
                "数据条数": len(history_data),
                "历史数据": history_data,
                "数据说明": f"获取了{len(history_data)}条{period}数据"
            }
            
            self._set_cache(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"获取股票历史数据失败: {e}")
            error_msg = SecurityConfig.sanitize_error_message(e)
            return {"error": f"获取历史数据失败: {error_msg}"}
    
    async def get_market_index(self) -> Dict[str, Any]:
        """获取主要市场指数"""
        cache_key = "market_index"
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        try:
            # 获取当前时间信息
            time_info = time_service.get_current_time()
            
            # 获取主要指数数据
            indices = {
                "上证指数": "000001",
                "深证成指": "399001", 
                "创业板指": "399006",
                "科创50": "000688",
                "沪深300": "000300",
                "中证500": "000905"
            }
            
            index_data = {}
            
            # 混合使用现货数据和历史数据
            try:
                # 先尝试从现货数据获取
                df_spot = ak.stock_zh_index_spot_em()
                spot_available = not df_spot.empty
                
                for name, code in indices.items():
                    try:
                        # 对于深证成指和创业板指，使用历史数据接口获取最新数据
                        if code in ['399001', '399006', '399005']:
                            # 获取最近3天的数据，确保能获取到最新交易日数据
                            end_date = datetime.now().strftime("%Y%m%d")
                            start_date = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
                            
                            df_hist = ak.index_zh_a_hist(symbol=code, period='daily', 
                                                       start_date=start_date, end_date=end_date)
                            
                            if not df_hist.empty:
                                latest = df_hist.iloc[-1]
                                prev = df_hist.iloc[-2] if len(df_hist) > 1 else latest
                                
                                change = latest['收盘'] - prev['收盘']
                                change_pct = (change / prev['收盘'] * 100) if prev['收盘'] != 0 else 0
                                
                                index_data[name] = {
                                    "代码": code,
                                    "最新价": self._safe_convert(latest['收盘'], float),
                                    "涨跌额": self._safe_convert(change, float),
                                    "涨跌幅": self._safe_convert(change_pct, float),
                                    "昨收": self._safe_convert(prev['收盘'], float),
                                    "今开": self._safe_convert(latest['开盘'], float),
                                    "最高": self._safe_convert(latest['最高'], float),
                                    "最低": self._safe_convert(latest['最低'], float),
                                    "成交量": self._safe_convert(latest['成交量'], int),
                                    "数据来源": "历史数据接口"
                                }
                            else:
                                index_data[name] = {"error": f"无法获取指数 {code} 的历史数据"}
                        
                        # 对于其他指数，尝试从现货数据获取
                        else:
                            if spot_available:
                                index_info = df_spot[df_spot['代码'].str.contains(code, na=False)]
                                
                                if not index_info.empty:
                                    row = index_info.iloc[0]
                                    index_data[name] = {
                                        "代码": str(row['代码']),
                                        "最新价": self._safe_convert(row['最新价'], float),
                                        "涨跌额": self._safe_convert(row['涨跌额'], float),
                                        "涨跌幅": self._safe_convert(row['涨跌幅'], float),
                                        "昨收": self._safe_convert(row['昨收'], float),
                                        "今开": self._safe_convert(row['今开'], float),
                                        "最高": self._safe_convert(row['最高'], float),
                                        "最低": self._safe_convert(row['最低'], float),
                                        "数据来源": "现货数据接口"
                                    }
                                else:
                                    # 如果现货数据中没有，也尝试历史数据
                                    end_date = datetime.now().strftime("%Y%m%d")
                                    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
                                    
                                    try:
                                        df_hist = ak.index_zh_a_hist(symbol=code, period='daily', 
                                                                   start_date=start_date, end_date=end_date)
                                        
                                        if not df_hist.empty:
                                            latest = df_hist.iloc[-1]
                                            prev = df_hist.iloc[-2] if len(df_hist) > 1 else latest
                                            
                                            change = latest['收盘'] - prev['收盘']
                                            change_pct = (change / prev['收盘'] * 100) if prev['收盘'] != 0 else 0
                                            
                                            index_data[name] = {
                                                "代码": code,
                                                "最新价": self._safe_convert(latest['收盘'], float),
                                                "涨跌额": self._safe_convert(change, float),
                                                "涨跌幅": self._safe_convert(change_pct, float),
                                                "昨收": self._safe_convert(prev['收盘'], float),
                                                "今开": self._safe_convert(latest['开盘'], float),
                                                "最高": self._safe_convert(latest['最高'], float),
                                                "最低": self._safe_convert(latest['最低'], float),
                                                "成交量": self._safe_convert(latest['成交量'], int),
                                                "数据来源": "历史数据接口(备用)"
                                            }
                                        else:
                                            index_data[name] = {"error": f"未找到指数代码 {code}"}
                                    except Exception as hist_e:
                                        index_data[name] = {"error": f"获取指数 {code} 失败: {str(hist_e)}"}
                            else:
                                index_data[name] = {"error": "现货数据源不可用"}
                    
                    except Exception as e:
                        logger.warning(f"获取指数 {name}({code}) 失败: {e}")
                        index_data[name] = {"error": f"获取失败: {str(e)}"}
                        
            except Exception as e:
                logger.warning(f"获取指数数据失败: {e}")
                # 提供基本的指数信息
                for name, code in indices.items():
                    index_data[name] = {
                        "代码": code,
                        "状态": "数据获取失败",
                        "错误": str(e)
                    }
            
            result = {
                "更新时间": time_info.get("当前时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                "是否交易时间": time_info.get("是否交易时间", False),
                "是否交易日": time_info.get("是否交易日", False),
                "下一个交易日": time_info.get("下一个交易日", ""),
                "指数数据": index_data,
                "数据说明": "实时指数" if time_info.get("是否交易时间") else "非交易时间指数"
            }
            
            self._set_cache(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"获取市场指数失败: {e}")
            return {"error": f"获取市场指数失败: {str(e)}"}
    
    async def search_stock(self, keyword: str) -> Dict[str, Any]:
        """搜索股票"""
        # 检查请求频率
        if not self._check_rate_limit():
            return {"error": "请求过于频繁，请稍后重试"}
        
        # 验证关键词
        if not keyword or len(keyword.strip()) < 1:
            return {"error": "搜索关键词不能为空"}
        
        keyword = keyword.strip()
        cache_key = f"search_{keyword}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        try:
            # 获取当前时间信息
            time_info = time_service.get_current_time()
            
            # 获取所有A股列表
            df = ak.stock_zh_a_spot_em()
            
            # 按名称或代码搜索
            result_df = df[
                df['名称'].str.contains(keyword, na=False) | 
                df['代码'].str.contains(keyword, na=False)
            ].head(20)  # 限制返回20条结果
            
            if result_df.empty:
                return {"error": f"未找到包含关键词 '{keyword}' 的股票"}
            
            search_results = []
            for _, row in result_df.iterrows():
                search_results.append({
                    "代码": str(row['代码']),
                    "名称": str(row['名称']),
                    "最新价": self._safe_convert(row['最新价'], float),
                    "涨跌幅": self._safe_convert(row['涨跌幅'], float),
                    "成交量": self._safe_convert(row['成交量'], int)
                })
            
            result = {
                "搜索关键词": keyword,
                "结果数量": len(search_results),
                "搜索结果": search_results,
                "搜索时间": time_info.get("当前时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                "是否交易时间": time_info.get("是否交易时间", False)
            }
            
            self._set_cache(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"搜索股票失败: {e}")
            error_msg = SecurityConfig.sanitize_error_message(e)
            return {"error": f"搜索失败: {error_msg}"}
    
    def calculate_ma(self, prices: List[float], period: int) -> List[float]:
        """计算移动平均线"""
        if len(prices) < period:
            return [None] * len(prices)
        
        ma_values = []
        for i in range(len(prices)):
            if i < period - 1:
                ma_values.append(None)
            else:
                ma = sum(prices[i-period+1:i+1]) / period
                ma_values.append(round(ma, 2))
        
        return ma_values
    
    def calculate_macd(self, prices: List[float], fast=12, slow=26, signal=9) -> Dict[str, List[float]]:
        """计算MACD指标"""
        if len(prices) < slow:
            return {"DIF": [None] * len(prices), "DEA": [None] * len(prices), "MACD": [None] * len(prices)}
        
        # 计算EMA
        def ema(data, period):
            alpha = 2 / (period + 1)
            ema_values = [data[0]]
            for i in range(1, len(data)):
                ema_values.append(alpha * data[i] + (1 - alpha) * ema_values[-1])
            return ema_values
        
        ema_fast = ema(prices, fast)
        ema_slow = ema(prices, slow)
        
        # 计算DIF
        dif = [ema_fast[i] - ema_slow[i] for i in range(len(prices))]
        
        # 计算DEA
        dea = ema(dif, signal)
        
        # 计算MACD柱
        macd = [(dif[i] - dea[i]) * 2 for i in range(len(prices))]
        
        return {
            "DIF": [round(x, 4) if x is not None else None for x in dif],
            "DEA": [round(x, 4) if x is not None else None for x in dea],
            "MACD": [round(x, 4) if x is not None else None for x in macd]
        }
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> List[float]:
        """计算RSI指标"""
        if len(prices) < period + 1:
            return [None] * len(prices)
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            gains.append(max(change, 0))
            losses.append(max(-change, 0))
        
        rsi_values = [None]  # 第一个值为None
        
        # 计算初始平均值
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        for i in range(period, len(gains)):
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            rsi_values.append(round(rsi, 2))
            
            # 更新平均值
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        # 补齐前面的None值
        while len(rsi_values) < len(prices):
            rsi_values.insert(1, None)
        
        return rsi_values
    
    def calculate_kdj(self, highs: List[float], lows: List[float], closes: List[float], 
                     period: int = 9) -> Dict[str, List[float]]:
        """计算KDJ指标"""
        if len(closes) < period:
            return {"K": [None] * len(closes), "D": [None] * len(closes), "J": [None] * len(closes)}
        
        k_values = []
        d_values = []
        j_values = []
        
        for i in range(len(closes)):
            if i < period - 1:
                k_values.append(None)
                d_values.append(None)
                j_values.append(None)
            else:
                # 计算RSV
                high_max = max(highs[i-period+1:i+1])
                low_min = min(lows[i-period+1:i+1])
                if high_max == low_min:
                    rsv = 50
                else:
                    rsv = (closes[i] - low_min) / (high_max - low_min) * 100
                
                # 计算K值
                if i == period - 1:
                    k = rsv
                else:
                    k = (2/3) * k_values[i-1] + (1/3) * rsv if k_values[i-1] is not None else rsv
                
                k_values.append(round(k, 2))
                
                # 计算D值
                if i == period - 1:
                    d = k
                else:
                    d = (2/3) * d_values[i-1] + (1/3) * k if d_values[i-1] is not None else k
                
                d_values.append(round(d, 2))
                
                # 计算J值
                j = 3 * k - 2 * d
                j_values.append(round(j, 2))
        
        return {"K": k_values, "D": d_values, "J": j_values}
    
    def calculate_bollinger_bands(self, prices: List[float], period: int = 20, 
                                 std_dev: float = 2) -> Dict[str, List[float]]:
        """计算布林带"""
        if len(prices) < period:
            return {"UPPER": [None] * len(prices), "MIDDLE": [None] * len(prices), "LOWER": [None] * len(prices)}
        
        upper = []
        middle = []
        lower = []
        
        for i in range(len(prices)):
            if i < period - 1:
                upper.append(None)
                middle.append(None)
                lower.append(None)
            else:
                # 计算移动平均
                ma = sum(prices[i-period+1:i+1]) / period
                
                # 计算标准差
                variance = sum([(x - ma) ** 2 for x in prices[i-period+1:i+1]]) / period
                std = variance ** 0.5
                
                upper.append(round(ma + std_dev * std, 2))
                middle.append(round(ma, 2))
                lower.append(round(ma - std_dev * std, 2))
        
        return {"UPPER": upper, "MIDDLE": middle, "LOWER": lower}
    
    async def get_technical_indicators(self, symbol: str, period: str = "daily", 
                                     days: int = 60) -> Dict[str, Any]:
        """获取技术指标"""
        # 检查请求频率
        if not self._check_rate_limit():
            return {"error": "请求过于频繁，请稍后重试"}
        
        # 验证股票代码格式
        if not symbol or len(symbol) != 6 or not symbol.isdigit():
            return {"error": "股票代码格式错误，请输入6位数字代码"}
        
        # 验证并限制days参数
        days = self._validate_days_parameter(days, SecurityConfig.MAX_TECHNICAL_DAYS)
        
        cache_key = f"technical_{symbol}_{period}_{days}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        try:
            # 获取历史数据（需要额外的数据来计算技术指标）
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=days+60)).strftime("%Y%m%d")
            
            df = ak.stock_zh_a_hist(symbol=symbol, period=period, 
                                   start_date=start_date, end_date=end_date)
            
            if df.empty:
                return {"error": f"未找到股票代码 {symbol} 的数据"}
            
            # 获取价格数据
            closes = [self._safe_convert(x, float) for x in df['收盘'].tolist()]
            highs = [self._safe_convert(x, float) for x in df['最高'].tolist()]
            lows = [self._safe_convert(x, float) for x in df['最低'].tolist()]
            
            # 计算技术指标
            ma5 = self.calculate_ma(closes, 5)
            ma10 = self.calculate_ma(closes, 10)
            ma20 = self.calculate_ma(closes, 20)
            ma60 = self.calculate_ma(closes, 60)
            
            macd_data = self.calculate_macd(closes)
            rsi = self.calculate_rsi(closes)
            kdj_data = self.calculate_kdj(highs, lows, closes)
            boll_data = self.calculate_bollinger_bands(closes)
            
            # 组织返回数据
            indicators = []
            for i in range(len(df)):
                indicators.append({
                    "日期": df.iloc[i]['日期'].strftime("%Y-%m-%d"),
                    "收盘价": closes[i],
                    "最高价": highs[i],
                    "最低价": lows[i],
                    "MA5": ma5[i],
                    "MA10": ma10[i],
                    "MA20": ma20[i],
                    "MA60": ma60[i],
                    "DIF": macd_data["DIF"][i],
                    "DEA": macd_data["DEA"][i],
                    "MACD": macd_data["MACD"][i],
                    "RSI": rsi[i],
                    "KDJ_K": kdj_data["K"][i],
                    "KDJ_D": kdj_data["D"][i],
                    "KDJ_J": kdj_data["J"][i],
                    "BOLL_UPPER": boll_data["UPPER"][i],
                    "BOLL_MIDDLE": boll_data["MIDDLE"][i],
                    "BOLL_LOWER": boll_data["LOWER"][i]
                })
            
            # 只返回最近的数据
            indicators = indicators[-days:] if len(indicators) > days else indicators
            
            # 计算当前技术状态
            latest = indicators[-1] if indicators else {}
            technical_status = self._analyze_technical_status(latest)
            
            # 获取时间信息
            time_info = time_service.get_current_time()
            
            result = {
                "股票代码": symbol,
                "指标数据": indicators,
                "技术状态": technical_status,
                "更新时间": time_info.get("当前时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                "是否交易时间": time_info.get("是否交易时间", False),
                "数据说明": f"获取{days}天技术指标数据"
            }
            
            self._set_cache(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"获取技术指标失败: {e}")
            error_msg = SecurityConfig.sanitize_error_message(e)
            return {"error": f"获取技术指标失败: {error_msg}"}
    
    def _analyze_technical_status(self, latest_data: Dict) -> Dict[str, str]:
        """分析技术状态"""
        if not latest_data:
            return {}
        
        status = {}
        
        # MA趋势分析
        if all(x is not None for x in [latest_data.get('MA5'), latest_data.get('MA10'), latest_data.get('MA20')]):
            ma5, ma10, ma20 = latest_data['MA5'], latest_data['MA10'], latest_data['MA20']
            if ma5 > ma10 > ma20:
                status["MA趋势"] = "多头排列"
            elif ma5 < ma10 < ma20:
                status["MA趋势"] = "空头排列"
            else:
                status["MA趋势"] = "震荡整理"
        
        # MACD分析
        if all(x is not None for x in [latest_data.get('DIF'), latest_data.get('DEA'), latest_data.get('MACD')]):
            dif, dea, macd = latest_data['DIF'], latest_data['DEA'], latest_data['MACD']
            if dif > dea and macd > 0:
                status["MACD"] = "金叉向上"
            elif dif < dea and macd < 0:
                status["MACD"] = "死叉向下"
            else:
                status["MACD"] = "震荡调整"
        
        # RSI分析
        if latest_data.get('RSI') is not None:
            rsi = latest_data['RSI']
            if rsi > 80:
                status["RSI"] = "超买区域"
            elif rsi < 20:
                status["RSI"] = "超卖区域"
            elif rsi > 50:
                status["RSI"] = "强势区域"
            else:
                status["RSI"] = "弱势区域"
        
        # KDJ分析
        if all(x is not None for x in [latest_data.get('KDJ_K'), latest_data.get('KDJ_D'), latest_data.get('KDJ_J')]):
            k, d, j = latest_data['KDJ_K'], latest_data['KDJ_D'], latest_data['KDJ_J']
            if k > 80 and d > 80:
                status["KDJ"] = "超买区域"
            elif k < 20 and d < 20:
                status["KDJ"] = "超卖区域"
            elif k > d:
                status["KDJ"] = "金叉向上"
            else:
                status["KDJ"] = "死叉向下"
        
        # 布林带分析
        if all(x is not None for x in [latest_data.get('收盘价'), latest_data.get('BOLL_UPPER'), 
                                      latest_data.get('BOLL_MIDDLE'), latest_data.get('BOLL_LOWER')]):
            price = latest_data['收盘价']
            upper, middle, lower = latest_data['BOLL_UPPER'], latest_data['BOLL_MIDDLE'], latest_data['BOLL_LOWER']
            
            if price > upper:
                status["布林带"] = "突破上轨"
            elif price < lower:
                status["布林带"] = "跌破下轨"
            elif price > middle:
                status["布林带"] = "中轨上方"
            else:
                status["布林带"] = "中轨下方"
        
        # 综合评价
        bullish_signals = 0
        bearish_signals = 0
        
        for key, value in status.items():
            if value in ["多头排列", "金叉向上", "强势区域", "突破上轨", "中轨上方"]:
                bullish_signals += 1
            elif value in ["空头排列", "死叉向下", "弱势区域", "跌破下轨", "中轨下方"]:
                bearish_signals += 1
        
        if bullish_signals > bearish_signals:
            status["综合评价"] = "偏多信号"
        elif bearish_signals > bullish_signals:
            status["综合评价"] = "偏空信号"
        else:
            status["综合评价"] = "信号中性"
        
        return status

# 创建股票数据服务实例
stock_service = StockDataService()

@mcp.tool()
async def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    """获取当前时间信息
    
    Args:
        timezone: 时区，默认为Asia/Shanghai（北京时间）
    
    Returns:
        当前时间信息的JSON字符串，包含时间、交易状态等
    """
    try:
        result = time_service.get_current_time(timezone)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"获取当前时间失败: {e}")
        return json.dumps({"error": f"获取时间失败: {str(e)}"}, ensure_ascii=False)

@mcp.tool()
async def get_stock_realtime(symbol: str) -> str:
    """获取股票实时行情数据
    
    Args:
        symbol: 股票代码，如 '000001' 或 '600000'
    
    Returns:
        股票实时行情数据的JSON字符串
    """
    try:
        result = await stock_service.get_stock_realtime(symbol)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"获取股票实时数据失败: {e}")
        error_msg = SecurityConfig.sanitize_error_message(e)
        return json.dumps({"error": f"获取数据失败: {error_msg}"}, ensure_ascii=False)

@mcp.tool()
async def get_stock_history(symbol: str, period: str = "daily", 
                           start_date: Optional[str] = None, 
                           end_date: Optional[str] = None,
                           days: Optional[int] = None) -> str:
    """获取股票历史数据
    
    Args:
        symbol: 股票代码，如 '000001' 或 '600000'
        period: 数据周期，daily(日线), weekly(周线), monthly(月线)，默认daily
        start_date: 开始日期，格式YYYYMMDD，默认为30天前
        end_date: 结束日期，格式YYYYMMDD，默认为今天
        days: 获取天数，如果指定则忽略start_date，基于交易日计算
    
    Returns:
        股票历史数据的JSON字符串
    """
    try:
        result = await stock_service.get_stock_history(symbol, period, start_date, end_date, days)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"获取股票历史数据失败: {e}")
        error_msg = SecurityConfig.sanitize_error_message(e)
        return json.dumps({"error": f"获取历史数据失败: {error_msg}"}, ensure_ascii=False)

@mcp.tool()
async def get_market_index(random_string: str = "dummy") -> str:
    """获取主要市场指数（上证指数、深证成指、创业板指、科创50）
    
    Args:
        random_string: 占位参数，用于无参数工具的兼容性
    
    Returns:
        市场指数数据的JSON字符串
    """
    try:
        result = await stock_service.get_market_index()
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"获取市场指数失败: {e}")
        error_msg = SecurityConfig.sanitize_error_message(e)
        return json.dumps({"error": f"获取市场指数失败: {error_msg}"}, ensure_ascii=False)

@mcp.tool()
async def search_stock(keyword: str) -> str:
    """根据关键词搜索股票
    
    Args:
        keyword: 搜索关键词，可以是股票名称或代码的一部分
    
    Returns:
        搜索结果的JSON字符串
    """
    try:
        result = await stock_service.search_stock(keyword)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"搜索股票失败: {e}")
        error_msg = SecurityConfig.sanitize_error_message(e)
        return json.dumps({"error": f"搜索失败: {error_msg}"}, ensure_ascii=False)

@mcp.tool()
async def get_technical_indicators(symbol: str, period: str = "daily", days: int = 30) -> str:
    """获取股票技术指标
    
    Args:
        symbol: 股票代码，如 '000001' 或 '600000'
        period: 数据周期，daily(日线), weekly(周线), monthly(月线)，默认daily
        days: 获取天数，默认30天，最大120天
    
    Returns:
        技术指标数据的JSON字符串，包含MA、MACD、RSI等指标和技术状态分析
    """
    try:
        result = await stock_service.get_technical_indicators(symbol, period, days)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"获取技术指标失败: {e}")
        error_msg = SecurityConfig.sanitize_error_message(e)
        return json.dumps({"error": f"获取技术指标失败: {error_msg}"}, ensure_ascii=False)

@mcp.tool()
async def get_comprehensive_analysis(symbol: str) -> str:
    """获取股票综合分析报告
    
    Args:
        symbol: 股票代码，如 '000001' 或 '600000'
    
    Returns:
        综合分析报告的JSON字符串，包含实时行情、技术指标、历史数据等
    """
    try:
        # 获取当前时间信息
        time_info = time_service.get_current_time()
        
        # 并行获取各种数据
        realtime_data = await stock_service.get_stock_realtime(symbol)
        technical_data = await stock_service.get_technical_indicators(symbol, days=30)
        
        # 智能获取历史数据范围
        current_time = datetime.now()
        end_date = current_time.strftime("%Y%m%d")
        start_date = (current_time - timedelta(days=15)).strftime("%Y%m%d")
        history_data = await stock_service.get_stock_history(symbol, "daily", start_date, end_date)
        
        comprehensive_report = {
            "股票代码": symbol,
            "分析时间": time_info.get("当前时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            "是否交易时间": time_info.get("是否交易时间", False),
            "是否交易日": time_info.get("是否交易日", False),
            "下一个交易日": time_info.get("下一个交易日", ""),
            "实时行情": realtime_data,
            "技术分析": technical_data,
            "近期历史": history_data,
            "分析说明": {
                "数据时效性": "实时数据" if time_info.get("是否交易时间") else "非交易时间数据",
                "技术指标": "基于30天历史数据计算",
                "历史数据": "近15天交易数据",
                "风险提示": "数据仅供参考，不构成投资建议"
            }
        }
        
        return json.dumps(comprehensive_report, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"获取综合分析失败: {e}")
        error_msg = SecurityConfig.sanitize_error_message(e)
        return json.dumps({"error": f"获取综合分析失败: {error_msg}"}, ensure_ascii=False)

if __name__ == "__main__":
    mcp.run() 