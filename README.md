# 中国股票市场行情 MCP 服务

🚀 基于 MCP (Model Context Protocol) 的中国股票市场行情服务，提供实时股票价格、历史数据、市场指数、技术分析等全面功能。

## ✨ 功能特性

### 📊 基础数据
- 🔄 **实时行情**: 获取A股实时价格、涨跌幅、成交量等数据
- 📈 **历史数据**: 查询股票历史K线数据（日线、周线、月线）
- 📊 **市场指数**: 获取上证指数、深证成指、创业板指、科创50、沪深300、中证500等主要指数
- 🔍 **股票搜索**: 根据股票名称或代码搜索相关股票
- ⏰ **时间服务**: 获取当前时间、交易状态、交易日判断等时间相关信息

### 📈 技术分析
- 📊 **移动平均线**: MA5、MA10、MA20、MA60
- 📈 **MACD指标**: DIF、DEA、MACD柱状图
- 📉 **RSI指标**: 相对强弱指数
- 🎯 **KDJ指标**: 随机指标K、D、J值
- 📊 **布林带**: 上轨、中轨、下轨
- 🎨 **技术状态**: 自动分析多头/空头排列、金叉死叉、超买超卖等

### 🔧 增强功能
- 📋 **综合分析**: 一键获取完整的股票分析报告
- ⚡ **数据缓存**: 内置缓存机制，提高数据获取效率
- 🛡️ **错误处理**: 完善的异常处理和NaN值处理
- 🎨 **智能分析**: 自动判断技术指标状态和趋势
- 📊 **综合评价**: 基于多个指标的综合信号分析
- 🔄 **混合数据源**: 智能切换数据源，确保数据完整性

## 📦 快速开始

```bash
# 克隆项目
git clone https://github.com/BigBossWHD/stock-mcp-service.git
cd stock-mcp-service

# 安装uv（如果尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装依赖
uv sync --no-install-project

# 运行服务
uv run python stock_mcp_server.py
```

## 🛠️ MCP工具

本服务提供7个MCP工具：

1. **get_current_time** - 时间服务
2. **get_stock_realtime** - 实时行情
3. **get_stock_history** - 历史数据
4. **get_market_index** - 市场指数
5. **search_stock** - 股票搜索
6. **get_technical_indicators** - 技术指标
7. **get_comprehensive_analysis** - 综合分析

## 🔧 MCP客户端配置

```json
{
  "mcpServers": {
    "stock-market": {
      "command": "uv",
      "args": ["--directory", "PATH_TO_PROJECT", "run", "stock_mcp_server.py"],
      "transportType": "stdio"
    }
  }
}
```

## 🏗️ 技术架构

- **MCP协议**: 使用FastMCP实现标准MCP协议通信
- **异步处理**: 基于asyncio的异步数据获取
- **数据缓存**: 内置5分钟缓存机制减少API调用
- **错误处理**: 完善的异常捕获和NaN值处理
- **技术计算**: 自研技术指标计算算法
- **智能分析**: 多指标综合技术状态分析
- **时间感知**: 集成时间服务，智能判断交易状态

## 📊 项目状态

### ✅ 已实现功能
- [x] 实时股票行情获取
- [x] 历史K线数据查询
- [x] 主要市场指数（6个指数全部支持）
- [x] 股票搜索功能
- [x] 完整技术指标计算（MA、MACD、RSI、KDJ、布林带）
- [x] 智能技术状态分析
- [x] 综合分析报告
- [x] 时间服务和交易状态判断
- [x] 数据缓存和错误处理
- [x] NaN值处理和类型安全
- [x] 混合数据源支持

### 🔧 最近修复
- [x] 深证成指数据源问题（使用历史数据接口）
- [x] 创业板指数据源问题（使用历史数据接口）
- [x] NaN值导致的数据转换错误
- [x] 实时行情在非交易时间的数据处理
- [x] 市场指数的数据完整性

## 📈 数据来源

本服务使用 [AKShare](https://github.com/akfamily/akshare) 库获取股票数据，数据来源包括：
- 东方财富
- 新浪财经
- 腾讯财经
- 等主流财经网站

## ⚠️ 免责声明

本项目仅供学习和研究使用，不构成任何投资建议。投资有风险，入市需谨慎。

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 🔗 相关链接

- [AKShare 文档](https://akshare.akfamily.xyz/)
- [MCP 协议规范](https://modelcontextprotocol.io/)
- [uv 包管理器](https://docs.astral.sh/uv/)