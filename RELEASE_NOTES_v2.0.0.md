# 🚀 股票MCP服务 v2.0.0 发布说明

## 📊 项目概述
中国股票市场行情MCP服务 - 基于Model Context Protocol的专业股票数据服务

## ✨ v2.0.0 主要新功能

### 📅 智能交易日历
- 🗓️ **真实交易日历**: 集成akshare获取中国股市官方交易日历
- 🎯 **节假日识别**: 自动识别春节、国庆等法定节假日和调休安排  
- 📊 **交易日计算**: 基于实际交易日数量计算历史数据范围
- 🔄 **智能回退**: 当交易日历不可用时自动回退到简化判断
- ⚡ **缓存优化**: 交易日历按年缓存，提高查询效率

### 🛡️ 安全增强
- 🚦 **请求频率限制**: 每分钟最多100次请求，防止过度使用
- 📏 **数据量限制**: 历史数据最大365天，技术指标最大120天
- 🔒 **输入验证**: 严格验证股票代码格式和参数范围
- 🛡️ **错误信息清理**: 自动清理敏感信息，避免内部细节泄露

### 🔧 技术改进
- ⚡ **智能缓存**: 5分钟数据缓存，减少重复请求
- 🔄 **增强错误处理**: 完善的异常处理和NaN值处理
- 📊 **数据验证**: 验证数据完整性和有效性
- 🎯 **参数限制**: 智能参数验证和范围限制

## 🛠️ 功能特性

### 📊 核心功能
- 🔄 **实时行情**: A股实时价格、涨跌幅、成交量
- 📈 **历史数据**: K线数据（日线、周线、月线）
- 📊 **市场指数**: 上证、深证、创业板、科创50等
- 🔍 **股票搜索**: 智能股票名称/代码搜索

### 📈 技术分析
- 📊 **移动平均线**: MA5、MA10、MA20、MA60
- 📈 **MACD指标**: DIF、DEA、MACD柱状图
- 📉 **RSI指标**: 相对强弱指数
- 🎯 **KDJ指标**: 随机指标K、D、J值
- 📊 **布林带**: 上轨、中轨、下轨

## 🚀 快速开始

### 安装
```bash
# 克隆仓库
git clone https://github.com/BigBossWHD/stock-mcp-service.git
cd stock-mcp-service

# 安装依赖
uv sync --no-install-project
```

### MCP配置
```json
{
  "mcpServers": {
    "stock-market": {
      "command": "uv",
      "args": ["--directory", "/path/to/stock_mcp_service", "run", "stock_mcp_server.py"],
      "transportType": "stdio"
    }
  }
}
```

## 🛡️ 安全部署

### 网络安全
- 🔒 不要直接暴露到公网
- 🛡️ 使用防火墙和VPN访问
- 📡 配置反向代理和SSL

### 访问控制  
- 👥 实施用户认证机制
- 🎫 使用API密钥或访问令牌
- 📊 记录审计日志

## 📋 使用限制
- 📏 历史数据: 最大365天
- 🔧 技术指标: 最大120天  
- 🚦 请求频率: 每分钟100次
- 💾 缓存时间: 5分钟自动过期

## 🔄 升级指南

### 从v1.0.0升级
1. 备份现有配置
2. 拉取最新代码: `git pull origin main`
3. 更新依赖: `uv sync --no-install-project`
4. 重启MCP服务

### 新增MCP工具
- `get_stock_history` 新增 `days` 参数支持
- 所有工具增强错误处理和安全验证

## 🐛 已知问题
- 部分市场指数（深证成指、创业板指）偶有网络连接问题
- 已实现智能重试和备用数据源机制

## 🔮 未来计划
- 增加更多技术指标（CCI、威廉指标等）
- 支持港股和美股数据
- 增加财务数据和基本面分析
- 优化数据源稳定性

## ⚠️ 免责声明
本服务提供的数据仅供参考，不构成投资建议。投资有风险，入市需谨慎。

## 📞 支持
- **GitHub Issues**: https://github.com/BigBossWHD/stock-mcp-service/issues
- **文档**: README.md
- **许可证**: MIT

---
**版本**: v2.0.0  
**发布日期**: 2025-06-12  
**开发者**: BigBossWHD 