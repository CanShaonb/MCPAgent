# 项目环境配置指南
## 项目框架
![img.png](readme_images/img.png)
## 环境准备

### 1. 激活 Conda 环境
```bash
conda create --name myenv python=3.12

conda activate myenv
```

### 2. 安装依赖包
```bash
cd MCPAgent

pip install -r requirements.txt
```

### 3. 部署 DuckDuckGo MCP 服务器
预先部署 [DuckDuckGo MCP Server](https://github.com/nickclyde/duckduckgo-mcp-server)，以便 Python 程序后续调用（建议运行时科学上网）。



## 使用说明见视频:

【金山文档 | WPS云文档】 6.15号框架与使用介绍
https://www.kdocs.cn/view/media/l/crGPOkOe56Wf