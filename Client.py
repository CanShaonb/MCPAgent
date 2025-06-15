import asyncio
import json
from typing import Dict
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=".env", override=False)
# MCP-client,并具有记忆历史对话功能，后期可拓展多个工具。
class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.sessions: Dict[str, Dict] = {}  # 存储多个服务端会话
        self.tools_map: Dict[str, str] = {}  # 工具映射：工具名称 -> 服务端 ID
        self.exit_stack = AsyncExitStack()
        self.openai = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY2"),
            base_url=os.getenv("OPENAI_BASE_URL2")
        )
        self.history_messages = []

    async def connect_to_server(self, server_id: str, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_id: 服务端标识符
            server_script_path: Path to the server script (.py or .js)
        """
        if server_id in self.sessions:
            print(f"Server {server_id} is already connected")
            return

        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        is_model = 0

        if not (is_python or is_js):
            is_model = 1

        if is_model == 1:
            command = "uvx"
            server_params = StdioServerParameters(
                command=command,
                args=[server_script_path],
            )
        else:
            command = "python" if is_python else "node"
            server_params = StdioServerParameters(
                command=command,
                args=[server_script_path],
                env=None
            )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        stdio, write = stdio_transport
        session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))

        await session.initialize()

        # 存储会话信息
        self.sessions[server_id] = {
            "session": session,
            "stdio": stdio,
            "write": write
        }

        # List available tools and update tools mapping
        response = await session.list_tools()
        tools = response.tools
        print(f"\nConnected to server {server_id} with tools:", [tool.name for tool in tools])

        # 更新工具映射
        for tool in tools:
            if tool.name in self.tools_map:
                print(
                    f"Warning: Tool {tool.name} already exists in {self.tools_map[tool.name]}, overriding with {server_id}")
            self.tools_map[tool.name] = server_id

    async def get_all_available_tools(self):
        """获取所有服务器的工具列表"""
        available_tools = []

        for server_id, session_info in self.sessions.items():
            session = session_info["session"]
            response = await session.list_tools()

            for tool in response.tools:
                available_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                })

        return available_tools

    async def process_query(self, query: str, history_messages=None) -> str:
        """Process a query using OpenAI and available tools"""
        if history_messages:
            self.history_messages = history_messages

        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        # 获取所有可用工具
        available_tools = await self.get_all_available_tools()
        #print(json.dumps(available_tools, indent=4, ensure_ascii=False))

        # Initial OpenAI API call
        response = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=1000,
            messages=self.history_messages + messages,
            tools=available_tools
        )

        # Process response and handle tool calls
        tool_results = []
        final_text = []

        message = response.choices[0].message

        if message.content:
            final_text.append(message.content)
        final_content = ""
        if message.tool_calls:
            # 添加助手消息到对话历史
            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": message.tool_calls
            })

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)  # 使用json.loads而不是eval

                # 根据工具名称找到对应的服务端
                server_id = self.tools_map.get(tool_name)
                if not server_id:
                    error_msg = f"Tool {tool_name} not found in any connected server"
                    print(error_msg)
                    final_text.append(f"[Error: {error_msg}]")
                    continue

                if server_id not in self.sessions:
                    error_msg = f"Server {server_id} not connected"
                    print(error_msg)
                    final_text.append(f"[Error: {error_msg}]")
                    continue

                print(f"Calling tool {tool_name} on server {server_id} with args {tool_args}")

                # Execute tool call on the correct server
                session = self.sessions[server_id]["session"]
                result = await session.call_tool(tool_name, tool_args)
                tool_results.append({"call": tool_name, "result": result, "server": server_id})
                final_text.append(f"[Calling tool {tool_name} on server {server_id} with args {tool_args}]")

                # 添加工具调用结果到对话历史
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result.content)
                })

            # Get final response from OpenAI with all tool results
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=1000,
                messages=self.history_messages + messages,
            )

            final_content = response.choices[0].message.content
            messages.append({
                "role": "assistant",
                "content": final_content,
            })

            final_text.append(final_content)

        # 更新历史消息
        self.history_messages = self.history_messages + messages.copy()
        #print("Updated history messages:", len(self.history_messages))
        return final_content #"\n".join(final_text)

    async def list_tools(self):
        """列出所有服务端的工具"""
        if not self.sessions:
            print("No connected servers")
            return

        print("\nConnected servers and their tools:")
        for tool_name, server_id in self.tools_map.items():
            print(f"Tool: {tool_name}, Server: {server_id}")

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries, 'tools' to list tools, or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break
                elif query.lower() == 'tools':
                    await self.list_tools()
                    continue

                response = await self.process_query(query)
                print("\n" + "Agent: "+response)

            except Exception as e:
                print(f"\nError: {str(e)}")
                import traceback
                traceback.print_exc()

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()
        self.sessions.clear()
        self.tools_map.clear()


async def main():
    client = MCPClient()
    try:
        # 连接多个服务器，每个都有唯一的server_id
        await client.connect_to_server("duckduckgo", "duckduckgo-mcp-server") # https://github.com/nickclyde/duckduckgo-mcp-server
        await client.connect_to_server("RAGFlow", "RAGFlow.py")
        # 列出所有工具
        await client.list_tools()
        # 开始聊天循环
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())