import asyncio
from fastmcp import Client


MCP_URL = "http://127.0.0.1:8001/mcp"


async def main():
    client = Client(MCP_URL)

    async with client:
        tools = await client.list_tools()

        print("TASK 14 - MCP CLIENT")
        print("=" * 60)

        print(f"MCP server: {MCP_URL}")
        print(f"Number of tools: {len(tools)}")

        for tool in tools:
            print(f"Tool ID: {tool.name}")

        print("=" * 60)

        loan_result = await client.call_tool(
            "get_loan_status",
            {"record_id": "CRD-0001"},
        )

        print("\nget_loan_status result:")
        print(loan_result)

        kb_result = await client.call_tool(
            "search_knowledge_base",
            {"query": "What are the KYC requirements?"},
        )

        print("\nsearch_knowledge_base result:")
        print(kb_result)

        tool_names = {tool.name for tool in tools}

        assert "get_loan_status" in tool_names
        assert "search_knowledge_base" in tool_names

        print("\n" + "=" * 60)
        print("TASK 14 VALIDATION: PASSED")


if __name__ == "__main__":
    asyncio.run(main())