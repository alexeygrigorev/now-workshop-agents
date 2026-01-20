from fastmcp import FastMCP
from toyaikit.tools import wrap_instance_methods
from search_tools import init_tools


def init_mcp():
    mcp = FastMCP("Demo")
    agent_tools = init_tools()
    wrap_instance_methods(mcp.tool, agent_tools)
    return mcp


if __name__ == "__main__":
    mcp = init_mcp()
    mcp.run()
