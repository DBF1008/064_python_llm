"""Tests for tool namespace and conflict detection."""
import json
import warnings

import pytest
from click.testing import CliRunner

import llm
from llm import cli, hookimpl, plugins, Tool, Toolbox


# --- helpers ---

def _make_plugin(name, tools_to_register):
    """Create a plugin class that registers the given tools."""

    class _Plugin:
        __name__ = name

        @hookimpl
        def register_tools(self, register):
            for t in tools_to_register:
                register(t)

    return _Plugin()


def ns_upper(text: str) -> str:
    """Convert text to uppercase."""
    return text.upper()


def ns_lower(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def unique_a(text: str) -> str:
    """Unique tool A."""
    return "a:" + text


def unique_b(text: str) -> str:
    """Unique tool B."""
    return "b:" + text


# --- Tests ---


class TestToolNamespaceNoConflict:
    """When no name conflict exists, tools are available by both short and qualified name."""

    def test_short_and_qualified_names(self):
        plugin = _make_plugin("my_plugin", [unique_a])
        try:
            plugins.pm.register(plugin, name="my_plugin")
            tools = llm.get_tools()
            # Short name present
            assert "unique_a" in tools, f"Keys: {list(tools.keys())}"
            # Namespace is set on the tool
            assert tools["unique_a"].namespace == "my_plugin"
            assert tools["unique_a"].qualified_name == "my_plugin:unique_a"
            # Qualified name lookup via get_tool()
            tool = llm.get_tool("my_plugin:unique_a")
            assert tool.name == "unique_a"
        finally:
            plugins.pm.unregister(name="my_plugin")

    def test_get_tool_helper(self):
        plugin = _make_plugin("helper_ns", [unique_b])
        try:
            plugins.pm.register(plugin, name="helper_ns")
            tool = llm.get_tool("unique_b")
            assert tool.name == "unique_b"
            assert tool.namespace == "helper_ns"
            # Also works with qualified name
            tool2 = llm.get_tool("helper_ns:unique_b")
            assert tool2 == tool
        finally:
            plugins.pm.unregister(name="helper_ns")

    def test_get_tool_not_found(self):
        with pytest.raises(KeyError, match="no_such_tool"):
            llm.get_tool("no_such_tool")


class TestToolNamespaceConflict:
    """When multiple plugins register the same tool name, conflict handling kicks in."""

    def test_conflict_uses_qualified_names(self):
        # Both plugins register a function named "upper"
        def upper_v1(text: str) -> str:
            """V1 upper."""
            return text.upper()

        def upper_v2(text: str) -> str:
            """V2 upper."""
            return text.upper() + "!"

        # Give them the same function name
        upper_v1.__name__ = "upper"
        upper_v2.__name__ = "upper"

        p1 = _make_plugin("plugin_a", [upper_v1])
        p2 = _make_plugin("plugin_b", [upper_v2])
        try:
            plugins.pm.register(p1, name="plugin_a")
            plugins.pm.register(p2, name="plugin_b")

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                tools = llm.get_tools()
                # Should have emitted a warning about "upper"
                conflict_warnings = [
                    x for x in w if "upper" in str(x.message)
                ]
                assert len(conflict_warnings) == 1
                assert "multiple plugins" in str(conflict_warnings[0].message)

            # Short name should NOT be present
            assert "upper" not in tools, f"Keys: {list(tools.keys())}"
            # Qualified names should be present
            assert "plugin_a:upper" in tools
            assert "plugin_b:upper" in tools
            # They should be different tool objects
            assert tools["plugin_a:upper"] is not tools["plugin_b:upper"]
            assert tools["plugin_a:upper"].description == "V1 upper."
            assert tools["plugin_b:upper"].description == "V2 upper."
        finally:
            plugins.pm.unregister(name="plugin_a")
            plugins.pm.unregister(name="plugin_b")

    def test_non_conflicting_tools_unaffected(self):
        def shared(x: str) -> str:
            """Shared tool."""
            return x

        shared2 = lambda x: x  # noqa: E731
        shared2.__name__ = "shared"
        shared2.__doc__ = "Shared tool 2."

        p1 = _make_plugin("ns1", [shared, unique_a])
        p2 = _make_plugin("ns2", [shared2, unique_b])
        try:
            plugins.pm.register(p1, name="ns1")
            plugins.pm.register(p2, name="ns2")

            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                tools = llm.get_tools()

            # "shared" is conflicted
            assert "shared" not in tools
            assert "ns1:shared" in tools
            assert "ns2:shared" in tools
            # Non-conflicting tools still have short names
            assert "unique_a" in tools
            assert "unique_b" in tools
        finally:
            plugins.pm.unregister(name="ns1")
            plugins.pm.unregister(name="ns2")


class TestToolNamespaceCLI:
    """CLI supports namespace:tool_name references."""

    def test_cli_qualified_name_reference(self):
        plugin = _make_plugin("cli_ns", [unique_a])
        try:
            plugins.pm.register(plugin, name="cli_ns")
            runner = CliRunner()
            result = runner.invoke(
                cli.cli,
                ["prompt", "-T", "cli_ns:unique_a", "--no-stream", "hi"],
            )
            # Should not fail with "not found" (may fail for other reasons
            # like missing API key, but the tool lookup itself should succeed)
            assert "not found" not in (result.output + str(result.exception or ""))
        finally:
            plugins.pm.unregister(name="cli_ns")

    def test_cli_conflict_short_name_error(self):
        def tool_x() -> str:
            """Tool X."""
            return "x"

        tool_x2 = lambda: "y"  # noqa: E731
        tool_x2.__name__ = "tool_x"
        tool_x2.__doc__ = "Tool X v2."

        p1 = _make_plugin("pa", [tool_x])
        p2 = _make_plugin("pb", [tool_x2])
        try:
            plugins.pm.register(p1, name="pa")
            plugins.pm.register(p2, name="pb")
            runner = CliRunner()

            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                result = runner.invoke(
                    cli.cli,
                    ["prompt", "-T", "tool_x", "--no-stream", "hi"],
                )
            assert result.exit_code != 0
            assert "multiple plugins" in result.output
            assert "pa:tool_x" in result.output
            assert "pb:tool_x" in result.output
        finally:
            plugins.pm.unregister(name="pa")
            plugins.pm.unregister(name="pb")

    def test_tools_list_json_includes_namespace(self):
        plugin = _make_plugin("json_ns", [unique_a])
        try:
            plugins.pm.register(plugin, name="json_ns")
            runner = CliRunner()
            result = runner.invoke(cli.cli, ["tools", "--json"])
            assert result.exit_code == 0
            data = json.loads(result.output)
            tool_names = {t["name"] for t in data["tools"]}
            assert "unique_a" in tool_names
            # Find the tool and check namespace field
            ua = [t for t in data["tools"] if t["name"] == "unique_a"][0]
            assert ua["namespace"] == "json_ns"
        finally:
            plugins.pm.unregister(name="json_ns")

    def test_tools_list_deduplicates(self):
        """Qualified-name aliases should not appear as separate entries."""
        plugin = _make_plugin("dedup_ns", [unique_a])
        try:
            plugins.pm.register(plugin, name="dedup_ns")
            runner = CliRunner()
            result = runner.invoke(cli.cli, ["tools"])
            assert result.exit_code == 0
            # unique_a should appear once, not twice
            assert result.output.count("unique_a") == 1
        finally:
            plugins.pm.unregister(name="dedup_ns")


class TestToolNamespaceToolbox:
    """Toolbox classes get namespace propagated to their tools."""

    def test_toolbox_namespace(self):
        class MyBox(Toolbox):
            def greet(self, name: str) -> str:
                """Say hello."""
                return f"Hello, {name}!"

        plugin = _make_plugin("box_ns", [MyBox])
        try:
            plugins.pm.register(plugin, name="box_ns")
            tools = llm.get_tools()
            assert "MyBox" in tools
            toolbox_cls = tools["MyBox"]
            assert toolbox_cls.namespace == "box_ns"
            # Qualified name lookup via get_tool()
            assert llm.get_tool("box_ns:MyBox") is not None

            # Instance tools also get the namespace
            instance = toolbox_cls()
            instance_tools = list(instance.tools())
            assert len(instance_tools) == 1
            assert instance_tools[0].namespace == "box_ns"
            assert instance_tools[0].qualified_name == "box_ns:MyBox_greet"
        finally:
            plugins.pm.unregister(name="box_ns")


class TestToolNamespaceExecution:
    """execute_tool_calls resolves both short and qualified names."""

    def test_execute_resolves_qualified_name(self):
        model = llm.get_model("echo")

        def adder(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        tool = Tool.function(adder)
        tool.namespace = "test_ns"

        # Simulate LLM calling by qualified name
        chain = model.chain(
            json.dumps({
                "tool_calls": [
                    {"name": "test_ns:adder", "arguments": {"a": 1, "b": 2}}
                ]
            }),
            tools=[tool],
        )
        output = chain.text()
        # Should not contain an error about tool not found
        assert "does not exist" not in output

    def test_execute_resolves_short_name(self):
        model = llm.get_model("echo")

        def subber(a: int, b: int) -> int:
            """Subtract two numbers."""
            return a - b

        tool = Tool.function(subber)
        tool.namespace = "test_ns"

        accumulated = []

        def after_call(tool, tool_call, tool_result):
            accumulated.append(tool_result.output)

        chain = model.chain(
            json.dumps({
                "tool_calls": [
                    {"name": "subber", "arguments": {"a": 5, "b": 3}}
                ]
            }),
            tools=[tool],
            after_call=after_call,
        )
        chain.text()
        assert accumulated == ["2"]
