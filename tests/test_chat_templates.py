from click.testing import CliRunner
import json
import sys
import llm.cli
import pytest


@pytest.mark.xfail(sys.platform == "win32", reason="Expected to fail on Windows")
def test_chat_template_system_only_no_duplicate_prompt(
    mock_model, logs_db, templates_path
):
    # Template that only sets a system prompt, no user prompt
    (templates_path / "wild-french.yaml").write_text(
        "system: Speak in French\n", "utf-8"
    )

    runner = CliRunner()
    mock_model.enqueue(["Bonjour !"])
    result = runner.invoke(
        llm.cli.cli,
        ["chat", "-m", "mock", "-t", "wild-french"],
        input="hi\nquit\n",
        catch_exceptions=False,
    )
    assert result.exit_code == 0

    # Ensure the logged prompt is not duplicated (no "hi\nhi")
    rows = list(logs_db["responses"].rows)
    assert len(rows) == 1
    assert rows[0]["prompt"] == "hi"
    assert rows[0]["system"] == "Speak in French"


@pytest.mark.xfail(sys.platform == "win32", reason="Expected to fail on Windows")
def test_chat_system_fragments_only_first_turn(tmpdir, mock_model, logs_db):
    # Create a system fragment file
    sys_frag_path = str(tmpdir / "sys.txt")
    with open(sys_frag_path, "w", encoding="utf-8") as fp:
        fp.write("System fragment content")

    runner = CliRunner()
    # Two responses queued for two turns
    mock_model.enqueue(["first"])
    mock_model.enqueue(["second"])
    result = runner.invoke(
        llm.cli.cli,
        ["chat", "-m", "mock", "--system-fragment", sys_frag_path],
        input="Hi\nHi two\nquit\n",
        catch_exceptions=False,
    )
    assert result.exit_code == 0

    # Verify only the first response has the system fragment
    responses = list(logs_db["responses"].rows)
    assert len(responses) == 2
    first_id = responses[0]["id"]
    second_id = responses[1]["id"]

    sys_frags = list(logs_db["system_fragments"].rows)
    # Exactly one system fragment row, attached to the first response only
    assert len(sys_frags) == 1
    assert sys_frags[0]["response_id"] == first_id
    assert sys_frags[0]["response_id"] != second_id


@pytest.mark.xfail(sys.platform == "win32", reason="Expected to fail on Windows")
def test_chat_template_loads_tools_into_logs(logs_db, templates_path):
    # Template that specifies tools; ensure chat picks them up
    (templates_path / "mytools.yaml").write_text(
        "model: echo\n" "tools:\n" "- llm_version\n" "- llm_time\n",
        "utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        llm.cli.cli,
        ["chat", "-t", "mytools"],
        input="hi\nquit\n",
        catch_exceptions=False,
    )
    assert result.exit_code == 0

    # Verify a single response was logged for the conversation
    responses = list(logs_db["responses"].rows)
    assert len(responses) == 1
    assert responses[0]["prompt"] == "hi"
    response_id = responses[0]["id"]

    # Tools from the template should be recorded against that response
    rows = list(
        logs_db.query(
            """
            select tools.name from tools
            join tool_responses tr on tr.tool_id = tools.id
            where tr.response_id = ?
            order by tools.name
            """,
            [response_id],
        )
    )
    assert [r["name"] for r in rows] == ["llm_time", "llm_version"]


@pytest.mark.xfail(sys.platform == "win32", reason="Expected to fail on Windows")
def test_chat_template_inherits_fragments(tmpdir, templates_path):
    """Template fragments should be included in the first chat message."""
    frag_path = str(tmpdir / "frag.txt")
    with open(frag_path, "w", encoding="utf-8") as fp:
        fp.write("FRAGMENT_CONTENT")

    (templates_path / "with_frag.yaml").write_text(
        f"prompt: $input\nfragments:\n- {frag_path}\n",
        "utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        llm.cli.cli,
        ["chat", "-m", "echo", "-t", "with_frag"],
        input="hello\nquit\n",
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    # Fragment should appear in the first message output
    assert "FRAGMENT_CONTENT" in result.output


@pytest.mark.xfail(sys.platform == "win32", reason="Expected to fail on Windows")
def test_chat_template_inherits_system_fragments(tmpdir, templates_path):
    """Template system_fragments should be included in the first chat message's system prompt."""
    sys_frag_path = str(tmpdir / "sys_frag.txt")
    with open(sys_frag_path, "w", encoding="utf-8") as fp:
        fp.write("SYS_FRAG_CONTENT")

    (templates_path / "with_sysfrag.yaml").write_text(
        f"prompt: $input\nsystem: Base system\nsystem_fragments:\n- {sys_frag_path}\n",
        "utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        llm.cli.cli,
        ["chat", "-m", "echo", "-t", "with_sysfrag"],
        input="hello\nquit\n",
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    # System fragment should be combined with the system prompt
    assert "SYS_FRAG_CONTENT" in result.output
    assert "Base system" in result.output


@pytest.mark.xfail(sys.platform == "win32", reason="Expected to fail on Windows")
def test_chat_template_inherits_schema(templates_path, logs_db, mock_model):
    """Template schema_object should be passed through to the chat response."""
    schema_json = json.dumps(
        {"type": "object", "properties": {"name": {"type": "string"}}}
    )
    (templates_path / "with_schema.yaml").write_text(
        f"prompt: $input\nschema_object: {schema_json}\n",
        "utf-8",
    )

    mock_model.enqueue(["hello response"])
    runner = CliRunner()
    result = runner.invoke(
        llm.cli.cli,
        ["chat", "-m", "mock", "-t", "with_schema"],
        input="hello\nquit\n",
        catch_exceptions=False,
    )
    assert result.exit_code == 0

    # The response should have a non-null schema_id
    responses = list(logs_db["responses"].rows)
    assert len(responses) == 1
    assert responses[0]["schema_id"] is not None


@pytest.mark.xfail(sys.platform == "win32", reason="Expected to fail on Windows")
def test_chat_template_inherits_options(templates_path):
    """Template options should be merged with CLI options, CLI taking priority."""
    (templates_path / "with_opts.yaml").write_text(
        "prompt: $input\nmodel: echo\noptions:\n  example_bool: true\n",
        "utf-8",
    )

    runner = CliRunner()
    # No CLI -o option, template option should be used
    result = runner.invoke(
        llm.cli.cli,
        ["chat", "-t", "with_opts"],
        input="hello\nquit\n",
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert '"example_bool": true' in result.output


@pytest.mark.xfail(sys.platform == "win32", reason="Expected to fail on Windows")
def test_chat_template_cli_options_override_template(templates_path):
    """CLI -o options should override template options."""
    (templates_path / "with_opts2.yaml").write_text(
        "prompt: $input\nmodel: echo\noptions:\n  example_bool: true\n",
        "utf-8",
    )

    runner = CliRunner()
    # CLI -o should override template option
    result = runner.invoke(
        llm.cli.cli,
        ["chat", "-t", "with_opts2", "-o", "example_bool", "false"],
        input="hello\nquit\n",
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert '"example_bool": false' in result.output


@pytest.mark.xfail(sys.platform == "win32", reason="Expected to fail on Windows")
def test_chat_template_fragments_merge_with_cli(tmpdir, templates_path):
    """Template fragments and CLI fragments should both appear, template first."""
    tmpl_frag = str(tmpdir / "tmpl_frag.txt")
    cli_frag = str(tmpdir / "cli_frag.txt")
    with open(tmpl_frag, "w", encoding="utf-8") as fp:
        fp.write("TMPL_FRAG")
    with open(cli_frag, "w", encoding="utf-8") as fp:
        fp.write("CLI_FRAG")

    (templates_path / "merge_frags.yaml").write_text(
        f"prompt: $input\nfragments:\n- {tmpl_frag}\n",
        "utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        llm.cli.cli,
        ["chat", "-m", "echo", "-t", "merge_frags", "-f", cli_frag],
        input="hello\nquit\n",
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert "TMPL_FRAG" in result.output
    assert "CLI_FRAG" in result.output


@pytest.mark.xfail(sys.platform == "win32", reason="Expected to fail on Windows")
def test_chat_remote_template_inherits_fragments(httpx_mock, tmpdir):
    """Remote (URL) templates should also inherit fragments in chat."""
    frag_path = str(tmpdir / "remote_frag.txt")
    with open(frag_path, "w", encoding="utf-8") as fp:
        fp.write("REMOTE_FRAG_CONTENT")

    template_yaml = f"prompt: $input\nfragments:\n- {frag_path}\n"
    httpx_mock.add_response(
        url="https://example.com/chat_tmpl.yaml",
        method="GET",
        text=template_yaml,
        status_code=200,
        is_reusable=True,
    )

    runner = CliRunner()
    result = runner.invoke(
        llm.cli.cli,
        ["chat", "-m", "echo", "-t", "https://example.com/chat_tmpl.yaml"],
        input="hello\nquit\n",
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert "REMOTE_FRAG_CONTENT" in result.output
