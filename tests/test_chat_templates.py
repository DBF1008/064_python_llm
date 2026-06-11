from click.testing import CliRunner
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
def test_chat_template_fragments(tmpdir, mock_model, logs_db, templates_path):
    """Template fragments are merged into chat's first message."""
    frag_path = str(tmpdir / "context.txt")
    with open(frag_path, "w", encoding="utf-8") as fp:
        fp.write("Important context fragment")

    (templates_path / "with-frags.yaml").write_text(
        "model: mock\nfragments:\n- {}\n".format(frag_path), "utf-8"
    )

    runner = CliRunner()
    mock_model.enqueue(["first reply"])
    mock_model.enqueue(["second reply"])
    result = runner.invoke(
        llm.cli.cli,
        ["chat", "-t", "with-frags"],
        input="hello\nworld\nquit\n",
        catch_exceptions=False,
    )
    assert result.exit_code == 0

    # Fragment should be in the first prompt
    first_prompt = mock_model.history[0][0]
    assert "Important context fragment" in first_prompt.prompt

    # Fragment should NOT be repeated in the second prompt
    second_prompt = mock_model.history[1][0]
    assert "Important context fragment" not in (second_prompt.prompt or "")


@pytest.mark.xfail(sys.platform == "win32", reason="Expected to fail on Windows")
def test_chat_template_system_fragments_from_template(
    tmpdir, mock_model, logs_db, templates_path
):
    """Template system_fragments are merged into chat's system prompt."""
    sys_frag_path = str(tmpdir / "sys_context.txt")
    with open(sys_frag_path, "w", encoding="utf-8") as fp:
        fp.write("System fragment from template")

    (templates_path / "with-sys-frags.yaml").write_text(
        "model: mock\nsystem_fragments:\n- {}\n".format(sys_frag_path), "utf-8"
    )

    runner = CliRunner()
    mock_model.enqueue(["first"])
    mock_model.enqueue(["second"])
    result = runner.invoke(
        llm.cli.cli,
        ["chat", "-t", "with-sys-frags"],
        input="hello\nworld\nquit\n",
        catch_exceptions=False,
    )
    assert result.exit_code == 0

    # System fragment should appear in the first response's system
    responses = list(logs_db["responses"].rows)
    assert len(responses) == 2
    first_id = responses[0]["id"]

    sys_frags = list(logs_db["system_fragments"].rows)
    # At least one system fragment row tied to the first response
    frag_for_first = [sf for sf in sys_frags if sf["response_id"] == first_id]
    assert len(frag_for_first) >= 1


@pytest.mark.xfail(sys.platform == "win32", reason="Expected to fail on Windows")
def test_chat_template_schema(mock_model, logs_db, templates_path):
    """Template schema_object is passed through to conversation.chain() in chat."""
    (templates_path / "with-schema.yaml").write_text(
        "model: mock\n"
        "schema_object:\n"
        "  type: object\n"
        "  properties:\n"
        "    name:\n"
        "      type: string\n",
        "utf-8",
    )

    runner = CliRunner()
    mock_model.enqueue(["result"])
    result = runner.invoke(
        llm.cli.cli,
        ["chat", "-t", "with-schema"],
        input="hello\nquit\n",
        catch_exceptions=False,
    )
    assert result.exit_code == 0

    first_prompt = mock_model.history[0][0]
    assert first_prompt.schema == {
        "type": "object",
        "properties": {"name": {"type": "string"}},
    }


@pytest.mark.xfail(sys.platform == "win32", reason="Expected to fail on Windows")
def test_chat_template_options(mock_model, logs_db, templates_path):
    """Template options are applied in chat."""
    (templates_path / "with-options.yaml").write_text(
        "model: mock\noptions:\n  max_tokens: 100\n", "utf-8"
    )

    runner = CliRunner()
    mock_model.enqueue(["reply"])
    result = runner.invoke(
        llm.cli.cli,
        ["chat", "-t", "with-options"],
        input="hello\nquit\n",
        catch_exceptions=False,
    )
    assert result.exit_code == 0

    first_prompt = mock_model.history[0][0]
    assert first_prompt.options.max_tokens == 100


@pytest.mark.xfail(sys.platform == "win32", reason="Expected to fail on Windows")
def test_chat_template_options_cli_override(mock_model, logs_db, templates_path):
    """CLI -o options override template options in chat."""
    (templates_path / "with-options-override.yaml").write_text(
        "model: mock\noptions:\n  max_tokens: 100\n", "utf-8"
    )

    runner = CliRunner()
    mock_model.enqueue(["reply"])
    result = runner.invoke(
        llm.cli.cli,
        ["chat", "-t", "with-options-override", "-o", "max_tokens", "200"],
        input="hello\nquit\n",
        catch_exceptions=False,
    )
    assert result.exit_code == 0

    first_prompt = mock_model.history[0][0]
    assert first_prompt.options.max_tokens == 200
