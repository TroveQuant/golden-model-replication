from __future__ import annotations

from golden_model.index_membership import _codes_from_html


def test_official_html_parser_accepts_whitespace_in_index_heading() -> None:
    html = """
    <p>中证 500 指数样本股调整名单</p>
    <table>
      <tr><th>调出代码</th><th>名称</th><th>调入代码</th><th>名称</th></tr>
      <tr><td>600001</td><td>甲</td><td>000002</td><td>乙</td></tr>
    </table>
    """
    additions, removals = _codes_from_html(html, "中证500")
    assert additions == ["000002.XSHE"]
    assert removals == ["600001.XSHG"]


def test_official_html_parser_fails_without_matching_table() -> None:
    html = "<p>中证 500 指数样本股调整名单</p><p>附件缺失</p>"
    try:
        _codes_from_html(html, "中证500")
    except ValueError as exc:
        assert "no table" in str(exc)
    else:
        raise AssertionError("missing official table must fail closed")


def test_official_html_parser_merges_multiple_temporary_tables() -> None:
    html = """
    <p>上证50指数临时调整</p>
    <table>
      <tr><th>指数代码</th><th>指数</th><th>调出</th><th>名称</th><th>调入</th><th>名称</th></tr>
      <tr><td>000016</td><td>上证50</td><td>601299</td><td>甲</td><td>600893</td><td>乙</td></tr>
    </table>
    <p>上证50指数另一项临时调整</p>
    <table>
      <tr><th>指数代码</th><th>指数</th><th>调出</th><th>名称</th><th>调入</th><th>名称</th></tr>
      <tr><td>000016</td><td>上证50</td><td>600832</td><td>丙</td><td>600583</td><td>丁</td></tr>
      <tr><td>000010</td><td>上证180</td><td>600832</td><td>丙</td><td>600660</td><td>戊</td></tr>
    </table>
    """
    additions, removals = _codes_from_html(html, "上证50", "000016")
    assert additions == ["600893.XSHG", "600583.XSHG"]
    assert removals == ["601299.XSHG", "600832.XSHG"]
