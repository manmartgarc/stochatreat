def test_import():
    from stochatreat import stochatreat  # noqa: PLC0415

    assert callable(stochatreat)
