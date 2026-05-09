def test_package_importable() -> None:
    import tooltuned_qwen

    assert tooltuned_qwen.__name__ == "tooltuned_qwen"
