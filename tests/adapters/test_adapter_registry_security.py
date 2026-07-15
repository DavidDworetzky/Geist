from adapters.adapter_registry import find_adapter_classes


def test_legacy_registry_advertises_only_declared_actions():
    actions_by_adapter = dict(find_adapter_classes())

    assert actions_by_adapter["SearchAdapter"] == ["search"]
    assert actions_by_adapter["ImageGenerationAdapter"] == ["generate_image"]
    assert actions_by_adapter["MarkdownFileAdapter"] == [
        "read_file",
        "write_file",
        "get_files",
    ]
    assert all(
        not action.startswith("_") and action != "enumerate_actions"
        for actions in actions_by_adapter.values()
        for action in actions
    )
