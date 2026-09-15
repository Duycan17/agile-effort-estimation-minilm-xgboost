from storypoint_oof.cli import parse_args


def test_table3_compatible_protocol_is_available() -> None:
    args = parse_args(
        [
            "--data",
            "tests/fixtures/tawos.csv",
            "--protocol",
            "table3-compatible",
        ]
    )

    assert args.protocol == "table3-compatible"
