from __future__ import annotations

from nanoforge.cli import build_parser


def test_public_cli_identity_is_latticekin_with_legacy_commands():
    parser = build_parser()
    assert parser.prog == "latticekin"
    args = parser.parse_args(["tui", "--run", "runs/example"])
    assert args.cmd == "tui"
    assert args.run == "runs/example"
