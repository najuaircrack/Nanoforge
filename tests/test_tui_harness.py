from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("textual")

from nanoforge.tui import NanoforgeApp


def test_tui_exposes_harness_screen():
    async def exercise() -> None:
        app = NanoforgeApp(project="configs/project-example.yaml")
        async with app.run_test(size=(140, 40)) as pilot:
            app.action_switch_screen("harness")
            await pilot.pause()
            assert app.screen.__class__.__name__ == "HarnessScreen"
            assert app.screen.query_one("#harness-project").value == "configs/project-example.yaml"

    asyncio.run(exercise())
