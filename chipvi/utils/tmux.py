"""Tmux-based job dispatching tools."""

from __future__ import annotations

import subprocess


def run_in_tmux(session: str, commands: list[str]) -> None:
    """Run commands in a tmux session.

    Args:
        session (str): The name of the tmux session to create.
        commands (list[str]): The commands to run in the session.

    """
    # 1) Start a detached session
    subprocess.run(["/usr/bin/tmux", "new-session", "-d", "-s", session], check=True)

    # 2) For each extra command, split & then tile immediately to avoid pane size errors.
    for _ in commands[1:]:
        subprocess.run(["/usr/bin/tmux", "split-window", "-t", f"{session}:0", "-v"], check=True)
        subprocess.run(
            ["/usr/bin/tmux", "select-layout", "-t", f"{session}:0", "tiled"],
            check=True,
        )

    # 3) Splits have been laid out; grab pane indices.
    pane_list = (
        subprocess.check_output(
            ["/usr/bin/tmux", "list-panes", "-t", f"{session}:0", "-F", "#{pane_index}"],
        )
        .decode()
        .split()
    )

    # 4) Dispatch each command into its corresponding pane.
    for idx, cmd in enumerate(commands):
        pane = pane_list[idx]
        subprocess.run(
            ["/usr/bin/tmux", "send-keys", "-t", f"{session}:0.{pane}", cmd, "C-m"],
            check=True,
        )
