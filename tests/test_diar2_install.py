"""install_mac.sh: source line goes into the login shell's startup file.

Runs the exact block between the "shell-eintrag" markers of install_mac.sh
with a temporary HOME; the rest of the installer needs macOS.
"""

import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
LINE = ".ohtools/diar2/diar2.sh"


def block():
    text = (HERE / "install_mac.sh").read_text(encoding="utf-8")
    start = text.index("# --- shell-eintrag: begin")
    end = text.index("# --- shell-eintrag: end")
    return text[start:end]


def run_block(home, shell):
    script = f'OHTOOLS="$HOME/Downloads/.ohtools"\n{block()}'
    env = {"HOME": str(home), "SHELL": shell, "PATH": "/usr/bin:/bin"}
    r = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


def count(path):
    return path.read_text().count(LINE) if path.exists() else 0


def test_bash_login_shell_writes_bash_profile(tmp_path):
    out = run_block(tmp_path, "/bin/bash")
    assert count(tmp_path / ".bash_profile") == 1
    assert not (tmp_path / ".zshrc").exists()
    assert "eingetragen in ~/.bash_profile" in out
    assert (tmp_path / ".bash_profile").read_text() == (
        f'source "{tmp_path}/Downloads/.ohtools/diar2/diar2.sh"  # diar2\n')


def test_zsh_login_shell_writes_zshrc(tmp_path):
    run_block(tmp_path, "/bin/zsh")
    assert count(tmp_path / ".zshrc") == 1
    assert not (tmp_path / ".bash_profile").exists()


@pytest.mark.parametrize("shell", ["/usr/local/bin/fish", ""])
def test_unknown_shell_writes_both(tmp_path, shell):
    run_block(tmp_path, shell)
    assert count(tmp_path / ".bash_profile") == 1 and count(tmp_path / ".zshrc") == 1


@pytest.mark.parametrize("shell,rc", [("/bin/bash", ".bash_profile"), ("/bin/zsh", ".zshrc")])
def test_second_run_adds_nothing(tmp_path, shell, rc):
    run_block(tmp_path, shell)
    before = (tmp_path / rc).read_text()
    out = run_block(tmp_path, shell)
    assert (tmp_path / rc).read_text() == before
    assert f"bereits eingetragen: ~/{rc}" in out


def test_existing_tilde_line_is_recognised(tmp_path):
    rc = tmp_path / ".bash_profile"
    rc.write_text('export PATH="$HOME/bin:$PATH"\nsource ~/Downloads/.ohtools/diar2/diar2.sh\n')
    run_block(tmp_path, "/bin/bash")
    assert count(rc) == 1
