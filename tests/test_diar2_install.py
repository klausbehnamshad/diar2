# SPDX-License-Identifier: Apache-2.0
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


# --- NeMo configure: Homebrew paths, no conda ------------------------------------

CONFIGURE_STUB = """#!/bin/bash
# records its arguments and PATH, then writes a CMakeCache like cmake would
printf '%s\\n' "$@" > "$STUB_LOG/configure_args"
echo "$PATH" > "$STUB_LOG/configure_path"
mkdir -p "build/$1"
printf 'absl_DIR:PATH=%s\\nSENTENCEPIECE_LIB:FILEPATH=%s\\n' "$STUB_ABSL" "$STUB_SP" \\
    > "build/$1/CMakeCache.txt"
"""


def cmake_block():
    text = (HERE / "install_mac.sh").read_text(encoding="utf-8")
    return text[text.index("# --- cmake-hilfen: begin"):text.index("# --- cmake-hilfen: end")]


@pytest.fixture()
def nemo(tmp_path):
    brew_prefix = tmp_path / "homebrew"
    conda_base = tmp_path / "anaconda3"
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "brew").write_text(
        f'#!/bin/sh\n[ "$2" = abseil ] && echo "{brew_prefix}/opt/abseil" || echo "{brew_prefix}"\n')
    (bin_ / "conda").write_text(f'#!/bin/sh\necho "{conda_base}"\n')
    for f in ("brew", "conda"):
        (bin_ / f).chmod(0o755)
    (conda_base / "bin").mkdir(parents=True)
    src = tmp_path / "src"
    (src / "scripts").mkdir(parents=True)
    (src / "scripts" / "configure.sh").write_text(CONFIGURE_STUB)
    (src / "scripts" / "configure.sh").chmod(0o755)
    log = tmp_path / "log"
    log.mkdir()
    env = {
        "HOME": str(tmp_path), "STUB_LOG": str(log),
        "PATH": f"{conda_base}/bin:{bin_}:/usr/bin:/bin",
        "CONDA_PREFIX": str(tmp_path / "anaconda3/envs/other"),
        "STUB_ABSL": f"{brew_prefix}/opt/abseil/lib/cmake/absl",
        "STUB_SP": f"{brew_prefix}/lib/libsentencepiece.dylib",
    }
    return {"tmp": tmp_path, "brew": brew_prefix, "conda": conda_base, "src": src,
            "log": log, "env": env, "bin": bin_}


def run_configure(n):
    script = (f'set -euo pipefail\nCONDA="{n["bin"]}/conda"\n'
              'die() { echo "install_mac.sh: $*" >&2; exit 1; }\n'
              f'{cmake_block()}\nconfigure_nemo "{n["src"]}" metal-diar\n')
    return subprocess.run(["bash", "-c", script], env=n["env"], capture_output=True, text=True)


def write_cache(n, absl):
    build = n["src"] / "build" / "metal-diar"
    build.mkdir(parents=True, exist_ok=True)
    (build / "CMakeCache.txt").write_text(
        f"absl_DIR:PATH={absl}\nSENTENCEPIECE_LIB:FILEPATH={n['brew']}/lib/libsentencepiece.dylib\n")
    (build / "marker.o").write_text("old object")
    return build


def test_cache_with_anaconda_absl_is_discarded(nemo):
    build = write_cache(nemo, f"{nemo['conda']}/lib/cmake/absl")
    r = run_configure(nemo)
    assert r.returncode == 0, r.stderr
    assert "alter Build-Ordner mit fremden Pfaden verworfen" in r.stdout
    assert f"absl_DIR={nemo['conda']}/lib/cmake/absl" in r.stdout
    assert not (build / "marker.o").exists()          # old build dir is gone
    assert (build / "CMakeCache.txt").exists()        # freshly configured


def test_cache_with_homebrew_paths_is_kept(nemo):
    build = write_cache(nemo, f"{nemo['brew']}/opt/abseil/lib/cmake/absl")
    r = run_configure(nemo)
    assert r.returncode == 0, r.stderr
    assert (build / "marker.o").exists()
    assert "verworfen" not in r.stdout


def test_configure_gets_homebrew_options_and_no_conda_on_path(nemo):
    r = run_configure(nemo)
    assert r.returncode == 0, r.stderr
    args = (nemo["log"] / "configure_args").read_text().splitlines()
    assert args[0] == "metal-diar"
    assert f"-DCMAKE_PREFIX_PATH={nemo['brew']}" in args
    assert f"-Dabsl_DIR={nemo['brew']}/opt/abseil/lib/cmake/absl" in args
    ignore = [a for a in args if a.startswith("-DCMAKE_IGNORE_PREFIX_PATH=")]
    assert ignore == [f"-DCMAKE_IGNORE_PREFIX_PATH={nemo['conda']};{nemo['conda']}/envs/other"]
    path = (nemo["log"] / "configure_path").read_text().strip().split(":")
    assert not any(p.startswith(str(nemo["conda"])) for p in path)
    assert str(nemo["bin"]) in path


def test_wrong_absl_after_configure_aborts_with_both_paths(nemo):
    nemo["env"]["STUB_ABSL"] = f"{nemo['conda']}/lib/cmake/absl"
    r = run_configure(nemo)
    assert r.returncode != 0
    err = [l for l in r.stderr.splitlines() if l.startswith("install_mac.sh:")]
    assert len(err) == 1
    assert f"absl_DIR={nemo['conda']}/lib/cmake/absl" in err[0]
    assert f"SENTENCEPIECE_LIB={nemo['brew']}/lib/libsentencepiece.dylib" in err[0]


def test_result_independent_of_active_conda(nemo):
    # same options whether a conda env is active (CONDA_PREFIX set) or not
    del nemo["env"]["CONDA_PREFIX"]
    nemo["env"]["PATH"] = f"{nemo['bin']}:/usr/bin:/bin"
    r = run_configure(nemo)
    assert r.returncode == 0, r.stderr
    args = (nemo["log"] / "configure_args").read_text().splitlines()
    assert f"-Dabsl_DIR={nemo['brew']}/opt/abseil/lib/cmake/absl" in args
    assert f"-DCMAKE_IGNORE_PREFIX_PATH={nemo['conda']}" in args


# --- Python der conda-Umgebung ---------------------------------------------------

CONDA_STUB = """#!/bin/bash
case "$1" in
  run)
    case "$STUB_MODE" in
      blank) printf '%s\\n\\n' "$STUB_PY" ;;
      extra) printf '2 channel Terms of Service accepted\\n%s\\n\\n' "$STUB_PY" ;;
      warn)  echo "/opt/x/site-packages/foo.py:1: FutureWarning" >&2; echo "$STUB_PY" ;;
      *)     echo "EnvironmentLocationNotFound" >&2 ;;
    esac ;;
  info) echo "$STUB_BASE"; echo ;;
  config) printf 'envs_dirs:\\n  - %s\\n  - %s\\n' "$STUB_DIR1" "$STUB_DIR2" ;;
esac
"""


def python_block():
    text = (HERE / "install_mac.sh").read_text(encoding="utf-8")
    return text[text.index("# --- python-pfad: begin"):text.index("# --- python-pfad: end")]


def fake_python(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return path


@pytest.fixture()
def conda(tmp_path):
    stub = tmp_path / "conda"
    stub.write_text(CONDA_STUB)
    stub.chmod(0o755)
    env = {"PATH": "/usr/bin:/bin", "STUB_BASE": str(tmp_path / "anaconda3"),
           "STUB_DIR1": str(tmp_path / "home/.conda/envs"),
           "STUB_DIR2": str(tmp_path / "anaconda3/envs"),
           "STUB_PY": str(fake_python(tmp_path / "anaconda3/envs/diar2/bin/python"))}
    return tmp_path, stub, env


def run_env_python(conda, mode):
    tmp, stub, env = conda
    script = (f'set -euo pipefail\nCONDA="{stub}"\n'
              'die() { echo "install_mac.sh: $*" >&2; exit 1; }\n'
              f'{python_block()}\nenv_python diar2\necho "PY=$PY"\n')
    return subprocess.run(["bash", "-c", script], env=dict(env, STUB_MODE=mode),
                          capture_output=True, text=True)


@pytest.mark.parametrize("mode", ["blank", "extra", "warn"])
def test_python_from_conda_run_output(conda, mode):
    r = run_env_python(conda, mode)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == f"PY={conda[2]['STUB_PY']}"


def test_no_output_falls_back_to_conda_base(conda):
    r = run_env_python(conda, "none")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == f"PY={conda[0]}/anaconda3/envs/diar2/bin/python"


def test_fallback_to_envs_dirs(conda):
    tmp, stub, env = conda
    (tmp / "anaconda3/envs/diar2/bin/python").unlink()
    py = fake_python(tmp / "home/.conda/envs/diar2/bin/python")
    r = run_env_python(conda, "none")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == f"PY={py}"


def test_nothing_found_aborts_with_raw_output(conda):
    tmp, stub, env = conda
    (tmp / "anaconda3/envs/diar2/bin/python").unlink()
    r = run_env_python(conda, "none")
    assert r.returncode != 0
    err = [l for l in r.stderr.splitlines() if l.startswith("install_mac.sh:")]
    assert len(err) == 1
    assert "Python der Umgebung diar2 nicht gefunden" in err[0]
    assert "EnvironmentLocationNotFound" in err[0]
