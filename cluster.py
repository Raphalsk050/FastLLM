#!/usr/bin/env python3
"""Run one llama.cpp model across many machines (Windows, Linux or macOS).

The main host runs llama-server / llama-bench. Every worker runs ggml-rpc-server
bound to 127.0.0.1, and the main host reaches it through an SSH tunnel, so the
unauthenticated RPC port is never exposed on the network.

Commands: init, add, remove, status, install, start, models, bench, serve, endpoint, stop, clean.
Run `python cluster.py <command> -h` for the options of each one.
"""

import argparse
import base64
import configparser
import hashlib
import http.server
import json
import os
import platform
import re
import secrets
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "cluster.json"
MODELS_PATH = HERE / "models.ini"
GENERATED = HERE / "generated"
RUNTIME = Path.home() / ".llama-cluster"
DOWNLOADS = RUNTIME / "downloads"
LOGS = RUNTIME / "logs"
STATE_PATH = RUNTIME / "state.json"
SCRIPT = Path(__file__).name
CMD = ("python " if os.name == "nt" else "python3 ") + SCRIPT  # how to run this tool, for the hints we print

DEFAULTS = {
    "build": "b11115",
    "ssh_key": "~/.ssh/llama_cluster_ed25519",
    "main_ip": "",
    "base_port": 50052,
    "remote_port": 50052,
    # Free VRAM left on each GPU. The main host runs the desktop and other apps, and on Windows a GPU that
    # gets too full silently spills to system RAM (generation drops below 1 token/s), so it gets more.
    "vram_margin_main_mib": 1536,
    "vram_margin_mib": 1024,
    "rpc_cache": True,
    "local_backend": "auto",
    "server_port": 8080,
    "workers": [],
}

# Model profile keys (named after the LM Studio settings) -> llama-server flag and value type.
PROFILE_FLAGS = {
    "context_length": ("-c", int),
    "gpu_offload": ("-ngl", str),
    "flash_attention": ("-fa", ("on", "off", "auto")),
    "k_cache_type": ("-ctk", str),
    "v_cache_type": ("-ctv", str),
    "cpu_threads": ("-t", int),
    "batch_size": ("-b", int),
    "ubatch_size": ("-ub", int),
    "moe_cpu_layers": ("--n-cpu-moe", int),
    "rope_freq_base": ("--rope-freq-base", float),
    "rope_freq_scale": ("--rope-freq-scale", float),
    "seed": ("--seed", int),
    "parallel": ("-np", int),
    "temperature": ("--temp", float),
    "top_k": ("--top-k", int),
    "top_p": ("--top-p", float),
    "min_p": ("--min-p", float),
    "repeat_penalty": ("--repeat-penalty", float),
    "repeat_last_n": ("--repeat-last-n", int),
    "presence_penalty": ("--presence-penalty", float),
    "frequency_penalty": ("--frequency-penalty", float),
    "max_tokens": ("-n", int),
    "thinking": ("--reasoning", ("on", "off", "auto")),
    "chat_template_file": ("--chat-template-file", str),
}
PROFILE_OTHER = ("model", "hf", "alias", "use_mmap", "keep_in_memory", "offload_kv_cache", "extra")

MODELS_TEMPLATE = """\
# Perfis de modelo, com as opcoes do LM Studio.
# Cada [secao] e um perfil:  python cluster.py serve <perfil>
# Linha comentada (#) ou ausente = padrao do llama.cpp.
# As opcoes de geracao sao o padrao do servidor; cada requisicao da API pode mandar as suas.
# Opcoes em [DEFAULT] valem para todos os perfis.

[exemplo]
# --- Modelo: arquivo GGUF local OU repositorio do Hugging Face ---
model = /caminho/para/modelo.gguf
# hf = Qwen/Qwen3-14B-GGUF:Q4_K_M
# Nome que os clientes da API veem (padrao: nome do perfil)
# alias = meu-modelo

# --- Carregamento (Load) ---
# Context Length, em tokens
context_length = 8192
# GPU Offload: auto, all ou um numero de camadas
# gpu_offload = auto
# Flash Attention: on, off, auto
# flash_attention = auto
# K/V Cache Quantization Type: f16, q8_0, q4_0 (quantizado exige flash_attention)
# k_cache_type = f16
# v_cache_type = f16
# Offload KV Cache to GPU Memory
# offload_kv_cache = true
# Try mmap() / Keep Model in Memory
# use_mmap = true
# keep_in_memory = false
# CPU Thread Pool Size
# cpu_threads = 8
# Evaluation Batch Size
# batch_size = 2048
# Force Model Expert Weights onto CPU: quantas camadas MoE ficam na CPU
# moe_cpu_layers = 0
# RoPE Frequency Base / Scale
# rope_freq_base = 1000000
# rope_freq_scale = 1.0
# Seed (-1 = aleatoria)
# seed = -1
# Requisicoes atendidas ao mesmo tempo
# parallel = 1

# --- Geracao (Inference) ---
temperature = 0.7
top_k = 20
top_p = 0.8
min_p = 0.0
repeat_penalty = 1.0
# repeat_last_n = 64
# presence_penalty = 0.0
# frequency_penalty = 0.0
# Limit Response Length, em tokens (-1 = sem limite)
max_tokens = -1
# Raciocinio (thinking): on, off, auto
# thinking = auto
# Prompt Template (arquivo Jinja)
# chat_template_file = /caminho/template.jinja

# --- Qualquer outro argumento do llama-server ---
# extra = --cache-reuse 256
"""

# Prebuilt llama.cpp release assets per (os, arch, backend); "{b}" is the build tag.
# The CUDA packages need the matching "cudart" package extracted in the same folder.
ASSETS = {
    ("windows", "x64", "cuda"): ["llama-{b}-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip"],
    ("windows", "x64", "vulkan"): ["llama-{b}-bin-win-vulkan-x64.zip"],
    ("windows", "x64", "cpu"): ["llama-{b}-bin-win-cpu-x64.zip"],
    ("windows", "arm64", "cpu"): ["llama-{b}-bin-win-cpu-arm64.zip"],
    ("linux", "x64", "cuda"): ["llama-{b}-bin-ubuntu-cuda-12.8-x64.tar.gz", "cudart-llama-{b}-bin-ubuntu-cuda-12.8-x64.tar.gz"],
    ("linux", "x64", "vulkan"): ["llama-{b}-bin-ubuntu-vulkan-x64.tar.gz"],
    ("linux", "x64", "cpu"): ["llama-{b}-bin-ubuntu-x64.tar.gz"],
    ("linux", "arm64", "cuda"): ["llama-{b}-bin-ubuntu-cuda-13.4-arm64.tar.gz", "cudart-llama-{b}-bin-ubuntu-cuda-13.4-arm64.tar.gz"],
    ("linux", "arm64", "vulkan"): ["llama-{b}-bin-ubuntu-vulkan-arm64.tar.gz"],
    ("linux", "arm64", "cpu"): ["llama-{b}-bin-ubuntu-arm64.tar.gz"],
    ("macos", "arm64", "metal"): ["llama-{b}-bin-macos-arm64.tar.gz"],
    ("macos", "x64", "cpu"): ["llama-{b}-bin-macos-x64.tar.gz"],
}

# The Linux CUDA packages are built on Ubuntu 24.04.
CUDA_LINUX_MIN_GLIBC = (2, 39)

WINDOWS_DETECT = r"""
$backend = 'cpu'; $gpus = ''
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
  $l = @(& nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>$null)
  if ($LASTEXITCODE -eq 0 -and $l.Count -gt 0) { $backend = 'cuda'; $gpus = ($l -join ';') }
}
if ($backend -eq 'cpu') {
  $v = @(Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'Radeon RX|Radeon Pro|Arc\(TM\) [AB]|Arc [AB]' })
  if ($v.Count -gt 0) { $backend = 'vulkan'; $gpus = ($v.Name -join ';') }
}
$arch = if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { 'arm64' } else { 'x64' }
[pscustomobject]@{ os = 'windows'; arch = $arch; home = $HOME; backend = $backend; gpus = $gpus; glibc = '' } | ConvertTo-Json -Compress
"""

POSIX_DETECT = r"""
os=$(uname -s); arch=$(uname -m); backend=cpu; gpus=""
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  backend=cuda; gpus=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | tr '\n' ';')
elif [ "$os" = "Darwin" ] && [ "$arch" = "arm64" ]; then
  backend=metal; gpus=$(sysctl -n machdep.cpu.brand_string 2>/dev/null)
elif command -v lspci >/dev/null 2>&1; then
  g=$(lspci | grep -Ei 'vga|3d controller|display' | grep -Ei 'radeon rx|radeon pro|navi|arc a|arc b' | sed 's/^[^:]*: //' | tr '\n' ';')
  if [ -n "$g" ]; then backend=vulkan; gpus=$g; fi
fi
glibc=$(ldd --version 2>/dev/null | head -n 1 | grep -Eo '[0-9]+\.[0-9]+$')
clean() { printf '%s' "$1" | tr -d '"\\' ; }
printf '{"os":"%s","arch":"%s","home":"%s","backend":"%s","gpus":"%s","glibc":"%s"}\n' \
  "$(clean "$os")" "$(clean "$arch")" "$(clean "$HOME")" "$backend" "$(clean "$gpus")" "$(clean "$glibc")"
"""


# ---------------------------------------------------------------- helpers

def die(msg):
    print(f"erro: {msg}", file=sys.stderr)
    sys.exit(1)


def load_config():
    if not CONFIG_PATH.exists():
        die(f"{CONFIG_PATH.name} nao existe. Rode primeiro: {CMD} init")
    cfg = dict(DEFAULTS)
    cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    return cfg


def save_config(cfg):
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"tunnels": []}


def save_state(state):
    RUNTIME.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def psq(s):
    """PowerShell single-quoted literal."""
    return "'" + str(s).replace("'", "''") + "'"


def shq(s):
    """POSIX single-quoted literal."""
    return "'" + str(s).replace("'", "'\\''") + "'"


def norm_arch(machine):
    m = machine.lower()
    if m in ("x86_64", "amd64", "x64"):
        return "x64"
    if m in ("aarch64", "arm64"):
        return "arm64"
    return m


def main_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # UDP connect only picks the route; nothing is sent
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def glibc_tuple(text):
    m = re.match(r"(\d+)\.(\d+)", text or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def clean_ps_stderr(text):
    """Windows PowerShell sends errors over SSH as CLIXML; keep only the messages."""
    if "#< CLIXML" not in text:
        return text.strip()
    parts = re.findall(r'<S S="Error">(.*?)</S>', text, flags=re.S)
    msg = "".join(parts).replace("_x000D__x000A_", "\n")
    return msg.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&").strip()


# ---------------------------------------------------------------- ssh

def key_path(cfg):
    return os.path.expanduser(cfg["ssh_key"])


def ssh_opts(cfg):
    return ["-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ServerAliveInterval=15", "-i", key_path(cfg)]


def dest(w):
    return f"{w['user']}@{w['host']}"


def ps_encoded(script):
    body = "$ProgressPreference = 'SilentlyContinue'\n" + script
    return "powershell -NoProfile -NonInteractive -EncodedCommand " + base64.b64encode(body.encode("utf-16-le")).decode()


def run_remote(cfg, w, script, timeout=300):
    """Run a PowerShell script (Windows) or a sh script (Linux/macOS) on a worker."""
    if w["os"] == "windows":
        cmd, stdin = ["ssh", *ssh_opts(cfg), dest(w), ps_encoded(script)], None
    else:
        # Bytes, not text: text mode on Windows would turn "\n" into "\r\n" and break sh.
        cmd, stdin = ["ssh", *ssh_opts(cfg), dest(w), "sh -s"], script.encode("utf-8")
    r = subprocess.run(cmd, input=stdin, capture_output=True, timeout=timeout)
    out = r.stdout.decode("utf-8", errors="replace")
    err = r.stderr.decode("utf-8", errors="replace")
    err = clean_ps_stderr(err) if w["os"] == "windows" else err.strip()
    return r.returncode, out, err


def last_json(text):
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    return None


def scp(cfg, w, local, remote_dir):
    if w["os"] == "windows":
        remote_dir = "/" + remote_dir.replace("\\", "/")
    target = f"{dest(w)}:{remote_dir.rstrip('/')}/"
    r = subprocess.run(["scp", "-q", *ssh_opts(cfg), str(local), target], capture_output=True, text=True)
    if r.returncode != 0:
        die(f"scp para {w['name']} falhou: {r.stderr.strip()}")


def remote_join(w, *parts):
    sep = "\\" if w["os"] == "windows" else "/"
    return sep.join([w["home"].rstrip("\\/"), *parts])


def remote_dir(cfg, w):
    return remote_join(w, "llama-cluster", f"{cfg['build']}-{w['backend']}")


def remote_downloads(cfg, w):
    return remote_join(w, "llama-cluster", "downloads", cfg["build"])


def kill_script(w):
    if w["os"] == "windows":
        return "Get-Process ggml-rpc-server -ErrorAction SilentlyContinue | Stop-Process -Force\n'stopped'"
    return "pkill -f ggml-rpc-server 2>/dev/null; echo stopped"


# ---------------------------------------------------------------- machines

def detect(cfg, w):
    r = subprocess.run(["ssh", *ssh_opts(cfg), dest(w), "uname -s"], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=60)
    if r.returncode == 255:
        die(f"sem acesso SSH a {dest(w)}: {r.stderr.strip()}\n"
            f"Rode o script de preparacao nessa maquina ({CMD} init gera os scripts).")
    posix = r.returncode == 0 and r.stdout.strip() in ("Linux", "Darwin")
    probe = dict(w, os="linux" if posix else "windows")
    rc, out, err = run_remote(cfg, probe, POSIX_DETECT if posix else WINDOWS_DETECT, timeout=120)
    info = last_json(out)
    if not info:
        die(f"nao consegui ler o hardware de {w['host']}: {err or out}")
    info["os"] = {"Linux": "linux", "Darwin": "macos"}.get(info["os"], info["os"])
    info["arch"] = norm_arch(info["arch"])
    return info


def local_machine(cfg):
    os_name = {"Windows": "windows", "Linux": "linux", "Darwin": "macos"}.get(platform.system())
    if not os_name:
        die(f"sistema nao suportado: {platform.system()}")
    arch = norm_arch(platform.machine())
    backend = cfg.get("local_backend", "auto")
    if backend == "auto":
        nvidia = shutil.which("nvidia-smi") and subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True).returncode == 0
        if nvidia:
            backend = "cuda"
            libc = glibc_tuple(platform.libc_ver()[1]) if os_name == "linux" else None
            if libc and libc < CUDA_LINUX_MIN_GLIBC:
                backend = "vulkan"  # the prebuilt Linux CUDA package needs glibc 2.39+
        elif os_name == "macos" and arch == "arm64":
            backend = "metal"
        else:
            backend = "cpu"
    return {"name": "principal", "os": os_name, "arch": arch, "backend": backend}


def local_dir(cfg, m):
    return RUNTIME / f"{cfg['build']}-{m['backend']}"


def local_exe(cfg, m, tool):
    p = local_dir(cfg, m) / (tool + (".exe" if m["os"] == "windows" else ""))
    if not p.exists():
        die(f"{p} nao existe. Rode: {CMD} install")
    return str(p)


def local_env(cfg, m):
    env = dict(os.environ)
    if m["os"] != "windows":
        d = str(local_dir(cfg, m))
        for var in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
            env[var] = d + (os.pathsep + env[var] if env.get(var) else "")
    return env


# ---------------------------------------------------------------- model profiles

def as_bool(key, value):
    v = value.strip().lower()
    if v in ("1", "true", "yes", "on", "sim"):
        return True
    if v in ("0", "false", "no", "off", "nao"):
        return False
    die(f"{MODELS_PATH.name}: '{key}' precisa ser true ou false (veio '{value}')")


def load_profiles():
    if not MODELS_PATH.exists():
        die(f"{MODELS_PATH.name} nao existe. Rode: {CMD} init")
    parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#", ";"))
    parser.read(MODELS_PATH, encoding="utf-8")
    return {name: dict(parser[name]) for name in parser.sections()}


def resolve_target(target):
    """Turn a profile name (models.ini) or a .gguf path into llama.cpp arguments."""
    if target.lower().endswith(".gguf") or os.path.isfile(target):
        if not os.path.isfile(target):
            die(f"arquivo nao encontrado: {target}")
        return {"alias": Path(target).stem, "model_args": ["-m", target], "args": []}
    profiles = load_profiles()
    if target not in profiles:
        die(f"perfil '{target}' nao existe em {MODELS_PATH.name}. Perfis: {', '.join(profiles) or 'nenhum'}")
    p = {k: v.strip() for k, v in profiles[target].items() if v.strip()}
    unknown = sorted(set(p) - set(PROFILE_FLAGS) - set(PROFILE_OTHER))
    if unknown:
        die(f"[{target}] opcoes desconhecidas: {', '.join(unknown)}. Validas: "
            f"{', '.join(sorted(set(PROFILE_FLAGS) | set(PROFILE_OTHER)))}")

    if "model" in p:
        if not os.path.isfile(p["model"]):
            die(f"[{target}] arquivo do modelo nao encontrado: {p['model']}")
        model_args = ["-m", p["model"]]
    elif "hf" in p:
        model_args = ["-hf", p["hf"]]
    else:
        die(f"[{target}] defina 'model' (arquivo .gguf) ou 'hf' (repositorio do Hugging Face)")

    args = []
    for key, (flag, kind) in PROFILE_FLAGS.items():
        if key not in p:
            continue
        value = p[key]
        if key == "gpu_offload" and value.lower() == "auto":
            continue  # llama.cpp fits the layers by itself
        if isinstance(kind, tuple):
            if value.lower() not in kind:
                die(f"[{target}] '{key}' aceita {', '.join(kind)} (veio '{value}')")
            value = value.lower()
        elif kind is not str:
            try:
                kind(value)
            except ValueError:
                die(f"[{target}] '{key}' precisa ser um numero (veio '{value}')")
        args += [flag, value]

    if "offload_kv_cache" in p and not as_bool("offload_kv_cache", p["offload_kv_cache"]):
        args.append("-nkvo")
    if "use_mmap" in p or "keep_in_memory" in p:
        mmap = as_bool("use_mmap", p.get("use_mmap", "true"))
        mlock = as_bool("keep_in_memory", p.get("keep_in_memory", "false"))
        args += ["--load-mode", {(True, True): "mmap+mlock", (True, False): "mmap",
                                 (False, True): "mlock", (False, False): "none"}[(mmap, mlock)]]
    if "extra" in p:
        tokens = shlex.split(p["extra"], posix=os.name != "nt")
        args += [t[1:-1] if len(t) > 1 and t[0] == t[-1] == '"' else t for t in tokens]
    return {"alias": p.get("alias", target), "model_args": model_args, "args": args}


def assets_for(m, build):
    names = ASSETS.get((m["os"], m["arch"], m["backend"]))
    if not names:
        die(f"nao existe pacote pronto do llama.cpp para {m['os']}/{m['arch']}/{m['backend']} ({m['name']})")
    return [n.format(b=build) for n in names]


# ---------------------------------------------------------------- downloads

def release_assets(build, refresh=False):
    cache = DOWNLOADS / build / "release.json"
    if cache.exists() and not refresh:
        data = json.loads(cache.read_text(encoding="utf-8"))
    else:
        url = f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{build}"
        req = urllib.request.Request(url, headers={"User-Agent": "llama-cluster",
                                                   "Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data), encoding="utf-8")
    return {a["name"]: a for a in data.get("assets", [])}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_asset(build, name):
    assets = release_assets(build)
    if name not in assets:
        assets = release_assets(build, refresh=True)
    if name not in assets:
        die(f"o release {build} do llama.cpp nao tem {name}")
    a = assets[name]
    want = (a.get("digest") or "").split(":")[-1]
    path = DOWNLOADS / build / name
    if path.exists() and (not want or sha256_file(path) == want):
        return path
    print(f"  baixando {name} ({a['size'] / 1e6:.0f} MB)")
    tmp = path.with_name(path.name + ".part")
    req = urllib.request.Request(a["browser_download_url"], headers={"User-Agent": "llama-cluster"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out, 1 << 20)
    if want and sha256_file(tmp) != want:
        tmp.unlink()
        die(f"sha256 nao confere em {name}")
    tmp.replace(path)
    return path


HF_PART_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")


def hf_request(url, extra_headers=None):
    headers = {"User-Agent": "fastllm"}
    if os.environ.get("HF_TOKEN"):  # gated models (Llama and others)
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    headers.update(extra_headers or {})
    return urllib.request.Request(url, headers=headers)


def hf_file_info(repo, path):
    """Size and sha256 of a file in a Hugging Face model repo."""
    folder = path.rsplit("/", 1)[0] if "/" in path else ""
    url = f"https://huggingface.co/api/models/{repo}/tree/main" + (f"/{folder}" if folder else "")
    try:
        with urllib.request.urlopen(hf_request(url), timeout=30) as r:
            items = json.load(r)
    except urllib.error.HTTPError as e:
        gated = " (modelo restrito? defina HF_TOKEN)" if e.code in (401, 403) else ""
        die(f"o Hugging Face respondeu {e.code} para {repo}{gated}")
    for item in items:
        if item.get("path") == path:
            lfs = item.get("lfs") or {}
            return lfs.get("size") or item.get("size"), lfs.get("oid")
    die(f"{path} nao existe em {repo}")


def download_file(url, dest, size, sha):
    """Resumable download with sha256 check."""
    if dest.exists() and dest.stat().st_size == size and (not sha or sha256_file(dest) == sha):
        print(f"  ja baixado: {dest}")
        return
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    with urllib.request.urlopen(hf_request(url, {"Range": f"bytes={have}-"} if have else None), timeout=60) as resp:
        if have and resp.status != 206:
            have = 0  # server ignored the range; start over
        done, last = have, 0.0
        with open(part, "ab" if have else "wb") as out:
            for chunk in iter(lambda: resp.read(1 << 20), b""):
                out.write(chunk)
                done += len(chunk)
                if time.time() - last > 10:
                    print(f"  {dest.name}: {done / 1e9:.1f} de {size / 1e9:.1f} GB")
                    last = time.time()
    if sha and sha256_file(part) != sha:
        part.unlink()
        die(f"sha256 nao confere em {dest.name}; rode de novo")
    part.replace(dest)
    print(f"  ok: {dest}")


def extract_local(files, target):
    target.mkdir(parents=True, exist_ok=True)
    for f in files:
        if f.name.endswith(".zip"):
            with zipfile.ZipFile(f) as z:
                z.extractall(target)
            continue
        with tarfile.open(f, "r:gz") as t:
            members = []
            for member in t.getmembers():
                parts = member.name.split("/", 1)  # drop the top-level folder of the release tarball
                if len(parts) == 2 and parts[1]:
                    member.name = parts[1]
                    members.append(member)
            try:
                t.extractall(target, members=members, filter="data")
            except TypeError:
                t.extractall(target, members=members)


# ---------------------------------------------------------------- commands

def cmd_init(args):
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    if args.key:
        cfg["ssh_key"] = args.key
    key = Path(key_path(cfg))
    if not key.exists():
        key.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "llama-cluster", "-f", str(key)],
                       check=True)
        print(f"chave SSH criada: {key}")
    pub = Path(str(key) + ".pub").read_text(encoding="utf-8").strip()
    cfg["main_ip"] = args.main_ip or cfg.get("main_ip") or main_ip()
    save_config(cfg)

    GENERATED.mkdir(exist_ok=True)
    for name in ("worker-setup-windows.ps1", "worker-setup-unix.sh"):
        text = (HERE / name).read_text(encoding="utf-8")
        text = text.replace("__MAIN_HOST_IP__", cfg["main_ip"]).replace("__PUBLIC_KEY__", pub)
        with open(GENERATED / name, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)

    if not MODELS_PATH.exists():
        MODELS_PATH.write_text(MODELS_TEMPLATE, encoding="utf-8")
    print(f"configuracao: {CONFIG_PATH}")
    print(f"perfis de modelo: {MODELS_PATH}")
    print(f"IP deste PC (principal): {cfg['main_ip']}  (mude com --main-ip se estiver errado)")
    print(f"scripts para os trabalhadores: {GENERATED}")
    print("\nPara registrar os trabalhadores:")
    print(f"  Linux/macOS: {CMD} enroll  (mostra um comando para colar em cada um)")
    print(f"               {CMD} bootstrap usuario@ip ...  (se ja aceitam SSH com senha)")
    print("  Windows: rode generated/worker-setup-windows.ps1 como administrador e depois a linha 'add' que ele imprime")


def register_worker(host, user, name=None, backend=None, replace_host=False):
    """Detect a worker over SSH and store it. replace_host re-registers a known host instead of failing."""
    cfg = load_config()
    name = name or host
    if replace_host:
        cfg["workers"] = [w for w in cfg["workers"] if w["host"] != host]
        base, n = name, 2
        while any(w["name"] == name for w in cfg["workers"]):
            name, n = f"{base}-{n}", n + 1
    elif any(w["name"] == name for w in cfg["workers"]):
        die(f"ja existe um trabalhador chamado {name}")
    w = {"name": name, "host": host, "user": user}
    w.update(detect(cfg, w))
    note = ""
    if backend:
        w["backend"] = backend
    elif w["os"] == "linux" and w["backend"] == "cuda":
        g = glibc_tuple(w.get("glibc"))
        if g and g < CUDA_LINUX_MIN_GLIBC:
            w["backend"] = "vulkan"
            note = (f"nota: glibc {w['glibc']} e antiga para o pacote CUDA pronto (pede 2.39+, Ubuntu 24.04); "
                    "a GPU NVIDIA vai rodar via Vulkan.")
    cfg["workers"].append(w)
    save_config(cfg)
    print(f"adicionado: {name}  {w['os']}/{w['arch']}  backend={w['backend']}  gpu={w['gpus'] or '-'}")
    if note:
        print(note)
    if w["backend"] == "cpu":
        print("aviso: sem GPU compativel; um trabalhador so com CPU costuma deixar o conjunto mais lento. "
              f"Para tirar: {CMD} remove {name}")
    return w


def cmd_add(args):
    register_worker(args.host, args.user, args.name, args.backend)


def cmd_bootstrap(args):
    """Prepare Linux/macOS workers that already accept SSH with a password, straight from the main host."""
    cfg = load_config()
    script = GENERATED / "worker-setup-unix.sh"
    if not script.exists():
        die(f"{script} nao existe. Rode: {CMD} init")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    remote = "/tmp/fastllm-worker.sh"
    for target in args.targets:
        user, _, host = target.rpartition("@")
        if not (user and HOST_RE.match(host) and USER_RE.match(user)):
            print(f"{target}: use usuario@ip")
            continue
        print(f"\n=== {target} (digite a senha desse usuario quando pedir) ===")
        opts = ["-o", "StrictHostKeyChecking=accept-new"]
        if os.name != "nt":  # one shared connection, so the password is typed once per machine
            opts += ["-o", "ControlMaster=auto", "-o", f"ControlPath={RUNTIME}/ssh-%C", "-o", "ControlPersist=60"]
        if subprocess.run(["scp", "-q", *opts, str(script), f"{target}:{remote}"]).returncode != 0:
            print(f"{target}: nao consegui copiar o script (o SSH com senha esta ligado nessa maquina?)")
            continue
        run = f"sudo sh {remote}; s=$?; rm -f {remote}; exit $s"
        if subprocess.run(["ssh", "-t", *opts, target, run]).returncode != 0:
            print(f"{target}: a preparacao falhou")
            continue
        r = subprocess.run(["ssh", *ssh_opts(cfg), target, "hostname -s || hostname"],
                           capture_output=True, text=True, timeout=60)
        name = r.stdout.strip().splitlines()[-1] if r.returncode == 0 and r.stdout.strip() else host
        try:
            register_worker(host, user, name if NAME_RE.match(name) else host, replace_host=True)
        except SystemExit:
            print(f"{target}: preparado, mas o registro falhou")


HOST_RE = re.compile(r"^[A-Za-z0-9.:-]{1,253}$")
USER_RE = re.compile(r"^[A-Za-z0-9._@\\-]{1,64}$")
NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def cmd_enroll(args):
    """Serve the Unix worker script and let each worker register itself with a one-time token."""
    cfg = load_config()
    script = GENERATED / "worker-setup-unix.sh"
    if not script.exists():
        die(f"{script} nao existe. Rode: {CMD} init")
    body = script.read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    token = secrets.token_urlsafe(12)
    base = f"http://{cfg.get('main_ip') or main_ip()}:{args.port}"

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            pass

        def reply(self, code, data, ctype="text/plain; charset=utf-8"):
            if isinstance(data, str):
                data = (data + "\n").encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/worker-setup-unix.sh":
                self.reply(200, body)
            else:
                self.reply(404, "nao encontrado")

        def do_POST(self):
            if self.path != "/enroll":
                return self.reply(404, "nao encontrado")
            size = min(int(self.headers.get("Content-Length") or 0), 4096)
            form = urllib.parse.parse_qs(self.rfile.read(size).decode("utf-8", "replace"))
            field = lambda k: (form.get(k) or [""])[0].strip()
            if not secrets.compare_digest(field("token"), token):
                return self.reply(403, "token invalido")
            host = field("host") or self.client_address[0]
            user, name = field("user"), field("name") or field("host") or self.client_address[0]
            if not (HOST_RE.match(host) and USER_RE.match(user) and NAME_RE.match(name)):
                return self.reply(400, "dados invalidos")
            print(f"\n{name} ({user}@{host}) pediu registro")
            try:
                w = register_worker(host, user, name, replace_host=True)
            except SystemExit:
                return self.reply(500, "falhou; veja o terminal do PC principal")
            self.reply(200, f"registrado como {w['name']} ({w['backend']}, {w['gpus'] or 'sem GPU'})")

    server = http.server.HTTPServer((args.bind, args.port), Handler)
    fetch = f"curl -fsSL {base}/worker-setup-unix.sh -o /tmp/fastllm-worker.sh"
    check = f"echo '{digest}  /tmp/fastllm-worker.sh'"
    run = f"sudo sh /tmp/fastllm-worker.sh {base} {token}"
    print("Em cada trabalhador, cole no terminal (vai pedir a senha do sudo):\n")
    print(f"Linux:\n{fetch} && {check} | sha256sum -c - && {run}\n")
    print(f"macOS:\n{fetch} && {check} | shasum -a 256 -c - && {run}\n")
    if os.name == "nt":
        print(f"(se o firewall bloquear, libere a porta {args.port} para a rede local enquanto registra)")
    else:
        print(f"(com ufw ativo neste PC: sudo ufw allow {args.port}/tcp enquanto registra; "
              f"depois sudo ufw delete allow {args.port}/tcp)")
    print(f"Os registros aparecem aqui. Ctrl+C quando terminar; depois: {CMD} install && {CMD} start")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    print(f"\n{len(load_config()['workers'])} trabalhador(es) registrados.")


def cmd_remove(args):
    cfg = load_config()
    before = len(cfg["workers"])
    cfg["workers"] = [w for w in cfg["workers"] if w["name"] != args.name]
    if len(cfg["workers"]) == before:
        die(f"nao existe trabalhador {args.name}")
    save_config(cfg)
    print(f"removido: {args.name}")


def check_script(cfg, w):
    d = remote_dir(cfg, w)
    if w["os"] == "windows":
        return (f"$b = Join-Path {psq(d)} 'BUILD'\n"
                "$inst = if (Test-Path -LiteralPath $b) { (Get-Content -LiteralPath $b -Raw).Trim() } else { '' }\n"
                "$run = [bool](Get-Process ggml-rpc-server -ErrorAction SilentlyContinue)\n"
                "[pscustomobject]@{ installed = $inst; running = $run } | ConvertTo-Json -Compress")
    return (f"inst=$(cat {shq(d + '/BUILD')} 2>/dev/null | tr -d '\\n\"')\n"
            "if pgrep -f ggml-rpc-server >/dev/null 2>&1; then run=true; else run=false; fi\n"
            "printf '{\"installed\":\"%s\",\"running\":%s}\\n' \"$inst\" \"$run\"")


def cmd_status(args):
    cfg = load_config()
    state = load_state()
    tunnels = {t["name"]: t for t in state.get("tunnels", [])}
    want = f"{cfg['build']} "
    print(f"build do llama.cpp: {cfg['build']}   trabalhadores: {len(cfg['workers'])}")
    srv = server_state()
    if srv:
        where = "rede" if srv["public"] else "so este PC"
        print(f"servidor: ligado, modelo {srv['alias']}, porta {srv['port']} ({where}). Detalhes: {CMD} endpoint")
    else:
        print("servidor: desligado")
    for w in cfg["workers"]:
        try:
            rc, out, err = run_remote(cfg, w, check_script(cfg, w), timeout=60)
            info = last_json(out) if rc == 0 else None
        except subprocess.TimeoutExpired:
            info = None
        if not info:
            print(f"- {w['name']:<14} {w['host']:<16} SEM ACESSO")
            continue
        installed = "instalado" if info["installed"].startswith(want) else "falta install"
        rpc = "rpc ligado" if info["running"] else "rpc parado"
        t = tunnels.get(w["name"])
        tunnel = f"tunel :{t['port']}" if t and pid_alive(t["pid"]) else "sem tunel"
        print(f"- {w['name']:<14} {w['host']:<16} {w['os']}/{w['backend']:<7} {installed:<14} {rpc:<11} "
              f"{tunnel:<12} {w['gpus'] or '-'}")


def install_remote(cfg, w, files):
    d = remote_dir(cfg, w)
    dl = remote_downloads(cfg, w)
    marker = f"{cfg['build']} {w['backend']}"
    if w["os"] == "windows":
        mk = f"New-Item -ItemType Directory -Force {psq(dl)}, {psq(d)} | Out-Null\n'ok'"
    else:
        mk = f"mkdir -p {shq(dl)} {shq(d)} && echo ok"
    rc, out, err = run_remote(cfg, w, mk)
    if rc != 0:
        die(f"{w['name']}: nao consegui criar as pastas: {err}")
    names = []
    for f in files:
        print(f"  copiando {f.name} para {w['name']}")
        scp(cfg, w, f, dl)
        names.append(remote_join(w, "llama-cluster", "downloads", cfg["build"], f.name))
    if w["os"] == "windows":
        lst = ", ".join(psq(n) for n in names)
        script = (f"$d = {psq(d)}\n"
                  f"foreach ($f in @({lst})) {{\n"
                  "  tar.exe -xf $f -C $d\n"
                  "  if ($LASTEXITCODE -ne 0) { Expand-Archive -Force -LiteralPath $f -DestinationPath $d }\n"
                  "}\n"
                  f"Set-Content -LiteralPath (Join-Path $d 'BUILD') -Value {psq(marker)}\n"
                  "$h = cmd /c \"`\"$d\\ggml-rpc-server.exe`\" --help 2>&1\" | Out-String\n"
                  "if ($h -match '--port') { 'INSTALL_OK' } else { 'INSTALL_FAIL ' + $h }")
    else:
        lst = " ".join(shq(n) for n in names)
        script = (f"d={shq(d)}\n"
                  f"for f in {lst}; do tar -xzf \"$f\" -C \"$d\" --strip-components=1 || exit 1; done\n"
                  f"printf '%s\\n' {shq(marker)} > \"$d/BUILD\"\n"
                  "h=$(env LD_LIBRARY_PATH=\"$d\" DYLD_LIBRARY_PATH=\"$d\" \"$d/ggml-rpc-server\" --help 2>&1)\n"
                  "case \"$h\" in *--port*) echo INSTALL_OK ;; *) echo \"INSTALL_FAIL $h\" | head -n 6 ;; esac")
    rc, out, err = run_remote(cfg, w, script, timeout=900)
    if "INSTALL_OK" not in out:
        detail = (out + "\n" + err).strip()
        hint = "\ndica: pacote CUDA para Linux pede glibc 2.39+ (Ubuntu 24.04)." if "GLIBC" in detail else ""
        die(f"{w['name']}: o ggml-rpc-server nao rodou depois de instalar:\n{detail[:800]}{hint}")


def cmd_install(args):
    cfg = load_config()
    build = cfg["build"]
    main = local_machine(cfg)
    print(f"principal: {main['os']}/{main['arch']}/{main['backend']}")
    marker = local_dir(cfg, main) / "BUILD"
    if args.force or not marker.exists():
        files = [fetch_asset(build, n) for n in assets_for(main, build)]
        extract_local(files, local_dir(cfg, main))
        marker.write_text(f"{build} {main['backend']}\n", encoding="utf-8")
    print(f"  ok: {local_dir(cfg, main)}")

    for w in cfg["workers"]:
        print(f"{w['name']}: {w['os']}/{w['arch']}/{w['backend']}")
        rc, out, err = run_remote(cfg, w, check_script(cfg, w), timeout=60)
        info = last_json(out) if rc == 0 else None
        if info is None:
            die(f"{w['name']}: sem acesso SSH ({err})")
        if info["installed"] == f"{build} {w['backend']}" and not args.force:
            print("  ja instalado")
            continue
        files = [fetch_asset(build, n) for n in assets_for(w, build)]
        install_remote(cfg, w, files)
        print(f"  ok: {remote_dir(cfg, w)}")


def pid_alive(pid):
    if os.name == "nt":
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def kill_pid(pid):
    if not pid_alive(pid):
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        import signal
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except OSError:
            pass


def spawn_detached(cmd, log, env=None):
    """Start a process that keeps running after this script (and its terminal) exits."""
    kw = {"stdin": subprocess.DEVNULL, "stdout": log, "stderr": subprocess.STDOUT, "env": env}
    if os.name != "nt":
        return subprocess.Popen(cmd, start_new_session=True, **kw)
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | 0x08000000  # CREATE_NO_WINDOW
    try:
        return subprocess.Popen(cmd, creationflags=flags | 0x01000000, **kw)  # CREATE_BREAKAWAY_FROM_JOB
    except OSError:
        return subprocess.Popen(cmd, creationflags=flags, **kw)


def rpc_command(cfg, w):
    d = remote_dir(cfg, w)
    flags = ["-H", "127.0.0.1", "-p", str(cfg["remote_port"])] + (["-c"] if cfg.get("rpc_cache", True) else [])
    if w["os"] == "windows":
        return ps_encoded("& " + psq(d + "\\ggml-rpc-server.exe") + " " + " ".join(flags))
    return (f"env LD_LIBRARY_PATH={shq(d)} DYLD_LIBRARY_PATH={shq(d)} "
            f"{shq(d + '/ggml-rpc-server')} {' '.join(flags)}")


DEV_RE = re.compile(r"RPC\d+:\s*127\.0\.0\.1:(\d+)\s*\((\d+) MiB, (\d+) MiB free\)")
LOCAL_DEV_RE = re.compile(r"^\s*([A-Za-z]+\d+):\s.*\(\d+ MiB, \d+ MiB free\)", re.M)


def list_devices(cfg, main, rpc=None):
    # --list-devices acts as soon as it is parsed, so --rpc has to come first.
    cmd = [local_exe(cfg, main, "llama-cli")] + (["--rpc", rpc] if rpc else []) + ["--list-devices"]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
                       env=local_env(cfg, main))
    return r.stdout + r.stderr


def probe_port(cfg, main, port):
    """Devices behind a tunnel as (total MiB, free MiB, count); None if not ready."""
    found = [(int(t), int(f)) for p, t, f in DEV_RE.findall(list_devices(cfg, main, f"127.0.0.1:{port}"))
             if int(p) == port]
    if not found:
        return None
    return sum(t for t, _ in found), sum(f for _, f in found), len(found)


def fit_targets(cfg, main, tunnels):
    """Per-device VRAM margins for --fit-target, in llama.cpp device order: local GPUs first, then RPC."""
    local = [d for d in LOCAL_DEV_RE.findall(list_devices(cfg, main)) if not d.startswith("RPC")]
    margins = [cfg["vram_margin_main_mib"]] * len(local)
    for t in tunnels:
        margins += [cfg["vram_margin_mib"]] * t.get("devices", 1)
    return ",".join(str(m) for m in margins) or str(cfg["vram_margin_main_mib"])


def port_free(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def cmd_start(args):
    cfg = load_config()
    if not cfg["workers"]:
        die(f"nenhum trabalhador. Use: {CMD} add <ip> <usuario>")
    main = local_machine(cfg)
    local_exe(cfg, main, "llama-cli")
    stop_all(cfg, quiet=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    state = {"tunnels": []}
    port = cfg["base_port"]
    for w in cfg["workers"]:
        while not port_free(port):  # something else already listens there; a probe would reach it
            port += 1
        cmd = ["ssh", *ssh_opts(cfg), "-o", "ExitOnForwardFailure=yes",
               "-L", f"{port}:127.0.0.1:{cfg['remote_port']}", dest(w), rpc_command(cfg, w)]
        log = open(LOGS / f"{w['name']}.log", "wb")
        p = spawn_detached(cmd, log)
        state["tunnels"].append({"name": w["name"], "pid": p.pid, "port": port})
        print(f"{w['name']}: servidor RPC iniciando (tunel na porta local {port})")
        port += 1
    save_state(state)

    total_free = 0
    for t in state["tunnels"]:
        deadline = time.time() + 60
        got = None
        while time.time() < deadline and got is None:
            if not pid_alive(t["pid"]):
                break
            got = probe_port(cfg, main, t["port"])
            if got is not None and not pid_alive(t["pid"]):
                got = None  # the answer did not come through our tunnel
            if got is None:
                time.sleep(2)
        if got is None:
            tail = (LOGS / f"{t['name']}.log").read_text(encoding="utf-8", errors="replace")[-600:]
            print(f"{t['name']}: NAO respondeu. Final do log:\n{tail}")
            continue
        total_free += got[1]
        t["devices"] = got[2]
        print(f"{t['name']}: pronto, {got[1] / 1024:.1f} GB livres de {got[0] / 1024:.1f} GB")
    save_state(state)
    print(f"memoria livre somada nos trabalhadores: {total_free / 1024:.1f} GB (+ as GPUs deste PC)")


def running_tunnels(cfg):
    state = load_state()
    tunnels = [t for t in state.get("tunnels", []) if pid_alive(t["pid"])]
    if not tunnels:
        die(f"nenhum servidor RPC ligado. Rode: {CMD} start")
    return tunnels


def rpc_list(tunnels):
    return ",".join(f"127.0.0.1:{t['port']}" for t in tunnels)


def cmd_pull(args):
    m = HF_PART_RE.search(args.file)
    files = ([HF_PART_RE.sub(f"-{i:05d}-of-{m.group(2)}.gguf", args.file) for i in range(1, int(m.group(2)) + 1)]
             if m else [args.file])
    target = Path(os.path.expanduser(args.dir))
    target.mkdir(parents=True, exist_ok=True)
    for f in files:
        size, sha = hf_file_info(args.repo, f)
        print(f"{f} ({size / 1e9:.1f} GB)")
        download_file(f"https://huggingface.co/{args.repo}/resolve/main/{f}", target / Path(f).name, size, sha)
    first = (target / Path(files[0]).name).resolve()
    if first.suffix != ".gguf":
        return
    name = args.profile or re.sub(r"-00001-of-\d{5}$", "", first.stem).lower()
    if MODELS_PATH.exists() and name in load_profiles():
        print(f"perfil '{name}' ja existe em {MODELS_PATH.name}")
    else:
        with open(MODELS_PATH, "a", encoding="utf-8") as fp:
            fp.write(f"\n[{name}]\nmodel = {first.as_posix()}\ncontext_length = {args.context}\n")
        print(f"perfil '{name}' criado em {MODELS_PATH.name}")
    print(f"Proximo: {CMD} serve {name} --detach --public")


def cmd_models(args):
    profiles = load_profiles()
    if not profiles:
        print(f"nenhum perfil em {MODELS_PATH}")
    for name, p in profiles.items():
        src = p.get("model") or p.get("hf") or "?"
        print(f"- {name:<16} contexto={p.get('context_length', 'padrao'):<8} {src}")


def cmd_bench(args):
    cfg = load_config()
    main = local_machine(cfg)
    target = resolve_target(args.target)
    cmd = [local_exe(cfg, main, "llama-bench"), *target["model_args"], "-fitt", str(cfg["vram_margin_main_mib"]),
           "-p", "512", "-n", "32" if args.quick else "128", "-r", "1" if args.quick else "2", "-o", "md"]
    if not args.local:
        cmd += ["--rpc", rpc_list(running_tunnels(cfg))]
    cmd += args.extra
    print("rodando: " + " ".join(cmd))
    sys.exit(subprocess.run(cmd, env=local_env(cfg, main)).returncode)


def local_http_status(port, path):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never route localhost via a proxy
    with opener.open(f"http://127.0.0.1:{port}{path}", timeout=3) as r:
        return r.status


def wait_health(port, pid, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not pid_alive(pid):
            return False
        try:
            if local_http_status(port, "/health") == 200:
                return True
        except Exception:
            pass  # 503 while the model loads
        time.sleep(2)
    return False


def print_endpoint(cfg, srv):
    ip = (cfg.get("main_ip") or main_ip()) if srv["public"] else "127.0.0.1"
    base = f"http://{ip}:{srv['port']}"
    key = srv.get("key")
    print("\nEndpoint compativel com a API da OpenAI")
    print(f"  URL base : {base}/v1")
    print(f"  Chave    : {key}" if key else "  Chave    : nenhuma (so este PC conecta; use --public para a rede)")
    print(f"  Modelo   : {srv['alias']}")
    print(f"  Chat web : {base}")
    auth = f" -H 'Authorization: Bearer {key}'" if key else ""
    body = json.dumps({"model": srv["alias"], "messages": [{"role": "user", "content": "oi"}]})
    print(f"  Teste    : curl {base}/v1/chat/completions{auth} -H 'Content-Type: application/json' -d '{body}'")
    if srv["public"]:
        port = srv["port"]
        if os.name == "nt":
            print("Para outros PCs conectarem, libere a porta uma vez (PowerShell como administrador):\n"
                  f'  New-NetFirewallRule -DisplayName "llama-cluster {port}" -Direction Inbound -Protocol TCP '
                  f"-LocalPort {port} -RemoteAddress LocalSubnet -Action Allow")
        else:
            net = ".".join(ip.split(".")[:3]) + ".0/24"
            print(f"Se o firewall estiver ativo, libere a porta:\n  sudo ufw allow proto tcp from {net} to any port {port}")


def server_state():
    srv = load_state().get("server")
    return srv if srv and pid_alive(srv["pid"]) else None


def stop_server():
    state = load_state()
    if state.get("server"):
        kill_pid(state["server"]["pid"])
    state["server"] = None
    save_state(state)


def cmd_serve(args):
    cfg = load_config()
    main = local_machine(cfg)
    target = resolve_target(args.target)
    port = args.port or cfg.get("server_port", 8080)
    tunnels = [] if args.local else running_tunnels(cfg)
    cmd = [local_exe(cfg, main, "llama-server"), *target["model_args"], *target["args"],
           "--alias", target["alias"], "-fitt", fit_targets(cfg, main, tunnels),
           "--host", "0.0.0.0" if args.public else "127.0.0.1", "--port", str(port)]
    if tunnels:
        cmd += ["--rpc", rpc_list(tunnels)]
    key = None
    if args.public:
        key = args.api_key or cfg.get("api_key") or secrets.token_urlsafe(24)
        if cfg.get("api_key") != key:
            cfg["api_key"] = key  # keep the same key across restarts
            save_config(cfg)
        cmd += ["--api-key", key]
    cmd += args.extra
    srv = {"port": port, "public": args.public, "alias": target["alias"], "key": key, "target": args.target}

    if not args.detach:
        print_endpoint(cfg, srv)
        print("\nCtrl+C para parar.\n")
        try:
            sys.exit(subprocess.run(cmd, env=local_env(cfg, main)).returncode)
        except KeyboardInterrupt:
            return

    stop_server()
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / "server.log"
    p = spawn_detached(cmd, open(log_path, "wb"), env=local_env(cfg, main))
    srv.update(pid=p.pid, log=str(log_path))
    state = load_state()
    state["server"] = srv
    save_state(state)
    print(f"carregando {target['alias']}... (log: {log_path})")
    if not wait_health(port, p.pid, args.timeout):
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
        stop_server()
        die(f"o servidor nao ficou pronto. Final do log:\n{tail}")
    print_endpoint(cfg, srv)
    print(f"\nRodando em segundo plano. Para desligar: {CMD} stop")


def cmd_endpoint(args):
    cfg = load_config()
    srv = server_state()
    if not srv:
        die(f"nenhum servidor em segundo plano. Rode: {CMD} serve <perfil> --detach")
    print_endpoint(cfg, srv)


def stop_all(cfg, quiet=False):
    stop_server()
    state = load_state()
    for t in state.get("tunnels", []):
        kill_pid(t["pid"])
    for w in cfg["workers"]:
        try:
            run_remote(cfg, w, kill_script(w), timeout=30)
        except subprocess.TimeoutExpired:
            if not quiet:
                print(f"{w['name']}: sem resposta ao desligar")
    save_state({"tunnels": [], "server": None})


def cmd_stop(args):
    cfg = load_config()
    stop_all(cfg)
    print("servidor, tuneis e servidores RPC desligados")


def cmd_clean(args):
    cfg = load_config()
    for w in cfg["workers"]:
        if w["os"] == "windows":
            script = ("$c = Join-Path $env:LOCALAPPDATA 'llama.cpp\\rpc'\n"
                      "if (Test-Path -LiteralPath $c) { Remove-Item -Recurse -Force -LiteralPath $c }\n'limpo'")
        else:
            script = ('if [ "$(uname -s)" = Darwin ]; then c="$HOME/Library/Caches/llama.cpp/rpc"; '
                      'else c="${XDG_CACHE_HOME:-$HOME/.cache}/llama.cpp/rpc"; fi\n'
                      'rm -rf "$c"; echo limpo')
        rc, out, err = run_remote(cfg, w, script, timeout=120)
        print(f"{w['name']}: {'cache RPC apagado' if 'limpo' in out else 'falhou: ' + err}")


def main():
    ap = argparse.ArgumentParser(description="Um modelo do llama.cpp rodando em varias maquinas.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="cria a configuracao, a chave SSH e os scripts dos trabalhadores")
    p.add_argument("--key", help="chave SSH privada existente (padrao: ~/.ssh/llama_cluster_ed25519)")
    p.add_argument("--main-ip", help="IP deste PC como os trabalhadores o veem")
    p.set_defaults(fn=cmd_init)

    p = sub.add_parser("add", help="registra um trabalhador ja preparado")
    p.add_argument("host")
    p.add_argument("user")
    p.add_argument("--name")
    p.add_argument("--backend", choices=["cuda", "vulkan", "cpu", "metal"], help="forca o tipo de GPU")
    p.set_defaults(fn=cmd_add)

    p = sub.add_parser("enroll", help="mostra um comando para cada trabalhador Linux/macOS se preparar e se registrar")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--bind", default="0.0.0.0", help="endereco onde escutar (padrao: todas as interfaces)")
    p.set_defaults(fn=cmd_enroll)

    p = sub.add_parser("bootstrap", help="prepara e registra trabalhadores Linux/macOS que ja aceitam SSH com senha")
    p.add_argument("targets", nargs="+", metavar="usuario@ip")
    p.set_defaults(fn=cmd_bootstrap)

    p = sub.add_parser("remove", help="tira um trabalhador da lista")
    p.add_argument("name")
    p.set_defaults(fn=cmd_remove)

    sub.add_parser("status", help="mostra cada trabalhador").set_defaults(fn=cmd_status)

    p = sub.add_parser("install", help="baixa o llama.cpp e instala aqui e nos trabalhadores")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_install)

    sub.add_parser("start", help="liga os servidores RPC e os tuneis").set_defaults(fn=cmd_start)
    sub.add_parser("models", help=f"lista os perfis de {MODELS_PATH.name}").set_defaults(fn=cmd_models)

    p = sub.add_parser("pull", help="baixa um GGUF do Hugging Face e cria o perfil")
    p.add_argument("repo", help="ex.: Qwen/Qwen3-32B-GGUF")
    p.add_argument("file", help="ex.: Qwen3-32B-Q4_K_M.gguf; arquivo em partes (-00001-of-0000N) baixa todas")
    p.add_argument("--dir", default="~/models", help="pasta de destino (padrao: ~/models)")
    p.add_argument("--profile", help="nome do perfil (padrao: nome do arquivo)")
    p.add_argument("--context", type=int, default=8192, help="context_length do perfil novo")
    p.set_defaults(fn=cmd_pull)

    p = sub.add_parser("bench", help="mede a velocidade de um perfil ou arquivo .gguf")
    p.add_argument("target", help=f"perfil de {MODELS_PATH.name} ou arquivo .gguf")
    p.add_argument("--quick", action="store_true", help="teste curto (~1 min)")
    p.add_argument("--local", action="store_true", help="so este PC, para comparar")
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("serve", help="sobe o servidor (API compativel com OpenAI) usando todas as GPUs")
    p.add_argument("target", help=f"perfil de {MODELS_PATH.name} ou arquivo .gguf")
    p.add_argument("--port", type=int, help="padrao: server_port do cluster.json (8080)")
    p.add_argument("--public", action="store_true", help="aceita conexoes da rede, com chave da API")
    p.add_argument("--api-key", help="chave fixa; sem isso, gera uma e guarda no cluster.json")
    p.add_argument("--detach", action="store_true", help="roda em segundo plano")
    p.add_argument("--local", action="store_true", help="so este PC, sem os trabalhadores")
    p.add_argument("--timeout", type=int, default=900, help="segundos esperando o modelo carregar (--detach)")
    p.set_defaults(fn=cmd_serve)

    sub.add_parser("endpoint", help="mostra URL, chave e modelo do servidor ligado").set_defaults(fn=cmd_endpoint)
    sub.add_parser("stop", help="desliga servidor, tuneis e servidores RPC").set_defaults(fn=cmd_stop)
    sub.add_parser("clean", help="apaga o cache de pesos nos trabalhadores").set_defaults(fn=cmd_clean)

    sys.stdout.reconfigure(line_buffering=True)  # keep our lines in order with the llama.cpp output
    # Everything after "--" goes untouched to llama-server / llama-bench.
    argv, extra = sys.argv[1:], []
    if "--" in argv:
        cut = argv.index("--")
        argv, extra = argv[:cut], argv[cut + 1:]
    args = ap.parse_args(argv)
    args.extra = extra
    args.fn(args)


if __name__ == "__main__":
    main()
