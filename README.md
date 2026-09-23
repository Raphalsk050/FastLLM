# FastLLM

Run **one** llama.cpp model split across several computers and serve it through an OpenAI-compatible endpoint. Each worker lends its GPU; the main host combines them and serves the model. The main host and the workers can run Windows, Linux or macOS, in any mix.

## How it works

- Every worker runs `ggml-rpc-server` bound to `127.0.0.1`.
- The main host opens an SSH tunnel to each worker and uses the remote GPUs as if they were local (`--rpc`).
- The model is split **by layers**: GPU memory adds up, and each token passes through every machine, one after the other.
- `llama-server` on the main host exposes the API (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings`) and a web chat.

## Requirements

| Machine | Needs |
|---|---|
| Main host | Python 3.8+, OpenSSH client (`ssh`, `scp`), access to GitHub |
| Windows worker | Windows 10/11, an administrator account during setup |
| Linux worker | x64 or arm64, `sudo` during setup, systemd |
| macOS worker | Apple Silicon, Remote Login enabled |
| Network | Wired. Wi-Fi hurts performance |

NVIDIA GPUs need the driver installed (`nvidia-smi` working); the CUDA Toolkit is not required. AMD Radeon RX/Pro and Intel Arc GPUs run through Vulkan. A machine without a supported GPU joins as CPU and usually slows the whole cluster down.

### Linux

- The prebuilt llama.cpp CUDA package needs glibc 2.39+ (Ubuntu 24.04, Debian 13, Fedora 40 or newer). On older systems, such as Ubuntu 22.04, the tool switches NVIDIA GPUs to the Vulkan package, which runs on glibc 2.35+. The setup script installs the Vulkan loader (`libvulkan1`).
- RHEL/Rocky/Alma 9 (glibc 2.34) cannot run the prebuilt packages; build llama.cpp from source there.
- Workers must not suspend. On GNOME: Settings → Power → Automatic Suspend off. On dedicated machines: `sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target`.
- Behind a proxy, run `export HTTPS_PROXY=http://proxy:port` before `install`.

### Tested combinations

| Main host | Workers |
|---|---|
| Windows 11 | Windows 11 (RTX 2060), Linux x64 (GTX 1050 Ti), macOS on Apple Silicon (M3) |
| Linux (Ubuntu 24.04) | Linux x64 (GTX 1050 Ti) |

## Quick start

1. On the main host:
   ```
   python cluster.py init
   ```
   This creates `cluster.json`, `models.ini`, an SSH key used only by the cluster, and the worker setup scripts in `generated/`.
2. Copy the matching script from `generated/` to each worker and run it as administrator:
   - Windows: `powershell -ExecutionPolicy Bypass -File worker-setup-windows.ps1`
   - Linux/macOS: `sudo sh worker-setup-unix.sh`

   Each one ends by printing a `python cluster.py add ...` line.
3. On the main host, run every `add` line that was printed.
4. Install the same llama.cpp build everywhere:
   ```
   python cluster.py install
   ```
5. Describe your models in `models.ini` (see below).
6. Start the workers and the server:
   ```
   python cluster.py start
   python cluster.py serve my-profile --detach --public
   ```
   The command prints the URL, the API key and the model name.
7. When you are done:
   ```
   python cluster.py stop
   ```

## Model profiles (`models.ini`)

Each `[section]` is a profile with the same settings LM Studio exposes. A commented or missing line keeps the llama.cpp default. Settings under `[DEFAULT]` apply to every profile.

```ini
[qwen14b]
model = /models/Qwen3-14B-Q4_K_M.gguf
context_length = 8192
temperature = 0.7
top_k = 20
top_p = 0.8
min_p = 0.0
thinking = off
```

| Key | LM Studio setting | Values |
|---|---|---|
| `model` / `hf` | Model | path to a `.gguf` file / Hugging Face `user/repo:quant` |
| `alias` | — | model name API clients see (default: the profile name) |
| `context_length` | Context Length | tokens |
| `gpu_offload` | GPU Offload | `auto`, `all` or a number of layers |
| `flash_attention` | Flash Attention | `on`, `off`, `auto` |
| `k_cache_type` / `v_cache_type` | K/V Cache Quantization Type | `f16`, `q8_0`, `q4_0`... |
| `offload_kv_cache` | Offload KV Cache to GPU Memory | `true` / `false` |
| `use_mmap` / `keep_in_memory` | Try mmap() / Keep Model in Memory | `true` / `false` |
| `cpu_threads` | CPU Thread Pool Size | number |
| `batch_size` / `ubatch_size` | Evaluation Batch Size | number |
| `moe_cpu_layers` | Force Model Expert Weights onto CPU | number of MoE layers kept on the CPU |
| `rope_freq_base` / `rope_freq_scale` | RoPE Frequency Base / Scale | number |
| `seed` | Seed | `-1` = random |
| `parallel` | — | requests served at the same time |
| `temperature`, `top_k`, `top_p`, `min_p` | Sampling | number |
| `repeat_penalty`, `repeat_last_n`, `presence_penalty`, `frequency_penalty` | Repeat Penalty and related | number |
| `max_tokens` | Limit Response Length | `-1` = no limit |
| `thinking` | Enable Thinking | `on`, `off`, `auto` |
| `chat_template_file` | Prompt Template | Jinja file |
| `extra` | — | any other `llama-server` argument |

Generation settings become the server defaults; every API request can still send its own (`temperature`, `top_p`, `max_tokens`, `top_k`, `min_p`...). The system prompt goes in the request itself (a `system` message) or in the web chat settings.

## Connecting to the endpoint

`serve --public` accepts connections from the network and requires an API key. The key is generated on first use and stored in `cluster.json`, so it survives restarts. `python cluster.py endpoint` prints the URL, key and model again.

```python
from openai import OpenAI

client = OpenAI(base_url="http://192.168.1.9:8080/v1", api_key="YOUR_KEY")
r = client.chat.completions.create(
    model="qwen14b",
    messages=[{"role": "user", "content": "hello"}],
    temperature=0.7,
)
print(r.choices[0].message.content)
```

Any app that accepts an OpenAI-compatible server (Open WebUI, editor extensions, LangChain and so on) uses the same base URL and key.

For other machines to connect, the port must be open in the main host's firewall. `serve --public` prints the exact command for your system.

## Commands

| Command | What it does |
|---|---|
| `init [--key FILE] [--main-ip IP]` | Creates the config, profiles, SSH key and worker scripts |
| `add IP USER [--name N] [--backend cuda\|vulkan\|cpu\|metal]` | Registers a worker and detects its OS and GPU |
| `remove NAME` | Removes a worker |
| `status` | Shows the server and, per worker, install state, RPC server and tunnel |
| `install [--force]` | Downloads llama.cpp (sha256 checked) and installs it here and on the workers |
| `start` | Starts the RPC servers and tunnels, and prints each worker's free memory |
| `models` | Lists the profiles in `models.ini` |
| `bench PROFILE\|FILE [--quick] [--local]` | Measures speed; `--local` uses only the main host, for comparison |
| `serve PROFILE\|FILE [--detach] [--public] [--port P] [--api-key K] [--local]` | Serves the API with every GPU |
| `endpoint` | Prints the URL, key and model of the running server |
| `stop` | Stops the server, the tunnels and the RPC servers |
| `clean` | Deletes the weight cache on the workers |

Arguments after `--` go straight to llama.cpp, e.g. `python cluster.py serve qwen14b -- --cache-reuse 256`.

## Configuration (`cluster.json`)

| Field | Default | Purpose |
|---|---|---|
| `build` | `b11115` | llama.cpp build. Every machine must run the same one |
| `vram_margin_main_mib` | `1536` | Free memory kept on each main host GPU (the desktop and other apps use it too) |
| `vram_margin_mib` | `1024` | Free memory kept on each worker GPU |
| `server_port` | `8080` | API port |
| `api_key` | generated | API key for `--public` |
| `rpc_cache` | `true` | Workers keep the weights on disk, so reloads are fast |
| `base_port` / `remote_port` | `50052` | Tunnel ports |
| `local_backend` | `auto` | Forces `cuda`, `vulkan`, `cpu` or `metal` on the main host |

## Security

- The RPC port has no authentication, so it is never exposed on the network: it only exists inside the SSH tunnel.
- Workers accept SSH only with the main host's key. On Windows the firewall only accepts the main host's IP; on Linux, too, when `ufw` or `firewalld` is active.
- `serve --public` requires an API key. Without `--public`, only the main host can connect.
- On a company network, clear it with IT first: the setup installs an SSH server and uses the GPUs of other people's machines.
- On Windows, Microsoft Entra ID accounts cannot log in with an SSH key. Set up the worker with a local administrator account.

## What to expect

- **More machines means more memory, not more speed.** With the model split by layers, generation time is roughly the sum of every machine's share. It only pays off when the model would not fit on fewer GPUs.
- **Leave slow machines out when you can.** Qwen3-14B ran at 26 tokens/s on two PCs and at 7 tokens/s after adding a GTX 1050 Ti and a MacBook Air on Wi-Fi.
- **First load:** the main host sends the weights to every worker (~90 MB/s on 1 Gbps, so 100 GB takes ~20 minutes). Later loads use each worker's cache (a 9 GB load went from ~60 s to ~15 s).
- **Full GPU:** on Windows, when a GPU fills up the system moves part of the model to RAM and generation drops below 1 token/s (`nvidia-smi dmon` shows ~10 GB/s of PCIe traffic). Raise the margins in `cluster.json` or lower `context_length`.

Measured with Qwen3-14B Q4_K_M (9 GB) on 1 Gbps Ethernet:

| Setup | Generation |
|---|---|
| RTX 3070 8 GB alone (rest in RAM) | 7.9 tokens/s |
| RTX 3070 + RTX 2060 6 GB, context 640 | 26.7 tokens/s |
| RTX 3070 + RTX 2060 6 GB, context 8192 (server) | 20.5 tokens/s |

Measured with Qwen3-32B Q4_K_M (19.8 GB), which does not fit on any single GPU here:

| Setup | Prompt (512 tokens) | Generation |
|---|---|---|
| RTX 3070 8 GB alone (rest in RAM) | 210 tokens/s | 1.9 tokens/s |
| RTX 3070 + RTX 2060 + GTX 1050 Ti + MacBook Air M3 on Wi-Fi | 53 tokens/s | 4.7 tokens/s |

Generation got 2.5x faster, while prompt processing got slower: activations cross the network and the slower GPUs compute their share. The first load took ~6.5 minutes, mostly weights going to the MacBook over Wi-Fi.

## Files

- `cluster.py`: the tool (runs on the main host).
- `worker-setup-windows.ps1`, `worker-setup-unix.sh`: templates for the worker setup scripts; `init` writes filled-in copies to `generated/`.
- `~/.llama-cluster/` on the main host: downloads, binaries, logs (tunnels and `server.log`) and state.
- `~/llama-cluster/` on the workers: binaries.
