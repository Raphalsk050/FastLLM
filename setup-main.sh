#!/bin/sh
# Prepares this Linux or macOS machine as the FastLLM main host: checks the tools,
# creates the config and SSH key, and installs llama.cpp here.
# Usage: sh setup-main.sh [--main-ip IP] [--key FILE]
set -eu
cd "$(dirname "$0")"

missing=""
for tool in python3 ssh scp ssh-keygen; do
    command -v "$tool" >/dev/null 2>&1 || missing="$missing $tool"
done
if [ -n "$missing" ]; then
    echo "Faltam:$missing"
    echo "Ubuntu/Debian: sudo apt install python3 openssh-client"
    echo "Fedora: sudo dnf install python3 openssh-clients"
    exit 1
fi
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' || {
    echo "Precisa de Python 3.8 ou mais novo."
    exit 1
}
if ! command -v nvidia-smi >/dev/null 2>&1 && command -v lspci >/dev/null 2>&1 && lspci | grep -qi nvidia; then
    echo "Aviso: GPU NVIDIA sem driver (nvidia-smi ausente). Este PC so vai usar a CPU ate o driver ser instalado."
fi

python3 cluster.py init "$@"
python3 cluster.py install

cat <<EOF

PC principal pronto. Proximos passos:
  1. Registrar os trabalhadores, de um destes jeitos:
       python3 cluster.py enroll                  mostra um comando para colar em cada trabalhador
       python3 cluster.py bootstrap user@ip ...   se eles ja aceitam SSH com senha
  2. python3 cluster.py install                   instala o llama.cpp nos trabalhadores
  3. python3 cluster.py start                     liga os trabalhadores e mostra a memoria livre
  4. python3 cluster.py pull Qwen/Qwen3-32B-GGUF Qwen3-32B-Q4_K_M.gguf --profile qwen32b
  5. python3 cluster.py serve qwen32b --detach --public
EOF
