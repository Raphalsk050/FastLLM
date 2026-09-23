#!/bin/sh
# Prepares a Linux or macOS machine to act as a llama-cluster worker.
# Run as root:  sudo sh worker-setup-unix.sh [ENROLL_URL TOKEN]
# With ENROLL_URL and TOKEN (printed by `cluster.py enroll`), the worker registers itself at the end.
set -eu
MAIN_HOST_IP='__MAIN_HOST_IP__'
PUBLIC_KEY='__PUBLIC_KEY__'
ENROLL_URL=${1:-}
ENROLL_TOKEN=${2:-}

if [ "$(id -u)" -ne 0 ]; then
    echo "Rode com sudo: sudo sh $0"
    exit 1
fi
TARGET_USER=${SUDO_USER:-root}
TARGET_HOME=$(eval echo "~$TARGET_USER")
OS=$(uname -s)

if [ "$OS" = Linux ]; then
    # 1. OpenSSH server
    if ! command -v sshd >/dev/null 2>&1 && [ ! -x /usr/sbin/sshd ]; then
        echo "Instalando o servidor OpenSSH..."
        if command -v apt-get >/dev/null 2>&1; then
            apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server
        elif command -v dnf >/dev/null 2>&1; then dnf install -y -q openssh-server
        elif command -v yum >/dev/null 2>&1; then yum install -y -q openssh-server
        elif command -v zypper >/dev/null 2>&1; then zypper -q install -y openssh
        elif command -v pacman >/dev/null 2>&1; then pacman -S --noconfirm --needed openssh
        else
            echo "Instale o servidor OpenSSH manualmente e rode de novo."
            exit 1
        fi
    fi
    systemctl enable --now ssh 2>/dev/null || systemctl enable --now sshd || {
        echo "Nao consegui ligar o servico SSH (sem systemd?). Ligue o sshd manualmente e rode de novo."
        exit 1
    }

    # 2. SSH only from the main host, when a firewall is active. The RPC port stays on 127.0.0.1.
    if command -v ufw >/dev/null 2>&1 && ufw status | grep -q 'Status: active'; then
        ufw allow from "$MAIN_HOST_IP" to any port 22 proto tcp
    fi
    if command -v firewall-cmd >/dev/null 2>&1 && firewall-cmd --state >/dev/null 2>&1; then
        firewall-cmd --permanent --add-rich-rule="rule family=ipv4 source address=$MAIN_HOST_IP port port=22 protocol=tcp accept"
        firewall-cmd --reload
    fi

    # Vulkan loader: GPUs run through Vulkan when the prebuilt CUDA package is too new for this glibc.
    if command -v apt-get >/dev/null 2>&1; then
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libvulkan1 >/dev/null 2>&1 ||
            { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libvulkan1 >/dev/null 2>&1; } || true
    elif command -v dnf >/dev/null 2>&1; then dnf install -y -q vulkan-loader >/dev/null 2>&1 || true
    elif command -v pacman >/dev/null 2>&1; then pacman -S --noconfirm --needed vulkan-icd-loader >/dev/null 2>&1 || true
    fi

    IP=$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for (i = 1; i < NF; i++) if ($i == "src") print $(i + 1)}')
    [ -n "$IP" ] || IP=$(hostname -I 2>/dev/null | awk '{print $1}')
    if ! command -v nvidia-smi >/dev/null 2>&1 && command -v lspci >/dev/null 2>&1 && lspci | grep -qi nvidia; then
        echo "Aviso: GPU NVIDIA sem driver (nvidia-smi ausente). Instale o driver para ela ser usada."
    fi
elif [ "$OS" = Darwin ]; then
    systemsetup -setremotelogin on >/dev/null 2>&1 ||
        echo "Ative Ajustes do Sistema > Geral > Compartilhamento > Sessao Remota, se ainda nao estiver ativo."
    IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || true)
else
    echo "Sistema nao suportado: $OS"
    exit 1
fi

# 3. Authorize the main host key for the user who ran sudo.
mkdir -p "$TARGET_HOME/.ssh"
touch "$TARGET_HOME/.ssh/authorized_keys"
grep -qxF "$PUBLIC_KEY" "$TARGET_HOME/.ssh/authorized_keys" ||
    printf '%s\n' "$PUBLIC_KEY" >> "$TARGET_HOME/.ssh/authorized_keys"
chown -R "$TARGET_USER:$(id -gn "$TARGET_USER")" "$TARGET_HOME/.ssh"
chmod 700 "$TARGET_HOME/.ssh"
chmod 600 "$TARGET_HOME/.ssh/authorized_keys"

NAME=$(hostname -s 2>/dev/null || hostname)
echo
if [ -n "$ENROLL_URL" ]; then
    DATA="token=$ENROLL_TOKEN&host=$IP&user=$TARGET_USER&name=$NAME"
    if command -v curl >/dev/null 2>&1; then
        R=$(curl -fsS --max-time 120 --data "$DATA" "$ENROLL_URL/enroll" 2>&1) || R="falhou: $R"
    else
        R=$(wget -qO- --timeout=120 --post-data="$DATA" "$ENROLL_URL/enroll" 2>&1) || R="falhou: $R"
    fi
    echo "Registro no PC principal: $R"
else
    echo "Pronto. No PC principal, rode:"
    echo "  python cluster.py add $IP $TARGET_USER --name $NAME"
fi
