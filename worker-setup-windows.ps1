# Prepares a Windows PC to act as a llama-cluster worker.
# Run in an elevated prompt:  powershell -ExecutionPolicy Bypass -File worker-setup-windows.ps1
param(
    [string]$MainHostIp = '__MAIN_HOST_IP__',
    [string]$PublicKey = '__PUBLIC_KEY__'
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$me = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $me.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host 'Abra o PowerShell como administrador e rode de novo.'
    exit 1
}
if ($env:USERDOMAIN -eq 'AzureAD') {
    Write-Host 'Aviso: contas Microsoft Entra ID nao entram por chave SSH. Rode com uma conta local de administrador.'
}

# 1. OpenSSH Server. On some Windows builds the first install leaves no sshd service; reinstalling fixes it.
$cap = 'OpenSSH.Server~~~~0.0.1.0'
if (-not (Get-Service sshd -ErrorAction SilentlyContinue)) {
    Write-Host 'Instalando o OpenSSH Server (pode levar alguns minutos)...'
    Add-WindowsCapability -Online -Name $cap | Out-Null
}
if (-not (Get-Service sshd -ErrorAction SilentlyContinue)) {
    Write-Host 'Servico sshd ausente; reinstalando o OpenSSH Server...'
    Remove-WindowsCapability -Online -Name $cap | Out-Null
    Add-WindowsCapability -Online -Name $cap | Out-Null
}
if (-not (Get-Service sshd -ErrorAction SilentlyContinue)) {
    Write-Host 'O servico sshd continua ausente. Reinicie o PC e rode este script de novo.'
    exit 1
}
Set-Service sshd -StartupType Automatic
Start-Service sshd

# 2. SSH only from the main host. The RPC port stays on 127.0.0.1 and needs no rule.
if (Get-NetFirewallRule -Name OpenSSH-Server-In-TCP -ErrorAction SilentlyContinue) {
    Set-NetFirewallRule -Name OpenSSH-Server-In-TCP -Enabled True -Profile Any -RemoteAddress $MainHostIp
} else {
    New-NetFirewallRule -Name OpenSSH-Server-In-TCP -DisplayName 'OpenSSH Server (sshd)' -Direction Inbound `
        -Protocol TCP -LocalPort 22 -Action Allow -Profile Any -RemoteAddress $MainHostIp | Out-Null
}

# 3. Authorize the main host key. Administrators read it from ProgramData, other users from their profile.
function Add-KeyLine($path) {
    if (-not (Test-Path -LiteralPath $path) -or -not (Select-String -LiteralPath $path -SimpleMatch $PublicKey -Quiet)) {
        Add-Content -LiteralPath $path -Value $PublicKey
    }
}
$adm = Join-Path $env:ProgramData 'ssh\administrators_authorized_keys'
Add-KeyLine $adm
icacls.exe $adm /inheritance:r /grant '*S-1-5-32-544:F' /grant '*S-1-5-18:F' | Out-Null
New-Item -ItemType Directory -Force (Join-Path $HOME '.ssh') | Out-Null
Add-KeyLine (Join-Path $HOME '.ssh\authorized_keys')

# 4. PowerShell as the SSH shell.
if (-not (Test-Path HKLM:\SOFTWARE\OpenSSH)) { New-Item HKLM:\SOFTWARE\OpenSSH | Out-Null }
New-ItemProperty -Path HKLM:\SOFTWARE\OpenSSH -Name DefaultShell -PropertyType String -Force `
    -Value "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" | Out-Null

# 5. What the main host needs.
$net = Get-NetIPConfiguration | Where-Object { $_.IPv4DefaultGateway -and $_.NetAdapter.Status -eq 'Up' } | Select-Object -First 1
$ip = @($net.IPv4Address.IPAddress)[0]
$user = if ($env:USERDOMAIN -and $env:USERDOMAIN -ne $env:COMPUTERNAME) { "$env:USERDOMAIN\$env:USERNAME" } else { $env:USERNAME }
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Write-Host 'Aviso: nvidia-smi nao encontrado. Sem GPU NVIDIA, este PC entra como Vulkan (AMD/Intel Arc) ou CPU.'
}
Write-Host ''
Write-Host 'Pronto. No PC principal, rode:'
Write-Host "  python cluster.py add $ip `"$user`" --name $env:COMPUTERNAME"
