# Rebuild the WSL2 NAT when WSL has no outbound network.
# Symptom: WSL -> Windows host works (e.g. /dev/tcp/<gateway>/22), but anything past the
# host (ping 1.1.1.1, curl https://...) times out; a reboot and a winnat restart do NOT help.
# Cause on fablab (2026-09-03): stale HNS "WSL" network of type ICS. Removing it and letting
# WSL recreate it on next start fixed it immediately. Must run from an ELEVATED PowerShell.
#Requires -RunAsAdministrator
$ErrorActionPreference = 'Continue'
Write-Host "1. wsl --shutdown";                         wsl --shutdown; Start-Sleep 3
Write-Host "2. remove HNS network 'WSL'";               Get-HnsNetwork | Where-Object Name -eq 'WSL' | Remove-HnsNetwork
Write-Host "3. cycle winnat + hns";                     sc.exe stop winnat | Out-Null; sc.exe start winnat | Out-Null; Restart-Service hns -Force; Start-Sleep 2
Write-Host "4. start WSL (HNS recreates the network)";  wsl -d Ubuntu-WSL2 -- true; Start-Sleep 3
Get-HnsNetwork | Where-Object Name -eq 'WSL' | ForEach-Object { Write-Host "   WSL net: $($_.Subnets.AddressPrefix) gw=$($_.Subnets.GatewayAddress) type=$($_.Type)" }
Write-Host "5. test from WSL (expect HTTP 301/200):"
wsl -d Ubuntu-WSL2 -- curl -sS -m 10 -o /dev/null -w 'WSL outbound -> HTTP %{http_code}\n' https://1.1.1.1
# Note: Restart-Service SharedAccess (ICS) refused to stop on fablab; the fix worked without it.
# If /etc/resolv.conf is missing afterwards: WSL generates it at boot unless /etc/wsl.conf has
# [network] generateResolvConf=false — the generated file points at the host gateway, which works.
