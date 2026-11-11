# Sentinel One-Line Installer for Windows (PowerShell)
# Usage: irm https://raw.githubusercontent.com/rimraf-adi/sentinel/main/install.ps1 | iex

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "Sentinel Installer (Windows)" -ForegroundColor Cyan
Write-Host ""

$CargoBin = Join-Path $env:USERPROFILE ".cargo\bin"

# 1. Ensure Rust / Cargo is available
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Host "-> Cargo not found. Installing Rust via rustup-init.exe..." -ForegroundColor Yellow
    $RustupExe = Join-Path $env:TEMP "rustup-init.exe"
    Invoke-WebRequest -Uri "https://win.rustup.rs/x86_64" -OutFile $RustupExe
    & $RustupExe -y
    Remove-Item $RustupExe -ErrorAction SilentlyContinue

    $env:Path = "$CargoBin;$env:Path"
    if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
        Write-Error "Cargo was installed to $CargoBin, but is not on PATH yet. Restart PowerShell and re-run this installer."
    }
}
Write-Host "✓ Rust/Cargo detected: $(cargo --version)" -ForegroundColor Green

# 2. Check Node/npm for dashboard build fallback
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Host "! Node.js/npm not detected. Installing with SENTINEL_SKIP_UI=1 (API + CLI mode)." -ForegroundColor Yellow
    $env:SENTINEL_SKIP_UI = "1"
} else {
    Write-Host "✓ Node.js/npm detected (will bundle embedded UI dashboard)" -ForegroundColor Green
}

# 3. Check Python (warn only, user can configure later)
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "! Python not detected on PATH. Install Python 3.10+ (tick 'Add to PATH') before running training jobs." -ForegroundColor Yellow
} else {
    Write-Host "✓ Python detected: $(python --version 2>&1)" -ForegroundColor Green
}

# 4. Install sentinel-cli from GitHub
Write-Host ""
Write-Host "-> Installing sentinel-cli from GitHub (compiling Rust binary)..." -ForegroundColor Cyan
cargo install --git https://github.com/rimraf-adi/sentinel.git sentinel-cli --locked --force

Write-Host ""
Write-Host "Sentinel installed cleanly!" -ForegroundColor Green
Write-Host ""
Write-Host "Quick start:"
Write-Host "  sentinel doctor        # verify Python, PyTorch, and GPU setup"
Write-Host "  sentinel init          # scaffold workspace in current folder"
Write-Host "  sentinel dev           # launch daemon & dashboard at http://localhost:2512"
Write-Host ""
Write-Host "If 'sentinel' is not recognized, restart PowerShell or ensure $CargoBin is in your PATH."
