# push_to_github.ps1
# One-click script to push D3-REC review code to GitHub
# Run from the project root: .\push_to_github.ps1

$ErrorActionPreference = "Continue"

$repoUrl = "https://github.com/CPSDSC-Lab/D3-REC.git"
$branch = "main"

Write-Host "========================================"
Write-Host " D3-REC GitHub Upload Script"
Write-Host " Target: $repoUrl"
Write-Host "========================================"

# 1. Check git is installed
git --version >$null 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Git not found. Please install Git first."
    exit 1
}
$gitVersion = git --version
Write-Host "[OK] Git detected: $gitVersion"

# 2. Ensure we are in the repo root
$repoRoot = git rev-parse --show-toplevel 2>$null
if ($LASTEXITCODE -ne 0 -or $repoRoot -ne (Get-Location).Path) {
    Write-Host "[INFO] Initializing git repository..."
    git init
}

# 3. Configure remote
$remotes = git remote
if ($remotes -contains "origin") {
    $currentUrl = git remote get-url origin
    if ($currentUrl -ne $repoUrl) {
        Write-Host "[INFO] Updating remote URL to $repoUrl"
        git remote set-url origin $repoUrl
    } else {
        Write-Host "[OK] Remote 'origin' already set to $repoUrl"
    }
} else {
    Write-Host "[INFO] Adding remote 'origin' -> $repoUrl"
    git remote add origin $repoUrl
}

# 4. Ensure branch name is main
$currentBranch = git branch --show-current 2>$null
if ($currentBranch -ne $branch) {
    Write-Host "[INFO] Renaming branch to '$branch'"
    git branch -M $branch
}

# 5. Stage all files (respects .gitignore)
Write-Host "[INFO] Staging files (training scripts & outputs excluded by .gitignore)..."
git add .

# 6. Show what will be committed
$staged = git diff --cached --name-only
$stagedCount = ($staged | Measure-Object).Count
Write-Host "[INFO] $stagedCount files staged for commit:"
git diff --cached --name-only | ForEach-Object { Write-Host "  - $_" }

# 7. Commit
$commitMsg = "Initial release: inference code, evaluation scripts, and core modules for D3-REC (CVPR 2025)"
Write-Host "[INFO] Committing with message: '$commitMsg'"
git commit -m "$commitMsg" >$null 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARN] Nothing new to commit (already committed). Continuing to push..."
}

# 8. Push
Write-Host "[INFO] Pushing to origin/$branch ..."
$pushErrFile = [System.IO.Path]::GetTempFileName()
git push -u origin $branch 2>$pushErrFile
$exitCode = $LASTEXITCODE

# Git writes progress to stderr; redirect it to a file and only show on actual failure
if ($exitCode -eq 0) {
    Remove-Item $pushErrFile -ErrorAction SilentlyContinue
    Write-Host ""
    Write-Host "========================================"
    Write-Host " SUCCESS! Code pushed to:"
    Write-Host " $repoUrl"
    Write-Host "========================================"
} else {
    $pushOutput = Get-Content $pushErrFile -Raw -ErrorAction SilentlyContinue
    Remove-Item $pushErrFile -ErrorAction SilentlyContinue

    Write-Host ""
    Write-Host "========================================"
    Write-Host " PUSH FAILED (exit code: $exitCode)"
    Write-Host "========================================"
    if ($pushOutput) { Write-Host $pushOutput }

    if ($pushOutput -match "403") {
        Write-Host ""
        Write-Host "[ERROR] HTTP 403 Forbidden"
        Write-Host "Cause: Current account lacks WRITE permission to this organization repo."
        Write-Host ""
        Write-Host "Solutions:"
        Write-Host "  1. Ask admin to add your account as collaborator at:"
        Write-Host "     https://github.com/CPSDSC-Lab/D3-REC/settings/access"
        Write-Host ""
        Write-Host "  2. Or run the fix script to switch account/token:"
        Write-Host "     .\fix_github_auth.ps1"
    } elseif ($pushOutput -match "404") {
        Write-Host ""
        Write-Host "[ERROR] Repository not found (404)."
        Write-Host "Create it first at: https://github.com/organizations/CPSDSC-Lab/repositories/new"
    }
}
