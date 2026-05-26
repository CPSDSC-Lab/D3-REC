# push_to_github.ps1
# One-click script to push D3-REC review code to GitHub
# Run from the project root: .\push_to_github.ps1

$ErrorActionPreference = "Stop"

$repoUrl = "https://github.com/CPSDSC-Lab/D3-REC.git"
$branch = "main"

Write-Host "========================================"
Write-Host " D3-REC GitHub Upload Script"
Write-Host " Target: $repoUrl"
Write-Host "========================================"

# 1. Check git is installed
try {
    $gitVersion = git --version
    Write-Host "[OK] Git detected: $gitVersion"
} catch {
    Write-Host "[ERROR] Git not found. Please install Git first."
    exit 1
}

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
git commit -m "$commitMsg"

if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARN] Nothing to commit or commit failed."
    exit 0
}

# 8. Push
Write-Host "[INFO] Pushing to origin/$branch ..."
try {
    git push -u origin $branch
    Write-Host ""
    Write-Host "========================================"
    Write-Host " SUCCESS! Code pushed to:"
    Write-Host " $repoUrl"
    Write-Host "========================================"
} catch {
    Write-Host "[ERROR] Push failed. Common causes:"
    Write-Host "  1. No write access to the repository."
    Write-Host "  2. Remote repository does not exist yet."
    Write-Host "  3. Authentication required (use Git Credential Manager or PAT)."
    Write-Host ""
    Write-Host "If the repo doesn't exist, create it first at:"
    Write-Host " https://github.com/organizations/CPSDSC-Lab/repositories/new"
}
