# fix_github_auth.ps1
# Diagnose and fix GitHub 403 permission issues for CPSDSC-Lab/D3-REC

$ErrorActionPreference = "Stop"
$repoUrl = "https://github.com/CPSDSC-Lab/D3-REC.git"

Write-Host "========================================"
Write-Host " GitHub Auth Fix for D3-REC"
Write-Host "========================================"
Write-Host ""

# 1. Show current remote
try {
    $currentUrl = git remote get-url origin
    Write-Host "Current remote URL: $currentUrl"
} catch {
    Write-Host "[WARN] No remote configured."
}

# 2. Show current git user
Write-Host ""
Write-Host "Current Git config:"
git config user.name 2>$null | ForEach-Object { Write-Host "  user.name : $_" }
git config user.email 2>$null | ForEach-Object { Write-Host "  user.email: $_" }

# 3. Show current credential helper
$credHelper = git config --get credential.helper 2>$null
if ($credHelper) {
    Write-Host "  credential.helper: $credHelper"
} else {
    Write-Host "  credential.helper: (not set)"
}

Write-Host ""
Write-Host "----------------------------------------"
Write-Host " Diagnosis: HTTP 403 means the GitHub"
Write-Host " account has NO WRITE permission to this"
Write-Host " organization repository."
Write-Host "----------------------------------------"
Write-Host ""

Write-Host "Choose a fix method:"
Write-Host "  [1] Clear saved credentials and retry (for PAT login)"
Write-Host "  [2] Switch remote to SSH (git@github.com:...)"
Write-Host "  [3] Switch remote back to HTTPS (if SSH fails)"
Write-Host "  [4] Exit and ask admin to add 'CharlesQian-ai' as collaborator"
Write-Host ""

$choice = Read-Host "Enter option (1-4)"

switch ($choice) {
    "1" {
        Write-Host ""
        Write-Host "[INFO] Clearing cached credentials for github.com..."
        git credential-manager reject https://github.com/CPSDSC-Lab/D3-REC.git 2>$null
        git credential-manager reject https://github.com 2>$null
        Write-Host "[OK] Credentials cleared."
        Write-Host ""
        Write-Host "Next steps:"
        Write-Host "  1. Run: git push -u origin main"
        Write-Host "  2. When prompted, enter your GitHub username"
        Write-Host "  3. For password, use a Personal Access Token (NOT your GitHub password)"
        Write-Host ""
        Write-Host "To generate a PAT:"
        Write-Host "  https://github.com/settings/tokens -> Generate new token (classic)"
        Write-Host "  Required scope: [repo]"
    }
    "2" {
        Write-Host ""
        Write-Host "[INFO] Switching remote to SSH..."
        git remote set-url origin git@github.com:CPSDSC-Lab/D3-REC.git
        Write-Host "[OK] Remote updated to SSH."
        Write-Host ""
        Write-Host "Testing SSH connection..."
        ssh -T git@github.com 2>&1 | ForEach-Object { Write-Host "  $_" }
        Write-Host ""
        Write-Host "If you see 'Permission denied', you need to add your SSH key to GitHub:"
        Write-Host "  https://github.com/settings/keys"
        Write-Host ""
        Write-Host "Then run: git push -u origin main"
    }
    "3" {
        Write-Host ""
        Write-Host "[INFO] Switching remote back to HTTPS..."
        git remote set-url origin $repoUrl
        Write-Host "[OK] Remote updated to HTTPS."
    }
    "4" {
        Write-Host ""
        Write-Host "[INFO] Please ask the repository admin to:"
        Write-Host "  1. Go to https://github.com/CPSDSC-Lab/D3-REC/settings/access"
        Write-Host "  2. Click 'Add people'"
        Write-Host "  3. Invite: CharlesQian-ai"
        Write-Host "  4. Grant 'Write' or 'Admin' role"
    }
    default {
        Write-Host "[INFO] Exiting without changes."
    }
}

Write-Host ""
Write-Host "========================================"
Write-Host " Done."
Write-Host "========================================"
