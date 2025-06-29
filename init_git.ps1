# PowerShell script to initialize Git repository

# Check if Git is installed
try {
    $gitVersion = git --version
    Write-Host "Git is installed: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "Git is not installed or not in PATH. Please install Git from https://git-scm.com/downloads" -ForegroundColor Red
    exit 1
}

# Initialize Git repository
Write-Host "Initializing Git repository..." -ForegroundColor Cyan
git init

# Add all files to staging
Write-Host "Adding files to Git..." -ForegroundColor Cyan
git add .

# Initial commit
Write-Host "Creating initial commit..." -ForegroundColor Cyan
git commit -m "Initial commit: EEG-based hand motion prediction project"

# Instructions for GitHub
Write-Host ""
Write-Host "Local Git repository initialized successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Create a new repository on GitHub (https://github.com/new)" -ForegroundColor Yellow
Write-Host "2. Run the following commands to link and push to GitHub:" -ForegroundColor Yellow
Write-Host "   git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY-NAME.git" -ForegroundColor White
Write-Host "   git branch -M main" -ForegroundColor White
Write-Host "   git push -u origin main" -ForegroundColor White
Write-Host ""
Write-Host "Replace YOUR-USERNAME and YOUR-REPOSITORY-NAME with your actual GitHub username and repository name." -ForegroundColor Yellow
Write-Host ""

# Prompt user if they want to configure Git user info
$configureGit = Read-Host "Do you want to configure Git user info? (y/n)"
if ($configureGit -eq "y") {
    $userName = Read-Host "Enter your name"
    $userEmail = Read-Host "Enter your email"
    
    git config --global user.name "$userName"
    git config --global user.email "$userEmail"
    
    Write-Host "Git user configured successfully!" -ForegroundColor Green
}

Write-Host "Script completed. See setup_github.md for detailed instructions." -ForegroundColor Green