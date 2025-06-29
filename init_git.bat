@echo off
echo Initializing Git repository for EEG Hand Motion Prediction project...

:: Check if Git is installed
git --version > nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Git is not installed or not in PATH. Please install Git from https://git-scm.com/downloads
    exit /b 1
)

:: Initialize Git repository
echo Initializing Git repository...
git init

:: Add all files to staging
echo Adding files to Git...
git add .

:: Initial commit
echo Creating initial commit...
git commit -m "Initial commit: EEG-based hand motion prediction project"

:: Instructions for GitHub
echo.
echo Local Git repository initialized successfully!
echo.
echo Next steps:
echo 1. Create a new repository on GitHub (https://github.com/new)
echo 2. Run the following commands to link and push to GitHub:
echo    git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY-NAME.git
echo    git branch -M main
echo    git push -u origin main
echo.
echo Replace YOUR-USERNAME and YOUR-REPOSITORY-NAME with your actual GitHub username and repository name.
echo.

:: Prompt user if they want to configure Git user info
set /p configureGit=Do you want to configure Git user info? (y/n): 
if /i "%configureGit%"=="y" (
    set /p userName=Enter your name: 
    set /p userEmail=Enter your email: 
    
    git config --global user.name "%userName%"
    git config --global user.email "%userEmail%"
    
    echo Git user configured successfully!
)

echo Script completed. See setup_github.md for detailed instructions.
echo.

pause