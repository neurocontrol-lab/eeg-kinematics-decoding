# GitHub Upload Guide

## Prerequisites
1. Install Git from [git-scm.com](https://git-scm.com/downloads) if you haven't already
2. Create a GitHub account at [github.com](https://github.com) if you don't have one
3. Configure Git with your username and email:
   ```
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```

## Steps to Upload to GitHub

### 1. Initialize Git Repository
Open a command prompt or PowerShell in your project directory and run:
```
cd "c:\Users\thowf\OneDrive\Desktop\handmotion_2"
git init
```

### 2. Add Files to Git
```
git add .
```

### 3. Commit Changes
```
git commit -m "Initial commit: EEG-based hand motion prediction project"
```

### 4. Create a New Repository on GitHub
1. Go to [github.com](https://github.com)
2. Click the '+' icon in the top right and select 'New repository'
3. Name your repository (e.g., "eeg-hand-motion-prediction")
4. Add a description (optional)
5. Choose public or private repository
6. Do NOT initialize with README, .gitignore, or license (we already have these)
7. Click 'Create repository'

### 5. Link Local Repository to GitHub
GitHub will show commands after repository creation. Use the commands under "…or push an existing repository from the command line":

```
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY-NAME.git
git branch -M main
git push -u origin main
```

Replace `YOUR-USERNAME` and `YOUR-REPOSITORY-NAME` with your actual GitHub username and repository name.

### 6. Enter GitHub Credentials
When prompted, enter your GitHub username and password or personal access token.

### 7. Verify Upload
Go to your GitHub repository URL to verify that all files have been uploaded successfully.

## Notes
- Large files (like the dataset) are excluded via .gitignore
- If you encounter issues with file size limits, consider using Git LFS (Large File Storage)
- For future updates, use the standard git workflow: `git add`, `git commit`, `git push`