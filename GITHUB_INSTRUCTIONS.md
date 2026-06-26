# Pushing to GitHub - Instructions

## 1. Create a new repository on GitHub

1. Go to https://github.com/new
2. Repository name: `zkml-verifiable-inference`
3. Description: `Verifiable Machine Learning Inference using Zero-Knowledge Proofs (BITS F463 Cryptography)`
4. Set to **Public** (or Private if you prefer)
5. Do NOT initialize with README (we already have one)
6. Click "Create repository"

## 2. Initialize Git and push (run in PowerShell)

```powershell
cd C:\ZKML_Project

# Initialize git
git init

# Create .gitignore first
# (we already have one - see below)

# Stage all files
git add .

# Commit
git commit -m "Initial commit: ZKML verifiable inference prototype"

# Add your GitHub repo as remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/zkml-verifiable-inference.git

# Push
git branch -M main
git push -u origin main
```

## 3. If you need to set up Git credentials

```powershell
# Set your identity
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# For authentication, use GitHub CLI or a personal access token:
# Option A: GitHub CLI
winget install GitHub.cli
gh auth login

# Option B: Personal Access Token
# Go to GitHub > Settings > Developer Settings > Personal Access Tokens
# Generate a token with 'repo' scope
# Use it as your password when pushing
```

## 4. Subsequent pushes

```powershell
git add .
git commit -m "your message here"
git push
```
