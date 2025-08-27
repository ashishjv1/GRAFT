#!/bin/bash
# Git push commands for GRAFT-Tests

echo "=== Git Setup and Push to Test Branch ==="

# Navigate to project directory
cd /Users/user_admin/Documents/GRAFT-Tests

# Initialize git if needed
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
fi

# Add remote origin
echo "Adding/updating remote origin..."
git remote add origin https://github.com/ashishjv1/GRAFT.git 2>/dev/null || git remote set-url origin https://github.com/ashishjv1/GRAFT.git

# Verify remote
echo "Remote repositories:"
git remote -v

# Fetch from remote
echo "Fetching from remote..."
git fetch origin

# Check current branch
echo "Current git status:"
git status

# Switch to or create test branch
echo "Switching to test branch..."
git checkout test 2>/dev/null || git checkout -b test

# Pull latest changes if test branch exists remotely
echo "Pulling latest changes..."
git pull origin test 2>/dev/null || echo "Test branch doesn't exist remotely yet"

# Add all changes
echo "Adding all changes..."
git add .

# Show status
echo "Files to be committed:"
git status

# Commit changes
echo "Committing changes..."
git commit -m "Fix import errors and prepare package for pip distribution

- Fixed relative imports in model_mapper.py, genindices.py, trainer.py
- Created proper package structure with graft/ directory  
- Added GitHub Actions workflow for TestPyPI/PyPI publishing
- Added CLI entry point and argument parser
- Fixed all ModuleNotFoundError issues
- Added .gitignore and test script

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Push to test branch
echo "Pushing to test branch..."
git push -u origin test

echo "=== Push completed! ==="
echo "Check your GitHub repository and Actions tab for the workflow."