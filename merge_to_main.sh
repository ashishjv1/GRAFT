#!/bin/bash
# Merge test branch to main for production PyPI release

echo "=== Merging Test Branch to Main for Production ==="

# Navigate to project directory
cd /Users/user_admin/Documents/GRAFT-Tests

# Check current status
echo "Current git status:"
git status

# Ensure we're on test branch
echo "Switching to test branch..."
git checkout test

# Pull latest changes
echo "Pulling latest changes..."
git pull origin test

# Switch to main branch (create if doesn't exist)
echo "Switching to main branch..."
git checkout main 2>/dev/null || git checkout -b main

# Merge test branch into main
echo "Merging test branch into main..."
git merge test

# Push main branch to remote
echo "Pushing main branch to remote..."
git push -u origin main

echo "=== Merge completed! ==="
echo ""
echo "✅ Test branch successfully merged to main"
echo "✅ Main branch pushed to GitHub"
echo "🚀 Ready for PyPI release creation!"
echo ""
echo "Next steps:"
echo "1. Verify OIDC trusted publishing is set up on PyPI"
echo "2. Create a GitHub release from main branch"
echo "3. PyPI will automatically publish via GitHub Actions"