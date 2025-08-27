#!/bin/bash
# Push linting fixes for GRAFT package

echo "=== Pushing Linting Fixes ==="

# Navigate to project directory
cd /Users/user_admin/Documents/GRAFT-Tests

# Check current status
echo "Current git status:"
git status

# Add all changes
echo "Adding all changes..."
git add .

# Commit the linting fixes
echo "Committing linting fixes..."
git commit -m "Fix flake8 linting errors and bump version to 0.1.2

- Fixed undefined names in resnet.py __all__ exports (lowercase -> uppercase)
- Excluded legacy trainer.py from root directory that was causing import errors
- Updated .gitignore to exclude legacy files (trainer.py, GRAFT.py, GRAFT_swinft.py)
- Updated MANIFEST.in to exclude legacy files from package distribution
- Updated GitHub Actions to only lint graft/ and tests/ directories
- Bumped version from 0.1.1 to 0.1.2
- This fixes F821 and F822 flake8 errors in CI/CD pipeline

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Push to test branch
echo "Pushing to test branch..."
git push origin test

echo "=== Push completed! ==="
echo "This should fix the flake8 linting errors in GitHub Actions."