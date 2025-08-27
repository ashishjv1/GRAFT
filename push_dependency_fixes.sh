#!/bin/bash
# Push dependency fixes for GRAFT package

echo "=== Pushing Dependency Fixes ==="

# Navigate to project directory
cd /Users/user_admin/Documents/GRAFT-Tests

# Check current status
echo "Current git status:"
git status

# Add all changes
echo "Adding all changes..."
git add .

# Commit the dependency fixes
echo "Committing dependency fixes..."
git commit -m "Fix missing dependencies and bump version to 0.1.1

- Add transformers>=4.0.0 dependency for BERT model support
- Add medmnist>=2.0.0 dependency for medical imaging datasets
- Updated requirements.txt and pyproject.toml dependencies
- Bumped version from 0.1.0 to 0.1.1 in all files
- This fixes the ModuleNotFoundError for transformers package

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Push to test branch
echo "Pushing to test branch..."
git push origin test

echo "=== Push completed! ==="
echo "This will trigger a new build with version 0.1.1 and correct dependencies."