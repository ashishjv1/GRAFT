#!/bin/bash
# Push test import fixes for GRAFT package

echo "=== Pushing Test Import Fixes ==="

# Navigate to project directory
cd /Users/user_admin/Documents/GRAFT-Tests

# Check current status
echo "Current git status:"
git status

# Add all changes
echo "Adding all changes..."
git add .

# Commit the test fixes
echo "Committing test import fixes..."
git commit -m "Fix test imports to use new graft package structure - v0.1.3

- Fixed test_genindices.py: from genindices import -> from graft.genindices import
- Fixed test_graft.py: from GRAFT import -> from graft import  
- Fixed test_graft_e2e.py: from GRAFT import -> from graft import
- Removed deprecated sys.path manipulations in favor of proper package imports
- Updated all test files to use the new graft package structure
- Bumped version from 0.1.2 to 0.1.3
- This fixes ModuleNotFoundError in pytest test collection

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Push to test branch
echo "Pushing to test branch..."
git push origin test

echo "=== Push completed! ==="
echo "This should fix the pytest import errors in GitHub Actions."