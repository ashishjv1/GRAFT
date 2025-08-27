#!/bin/bash
# Push test warnings and failure fixes for GRAFT package

echo "=== Pushing Test Warnings and Failure Fixes ==="

# Navigate to project directory
cd /Users/user_admin/Documents/GRAFT-Tests

# Check current status
echo "Current git status:"
git status

# Add all changes
echo "Adding all changes..."
git add .

# Commit the test fixes
echo "Committing test warning and failure fixes..."
git commit -m "Fix test warnings and failures - v0.1.4

- Fixed pytest collection warning: renamed TestModel -> MockModel to avoid __init__ confusion
- Fixed NumPy 2.0 deprecation warning in grad_dist.py: replaced @ operator with np.matmul()
- Fixed failing test_sample_selection_deterministic by adding np.random.seed() alongside torch.manual_seed()
- The deterministic test was failing because sample_selection uses np.random.choice() but only torch seed was set
- Bumped version from 0.1.3 to 0.1.4
- This should make all tests pass cleanly without warnings

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Push to test branch
echo "Pushing to test branch..."
git push origin test

echo "=== Push completed! ==="
echo "All tests should now pass without warnings or failures."