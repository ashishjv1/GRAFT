#!/bin/bash
# Push OIDC workflow updates for GRAFT package

echo "=== Pushing OIDC Workflow Updates ==="

# Navigate to project directory
cd /Users/user_admin/Documents/GRAFT-Tests

# Check current status
echo "Current git status:"
git status

# Add all changes
echo "Adding all changes..."
git add .

# Commit the OIDC workflow updates
echo "Committing OIDC workflow updates..."
git commit -m "Implement OIDC trusted publishing for secure PyPI deployment - v0.1.5

🔐 Security Improvements:
- Replaced API token authentication with OIDC trusted publishing
- Added id-token: write permissions for both workflows  
- Updated to latest action versions (v4/v5)
- Eliminated need for stored API tokens/secrets

📦 Workflow Changes:
- publish.yml: Production PyPI publishing on GitHub releases (pypi environment)
- test.yml: Automatic TestPyPI publishing on test branch pushes (testpypi environment)
- Added comprehensive OIDC_SETUP.md with configuration instructions

🚀 Publishing Flow:
- test branch → TestPyPI (automatic after tests pass)
- GitHub release → Production PyPI (with environment protection)

- Bumped version from 0.1.4 to 0.1.5
- No more API tokens needed - more secure and easier to maintain!

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Push to test branch
echo "Pushing to test branch..."
git push origin test

echo "=== Push completed! ==="
echo ""
echo "🔐 IMPORTANT: OIDC Setup Required!"
echo "Before workflows can publish, you need to configure OIDC:"
echo ""
echo "1. Read OIDC_SETUP.md for detailed instructions"
echo "2. Configure PyPI trusted publisher: https://pypi.org/manage/account/publishing/"
echo "3. Configure TestPyPI trusted publisher: https://test.pypi.org/manage/account/publishing/"
echo "4. Create GitHub environments: pypi & testpypi"
echo ""
echo "Once configured, this push will automatically publish to TestPyPI!"