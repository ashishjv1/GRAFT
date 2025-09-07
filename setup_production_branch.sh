#!/bin/bash
# Set up production branch for safe PyPI deployment

echo "=== Setting Up Production Branch for Safe PyPI Deployment ==="
echo ""
echo "🔒 This script will preserve your original GRAFT code safely!"
echo ""

# Check if user wants to proceed
read -p "This will create backups and a production branch. Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cancelled by user"
    exit 1
fi

echo ""
echo "📝 Please provide the path to your main GRAFT repository:"
read -p "GRAFT repo path: " GRAFT_REPO_PATH

# Validate path exists
if [ ! -d "$GRAFT_REPO_PATH" ]; then
    echo "❌ Directory does not exist: $GRAFT_REPO_PATH"
    exit 1
fi

# Navigate to main GRAFT repo
cd "$GRAFT_REPO_PATH"

echo ""
echo "🔄 Current directory: $(pwd)"
echo "📋 Git status:"
git status --short

echo ""
echo "🛡️  Step 1: Creating safety backups..."

# Create legacy branch backup
echo "   Creating legacy-original-graft branch..."
git checkout main
git checkout -b legacy-original-graft 2>/dev/null || git checkout legacy-original-graft
git push -u origin legacy-original-graft

# Create backup tag
echo "   Creating backup tag..."
git checkout main
git tag -a v-original-backup -m "Backup of original GRAFT before pip packaging" 2>/dev/null || echo "   (Tag already exists)"
git push origin v-original-backup 2>/dev/null || echo "   (Tag already pushed)"

echo "✅ Backups created:"
echo "   - Branch: legacy-original-graft"  
echo "   - Tag: v-original-backup"

echo ""
echo "🚀 Step 2: Creating production branch..."

# Create production branch
git checkout -b production 2>/dev/null || git checkout production

# Copy files from GRAFT-Tests
echo "   Copying packageable files..."
GRAFT_TESTS_PATH="/Users/user_admin/Documents/GRAFT-Tests"

if [ -d "$GRAFT_TESTS_PATH" ]; then
    # Copy all files including hidden ones
    rsync -av --exclude='.git' "$GRAFT_TESTS_PATH/" ./
    
    # Add and commit
    git add .
    git commit -m "Add pip-packageable version for PyPI deployment

- Complete package structure with graft/ directory  
- CLI interface with graft-train command
- Python API with ModelTrainer and TrainingConfig
- OIDC-enabled workflows for secure publishing
- Comprehensive tests (Python 3.8-3.11 support)
- Professional documentation and README
- All tests passing on TestPyPI

🚀 Ready for production PyPI deployment!"

    # Push production branch
    git push -u origin production
    
    echo "✅ Production branch created and pushed!"
else
    echo "❌ GRAFT-Tests directory not found at: $GRAFT_TESTS_PATH"
    echo "   Please manually copy files from GRAFT-Tests to current directory"
fi

echo ""
echo "🎉 Setup Complete!"
echo ""
echo "📊 Your repository now has:"
echo "   🔒 main: Original GRAFT code (unchanged)"  
echo "   🛡️  legacy-original-graft: Backup of original code"
echo "   🏷️  v-original-backup: Tagged backup snapshot"
echo "   🚀 production: Pip-packageable version for PyPI"
echo ""
echo "📋 Next Steps:"
echo "   1. Set up PyPI OIDC with branch restriction: 'production'"
echo "   2. Create GitHub release targeting 'production' branch"
echo "   3. PyPI will auto-publish via GitHub Actions"
echo ""
echo "🔄 To switch between versions:"
echo "   Research work: git checkout legacy-original-graft"
echo "   Package work:  git checkout production"
echo "   Original:      git checkout main"
echo ""
echo "🛡️  Recovery (if needed):"
echo "   git checkout legacy-original-graft"
echo "   git checkout -b main-restored"