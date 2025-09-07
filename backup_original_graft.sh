#!/bin/bash
# Backup original GRAFT code before productionizing pip package

echo "=== Backing Up Original GRAFT Code ==="

# Navigate to your main GRAFT repository (not GRAFT-Tests)
# You'll need to run this in your main GRAFT repo directory
echo "📍 Navigate to your main GRAFT repository first:"
echo "cd path/to/your/main/GRAFT/repository"
echo ""

echo "Then run these commands:"
echo ""

echo "# 1. Create a backup of current main branch"
echo "git checkout main"
echo "git pull origin main"  
echo "git checkout -b legacy-original-graft"
echo "git push -u origin legacy-original-graft"
echo ""

echo "# 2. Create a backup tag as well (extra safety)"
echo "git tag -a v-original-backup -m 'Backup of original GRAFT before pip packaging'"
echo "git push origin v-original-backup"
echo ""

echo "# 3. Create production branch from your test branch"
echo "git checkout -b production"
echo ""

echo "# 4. Pull the packageable version from your GRAFT-Tests"
echo "# (You'll need to manually copy files or set up a remote)"
echo ""

echo "=== Backup Strategy Explained ==="
echo "✅ legacy-original-graft branch: Contains your original working GRAFT"
echo "✅ v-original-backup tag: Permanent snapshot of original code"  
echo "✅ production branch: Will contain the pip-packageable version"
echo "✅ main branch: Remains unchanged until you're confident"
echo ""
echo "This way you have multiple layers of backup!"