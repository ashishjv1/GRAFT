# Safe Production Deployment Options

## Current Situation
- **main branch**: Original GRAFT code (working, stable)  
- **test branch**: New pip-packageable version (tested on TestPyPI)
- **Goal**: Deploy to PyPI while preserving original code

## Option 1: Use Production Branch (Recommended - Safest)

### Advantages:
- ✅ Original `main` branch completely untouched
- ✅ Can easily switch between versions
- ✅ Zero risk to existing code
- ✅ Clear separation of concerns

### Steps:
1. **Backup original code**:
   ```bash
   # In your main GRAFT repo
   git checkout main
   git checkout -b legacy-original-graft
   git push -u origin legacy-original-graft
   git tag -a v-original-backup -m "Original GRAFT backup"
   git push origin v-original-backup
   ```

2. **Create production branch**:
   ```bash
   git checkout -b production
   # Copy files from GRAFT-Tests or merge test branch here
   git push -u origin production
   ```

3. **Update OIDC settings** on PyPI to use:
   - **Workflow**: `publish.yml`
   - **Environment**: `pypi`
   - **Branch**: `production` (instead of `main`)

4. **Create release from production branch**

### OIDC Configuration:
```
PyPI Project Name: graft-pytorch
Owner: ashishjv1
Repository: GRAFT
Workflow: publish.yml  
Environment: pypi
Branch restriction: production  # Add this for extra safety
```

## Option 2: Legacy Branch Backup

### Steps:
1. **Create legacy branch from current main**:
   ```bash
   git checkout main
   git checkout -b legacy-original-graft  
   git push -u origin legacy-original-graft
   ```

2. **Replace main with packaged version**:
   ```bash
   git checkout main
   # Copy files from GRAFT-Tests
   git add .
   git commit -m "Replace with pip-packageable version"
   git push origin main
   ```

3. **Use standard PyPI deployment** (targets main branch)

## Option 3: Separate Repository (Ultra-Safe)

Create `GRAFT-Production` repository:
- Keep original GRAFT repo completely unchanged
- New repo only for pip package
- Independent development and releases

### Advantages:
- ✅ Original repo completely untouched
- ✅ Clean separation
- ✅ Different issue tracking
- ✅ Can have different collaborators

## Option 4: Fork Strategy

1. **Fork your own repository** to `GRAFT-Package`
2. **Use the fork** for pip packaging
3. **Keep original** for research/development

## Recommended Workflow: Option 1 (Production Branch)

### Setup Commands:

```bash
# 1. In your main GRAFT repository
cd /path/to/your/main/GRAFT/repo
git checkout main
git pull origin main

# 2. Create safety backups
git checkout -b legacy-original-graft
git push -u origin legacy-original-graft
git checkout main
git tag -a v-original-backup -m "Backup before pip packaging"
git push origin v-original-backup

# 3. Create production branch  
git checkout -b production

# 4. Copy content from GRAFT-Tests
# (Manual step - copy all files from GRAFT-Tests to current directory)
cp -r /Users/user_admin/Documents/GRAFT-Tests/* .
cp /Users/user_admin/Documents/GRAFT-Tests/.[^.]* . 2>/dev/null || true

# 5. Commit and push
git add .
git commit -m "Add pip-packageable version for PyPI deployment

- Complete package structure with graft/ directory
- CLI interface and Python API
- OIDC-enabled workflows for secure publishing  
- Comprehensive tests and documentation
- Professional README and setup files

🚀 Ready for PyPI production deployment!"

git push -u origin production
```

### PyPI OIDC Setup:
- **Branch restriction**: `production`
- **Workflow**: `publish.yml` 
- **Environment**: `pypi`

### Release Creation:
- **Target branch**: `production`
- **Tag**: `v0.1.7`
- **Title**: `GRAFT v0.1.7 - Production Release`

## Recovery Plan (If Something Goes Wrong)

### Quick Rollback:
```bash
# Switch back to original code
git checkout legacy-original-graft
git checkout -b main-restored
git push -u origin main-restored
```

### Or restore from tag:
```bash
git checkout v-original-backup
git checkout -b main-restored  
git push -u origin main-restored
```

## Monitoring After Deployment

1. **Test PyPI installation**:
   ```bash
   pip install graft-pytorch
   python -c "import graft; print('Success!')"
   ```

2. **Monitor GitHub Issues** for user problems

3. **Keep both versions documented**:
   - `main`/`legacy-original-graft`: Original research code
   - `production`: PyPI package version

4. **Easy switching**:
   ```bash
   # For package development
   git checkout production
   
   # For research/original work  
   git checkout legacy-original-graft
   ```

This way you have maximum safety with easy recovery options!