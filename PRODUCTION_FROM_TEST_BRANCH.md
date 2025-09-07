# Production PyPI Deployment from Test Branch

## Strategy Overview

- **main branch**: Core GRAFT research code (preserved exactly as is)
- **test branch**: PyPI package version (already tested on TestPyPI)
- **Releases from test branch**: Automatically publish to production PyPI

## Configuration Changes Needed

### 1. PyPI OIDC Configuration

Go to https://pypi.org/manage/account/publishing/ and add:

```
PyPI Project Name: graft-pytorch
Owner: ashishjv1
Repository name: GRAFT
Workflow filename: publish.yml
Environment name: pypi
```

**Important**: Leave branch field empty or specify `test` if the option is available.

### 2. GitHub Environment Setup

1. Go to your GitHub repo → Settings → Environments
2. Create environment named `pypi` (if not exists)
3. Add protection rules:
   - ✅ Required reviewers: Add yourself
   - ✅ Restrict to protected branches: Select `test`
   - ✅ Wait timer: 5 minutes (optional)

### 3. Update Workflow (Optional Enhancement)

The current `publish.yml` will work fine, but we can make it clearer:

```yaml
name: Publish to PyPI

on:
  # Trigger on new releases (from any branch, but we'll create from test)
  release:
    types: [published]
  
  # Allow manual triggering
  workflow_dispatch:

permissions:
  id-token: write  # IMPORTANT: this permission is mandatory for trusted publishing

jobs:
  # ... existing jobs remain the same
```

## Deployment Process

### Step 1: Create GitHub Release from Test Branch

1. **Go to**: https://github.com/ashishjv1/GRAFT/releases
2. **Click**: "Create a new release"
3. **Configure**:
   - **Tag version**: `v0.1.7`
   - **Target**: `test` branch (this is the key!)
   - **Release title**: `GRAFT v0.1.7 - Production Release`
   - **Description**: Use the comprehensive release notes

### Step 2: Automatic Publication

Once you create the release:
- GitHub Actions triggers `publish.yml` workflow
- OIDC authenticates with PyPI automatically
- Package builds from test branch code
- Uploads to https://pypi.org/project/graft-pytorch/

### Step 3: Verification

```bash
# Test installation
pip install graft-pytorch

# Verify import
python -c "import graft; print(f'GRAFT v{graft.__version__} installed successfully!')"

# Test CLI
graft-train --help
```

## Branch Management Going Forward

### For Research/Core Development:
```bash
git checkout main
# Work on core GRAFT research code
git add .
git commit -m "Research improvements"
git push origin main
```

### For Package Updates:
```bash
git checkout test
# Make package improvements, update version numbers
git add .
git commit -m "Package improvements for v0.1.8"
git push origin test

# When ready for PyPI release:
# Create GitHub release from test branch
```

### Version Updates (Future Releases):

1. **Update version** in test branch:
   ```bash
   git checkout test
   # Edit pyproject.toml, setup.py, graft/__init__.py
   # Increment version: 0.1.7 → 0.1.8
   git commit -m "Bump version to 0.1.8"
   git push origin test
   ```

2. **Create release** with new tag targeting test branch

## Benefits of This Approach

✅ **Clean Separation**: 
- `main` = Research code (papers, experiments)
- `test` = Package code (PyPI distribution)

✅ **Zero Risk to Core**:
- Your main research code never changes
- No chance of breaking existing workflows

✅ **Proven Stability**:
- Test branch already validated on TestPyPI
- All tests pass, all imports work

✅ **Simple Workflow**:
- One command: Create release from test branch
- Everything else happens automatically

✅ **Easy Maintenance**:
- Package updates only touch test branch
- Research updates only touch main branch
- Clear responsibility separation

## Troubleshooting

### If OIDC Fails:
1. Check PyPI trusted publisher settings match exactly
2. Verify GitHub environment `pypi` exists
3. Ensure release was created from `test` branch

### If Build Fails:
1. Check GitHub Actions logs in repo's Actions tab
2. Verify all required files exist in test branch
3. Test locally: `python -m build`

### If Import Fails After Installation:
1. Check version published: `pip show graft-pytorch`
2. Verify dependencies: `pip install graft-pytorch[all]`
3. Test in clean virtual environment

## Ready to Deploy!

Your setup is perfect:
- ✅ Core code safe in main branch
- ✅ Package code tested and ready in test branch
- ✅ OIDC workflow configured
- ✅ All tests passing

Just configure PyPI OIDC and create the release from test branch!