# PyPI Release Guide

## Prerequisites Checklist

Before creating a release, ensure:

- [ ] OIDC trusted publishing configured on PyPI.org
- [ ] GitHub environment `pypi` created with protection rules
- [ ] Test branch merged to `main` branch
- [ ] All tests passing on main branch
- [ ] Package version is correct (currently v0.1.7)

## Creating a GitHub Release

### Option 1: Via GitHub Web Interface (Recommended)

1. **Go to your GitHub repository**: https://github.com/ashishjv1/GRAFT
2. **Click "Releases"** in the right sidebar
3. **Click "Create a new release"**
4. **Fill in the release form**:

   **Tag version**: `v0.1.7`
   **Release title**: `GRAFT v0.1.7 - Production Release`
   **Description**:
   ```markdown
   # GRAFT v0.1.7 - Production Release
   
   ## 🚀 First stable release of GRAFT for PyPI
   
   GRAFT (Gradient-Aware Fast MaxVol Technique) provides smart sampling for efficient deep learning training.
   
   ### 📦 Installation
   ```bash
   pip install graft-pytorch
   ```
   
   ### ✨ Key Features
   - Smart sample selection using gradient-based importance scoring
   - Multi-architecture support (ResNet, ResNeXT, EfficientNet, BERT)
   - 30-50% faster training time with maintained accuracy
   - Built-in experiment tracking and carbon footprint monitoring
   
   ### 🔧 What's Included
   - Complete package restructure for pip distribution
   - Command-line interface (`graft-train`)
   - Python API with ModelTrainer and TrainingConfig
   - Comprehensive test suite (Python 3.8-3.11 support)
   - OIDC-enabled secure publishing pipeline
   - Professional documentation
   
   ### 📚 Documentation
   - [PyPI Package](https://pypi.org/project/graft-pytorch/)
   - [GitHub Repository](https://github.com/ashishjv1/GRAFT)
   - [Research Paper](https://arxiv.org/abs/2508.13653)
   
   ### 🐛 Bug Reports
   Please report issues at: https://github.com/ashishjv1/GRAFT/issues
   ```

5. **Target branch**: Select `main`
6. **Click "Publish release"**

### Option 2: Via Command Line

```bash
# Using GitHub CLI (if installed)
gh release create v0.1.7 \
  --title "GRAFT v0.1.7 - Production Release" \
  --notes-file RELEASE_NOTES.md \
  --target main
```

## What Happens After Release

1. **GitHub Actions Trigger**: The `publish.yml` workflow will trigger automatically
2. **OIDC Authentication**: GitHub will authenticate with PyPI using OIDC
3. **Package Build**: The workflow builds the package using `python -m build`
4. **PyPI Upload**: Package uploads to https://pypi.org/project/graft-pytorch/
5. **Confirmation**: Check PyPI for successful publication

## Monitoring the Release

1. **GitHub Actions**: Go to Actions tab to monitor the workflow
2. **PyPI**: Check https://pypi.org/project/graft-pytorch/ for the new version
3. **Installation Test**: `pip install graft-pytorch==0.1.7`

## If Something Goes Wrong

### Common Issues:

1. **OIDC not configured**: Ensure PyPI trusted publishing is set up
2. **Environment protection**: Check if `pypi` environment needs approval
3. **Version conflict**: Ensure version 0.1.7 doesn't already exist on PyPI
4. **Build errors**: Check GitHub Actions logs for detailed errors

### Troubleshooting:

- Check OIDC_SETUP.md for detailed configuration help
- Review GitHub Actions logs in the Actions tab
- Verify PyPI project settings match exactly
- Ensure all required files are in the main branch

## Post-Release Steps

1. **Test Installation**: 
   ```bash
   pip install graft-pytorch
   python -c "import graft; print(f'GRAFT v{graft.__version__} installed successfully!')"
   ```

2. **Update Documentation**: Update any external docs with new PyPI links

3. **Announce**: Share the release with your community/colleagues

4. **Monitor**: Watch for user feedback and issues

## Version Management

For future releases:
1. Update version in `pyproject.toml`, `setup.py`, and `graft/__init__.py`
2. Commit changes to test branch first
3. Test on TestPyPI
4. Merge to main when ready
5. Create new GitHub release with incremented version tag