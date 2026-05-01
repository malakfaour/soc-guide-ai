# Project Rename Guide - SOC-GUIDE-AI → SOC Intelligence

## Summary of Changes

✅ **COMPLETED - Local Files Updated:**
- README.md - Header and description updated
- STATUS.md - GitHub URL updated to soc-intelligence
- TABNET_EXPLAINABILITY_SUMMARY.md - Project name updated

---

## Step 1: Rename GitHub Repository

GitHub repository renaming must be done through the GitHub web interface:

### Instructions:
1. Go to https://github.com/malakfaour/soc-guide-ai
2. Click **Settings** (gear icon in top right)
3. In the "Repository name" field, change:
   - **FROM**: `soc-guide-ai`
   - **TO**: `soc-intelligence`
4. Click **Rename**

### GitHub will:
- ✓ Update the repository name
- ✓ Automatically redirect old URLs to the new URL
- ✓ Update your local git remote (see "Fix Local Git" below)

---

## Step 2: Update Local Git Remote

After renaming on GitHub, update your local repository:

```bash
# Option A: Update existing remote
git remote set-url origin https://github.com/malakfaour/soc-intelligence.git

# Option B: Verify it worked
git remote -v
# Should show:
# origin  https://github.com/malakfaour/soc-intelligence.git (fetch)
# origin  https://github.com/malakfaour/soc-intelligence.git (push)
```

---

## Step 3: Rename Local Folder (OPTIONAL)

If you want to rename the local folder from `soc-guide-ai` to `soc-intelligence`:

### Option A: Using File Explorer (Recommended)
1. Close VS Code
2. Right-click `C:\Users\malty\Projects\soc-guide-ai`
3. Select **Rename**
4. Change to `soc-intelligence`
5. Re-open the project in VS Code

### Option B: Using PowerShell
```powershell
cd C:\Users\malty\Projects
Rename-Item -Path "soc-guide-ai" -NewName "soc-intelligence"
```

### Option C: Keep folder as-is
You can leave the folder name as `soc-guide-ai` - the repository name on GitHub is what matters most.

---

## Step 4: Update Local Workspace Reference (if folder renamed)

If you renamed the folder, you may need to:
1. Close the current workspace in VS Code
2. Open the new folder: `C:\Users\malty\Projects\soc-intelligence`

---

## Files Already Updated

### README.md
```
# SOC Intelligence

A machine learning platform for SOC incident classification and explainability analysis.
```

### STATUS.md
```
- **Pushed to**: https://github.com/malakfaour/soc-intelligence/tree/main
```

### TABNET_EXPLAINABILITY_SUMMARY.md
```
A complete, production-ready explainability module for PyTorch TabNet models 
in the SOC Intelligence project.
```

---

## Verification Checklist

After completing all steps, verify:

- [ ] GitHub repository renamed to `soc-intelligence`
- [ ] Local git remote updated: `git remote -v` shows correct URL
- [ ] Local folder renamed (if desired)
- [ ] VS Code reopened in new folder location (if folder renamed)
- [ ] Can successfully: `git push origin main`

---

## What Happens on GitHub After Rename

✓ Old URL redirects to new URL automatically  
✓ All issues, PRs, and discussions are preserved  
✓ All commit history is preserved  
✓ All collaborators retain access  

---

## Verify Everything Works

Once completed, run:

```bash
# Check git remote
git remote -v

# Check current directory
pwd

# List project files
ls -la

# Verify you can commit
git status
```

---

## Quick Reference

| Item | Old | New |
|------|-----|-----|
| GitHub Repo | soc-guide-ai | soc-intelligence |
| Local Folder | soc-guide-ai | soc-intelligence (optional) |
| Project Name | SOC GUIDE AI | SOC Intelligence |
| Documentation | ✓ Updated | ✓ Updated |

---

## Need Help?

If the local git remote needs to be reset after GitHub rename:

```bash
# Remove old remote
git remote remove origin

# Add new remote
git remote add origin https://github.com/malakfaour/soc-intelligence.git

# Verify
git remote -v
```

---

**Next Steps:**
1. Navigate to https://github.com/malakfaour/soc-guide-ai
2. Go to Settings and rename the repository
3. Run `git remote set-url origin https://github.com/malakfaour/soc-intelligence.git` locally
4. Done! 🎉
