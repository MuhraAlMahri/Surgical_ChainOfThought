#!/bin/bash

# ============================================================================
# GitHub Push Script for Surgical_COT Repository
# ============================================================================
# 
# This script will:
# 1. Initialize git repository (if not already done)
# 2. Add all files
# 3. Create initial commit
# 4. Set up remote repository
# 5. Push to GitHub
#
# Usage:
#   chmod +x push_to_github.sh
#   ./push_to_github.sh
#
# ============================================================================

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                                                                          ║${NC}"
echo -e "${BLUE}║              🚀 Surgical COT - GitHub Push Script                       ║${NC}"
echo -e "${BLUE}║                                                                          ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Navigate to repository root
cd /l/users/muhra.almahri/Surgical_COT

echo -e "${YELLOW}📁 Current directory: $(pwd)${NC}"
echo ""

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo -e "${RED}❌ Error: git is not installed${NC}"
    echo "Please install git first: https://git-scm.com/downloads"
    exit 1
fi

echo -e "${GREEN}✓ Git is installed${NC}"
echo ""

# Step 1: Initialize git repository (if not already)
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Step 1: Initialize Git Repository${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if [ -d .git ]; then
    echo -e "${YELLOW}⚠️  Git repository already exists${NC}"
    echo -e "${YELLOW}   Do you want to reinitialize? (y/N): ${NC}"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        rm -rf .git
        git init
        echo -e "${GREEN}✓ Repository reinitialized${NC}"
    else
        echo -e "${BLUE}➜ Keeping existing repository${NC}"
    fi
else
    git init
    echo -e "${GREEN}✓ Git repository initialized${NC}"
fi
echo ""

# Step 2: Configure git (if not already configured)
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Step 2: Configure Git User${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check if user name is configured
if ! git config user.name &> /dev/null; then
    echo -e "${YELLOW}Enter your name (for git commits): ${NC}"
    read -r git_name
    git config user.name "$git_name"
    echo -e "${GREEN}✓ Git user name set to: $git_name${NC}"
else
    echo -e "${GREEN}✓ Git user already configured: $(git config user.name)${NC}"
fi

# Check if user email is configured
if ! git config user.email &> /dev/null; then
    echo -e "${YELLOW}Enter your email (for git commits): ${NC}"
    read -r git_email
    git config user.email "$git_email"
    echo -e "${GREEN}✓ Git user email set to: $git_email${NC}"
else
    echo -e "${GREEN}✓ Git email already configured: $(git config user.email)${NC}"
fi
echo ""

# Step 3: Add files to git
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Step 3: Add Files to Git${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo -e "${YELLOW}Adding all files (respecting .gitignore)...${NC}"
git add .

# Show what will be committed
echo ""
echo -e "${BLUE}Files to be committed:${NC}"
git status --short | head -20
echo ""
total_files=$(git status --short | wc -l)
echo -e "${GREEN}✓ Total files staged: $total_files${NC}"
echo ""

# Step 4: Create commit
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Step 4: Create Initial Commit${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

COMMIT_MESSAGE="Initial commit: Complete Surgical COT experiments

- 4 complete experiments: Random, Qwen Ordering, CXRTrek Sequential, Curriculum Learning
- CXRTrek Sequential winner: 77.59% accuracy
- Verified results with job logs and prediction files
- Complete documentation: README, setup guide, contributing guide
- Production-ready code and evaluation scripts
- MIT License, open source ready"

echo -e "${YELLOW}Commit message:${NC}"
echo "$COMMIT_MESSAGE"
echo ""

git commit -m "$COMMIT_MESSAGE"
echo -e "${GREEN}✓ Initial commit created${NC}"
echo ""

# Step 5: Set up remote
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Step 5: Set Up Remote Repository${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

REPO_URL="https://github.com/MuhraAlMahri/Surgical_COT.git"

# Remove existing remote if it exists
if git remote | grep -q origin; then
    echo -e "${YELLOW}⚠️  Remote 'origin' already exists${NC}"
    git remote remove origin
    echo -e "${BLUE}➜ Removed old remote${NC}"
fi

git remote add origin "$REPO_URL"
echo -e "${GREEN}✓ Remote repository added: $REPO_URL${NC}"
echo ""

# Step 6: Create and switch to main branch
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Step 6: Set Main Branch${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

git branch -M main
echo -e "${GREEN}✓ Branch set to 'main'${NC}"
echo ""

# Step 7: Push to GitHub
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Step 7: Push to GitHub${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo -e "${YELLOW}⚠️  IMPORTANT: You need to authenticate with GitHub${NC}"
echo ""
echo "Options for authentication:"
echo "  1. Personal Access Token (Recommended)"
echo "     - Go to: https://github.com/settings/tokens"
echo "     - Create token with 'repo' permissions"
echo "     - Use token as password when prompted"
echo ""
echo "  2. SSH Key"
echo "     - Set up SSH key first"
echo "     - Change remote URL to: git@github.com:MuhraAlMahri/Surgical_COT.git"
echo ""
echo -e "${YELLOW}Press Enter when ready to push...${NC}"
read -r

echo ""
echo -e "${YELLOW}Pushing to GitHub...${NC}"
echo ""

if git push -u origin main; then
    echo ""
    echo -e "${GREEN}✓ Successfully pushed to GitHub!${NC}"
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}                      🎉 SUCCESS! 🎉${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${GREEN}Your repository is now live at:${NC}"
    echo -e "${BLUE}https://github.com/MuhraAlMahri/Surgical_COT${NC}"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "  1. Visit your repository on GitHub"
    echo "  2. Add topics: medical-ai, vision-language-models, vqa, curriculum-learning"
    echo "  3. Add repository description"
    echo "  4. Enable Issues and Discussions (optional)"
    echo "  5. Share with your advisor and colleagues!"
    echo ""
else
    echo ""
    echo -e "${RED}❌ Push failed${NC}"
    echo ""
    echo -e "${YELLOW}Common issues:${NC}"
    echo "  1. Repository doesn't exist on GitHub yet"
    echo "     → Create it at: https://github.com/new"
    echo "     → Name: Surgical_COT"
    echo "     → Don't initialize with README"
    echo ""
    echo "  2. Authentication failed"
    echo "     → Use Personal Access Token"
    echo "     → Generate at: https://github.com/settings/tokens"
    echo ""
    echo "  3. Permission denied"
    echo "     → Ensure you're MuhraAlMahri on GitHub"
    echo "     → Check repository permissions"
    echo ""
    exit 1
fi

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""









