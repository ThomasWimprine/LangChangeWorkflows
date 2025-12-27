#!/usr/bin/env bash
# Deploy LangChainWorkflows to ~/bin/workflows/prp
# Usage: ./scripts/deploy.sh
#
# This script:
# 1. Removes the existing deployed copy
# 2. Copies the current repo state to ~/bin/workflows/prp
# 3. Creates/recreates the virtual environment
# 4. Installs dependencies
#
# Run after pulling changes or merging to main.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEPLOY_TARGET="${DEPLOY_TARGET:-$HOME/bin/workflows/prp}"

echo -e "${GREEN}=== LangChainWorkflows Deploy ===${NC}"
echo "Source: $REPO_ROOT"
echo "Target: $DEPLOY_TARGET"
echo ""

# Verify we're in the right repo
if [[ ! -f "$REPO_ROOT/workflows/prp-draft.py" ]]; then
    echo -e "${RED}Error: Not in LangChainWorkflows repository${NC}"
    exit 1
fi

# Check for uncommitted changes (warning only)
if [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; then
    echo -e "${YELLOW}Warning: Uncommitted changes in repository${NC}"
    echo "Deploying current working tree state..."
    echo ""
fi

# Show what version we're deploying
COMMIT_HASH=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo "unknown")
COMMIT_MSG=$(git -C "$REPO_ROOT" log -1 --format='%s' 2>/dev/null || echo "unknown")
echo "Deploying commit: $COMMIT_HASH - $COMMIT_MSG"
echo ""

# Step 1: Remove existing deployment
if [[ -d "$DEPLOY_TARGET" ]]; then
    echo -e "${YELLOW}Removing existing deployment...${NC}"
    rm -rf "$DEPLOY_TARGET"
fi

# Step 2: Create target directory
echo "Creating deployment directory..."
mkdir -p "$DEPLOY_TARGET"

# Step 3: Copy files (excluding .git, .venv, __pycache__, etc.)
echo "Copying files..."
rsync -a --exclude='.git' \
         --exclude='.venv' \
         --exclude='__pycache__' \
         --exclude='*.pyc' \
         --exclude='.pytest_cache' \
         --exclude='.mypy_cache' \
         --exclude='*.egg-info' \
         --exclude='.coverage' \
         --exclude='htmlcov' \
         --exclude='logs/*.log' \
         "$REPO_ROOT/" "$DEPLOY_TARGET/"

# Step 4: Create virtual environment
echo "Creating virtual environment..."
python3 -m venv "$DEPLOY_TARGET/.venv"

# Step 5: Install dependencies
echo "Installing dependencies..."
"$DEPLOY_TARGET/.venv/bin/pip" install --quiet --upgrade pip
if [[ -f "$DEPLOY_TARGET/pyproject.toml" ]]; then
    "$DEPLOY_TARGET/.venv/bin/pip" install --quiet -e "$DEPLOY_TARGET"
elif [[ -f "$DEPLOY_TARGET/requirements.txt" ]]; then
    "$DEPLOY_TARGET/.venv/bin/pip" install --quiet -r "$DEPLOY_TARGET/requirements.txt"
fi

# Step 6: Record deployment info
DEPLOY_INFO="$DEPLOY_TARGET/.deploy-info"
cat > "$DEPLOY_INFO" << EOF
{
  "deployed_at": "$(date -Iseconds)",
  "commit": "$COMMIT_HASH",
  "commit_message": "$COMMIT_MSG",
  "source": "$REPO_ROOT",
  "deployed_by": "$USER"
}
EOF

# Step 7: Update symlinks in ~/bin
echo "Updating symlinks in ~/bin..."
mkdir -p "$HOME/bin"

# Remove old symlinks (may point to repo)
rm -f "$HOME/bin/prp" "$HOME/bin/prp-draft" "$HOME/bin/prp-workflow"

# Create new symlinks pointing to deployed copy
ln -sf "$DEPLOY_TARGET/scripts/bin/prp" "$HOME/bin/prp"
ln -sf "$DEPLOY_TARGET/scripts/bin/prp-draft" "$HOME/bin/prp-draft"
ln -sf "$DEPLOY_TARGET/scripts/bin/prp-workflow" "$HOME/bin/prp-workflow"

# Step 8: Verify deployment
echo ""
echo -e "${GREEN}=== Deployment Complete ===${NC}"
echo "Deployed to: $DEPLOY_TARGET"
echo "Version: $COMMIT_HASH"
echo ""

# Verify symlinks
echo "Symlinks:"
ls -la "$HOME/bin/prp" "$HOME/bin/prp-draft" "$HOME/bin/prp-workflow" 2>/dev/null | sed 's/^/  /'
echo ""

# Show usage
echo "Usage:"
echo "  prp-draft \"Feature description\""
echo "  prp draft \"Feature description\""
echo "  prp workflow"
echo ""

# Verify prp-draft works
if "$DEPLOY_TARGET/.venv/bin/python" -c "import sys; sys.path.insert(0, '$DEPLOY_TARGET'); from workflows import prp_draft" 2>/dev/null; then
    echo -e "${GREEN}Verification: prp-draft module loads successfully${NC}"
else
    echo -e "${YELLOW}Warning: Could not verify prp-draft module${NC}"
fi
