#!/bin/bash

# Setup script to connect local repository to GitHub
# Replace 'ctekinay' with your actual GitHub username if different

GITHUB_USER="ctekinay"
REPO_NAME="AInsteinDSO"

echo "Setting up GitHub remote for $GITHUB_USER/$REPO_NAME..."

# Add the remote origin
git remote add origin "https://github.com/$GITHUB_USER/$REPO_NAME.git"

# Verify the remote was added
echo "Remote configuration:"
git remote -v

# Push all branches to GitHub
echo "Pushing to GitHub..."
git push -u origin feat/ea-assistant-implementation

echo "Done! Your repository should now be visible at:"
echo "https://github.com/$GITHUB_USER/$REPO_NAME"