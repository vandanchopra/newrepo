#!/bin/bash
#
# DroidRun Termux Setup Script
#
# Run this on Termux to set up everything:
#   curl -sL https://raw.githubusercontent.com/droidrun/droidrun/main/termux-setup.sh | bash
#
# Or manually:
#   bash termux-setup.sh
#

set -e

echo "🚀 DroidRun Termux Setup"
echo "========================"
echo ""

# Update packages
echo "📦 Updating packages..."
pkg update -y

# Install required packages
echo "📦 Installing dependencies..."
pkg install -y python android-tools nodejs-lts git

# Reinstall to fix library linking issues
echo "🔧 Fixing library dependencies..."
pkg reinstall -y nodejs-lts icu android-tools 2>/dev/null || true

# Install Claude Code CLI
echo "🤖 Installing Claude Code CLI..."
npm install -g @anthropic-ai/claude-code

# Install DroidRun (without heavy scipy dependencies)
echo "📱 Installing DroidRun..."
pip install droidrun

# Create convenience alias
echo "⚡ Creating 'auto' alias..."
grep -q 'alias auto=' ~/.bashrc 2>/dev/null || echo 'alias auto="droidrun-auto"' >> ~/.bashrc

# Setup ADB for localhost
echo "🔧 Configuring ADB..."
grep -q 'ADB_HOST=' ~/.bashrc 2>/dev/null || echo 'export ADB_HOST=localhost' >> ~/.bashrc
grep -q 'ADB_PORT=' ~/.bashrc 2>/dev/null || echo 'export ADB_PORT=5037' >> ~/.bashrc

# Instructions
echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Sign into Claude Code: claude"
echo "   2. Enable ADB over TCP in developer options:"
echo "      Settings > Developer Options > ADB over network"
echo "   3. Run: source ~/.bashrc"
echo "   4. Test: auto \"open calculator\""
echo ""
echo "🎯 Usage examples:"
echo "   auto \"open chrome and search for weather\""
echo "   auto \"send whatsapp message to mom saying hello\""
echo "   auto \"open settings and turn on wifi\""
echo ""
echo "📝 Note: Phoenix tracing is optional. To enable:"
echo "   pip install droidrun[phoenix]"
echo ""
