#!/bin/bash
#
# DroidRun Termux Setup Script
#
# Run this on Termux to set up everything:
#   curl -sL https://raw.githubusercontent.com/.../termux-setup.sh | bash
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
pkg install -y python android-tools nodejs git

# Install Claude Code CLI
echo "🤖 Installing Claude Code CLI..."
npm install -g @anthropic-ai/claude-code

# Install DroidRun
echo "📱 Installing DroidRun..."
pip install droidrun

# Create convenience alias
echo "⚡ Creating 'auto' alias..."
echo 'alias auto="droidrun-auto"' >> ~/.bashrc

# Setup ADB for localhost
echo "🔧 Configuring ADB..."
echo 'export ADB_HOST=localhost' >> ~/.bashrc
echo 'export ADB_PORT=5037' >> ~/.bashrc

# Instructions
echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Sign into Claude Code: claude"
echo "   2. Enable ADB over TCP in developer options"
echo "   3. Run: source ~/.bashrc"
echo "   4. Test: auto \"open calculator\""
echo ""
echo "🎯 Usage examples:"
echo "   auto \"open chrome and search for weather\""
echo "   auto \"send whatsapp message to mom saying hello\""
echo "   auto \"open settings and turn on wifi\""
echo ""
