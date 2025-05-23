@echo off
echo ===== Checking Node.js Environment =====

echo Node.js version:
node --version

echo npm version:
npm --version

echo Checking global npm packages:
npm list -g --depth=0

echo Checking local packages:
npm list --depth=0

echo ===== Environment Check Completed =====
pause
