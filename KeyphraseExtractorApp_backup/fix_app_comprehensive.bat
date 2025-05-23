@echo off
echo ===== Keyphrase Extractor App Comprehensive Fix Script =====

echo This script will:
echo 1. Update the package.json to use Expo 49 (more stable version)
echo 2. Clean up node_modules and package-lock.json
echo 3. Install all dependencies with the correct versions
echo 4. Clear the React Native cache
echo.
echo Press any key to continue or Ctrl+C to cancel...
pause > nul

echo Step 1: Updating package.json...
call update_package_json.bat

echo Step 2: Cleaning up...
rmdir /s /q node_modules
del package-lock.json
if exist .expo rmdir /s /q .expo

echo Step 3: Installing dependencies...
npm install --force

echo Step 4: Clearing React Native cache...
npx react-native start --reset-cache

echo ===== Comprehensive Fix completed! =====
echo.
echo Please run 'npx expo start --web' to start the app
echo.
pause
