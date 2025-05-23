@echo off
echo ===== COMPLETE FIX FOR KEYPHRASE EXTRACTOR APP =====
echo This script will:
echo 1. Install Expo CLI globally
echo 2. Downgrade to Expo 49 (stable version)
echo 3. Fix all dependencies
echo 4. Clear caches
echo.
echo Press any key to continue or Ctrl+C to cancel...
pause > nul

echo Step 1: Installing Expo CLI globally...
call npm install -g expo-cli
call npm install -g @expo/cli

echo Step 2: Cleaning up project...
if exist node_modules rmdir /s /q node_modules
if exist package-lock.json del package-lock.json
if exist yarn.lock del yarn.lock
if exist .expo rmdir /s /q .expo

echo Step 3: Updating package.json to Expo 49...
echo {> package.json.new
echo   "name": "keyphraseextractorapp",>> package.json.new
echo   "version": "1.0.0",>> package.json.new
echo   "main": "index.ts",>> package.json.new
echo   "scripts": {>> package.json.new
echo     "start": "npx expo start",>> package.json.new
echo     "android": "npx expo run:android",>> package.json.new
echo     "ios": "npx expo run:ios",>> package.json.new
echo     "web": "npx expo start --web",>> package.json.new
echo     "start-alt": "node node_modules/expo/bin/cli.js start",>> package.json.new
echo     "web-alt": "node node_modules/expo/bin/cli.js start --web">> package.json.new
echo   },>> package.json.new
echo   "dependencies": {>> package.json.new
echo     "@expo/webpack-config": "^18.0.1",>> package.json.new
echo     "@react-native-async-storage/async-storage": "1.18.2",>> package.json.new
echo     "@react-navigation/native": "^6.1.9",>> package.json.new
echo     "@react-navigation/native-stack": "^6.9.17",>> package.json.new
echo     "@react-navigation/stack": "^6.3.20",>> package.json.new
echo     "axios": "^1.6.2",>> package.json.new
echo     "buffer": "^6.0.3",>> package.json.new
echo     "crypto-browserify": "^3.12.1",>> package.json.new
echo     "expo": "~49.0.15",>> package.json.new
echo     "expo-constants": "~14.4.2",>> package.json.new
echo     "expo-file-system": "~15.4.5",>> package.json.new
echo     "expo-haptics": "~12.4.0",>> package.json.new
echo     "expo-linear-gradient": "~12.3.0",>> package.json.new
echo     "expo-linking": "~5.0.2",>> package.json.new
echo     "expo-screen-capture": "~5.3.0",>> package.json.new
echo     "expo-status-bar": "~1.6.0",>> package.json.new
echo     "expo-task-manager": "~11.3.0",>> package.json.new
echo     "react": "18.2.0",>> package.json.new
echo     "react-dom": "18.2.0",>> package.json.new
echo     "react-native": "0.72.6",>> package.json.new
echo     "react-native-gesture-handler": "~2.12.0",>> package.json.new
echo     "react-native-paper": "^5.11.1",>> package.json.new
echo     "react-native-reanimated": "~3.3.0",>> package.json.new
echo     "react-native-safe-area-context": "4.6.3",>> package.json.new
echo     "react-native-screens": "~3.22.0",>> package.json.new
echo     "react-native-vector-icons": "^10.0.0",>> package.json.new
echo     "react-native-web": "~0.19.6",>> package.json.new
echo     "stream-browserify": "^3.0.0",>> package.json.new
echo     "vm-browserify": "^1.1.2">> package.json.new
echo   },>> package.json.new
echo   "devDependencies": {>> package.json.new
echo     "@babel/core": "^7.20.0",>> package.json.new
echo     "@expo/cli": "^0.10.16",>> package.json.new
echo     "@react-native-community/cli": "^11.3.9",>> package.json.new
echo     "@types/react": "~18.2.14",>> package.json.new
echo     "typescript": "^5.1.3">> package.json.new
echo   },>> package.json.new
echo   "private": true,>> package.json.new
echo   "browser": {>> package.json.new
echo     "crypto": "crypto-browserify",>> package.json.new
echo     "stream": "stream-browserify",>> package.json.new
echo     "buffer": "buffer",>> package.json.new
echo     "vm": "vm-browserify">> package.json.new
echo   }>> package.json.new
echo }>> package.json.new

echo Backing up original package.json...
copy package.json package.json.bak
echo Replacing with new package.json...
move /y package.json.new package.json

echo Step 4: Installing dependencies...
call npm install --force

echo Step 5: Installing specific versions of critical packages...
call npm install expo@~49.0.15 --force
call npm install @expo/cli@latest --save-dev --force
call npm install @react-native-community/cli@latest --save-dev --force

echo Step 6: Clearing npm cache...
call npm cache clean --force

echo Step 7: Setting up PATH temporarily...
set PATH=%PATH%;%CD%\node_modules\.bin

echo ===== FIX COMPLETED! =====
echo.
echo To run your app, use one of these commands:
echo 1. npm run web
echo 2. npx expo start --web
echo 3. node node_modules/expo/bin/cli.js start --web
echo.
pause
