@echo off
echo ===== Keyphrase Extractor App Fix Script =====

echo Installing @react-native-community/cli...
npm install @react-native-community/cli@latest --save-dev --legacy-peer-deps

echo Cleaning up node_modules...
rmdir /s /q node_modules

echo Cleaning up package-lock.json...
del package-lock.json

echo Cleaning up .expo folder if it exists...
if exist .expo rmdir /s /q .expo

echo Installing dependencies with legacy peer deps...
npm install --legacy-peer-deps

echo ===== Fix completed! =====
echo.
echo Please run 'npx expo start --web' to start the app
echo.
pause
