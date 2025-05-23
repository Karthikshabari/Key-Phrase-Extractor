@echo off
echo ===== Installing Expo CLI and Dependencies =====

echo Installing Expo CLI globally...
npm install -g expo-cli

echo Installing required local dependencies...
npm install expo --force
npm install @expo/cli --force

echo Cleaning up node_modules...
rmdir /s /q node_modules

echo Cleaning up package-lock.json...
del package-lock.json

echo Installing all dependencies...
npm install --force

echo ===== Installation completed! =====
echo.
echo Please try running 'npx expo start' again
echo.
pause
