@echo off
echo ===== Keyphrase Extractor App UI Update Script =====

echo Installing expo-linear-gradient if not already installed...
npm install expo-linear-gradient --legacy-peer-deps

echo Clearing React Native cache...
npx react-native start --reset-cache

echo ===== UI Update completed! =====
echo.
echo Please run 'npx expo start --web' to see the updated UI
echo.
pause
