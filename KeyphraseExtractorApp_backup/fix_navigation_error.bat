@echo off
echo ===== Keyphrase Extractor App Navigation Error Fix Script =====

echo Installing @react-native-community/cli...
npm install @react-native-community/cli@latest --save-dev --legacy-peer-deps

echo Clearing React Native cache...
npx react-native start --reset-cache

echo ===== Navigation Error Fix completed! =====
echo.
echo Please run 'npx expo start --web' to see the fixed app
echo.
pause
