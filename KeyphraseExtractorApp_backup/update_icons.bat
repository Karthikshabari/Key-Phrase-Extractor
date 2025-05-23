@echo off
echo ===== Keyphrase Extractor App Icon Update Script =====

echo Clearing React Native cache...
npx react-native start --reset-cache

echo ===== Icon Update completed! =====
echo.
echo Please run 'npx expo start --web' to see the updated icons
echo.
pause
