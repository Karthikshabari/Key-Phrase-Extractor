@echo off
echo ===== Keyphrase Extractor App Dependencies Update Script =====

echo Cleaning up node_modules...
rmdir /s /q node_modules

echo Cleaning up package-lock.json...
del package-lock.json

echo Cleaning up .expo folder if it exists...
if exist .expo rmdir /s /q .expo

echo Installing dependencies with force flag...
npm install --force

echo ===== Dependencies Update completed! =====
echo.
echo Please run 'npx expo start --web' to start the app
echo.
pause
