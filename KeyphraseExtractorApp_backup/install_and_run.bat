@echo off
echo ===== Installing Expo CLI Locally and Running App =====

echo Installing @expo/cli locally...
npm install --save-dev @expo/cli

echo Installing expo package...
npm install --save expo

echo Adding @expo/cli to PATH temporarily...
set PATH=%PATH%;%CD%\node_modules\.bin

echo Running the app...
npm run web

echo If the above command fails, try:
echo npm run web-alt

echo ===== End of script =====
pause
