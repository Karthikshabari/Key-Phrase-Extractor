@echo off
echo ===== Running Keyphrase Extractor App (Alternative Method) =====

echo Installing required packages...
npm install --save-dev @expo/cli
npm install --save expo

echo Starting the app using node_modules path...
node node_modules\expo\bin\cli.js start

echo If the above command fails, try:
echo node node_modules\@expo\cli\build\src\start.js

echo ===== End of script =====
pause
