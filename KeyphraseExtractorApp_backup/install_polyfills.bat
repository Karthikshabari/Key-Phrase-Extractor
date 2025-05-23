@echo off
echo Installing polyfill packages for web compatibility...
cd KeyphraseExtractorApp
npm install --save crypto-browserify stream-browserify buffer --force
echo Polyfill packages installed successfully!
echo.
echo Now try running the app again with:
echo npm run web
echo.
pause
