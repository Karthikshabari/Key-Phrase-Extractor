@echo off
echo Installing vm-browserify polyfill package...
cd KeyphraseExtractorApp
npm install --save vm-browserify --force
echo VM polyfill package installed successfully!
echo.
echo Now try running the app again with:
echo npm run web
echo.
pause
