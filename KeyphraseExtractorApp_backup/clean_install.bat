@echo off
echo Cleaning and reinstalling dependencies for Keyphrase Extractor App...
cd KeyphraseExtractorApp

echo Removing node_modules folder...
if exist node_modules rmdir /s /q node_modules

echo Removing package-lock.json...
if exist package-lock.json del package-lock.json

echo Installing dependencies...
npm install --force

echo Dependencies installed successfully!
echo.
echo To run the app, use one of the following commands:
echo - For web: npm run web
echo - For Android: npm run android
echo - For iOS: npm run ios
echo.
pause
