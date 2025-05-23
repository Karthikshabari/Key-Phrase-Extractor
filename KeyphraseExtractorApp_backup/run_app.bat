@echo off
echo Starting Keyphrase Extractor App...
cd KeyphraseExtractorApp

echo Select platform:
echo 1. Web
echo 2. Android
echo 3. iOS
echo.

set /p platform="Enter your choice (1-3): "

if "%platform%"=="1" (
    echo Starting web app...
    npm run web
) else if "%platform%"=="2" (
    echo Starting Android app...
    npm run android
) else if "%platform%"=="3" (
    echo Starting iOS app...
    npm run ios
) else (
    echo Invalid choice. Starting web app by default...
    npm run web
)
