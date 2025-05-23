@echo off
echo ===== Direct Run Method for Keyphrase Extractor App =====

echo Starting Metro bundler directly...
node node_modules\@react-native-community\cli\build\bin.js start --reset-cache

echo If the above command fails, try:
echo node node_modules\react-native\cli.js start --reset-cache

echo ===== End of script =====
pause
