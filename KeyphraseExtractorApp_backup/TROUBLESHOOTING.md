# Troubleshooting Guide

This document provides solutions to common issues you might encounter when running the Keyphrase Extractor App.

## Installation Issues

### Error: ERESOLVE unable to resolve dependency tree

**Error Message:**
```
npm ERR! code ERESOLVE
npm ERR! ERESOLVE unable to resolve dependency tree
npm ERR!
npm ERR! While resolving: keyphraseextractorapp@1.0.0
npm ERR! Found: expo@49.0.23
npm ERR! node_modules/expo
npm ERR!   expo@"~49.0.15" from the root project
npm ERR!
npm ERR! Could not resolve dependency:
npm ERR! peer expo@"^48.0.17" from @expo/webpack-config@18.1.4
```

**Solution:**
1. Use the `--force` flag when installing dependencies:
   ```
   npm install --force
   ```
2. Or use the provided installation script:
   ```
   clean_install.bat
   ```

## Web Build Issues

### Error: Module not found: Can't resolve 'crypto'

**Error Message:**
```
ERROR in ./node_modules/expo-modules-core/build/uuid/uuid.web.js:9:8
Module not found: Can't resolve 'crypto'
```

**Solution:**
1. Install the crypto-browserify polyfill:
   ```
   npm install --save crypto-browserify --force
   ```
2. Or use the provided script:
   ```
   install_polyfills.bat
   ```

### Error: Module not found: Can't resolve 'vm'

**Error Message:**
```
WARNING in ./node_modules/asn1.js/lib/asn1/api.js:21
Module not found: Can't resolve 'vm'
```

**Solution:**
1. Install the vm-browserify polyfill:
   ```
   npm install --save vm-browserify --force
   ```
2. Or use the provided script:
   ```
   install_vm_polyfill.bat
   ```

## API Connection Issues

### Error: No response received from server

**Possible Causes:**
1. The API server is not running
2. The ngrok tunnel has expired or changed
3. Network connectivity issues

**Solutions:**
1. Verify your API server is running
2. Check if the ngrok URL has changed and update it in the Settings screen
3. Enable Debug Mode in the Settings screen to see detailed error information

### Error: Server responded with status 404/500

**Possible Causes:**
1. Incorrect API endpoint
2. Server-side error in processing the request

**Solutions:**
1. Verify the API endpoint is correct (`/extract_keyphrases`)
2. Check the server logs for errors
3. Enable Debug Mode to see the full error response

## API Response Format Issues

### Error: Cannot read properties of undefined

This often happens when the API response format doesn't match what the app expects.

**Expected API Response Format:**
```json
{
  "keyphrases": [
    ["keyphrase1", 0.85],
    ["keyphrase2", 0.72]
  ]
}
```

**Solutions:**
1. Ensure your API returns the exact format shown above
2. Check the browser console for the actual response format
3. If needed, modify the `api.ts` file to match your API's response format

## Other Issues

If you encounter other issues:

1. Enable Debug Mode in the Settings screen
2. Check the browser console (F12 in most browsers) for error messages
3. Try running the app on a different platform (web/Android/iOS)
4. Clear your browser cache or reinstall the app on your device
