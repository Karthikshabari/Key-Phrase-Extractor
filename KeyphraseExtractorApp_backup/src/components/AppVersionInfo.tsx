import React, { useState, useEffect } from 'react';
import { View, StyleSheet, Linking, Platform } from 'react-native';
import { Text, Button, useTheme, Surface } from 'react-native-paper';

// Conditionally import expo-constants
let Constants: any = null;

// Only import these modules on native platforms
if (Platform.OS !== 'web') {
  try {
    Constants = require('expo-constants');
  } catch (error) {
    console.warn('expo-constants not available');
  }
} else {
  // Mock for web platform
  Constants = {
    expoConfig: { version: '1.0.0' }
  };
}

interface AppVersionInfoProps {
  showUpdateButton?: boolean;
}

const AppVersionInfo: React.FC<AppVersionInfoProps> = ({
  showUpdateButton = false
}) => {
  const theme = useTheme();

  // Get app version from app.json
  const appVersion = Constants.expoConfig?.version || '1.0.0';

  return (
    <View style={styles.container}>
      <Text style={[styles.versionText, { color: theme.colors.onSurfaceVariant }]}>
        Version {appVersion}
      </Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    padding: 16,
  },
  versionText: {
    fontSize: 12,
    opacity: 0.7,
  },
});

export default AppVersionInfo;
