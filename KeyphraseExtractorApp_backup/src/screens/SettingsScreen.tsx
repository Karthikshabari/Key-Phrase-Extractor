import React, { useState, useEffect } from 'react';
import { View, StyleSheet, ScrollView } from 'react-native';
import { Text, TextInput, Button, Divider, Switch } from 'react-native-paper';
import { NativeStackScreenProps } from '@react-navigation/native-stack/lib/typescript/src/types';
import { RootStackParamList } from '../../App';
import { saveApiUrl, getApiUrl, saveDebugMode, getDebugMode } from '../utils/storage';
import { updateApiUrl } from '../services/api';

type SettingsScreenProps = NativeStackScreenProps<RootStackParamList, 'Settings'>;

/**
 * Settings screen for configuring the app
 */
const SettingsScreen: React.FC<SettingsScreenProps> = ({ navigation }) => {
  const [apiUrl, setApiUrl] = useState('');
  const [darkMode, setDarkMode] = useState(false);
  const [debugMode, setDebugMode] = useState(false);
  const [saveEnabled, setSaveEnabled] = useState(false);

  // Load saved settings when the screen mounts
  useEffect(() => {
    const loadSettings = async () => {
      // Load API URL
      const savedUrl = await getApiUrl();
      if (savedUrl) {
        setApiUrl(savedUrl);
      }

      // Load debug mode
      const isDebugMode = await getDebugMode();
      setDebugMode(isDebugMode);
    };

    loadSettings();
  }, []);

  // Enable save button when API URL is valid
  useEffect(() => {
    const isValidUrl = apiUrl.trim().startsWith('http');
    setSaveEnabled(isValidUrl);
  }, [apiUrl]);

  // Save settings
  const handleSaveSettings = async () => {
    if (saveEnabled) {
      // Save API URL
      await saveApiUrl(apiUrl);
      updateApiUrl(apiUrl);

      // Save debug mode
      await saveDebugMode(debugMode);

      navigation.goBack();
    }
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>API Settings</Text>
        <Text style={styles.description}>
          Enter the URL of your keyphrase extraction API. This should be the ngrok URL or other publicly accessible endpoint.
        </Text>

        <TextInput
          mode="outlined"
          label="API URL"
          value={apiUrl}
          onChangeText={setApiUrl}
          placeholder="https://your-api-url.ngrok.io"
          autoCapitalize="none"
          keyboardType="url"
          style={styles.input}
        />

        <Text style={styles.helperText}>
          Example: https://1234-abcd-5678.ngrok.io
        </Text>
      </View>

      <Divider />

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>App Settings</Text>

        <View style={styles.settingRow}>
          <Text>Dark Mode</Text>
          <Switch
            value={darkMode}
            onValueChange={setDarkMode}
          />
        </View>
        <Text style={styles.helperText}>
          Note: Dark mode is not fully implemented yet.
        </Text>

        <View style={styles.settingRow}>
          <Text>Debug Mode</Text>
          <Switch
            value={debugMode}
            onValueChange={setDebugMode}
          />
        </View>
        <Text style={styles.helperText}>
          Enable to see detailed logs and error messages.
        </Text>
      </View>

      <View style={styles.buttonContainer}>
        <Button
          mode="contained"
          onPress={handleSaveSettings}
          disabled={!saveEnabled}
          style={styles.button}
        >
          Save Settings
        </Button>

        <Button
          mode="outlined"
          onPress={() => navigation.goBack()}
          style={styles.button}
        >
          Cancel
        </Button>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  section: {
    padding: 16,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  description: {
    fontSize: 14,
    marginBottom: 16,
    opacity: 0.7,
  },
  input: {
    marginBottom: 8,
  },
  helperText: {
    fontSize: 12,
    opacity: 0.5,
    marginBottom: 8,
  },
  settingRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
  },
  buttonContainer: {
    padding: 16,
    marginTop: 16,
  },
  button: {
    marginBottom: 12,
  },
});

export default SettingsScreen;
