import AsyncStorage from '@react-native-async-storage/async-storage';
import { Keyphrase } from '../services/api';
import * as FileSystem from 'expo-file-system';
import { Platform } from 'react-native';

// Keys for AsyncStorage
const HISTORY_KEY = '@KeyphraseExtractor:history';
const API_URL_KEY = '@KeyphraseExtractor:apiUrl';
const DEBUG_MODE_KEY = '@KeyphraseExtractor:debugMode';

// File paths for file-based storage
const HISTORY_DIRECTORY = FileSystem.documentDirectory + 'history/';
const HISTORY_FILE = HISTORY_DIRECTORY + 'history.json';

// Create history directory if it doesn't exist
const ensureDirectoryExists = async () => {
  if (Platform.OS !== 'web') {
    const dirInfo = await FileSystem.getInfoAsync(HISTORY_DIRECTORY);
    if (!dirInfo.exists) {
      await FileSystem.makeDirectoryAsync(HISTORY_DIRECTORY, { intermediates: true });
    }
  }
};

// Interface for a history item
export interface HistoryItem {
  id: string;
  text: string;
  keyphrases: Keyphrase[];
  timestamp: number;
}

/**
 * Save a history item to both AsyncStorage and file system
 * @param item The history item to save
 */
export const saveHistoryItem = async (item: HistoryItem): Promise<void> => {
  try {
    // Get existing history
    const history = await getHistory();

    // Add new item to the beginning of the array
    const updatedHistory = [item, ...history];

    // Limit history to 20 items
    const limitedHistory = updatedHistory.slice(0, 20);

    // Save to AsyncStorage for web compatibility
    await AsyncStorage.setItem(HISTORY_KEY, JSON.stringify(limitedHistory));

    // Save to file system for persistence on mobile
    if (Platform.OS !== 'web') {
      await ensureDirectoryExists();
      await FileSystem.writeAsStringAsync(
        HISTORY_FILE,
        JSON.stringify(limitedHistory, null, 2)
      );

      // Also save individual history item as separate file for easier access
      const itemFile = HISTORY_DIRECTORY + item.id + '.json';
      await FileSystem.writeAsStringAsync(
        itemFile,
        JSON.stringify(item, null, 2)
      );
    }
  } catch (error) {
    console.error('Error saving history item:', error);
  }
};

/**
 * Get all history items from storage
 * @returns An array of history items
 */
export const getHistory = async (): Promise<HistoryItem[]> => {
  try {
    // Try to get from file system first (on mobile)
    if (Platform.OS !== 'web') {
      await ensureDirectoryExists();
      const fileInfo = await FileSystem.getInfoAsync(HISTORY_FILE);

      if (fileInfo.exists) {
        const fileContent = await FileSystem.readAsStringAsync(HISTORY_FILE);
        return JSON.parse(fileContent);
      }
    }

    // Fall back to AsyncStorage (or for web platform)
    const historyJson = await AsyncStorage.getItem(HISTORY_KEY);
    return historyJson ? JSON.parse(historyJson) : [];
  } catch (error) {
    console.error('Error getting history:', error);
    return [];
  }
};

/**
 * Clear all history items from storage
 */
export const clearHistory = async (): Promise<void> => {
  try {
    // Clear from AsyncStorage
    await AsyncStorage.removeItem(HISTORY_KEY);

    // Clear from file system on mobile
    if (Platform.OS !== 'web') {
      const dirInfo = await FileSystem.getInfoAsync(HISTORY_DIRECTORY);
      if (dirInfo.exists) {
        await FileSystem.deleteAsync(HISTORY_DIRECTORY, { idempotent: true });
        // Recreate the empty directory
        await ensureDirectoryExists();
      }
    }
  } catch (error) {
    console.error('Error clearing history:', error);
  }
};

/**
 * Get a specific history item by ID
 * @param id The ID of the history item to get
 * @returns The history item or null if not found
 */
export const getHistoryItem = async (id: string): Promise<HistoryItem | null> => {
  try {
    // Try to get from file system first (on mobile)
    if (Platform.OS !== 'web') {
      const itemFile = HISTORY_DIRECTORY + id + '.json';
      const fileInfo = await FileSystem.getInfoAsync(itemFile);

      if (fileInfo.exists) {
        const fileContent = await FileSystem.readAsStringAsync(itemFile);
        return JSON.parse(fileContent);
      }
    }

    // Fall back to searching in the full history
    const history = await getHistory();
    return history.find(item => item.id === id) || null;
  } catch (error) {
    console.error(`Error getting history item ${id}:`, error);
    return null;
  }
};

/**
 * Export history to a downloadable file (web only)
 * @returns URL to the exported file or null if not on web
 */
export const exportHistory = async (): Promise<string | null> => {
  if (Platform.OS !== 'web') {
    return null; // Not supported on mobile yet
  }

  try {
    const history = await getHistory();
    const historyBlob = new Blob([JSON.stringify(history, null, 2)], { type: 'application/json' });
    return URL.createObjectURL(historyBlob);
  } catch (error) {
    console.error('Error exporting history:', error);
    return null;
  }
};

/**
 * Save the API URL to AsyncStorage
 * @param url The API URL to save
 */
export const saveApiUrl = async (url: string): Promise<void> => {
  try {
    await AsyncStorage.setItem(API_URL_KEY, url);
  } catch (error) {
    console.error('Error saving API URL:', error);
  }
};

/**
 * Get the API URL from AsyncStorage
 * @returns The saved API URL or null if not found
 */
export const getApiUrl = async (): Promise<string | null> => {
  try {
    return await AsyncStorage.getItem(API_URL_KEY);
  } catch (error) {
    console.error('Error getting API URL:', error);
    return null;
  }
};

/**
 * Save the debug mode setting to AsyncStorage
 * @param enabled Whether debug mode is enabled
 */
export const saveDebugMode = async (enabled: boolean): Promise<void> => {
  try {
    await AsyncStorage.setItem(DEBUG_MODE_KEY, JSON.stringify(enabled));
  } catch (error) {
    console.error('Error saving debug mode:', error);
  }
};

/**
 * Get the debug mode setting from AsyncStorage
 * @returns Whether debug mode is enabled
 */
export const getDebugMode = async (): Promise<boolean> => {
  try {
    const value = await AsyncStorage.getItem(DEBUG_MODE_KEY);
    return value ? JSON.parse(value) : false;
  } catch (error) {
    console.error('Error getting debug mode:', error);
    return false;
  }
};
