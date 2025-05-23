import { Platform } from 'react-native';

// Conditionally import expo-haptics
let Haptics: any = null;

// Only import on native platforms
if (Platform.OS !== 'web') {
  try {
    Haptics = require('expo-haptics');
  } catch (error) {
    console.warn('expo-haptics not available');
  }
}

// Create mock implementations for web
const mockHaptics = {
  impactAsync: () => Promise.resolve(),
  notificationAsync: () => Promise.resolve(),
  selectionAsync: () => Promise.resolve(),
  ImpactFeedbackStyle: {
    Light: 'light',
    Medium: 'medium',
    Heavy: 'heavy'
  },
  NotificationFeedbackType: {
    Success: 'success',
    Warning: 'warning',
    Error: 'error'
  }
};

/**
 * Utility functions for haptic feedback
 */

// Use Haptics if available, otherwise use mock
const haptics = Haptics || mockHaptics;

// Light impact for regular interactions
export const lightImpact = () => {
  if (Platform.OS !== 'web') {
    haptics.impactAsync(haptics.ImpactFeedbackStyle.Light);
  }
};

// Medium impact for more significant interactions
export const mediumImpact = () => {
  if (Platform.OS !== 'web') {
    haptics.impactAsync(haptics.ImpactFeedbackStyle.Medium);
  }
};

// Heavy impact for major interactions
export const heavyImpact = () => {
  if (Platform.OS !== 'web') {
    haptics.impactAsync(haptics.ImpactFeedbackStyle.Heavy);
  }
};

// Success notification
export const successNotification = () => {
  if (Platform.OS !== 'web') {
    haptics.notificationAsync(haptics.NotificationFeedbackType.Success);
  }
};

// Warning notification
export const warningNotification = () => {
  if (Platform.OS !== 'web') {
    haptics.notificationAsync(haptics.NotificationFeedbackType.Warning);
  }
};

// Error notification
export const errorNotification = () => {
  if (Platform.OS !== 'web') {
    haptics.notificationAsync(haptics.NotificationFeedbackType.Error);
  }
};

// Selection feedback
export const selectionFeedback = () => {
  if (Platform.OS !== 'web') {
    haptics.selectionAsync();
  }
};
