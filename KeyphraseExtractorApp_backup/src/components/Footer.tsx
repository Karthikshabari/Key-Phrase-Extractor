import React from 'react';
import { View, StyleSheet } from 'react-native';
import { Text, useTheme } from 'react-native-paper';

const Footer = () => {
  const theme = useTheme();
  
  return (
    <View style={styles.container}>
      <Text style={[styles.text, { color: theme.colors.onSurfaceVariant }]}>
        Designed and developed by Karthik Shabari
      </Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    padding: 16,
    alignItems: 'center',
    justifyContent: 'center',
    borderTopWidth: 1,
    borderTopColor: 'rgba(0,0,0,0.05)',
  },
  text: {
    fontSize: 12,
    opacity: 0.8,
    letterSpacing: 0.2,
  }
});

export default Footer;
