import React, { useState } from 'react';
import { Animated, StyleSheet, Pressable, ViewStyle, TextStyle, View } from 'react-native';
import { Text, useTheme } from 'react-native-paper';
import { mediumImpact } from '../utils/haptics';

interface AnimatedButtonProps {
  onPress: () => void;
  title: string;
  icon?: React.ReactNode;
  style?: ViewStyle;
  textStyle?: TextStyle;
  disabled?: boolean;
  mode?: 'contained' | 'outlined' | 'text';
  color?: string;
  loading?: boolean;
}

const AnimatedButton: React.FC<AnimatedButtonProps> = ({
  onPress,
  title,
  icon,
  style,
  textStyle,
  disabled = false,
  mode = 'contained',
  color,
  loading = false,
}) => {
  const theme = useTheme();
  const [scaleAnim] = useState(new Animated.Value(1));

  // Get background color based on mode and provided color
  const getBackgroundColor = () => {
    if (disabled) return theme.colors.surfaceDisabled;

    if (mode === 'contained') {
      return color || theme.colors.primary;
    } else if (mode === 'outlined') {
      return 'transparent';
    } else {
      return 'transparent';
    }
  };

  // Get text color based on mode and provided color
  const getTextColor = () => {
    if (disabled) return theme.colors.onSurfaceDisabled;

    if (mode === 'contained') {
      return theme.colors.onPrimary;
    } else {
      return color || theme.colors.primary;
    }
  };

  // Get border style based on mode
  const getBorderStyle = () => {
    if (mode === 'outlined') {
      return {
        borderWidth: 1.5,
        borderColor: disabled ? theme.colors.surfaceDisabled : (color || theme.colors.primary),
      };
    }
    return {};
  };

  // Handle press animation
  const handlePressIn = () => {
    if (disabled || loading) return;

    Animated.spring(scaleAnim, {
      toValue: 0.95,
      useNativeDriver: true,
      speed: 50,
      bounciness: 5,
    }).start();
  };

  const handlePressOut = () => {
    if (disabled || loading) return;

    Animated.spring(scaleAnim, {
      toValue: 1,
      useNativeDriver: true,
      speed: 50,
      bounciness: 5,
    }).start();
  };

  const handlePress = () => {
    if (disabled || loading) return;

    mediumImpact();
    onPress();
  };

  return (
    <Animated.View
      style={[
        styles.container,
        {
          transform: [{ scale: scaleAnim }],
        },
        style,
      ]}
    >
      <Pressable
        onPressIn={handlePressIn}
        onPressOut={handlePressOut}
        onPress={handlePress}
        style={[
          styles.button,
          {
            backgroundColor: getBackgroundColor(),
            opacity: disabled ? 0.6 : 1,
          },
          getBorderStyle(),
        ]}
        disabled={disabled || loading}
      >
        {loading ? (
          <Animated.View
            style={styles.loadingIndicator}
            // Rotate animation for loading indicator
            // This would be implemented with a custom loading indicator
          />
        ) : (
          <>
            {icon && <View style={styles.iconContainer}>{icon}</View>}
            <Text
              style={[
                styles.text,
                {
                  color: getTextColor(),
                },
                textStyle,
              ]}
            >
              {title}
            </Text>
          </>
        )}
      </Pressable>
    </Animated.View>
  );
};

const styles = StyleSheet.create({
  container: {
    overflow: 'hidden',
  },
  button: {
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 12,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    elevation: 2,
    shadowColor: 'rgba(0,0,0,0.3)',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 2,
  },
  text: {
    fontSize: 16,
    fontWeight: '600',
    textAlign: 'center',
  },
  iconContainer: {
    marginRight: 8,
  },
  loadingIndicator: {
    width: 24,
    height: 24,
    borderRadius: 12,
    borderWidth: 2,
    borderTopColor: 'transparent',
  },
});

export default AnimatedButton;
