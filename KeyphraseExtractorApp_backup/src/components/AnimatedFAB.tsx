import React, { useState, useEffect } from 'react';
import { Animated, StyleSheet, Pressable, View } from 'react-native';
import { IconButton, Text, useTheme } from 'react-native-paper';
import { mediumImpact } from '../utils/haptics';

interface AnimatedFABProps {
  icon: string;
  onPress: () => void;
  label?: string;
  extended?: boolean;
  visible?: boolean;
  color?: string;
  backgroundColor?: string;
}

const AnimatedFAB: React.FC<AnimatedFABProps> = ({
  icon,
  onPress,
  label,
  extended = false,
  visible = true,
  color,
  backgroundColor,
}) => {
  const theme = useTheme();
  const [scaleAnim] = useState(new Animated.Value(1));
  const [visibilityAnim] = useState(new Animated.Value(visible ? 1 : 0));
  const [widthAnim] = useState(new Animated.Value(extended ? 1 : 0));
  
  // Update visibility animation when visible prop changes
  useEffect(() => {
    Animated.spring(visibilityAnim, {
      toValue: visible ? 1 : 0,
      useNativeDriver: true,
      tension: 80,
      friction: 7,
    }).start();
  }, [visible]);
  
  // Update width animation when extended prop changes
  useEffect(() => {
    Animated.spring(widthAnim, {
      toValue: extended ? 1 : 0,
      useNativeDriver: false,
      tension: 80,
      friction: 7,
    }).start();
  }, [extended]);
  
  // Handle press animation
  const handlePressIn = () => {
    Animated.spring(scaleAnim, {
      toValue: 0.9,
      useNativeDriver: true,
      speed: 50,
      bounciness: 5,
    }).start();
  };
  
  const handlePressOut = () => {
    Animated.spring(scaleAnim, {
      toValue: 1,
      useNativeDriver: true,
      speed: 50,
      bounciness: 5,
    }).start();
  };
  
  const handlePress = () => {
    mediumImpact();
    onPress();
  };
  
  // Calculate width based on animation value and label length
  const width = widthAnim.interpolate({
    inputRange: [0, 1],
    outputRange: [56, label ? Math.max(56, label.length * 10 + 80) : 56],
  });
  
  return (
    <Animated.View
      style={[
        styles.container,
        {
          transform: [
            { scale: scaleAnim },
            { scale: visibilityAnim },
          ],
          opacity: visibilityAnim,
        },
      ]}
    >
      <Pressable
        onPressIn={handlePressIn}
        onPressOut={handlePressOut}
        onPress={handlePress}
        style={({ pressed }) => [
          styles.button,
          {
            backgroundColor: backgroundColor || theme.colors.primary,
            width,
          },
        ]}
      >
        <IconButton
          icon={icon}
          size={24}
          iconColor={color || theme.colors.onPrimary}
          style={styles.icon}
        />
        
        {extended && label && (
          <Animated.View
            style={[
              styles.labelContainer,
              {
                opacity: widthAnim,
              },
            ]}
          >
            <Text
              style={[
                styles.label,
                {
                  color: color || theme.colors.onPrimary,
                },
              ]}
              numberOfLines={1}
            >
              {label}
            </Text>
          </Animated.View>
        )}
      </Pressable>
    </Animated.View>
  );
};

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    bottom: 16,
    right: 16,
    zIndex: 1000,
    elevation: 6,
    shadowColor: 'rgba(0,0,0,0.5)',
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
  },
  button: {
    height: 56,
    borderRadius: 28,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 16,
  },
  icon: {
    margin: 0,
    width: 24,
    height: 24,
  },
  labelContainer: {
    marginLeft: 4,
    marginRight: 8,
  },
  label: {
    fontSize: 14,
    fontWeight: '500',
  },
});

export default AnimatedFAB;
