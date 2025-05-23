import React, { useState, useEffect, useRef } from 'react';
import { Animated, StyleSheet, ViewStyle, TextStyle, View } from 'react-native';
import { TextInput, Text, useTheme } from 'react-native-paper';

interface AnimatedTextInputProps {
  value: string;
  onChangeText: (text: string) => void;
  label: string;
  placeholder?: string;
  multiline?: boolean;
  numberOfLines?: number;
  style?: ViewStyle;
  inputStyle?: TextStyle;
  error?: string;
  autoFocus?: boolean;
  secureTextEntry?: boolean;
  keyboardType?: 'default' | 'email-address' | 'numeric' | 'phone-pad' | 'url';
  maxLength?: number;
  disabled?: boolean;
  onFocus?: () => void;
  onBlur?: () => void;
}

const AnimatedTextInput: React.FC<AnimatedTextInputProps> = ({
  value,
  onChangeText,
  label,
  placeholder,
  multiline = false,
  numberOfLines = 1,
  style,
  inputStyle,
  error,
  autoFocus = false,
  secureTextEntry = false,
  keyboardType = 'default',
  maxLength,
  disabled = false,
  onFocus,
  onBlur,
}) => {
  const theme = useTheme();
  const [isFocused, setIsFocused] = useState(false);
  const focusAnim = useRef(new Animated.Value(0)).current;
  const errorAnim = useRef(new Animated.Value(0)).current;
  
  // Animate focus state
  useEffect(() => {
    Animated.timing(focusAnim, {
      toValue: isFocused ? 1 : 0,
      duration: 200,
      useNativeDriver: false,
    }).start();
  }, [isFocused]);
  
  // Animate error state
  useEffect(() => {
    Animated.timing(errorAnim, {
      toValue: error ? 1 : 0,
      duration: 200,
      useNativeDriver: false,
    }).start();
  }, [error]);
  
  // Handle focus and blur
  const handleFocus = () => {
    setIsFocused(true);
    if (onFocus) onFocus();
  };
  
  const handleBlur = () => {
    setIsFocused(false);
    if (onBlur) onBlur();
  };
  
  // Interpolate border color based on focus and error state
  const borderColor = Animated.color(
    Animated.add(
      Animated.multiply(focusAnim, 123), // Primary color R component
      Animated.multiply(errorAnim, 230)  // Error color R component
    ),
    Animated.add(
      Animated.multiply(focusAnim, 104), // Primary color G component
      Animated.multiply(errorAnim, 57)   // Error color G component
    ),
    Animated.add(
      Animated.multiply(focusAnim, 238), // Primary color B component
      Animated.multiply(errorAnim, 70)   // Error color B component
    )
  );
  
  return (
    <View style={[styles.container, style]}>
      <Animated.View
        style={[
          styles.inputContainer,
          {
            borderColor,
            backgroundColor: disabled ? theme.colors.surfaceDisabled : theme.colors.surface,
          },
        ]}
      >
        <TextInput
          mode="outlined"
          value={value}
          onChangeText={onChangeText}
          label={label}
          placeholder={placeholder}
          multiline={multiline}
          numberOfLines={numberOfLines}
          style={[styles.input, inputStyle]}
          outlineStyle={styles.outline}
          onFocus={handleFocus}
          onBlur={handleBlur}
          autoFocus={autoFocus}
          secureTextEntry={secureTextEntry}
          keyboardType={keyboardType}
          maxLength={maxLength}
          disabled={disabled}
          activeOutlineColor={theme.colors.primary}
          outlineColor={theme.colors.outline}
          error={!!error}
        />
      </Animated.View>
      
      {error && (
        <Animated.View
          style={[
            styles.errorContainer,
            {
              opacity: errorAnim,
              transform: [
                {
                  translateY: errorAnim.interpolate({
                    inputRange: [0, 1],
                    outputRange: [-10, 0],
                  }),
                },
              ],
            },
          ]}
        >
          <Text style={[styles.errorText, { color: theme.colors.error }]}>
            {error}
          </Text>
        </Animated.View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    marginBottom: 16,
  },
  inputContainer: {
    borderRadius: 8,
    overflow: 'hidden',
  },
  input: {
    backgroundColor: 'transparent',
  },
  outline: {
    borderRadius: 8,
  },
  errorContainer: {
    marginTop: 4,
    paddingHorizontal: 8,
  },
  errorText: {
    fontSize: 12,
  },
});

export default AnimatedTextInput;
