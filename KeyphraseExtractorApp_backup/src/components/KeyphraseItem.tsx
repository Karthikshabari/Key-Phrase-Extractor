import React, { useEffect } from 'react';
import { View, StyleSheet, Animated } from 'react-native';
import { Chip, Text, useTheme, Surface, IconButton } from 'react-native-paper';
import { Keyphrase } from '../services/api';
// No longer using LinearGradient

interface KeyphraseItemProps {
  keyphrase: Keyphrase;
  index: number;
}

/**
 * Component to display a single keyphrase with its score
 */
const KeyphraseItem: React.FC<KeyphraseItemProps> = ({ keyphrase, index }) => {
  // Debug log the keyphrase
  console.log(`KeyphraseItem rendering: ${index}`, JSON.stringify(keyphrase, null, 2));

  const theme = useTheme();

  // Animation value for fade-in effect
  const fadeAnim = React.useRef(new Animated.Value(0)).current;
  const slideAnim = React.useRef(new Animated.Value(20)).current;

  // Start animations when component mounts
  useEffect(() => {
    // Add a slight delay based on index for staggered animation
    const delay = index * 100;

    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 500,
        delay,
        useNativeDriver: true,
      }),
      Animated.timing(slideAnim, {
        toValue: 0,
        duration: 500,
        delay,
        useNativeDriver: true,
      })
    ]).start();
  }, []);

  // Calculate colors based on the score
  const getScoreColors = (score: number) => {
    if (score >= 0.8) {
      return [theme.colors.primary, '#7B68EE']; // Purple gradient for high scores
    } else if (score >= 0.6) {
      return [theme.colors.secondary, '#00B8A9']; // Teal gradient for good scores
    } else if (score >= 0.4) {
      return [theme.colors.tertiary, '#FF85A2']; // Pink gradient for medium scores
    } else if (score >= 0.2) {
      return [theme.colors.accent || '#FFD166', '#F4A261']; // Gold gradient for low scores
    } else {
      return [theme.colors.error, '#E63946']; // Red gradient for very low scores
    }
  };

  // Make sure we have a valid score and keyphrase
  const score = typeof keyphrase?.score === 'number' ? keyphrase.score : 0.5;
  const keyphraseText = keyphrase?.keyphrase || 'Unknown';

  // Get score color gradient
  const scoreColors = getScoreColors(score);

  // Calculate score bar width as percentage
  const scoreWidth = `${Math.min(score * 100, 100)}%`;

  // Get appropriate icon based on score
  const getScoreIcon = (score: number) => {
    if (score >= 0.8) return 'star';
    if (score >= 0.6) return 'star-half-full';
    if (score >= 0.4) return 'thumb-up';
    if (score >= 0.2) return 'thumb-up-outline';
    return 'thumbs-up-down';
  };

  // Get the icon based on the score
  const scoreIcon = getScoreIcon(score);

  return (
    <Animated.View
      style={{
        opacity: fadeAnim,
        transform: [{ translateY: slideAnim }]
      }}
    >
      <Surface style={styles.container} elevation={3}>
        <LinearGradient
          colors={['rgba(255,255,255,0.8)', 'rgba(255,255,255,0.4)']}
          start={{ x: 0, y: 0 }}
          end={{ x: 0, y: 1 }}
          style={styles.backgroundGradient}
        />
        <View style={styles.contentContainer}>
          {/* Keyphrase chip */}
          <View style={styles.keyphraseContainer}>
            <Chip
              mode="flat"
              style={[
                styles.chip,
                { backgroundColor: `${scoreColors[0]}15` } // Very light version of the score color
              ]}
              icon={() => (
                <IconButton
                  icon={scoreIcon}
                  size={16}
                  iconColor={scoreColors[0]}
                  style={styles.chipIcon}
                />
              )}
            >
              <Text
                style={[styles.keyphraseText, { color: theme.colors.onSurface }]}
                numberOfLines={1}
                ellipsizeMode="tail"
              >
                {keyphraseText}
              </Text>
            </Chip>
          </View>

          {/* Icon to indicate importance based on score */}
          <View style={styles.tagContainer}>
            <IconButton
              icon={score >= 0.7 ? "key-variant" : score >= 0.5 ? "key-outline" : "label-outline"}
              size={18}
              iconColor={scoreColors[0]}
              style={styles.tagIcon}
            />
          </View>
        </View>
      </Surface>
    </Animated.View>
  );
};

const styles = StyleSheet.create({
  container: {
    margin: 8,
    borderRadius: 16,
    overflow: 'hidden',
    position: 'relative',
  },
  backgroundGradient: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
  },
  contentContainer: {
    padding: 16,
  },
  keyphraseContainer: {
    marginBottom: 12,
  },
  chip: {
    alignSelf: 'flex-start',
    borderRadius: 8,
    height: 40,
    paddingHorizontal: 4,
  },
  chipIcon: {
    margin: 0,
    padding: 0,
    width: 20,
    height: 20,
  },
  keyphraseText: {
    fontSize: 16,
    fontWeight: '500',
    letterSpacing: 0.25,
  },
  tagContainer: {
    alignItems: 'flex-end',
    justifyContent: 'center',
  },
  tagIcon: {
    margin: 0,
    opacity: 0.7,
  },
});

export default KeyphraseItem;
