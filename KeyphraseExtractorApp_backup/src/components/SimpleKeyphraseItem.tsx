import React, { useState } from 'react';
import { View, StyleSheet, TouchableOpacity } from 'react-native';
import { Text, Surface, useTheme, IconButton, Chip, Divider } from 'react-native-paper';
import { Keyphrase } from '../services/api';

interface SimpleKeyphraseItemProps {
  keyphrase: Keyphrase;
  index: number;
}

/**
 * A simplified component to display a single keyphrase with expandable related terms
 */
const SimpleKeyphraseItem: React.FC<SimpleKeyphraseItemProps> = ({ keyphrase, index }) => {
  const theme = useTheme();
  const [expanded, setExpanded] = useState(false);

  // Make sure we have a valid keyphrase
  const keyphraseText = keyphrase?.keyphrase || 'Unknown';
  const score = typeof keyphrase?.score === 'number' ? keyphrase.score : 0.5;

  // Filter out expansions that match the original keyphrase
  const filteredExpansions = keyphrase?.expansions ?
    keyphrase.expansions.filter(expansion =>
      expansion.toLowerCase() !== keyphraseText.toLowerCase()
    ) : [];

  const hasExpansions = filteredExpansions.length > 0;

  // Get appropriate icon based on score
  const getIcon = (score: number) => {
    if (score >= 0.7) return 'key-variant';
    if (score >= 0.5) return 'key-outline';
    return 'label-outline';
  };

  // Get color based on score
  const getColor = (score: number) => {
    if (score >= 0.7) return theme.colors.primary;
    if (score >= 0.5) return theme.colors.secondary;
    return theme.colors.tertiary;
  };

  const icon = getIcon(score);
  const color = getColor(score);

  // Toggle expanded state
  const toggleExpanded = () => {
    if (hasExpansions) {
      setExpanded(!expanded);
    }
  };

  return (
    <Surface style={styles.container} elevation={1}>
      <TouchableOpacity
        style={styles.content}
        onPress={toggleExpanded}
        disabled={!hasExpansions}
      >
        <View style={styles.iconContainer}>
          <IconButton
            icon={icon}
            size={20}
            iconColor={color}
            style={styles.icon}
          />
        </View>
        <View style={styles.textContainer}>
          <Text style={styles.keyphraseText}>{keyphraseText}</Text>
        </View>
        {hasExpansions && (
          <View style={styles.expandButtonContainer}>
            <Surface style={styles.expandButtonSurface} elevation={2}>
              <IconButton
                icon={expanded ? 'chevron-up' : 'chevron-down'}
                size={20}
                iconColor="white"
                style={styles.expandButton}
              />
              <Text style={styles.expandButtonText}>
                {expanded ? 'Hide' : 'See'} Expansion
              </Text>
            </Surface>
          </View>
        )}
      </TouchableOpacity>

      {/* Expanded content with related terms */}
      {expanded && hasExpansions && (
        <View style={styles.expandedContent}>
          <Divider style={styles.divider} />
          <View style={styles.relatedTermsContainer}>
            <Text style={styles.relatedTermsTitle}>Related Terms:</Text>
            <View style={styles.chipContainer}>
              {filteredExpansions.map((expansion, idx) => (
                <Chip
                  key={idx}
                  style={styles.chip}
                  mode="outlined"
                  icon="tag-text"
                >
                  {expansion}
                </Chip>
              ))}
              {filteredExpansions.length === 0 && (
                <Text style={styles.noExpansionsText}>No related terms found</Text>
              )}
            </View>
          </View>
        </View>
      )}
    </Surface>
  );
};

const styles = StyleSheet.create({
  container: {
    marginVertical: 6,
    marginHorizontal: 12,
    borderRadius: 12,
    overflow: 'hidden',
  },
  content: {
    padding: 12,
    flexDirection: 'row',
    alignItems: 'center',
  },
  iconContainer: {
    marginRight: 8,
  },
  icon: {
    margin: 0,
  },
  textContainer: {
    flex: 1,
  },
  keyphraseText: {
    fontSize: 16,
    fontWeight: '500',
  },
  expandButtonContainer: {
    marginLeft: 8,
  },
  expandButtonSurface: {
    borderRadius: 16,
    backgroundColor: '#6A5ACD',
    flexDirection: 'row',
    alignItems: 'center',
    paddingRight: 12,
    paddingLeft: 4,
    overflow: 'hidden',
    elevation: 3,
    shadowColor: '#6A5ACD',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 3,
  },
  expandButton: {
    margin: 0,
  },
  expandButtonText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '500',
  },
  expandedContent: {
    paddingHorizontal: 12,
    paddingBottom: 12,
  },
  divider: {
    marginVertical: 8,
  },
  relatedTermsContainer: {
    marginTop: 4,
  },
  relatedTermsTitle: {
    fontSize: 14,
    fontWeight: '500',
    marginBottom: 8,
    opacity: 0.8,
  },
  chipContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginHorizontal: -4,
  },
  chip: {
    margin: 4,
    backgroundColor: 'rgba(106, 90, 205, 0.1)',
    borderColor: 'rgba(106, 90, 205, 0.3)',
  },
  noExpansionsText: {
    fontStyle: 'italic',
    opacity: 0.6,
    padding: 4,
  },
});

export default SimpleKeyphraseItem;
