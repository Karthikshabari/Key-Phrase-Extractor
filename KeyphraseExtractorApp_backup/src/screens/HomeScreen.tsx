import React, { useState, useEffect } from 'react';
import { View, StyleSheet, ScrollView } from 'react-native';
import { Button, TextInput, Text, useTheme, IconButton, Card } from 'react-native-paper';
import { getSafeIconName } from '../utils/iconUtils';
import { LinearGradient } from 'expo-linear-gradient';
import { NativeStackScreenProps } from '@react-navigation/native-stack/lib/typescript/src/types';
import { RootStackParamList } from '../../App';
import { extractKeyphrases } from '../services/api';
import LoadingOverlay from '../components/LoadingOverlay';
import Footer from '../components/Footer';
import AppVersionInfo from '../components/AppVersionInfo';
import { saveHistoryItem, getDebugMode } from '../utils/storage';
import { mediumImpact, successNotification, errorNotification } from '../utils/haptics';

type HomeScreenProps = NativeStackScreenProps<RootStackParamList, 'Home'>;

/**
 * Home screen with text input for keyphrase extraction
 */
const HomeScreen: React.FC<HomeScreenProps> = ({ navigation }) => {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isDebugMode, setIsDebugMode] = useState(false);
  const theme = useTheme();

  // Load debug mode setting
  useEffect(() => {
    const loadDebugMode = async () => {
      const debugMode = await getDebugMode();
      setIsDebugMode(debugMode);
    };

    loadDebugMode();
  }, []);

  // Function to count words in text
  const countWords = (text: string): number => {
    return text.trim().split(/\s+/).filter(word => word.length > 0).length;
  };

  const handleExtract = async () => {
    if (!text.trim()) {
      setError('Please enter some text to extract keyphrases from.');
      errorNotification();
      return;
    }

    // Count words and validate
    const wordCount = countWords(text);
    if (wordCount < 300) {
      setError(`Your text has only ${wordCount} words. Please enter at least 300 words for optimal results.`);
      errorNotification();
      return;
    }

    if (wordCount > 500) {
      setError(`Your text has ${wordCount} words. For optimal results, please limit your text to 500 words.`);
      errorNotification();
      return;
    }

    setLoading(true);
    setError(null);
    mediumImpact();

    try {
      if (isDebugMode) console.log('Sending text to API:', text);
      // Extract keyphrases with domain set to 'auto' for automatic detection
      const keyphrases = await extractKeyphrases(text, 'auto');
      // Always log keyphrases for debugging
      console.log('Received keyphrases:', JSON.stringify(keyphrases, null, 2));

      // Save to history
      const historyItem = {
        id: Date.now().toString(),
        text,
        keyphrases,
        timestamp: Date.now(),
      };
      await saveHistoryItem(historyItem);

      // Provide success feedback
      successNotification();

      // Navigate to results screen
      navigation.navigate('Results', { keyphrases, text });
    } catch (err: any) {
      console.error('Error extracting keyphrases:', err);
      let errorMessage = 'Failed to extract keyphrases. Please check your API settings and try again.';

      // Add more detailed error information
      if (err.response) {
        // The request was made and the server responded with a status code
        // that falls out of the range of 2xx
        if (isDebugMode) console.error('Error response data:', err.response.data);
        if (isDebugMode) console.error('Error response status:', err.response.status);
        errorMessage += ` Server responded with status ${err.response.status}.`;
      } else if (err.request) {
        // The request was made but no response was received
        if (isDebugMode) console.error('Error request:', err.request);
        errorMessage += ' No response received from server.';
      } else {
        // Something happened in setting up the request that triggered an Error
        if (isDebugMode) console.error('Error message:', err.message);
        errorMessage += ` Error: ${err.message}`;
      }

      // Always show detailed error in debug mode
      if (isDebugMode) {
        errorMessage += '\n\nDebug Info: ' + JSON.stringify(err, null, 2);
      }

      setError(errorMessage);
      errorNotification();
    } finally {
      setLoading(false);
    }
  };

  const handleViewHistory = () => {
    navigation.navigate('History');
  };

  return (
    <View style={styles.container}>
      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {/* Header with subtle gradient */}
        <Card style={styles.headerCard} elevation={5}>
          <LinearGradient
            colors={[
              '#6A5ACD', // SlateBlue
              '#9370DB'  // MediumPurple
            ]}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 0 }}
            style={styles.headerGradient}
          >
            <View style={styles.headerOverlay}>
              <Card.Content style={styles.headerContent}>
                <View style={styles.headerIconContainer}>
                  <IconButton
                    icon="text-search"
                    size={36}
                    iconColor="#fff"
                    style={styles.headerIcon}
                  />
                </View>
                <Text style={styles.title} variant="titleLarge">
                  Keyphrase Extractor
                </Text>
                <Text style={styles.subtitle} variant="bodyLarge">
                  Extract key concepts and phrases from news articles
                </Text>

                <View style={styles.headerDecoration}>
                  <View style={styles.decorationDot} />
                  <View style={styles.decorationLine} />
                  <View style={styles.decorationDot} />
                </View>
              </Card.Content>
            </View>
          </LinearGradient>
        </Card>

        {/* Text Input Card */}
        <Card style={styles.inputCard} elevation={2}>
          <Card.Content style={styles.cardContent}>
            <View style={styles.sectionTitleContainer}>
              <View style={styles.sectionIconBackground}>
                <IconButton
                  icon={getSafeIconName('text-box-edit-outline')}
                  size={20}
                  iconColor={theme.colors.primary}
                  style={styles.sectionIcon}
                />
              </View>
              <Text style={styles.sectionTitle} variant="titleMedium">
                Text Input
              </Text>
            </View>
            <TextInput
              mode="outlined"
              label="Text to analyze (300-500 words)"
              value={text}
              onChangeText={setText}
              multiline
              numberOfLines={12} // Increased for better visibility
              style={styles.textInput}
              placeholder="Paste your article, document, or any text here..."
              outlineStyle={styles.textInputOutline}
              placeholderTextColor={theme.colors.onSurfaceVariant}
              activeOutlineColor={theme.colors.primary}
              outlineColor="rgba(0,0,0,0.12)"
              autoCapitalize="sentences"
              textAlignVertical="top" // Ensures text starts from the top
              selectionColor={theme.colors.primary} // Cursor and selection color
            />

            {/* Word counter */}
            <View style={styles.wordCountContainer}>
              <View style={styles.wordCountBadge}>
                <IconButton
                  icon="counter"
                  size={16}
                  iconColor={text.trim() ? (
                    countWords(text) < 300 || countWords(text) > 500 ? theme.colors.error : theme.colors.primary
                  ) : theme.colors.onSurfaceVariant}
                  style={styles.wordCountIcon}
                />
                <Text style={[styles.wordCount, {
                  color: text.trim() ? (
                    countWords(text) < 300 || countWords(text) > 500 ? theme.colors.error : theme.colors.primary
                  ) : theme.colors.onSurfaceVariant
                }]}>
                  {countWords(text)} words
                </Text>
              </View>
              <View style={styles.wordCountHintContainer}>
                <IconButton
                  icon="information-outline"
                  size={16}
                  iconColor={theme.colors.onSurfaceVariant}
                  style={styles.wordCountInfoIcon}
                />
                <Text style={styles.wordCountHint}>
                  Optimal: 300-500 words
                </Text>
              </View>
            </View>

            {error && (
              <View style={styles.errorContainer}>
                <IconButton
                  icon="alert-circle-outline"
                  size={20}
                  iconColor={theme.colors.error}
                  style={styles.errorIcon}
                />
                <Text style={[styles.errorText, { color: theme.colors.error }]}>
                  {error}
                </Text>
              </View>
            )}

            <View style={styles.buttonContainer}>
              <Button
                mode="contained"
                onPress={handleExtract}
                style={styles.extractButton}
                contentStyle={styles.buttonContent}
                disabled={loading || !text.trim() || countWords(text) < 300 || countWords(text) > 500}
                icon="text-box-search"
                labelStyle={styles.buttonLabel}
                elevation={3}
              >
                Extract Keyphrases
              </Button>

              <Button
                mode="outlined"
                onPress={handleViewHistory}
                style={styles.historyButton}
                contentStyle={styles.buttonContent}
                icon="history"
                labelStyle={styles.buttonLabel}
                textColor={theme.colors.primary}
              >
                View History
              </Button>
            </View>
          </Card.Content>
        </Card>

        {/* Tips Card */}
        <Card style={styles.tipsCard} elevation={2}>
          <Card.Content style={styles.cardContent}>
            {/* Tips Header with Lightbulb Icon */}
            <View style={styles.tipsHeader}>
              <View style={styles.tipsHeaderIconContainer}>
                <View style={styles.tipsHeaderIconBackground}>
                  <IconButton
                    icon="lightbulb-on-outline"
                    size={22}
                    iconColor={theme.colors.primary}
                    style={styles.tipsHeaderIcon}
                  />
                </View>
              </View>
              <Text style={styles.sectionTitle} variant="titleMedium">
                Tips for Best Results
              </Text>
            </View>

            {/* Tip 1 - Optimal Length */}
            <View style={styles.tipItem}>
              <View style={styles.tipIconContainer}>
                <View style={styles.tipIconBackground}>
                  <IconButton
                    icon="ruler"
                    size={20}
                    iconColor={theme.colors.primary}
                    style={styles.tipIcon}
                  />
                </View>
              </View>
              <View style={styles.tipContent}>
                <Text style={styles.tipTitle}>
                  Optimal Length
                </Text>
                <Text style={styles.tipText}>
                  Use text between 350-500 words for best results
                </Text>
              </View>
            </View>

            {/* Tip 2 - Content Type */}
            <View style={styles.tipItem}>
              <View style={styles.tipIconContainer}>
                <View style={styles.tipIconBackground}>
                  <IconButton
                    icon="file-document-outline"
                    size={20}
                    iconColor={theme.colors.secondary}
                    style={styles.tipIcon}
                  />
                </View>
              </View>
              <View style={styles.tipContent}>
                <Text style={styles.tipTitle}>
                  Content Type
                </Text>
                <Text style={styles.tipText}>
                  Optimized for news articles in specific domains
                </Text>
              </View>
            </View>

            {/* Tip 3 - Specialized Domains */}
            <View style={styles.tipItem}>
              <View style={styles.tipIconContainer}>
                <View style={styles.tipIconBackground}>
                  <IconButton
                    icon="domain"
                    size={20}
                    iconColor={theme.colors.tertiary}
                    style={styles.tipIcon}
                  />
                </View>
              </View>
              <View style={styles.tipContent}>
                <Text style={styles.tipTitle}>
                  Specialized Domains
                </Text>
                <View style={styles.domainTagsContainer}>
                  {[
                    'AI', 'Automotive', 'Cybersecurity', 'Food',
                    'Environment', 'Real Estate', 'Entertainment'
                  ].map((domain) => (
                    <View
                      key={domain}
                      style={[styles.domainTag, { backgroundColor: '#EDE7F6' }]}
                    >
                      <Text style={[styles.domainTagText, { color: theme.colors.primary }]}>
                        {domain}
                      </Text>
                    </View>
                  ))}
                </View>
              </View>
            </View>
          </Card.Content>
        </Card>

        <View>
          <Footer />
          <AppVersionInfo />
        </View>
      </ScrollView>

      <LoadingOverlay visible={loading} message="Extracting keyphrases..." />
    </View>
  );
};

// Get theme colors for use in styles
const themeColors = {
  background: '#F8F9FA',
  onSurface: '#343A40',
  onSurfaceVariant: '#6C757D',
  error: '#E63946',
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: themeColors.background,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: 24,
    paddingBottom: 40,
  },
  // Header styles
  headerCard: {
    marginBottom: 24,
    borderRadius: 20,
    overflow: 'hidden',
  },
  headerGradient: {
    padding: 0,
  },
  headerOverlay: {
    backgroundColor: 'rgba(0, 0, 0, 0.05)',
    padding: 28,
  },
  headerContent: {
    alignItems: 'center',
  },
  headerIconContainer: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'rgba(255, 255, 255, 0.25)',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 20,
    elevation: 4,
    shadowColor: 'rgba(0, 0, 0, 0.3)',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
  },
  headerIcon: {
    margin: 0,
  },
  title: {
    marginBottom: 12,
    textAlign: 'center',
    color: '#fff',
    fontSize: 32,
    fontWeight: 'bold',
    letterSpacing: 0.5,
    textShadowColor: 'rgba(0, 0, 0, 0.2)',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 3,
  },
  subtitle: {
    textAlign: 'center',
    color: '#fff',
    fontSize: 16,
    letterSpacing: 0.25,
    marginBottom: 20,
  },
  headerDecoration: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    width: '100%',
    marginTop: 10,
  },
  decorationDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: 'rgba(255, 255, 255, 0.7)',
  },
  decorationLine: {
    height: 2,
    width: 40,
    backgroundColor: 'rgba(255, 255, 255, 0.5)',
    marginHorizontal: 8,
  },
  // Input card styles
  inputCard: {
    marginBottom: 24,
    borderRadius: 20,
    overflow: 'hidden',
  },
  cardContent: {
    padding: 20,
  },
  sectionTitleContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 20,
  },
  sectionIconBackground: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: 'rgba(123, 104, 238, 0.1)',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  sectionIcon: {
    margin: 0,
  },
  sectionTitle: {
    color: themeColors.onSurface,
  },
  textInput: {
    marginBottom: 8, // Reduced to make room for word counter
    minHeight: 240, // Increased height for better visibility
    backgroundColor: '#fff',
    fontSize: 16, // Increased font size for better readability
    lineHeight: 24, // Added line height for better readability
  },
  textInputOutline: {
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(0,0,0,0.2)', // Darker border for better visibility
  },
  // Word counter styles
  wordCountContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
    paddingHorizontal: 4,
  },
  wordCountBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.05)',
    borderRadius: 16,
    paddingRight: 12,
  },
  wordCountIcon: {
    margin: 0,
  },
  wordCount: {
    fontSize: 14,
    fontWeight: '500',
  },
  wordCountHintContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  wordCountInfoIcon: {
    margin: 0,
  },
  wordCountHint: {
    fontSize: 12,
    opacity: 0.7,
  },
  errorContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 20,
    padding: 12,
    borderRadius: 8,
    backgroundColor: 'rgba(230, 57, 70, 0.08)',
    borderLeftWidth: 4,
    borderLeftColor: themeColors.error,
  },
  errorIcon: {
    margin: 0,
    marginRight: 8,
  },
  errorText: {
    fontSize: 14,
    flex: 1,
  },
  // Button styles
  buttonContainer: {
    marginTop: 20,
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  extractButton: {
    flex: 1,
    marginRight: 12,
    borderRadius: 12,
    elevation: 2,
  },
  historyButton: {
    flex: 1,
    marginLeft: 12,
    borderRadius: 12,
    borderWidth: 1.5,
    backgroundColor: 'rgba(255, 255, 255, 0.8)',
  },
  buttonContent: {
    paddingVertical: 10,
    height: 50,
  },
  buttonLabel: {
    fontSize: 15,
    fontWeight: '600',
    letterSpacing: 0.5,
  },
  // Tips card styles
  tipsCard: {
    borderRadius: 20,
    overflow: 'hidden',
    marginBottom: 24,
    elevation: 2,
    shadowColor: 'rgba(0,0,0,0.1)',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 8,
  },
  tipsHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 24,
  },
  tipsHeaderIconContainer: {
    marginRight: 12,
  },
  tipsHeaderIconBackground: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: '#F0E6FF',
    alignItems: 'center',
    justifyContent: 'center',
    elevation: 2,
    shadowColor: 'rgba(0,0,0,0.1)',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.2,
    shadowRadius: 2,
  },
  tipsHeaderIcon: {
    margin: 0,
  },
  divider: {
    marginBottom: 20,
    height: 1.5,
    backgroundColor: 'rgba(0,0,0,0.05)',
  },
  tipItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 24,
    paddingVertical: 4,
  },
  tipIconContainer: {
    marginRight: 16,
    alignItems: 'center',
    justifyContent: 'center',
  },
  tipIconBackground: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#F5F5F5',
    alignItems: 'center',
    justifyContent: 'center',
    elevation: 2,
    shadowColor: 'rgba(0,0,0,0.1)',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.2,
    shadowRadius: 2,
  },
  tipIcon: {
    margin: 0,
    alignItems: 'center',
    justifyContent: 'center',
  },
  tipContent: {
    flex: 1,
  },
  tipTitle: {
    fontWeight: '600',
    marginBottom: 6,
    fontSize: 16,
    color: themeColors.onSurface,
  },
  tipText: {
    opacity: 0.87,
    lineHeight: 20,
  },
  domainTagsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginTop: 8,
  },
  domainTag: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    marginRight: 8,
    marginBottom: 8,
  },
  domainTagText: {
    fontSize: 13,
    fontWeight: '500',
    color: themeColors.onSurface,
  },
});

export default HomeScreen;
