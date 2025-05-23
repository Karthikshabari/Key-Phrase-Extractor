import React, { useState, useEffect } from 'react';
import { View, StyleSheet, FlatList, Animated } from 'react-native';
import { Text, Divider, Button, Chip, useTheme, Card, IconButton, Surface } from 'react-native-paper';
import { LinearGradient } from 'expo-linear-gradient';
import { NativeStackScreenProps } from '@react-navigation/native-stack/lib/typescript/src/types';
import { RootStackParamList } from '../../App';
import KeyphraseItem from '../components/KeyphraseItem';
import SimpleKeyphraseItem from '../components/SimpleKeyphraseItem';
import KeyphraseSkeleton from '../components/KeyphraseSkeleton';
import Footer from '../components/Footer';
import { Keyphrase } from '../services/api';

type ResultsScreenProps = NativeStackScreenProps<RootStackParamList, 'Results'>;

/**
 * Results screen to display extracted keyphrases
 */
const ResultsScreen: React.FC<ResultsScreenProps> = ({ route, navigation }) => {
  const { keyphrases, text } = route.params;
  // State for alphabetical sort order
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('asc');
  const theme = useTheme();

  // Debug log the received keyphrases
  console.log('ResultsScreen received keyphrases:', JSON.stringify(keyphrases, null, 2));

  // Ensure keyphrases is an array and has the expected format
  const validKeyphrases = Array.isArray(keyphrases) ? keyphrases : [];

  // Sort keyphrases alphabetically based on sort order
  const sortedKeyphrases = [...validKeyphrases].sort((a, b) => {
    // Get the keyphrases, defaulting to empty string if undefined
    const keyphraseA = (a?.keyphrase || '').toLowerCase();
    const keyphraseB = (b?.keyphrase || '').toLowerCase();

    // Sort based on current sort order
    return sortOrder === 'asc'
      ? keyphraseA.localeCompare(keyphraseB)
      : keyphraseB.localeCompare(keyphraseA);
  });

  // Calculate some statistics
  const averageScore = validKeyphrases.length > 0
    ? validKeyphrases.reduce((sum, kp) => sum + (kp?.score || 0), 0) / validKeyphrases.length
    : 0;

  const topKeyphrases = [...validKeyphrases]
    .sort((a, b) => (b?.score || 0) - (a?.score || 0))
    .slice(0, Math.min(3, validKeyphrases.length));

  // Animation value for fade-in effect
  const fadeAnim = React.useRef(new Animated.Value(0)).current;
  const [loading, setLoading] = useState(true);

  // Start fade-in animation when component mounts
  useEffect(() => {
    // Simulate loading time for skeleton effect
    const timer = setTimeout(() => {
      setLoading(false);
    }, 1500);

    Animated.timing(fadeAnim, {
      toValue: 1,
      duration: 800,
      useNativeDriver: true,
    }).start();

    return () => clearTimeout(timer);
  }, []);

  return (
    <View style={styles.container}>
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
                  icon="format-list-bulleted"
                  size={36}
                  iconColor="#fff"
                  style={styles.headerIcon}
                />
              </View>
              <Text style={styles.title}>Extracted Keyphrases</Text>
              <Text style={styles.subtitle}>
                {keyphrases.length} keyphrases extracted
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

      <Card style={styles.controlCard}>
        <Card.Content>
          <View style={styles.sortContainer}>
            <View style={styles.sortLabelContainer}>
              <IconButton
                icon="sort-alphabetical-ascending"
                size={20}
                iconColor={theme.colors.primary}
                style={styles.sortIcon}
              />
              <Text style={styles.sortLabel}>Sort: {sortOrder === 'asc' ? 'A-Z' : 'Z-A'}</Text>
            </View>
            <Button
              mode="outlined"
              onPress={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
              icon={sortOrder === 'asc' ? 'sort-alphabetical-descending' : 'sort-alphabetical-ascending'}
              style={styles.sortButton}
              compact
            >
              {sortOrder === 'asc' ? 'Reverse (Z-A)' : 'Reverse (A-Z)'}
            </Button>
          </View>
        </Card.Content>
      </Card>

      <Card style={styles.statsCard}>
        <Card.Content>
          <View style={styles.statsTitleContainer}>
            <View style={styles.statsIconBackground}>
              <IconButton
                icon="key-star"
                size={20}
                iconColor={theme.colors.primary}
                style={styles.statsIcon}
              />
            </View>
            <Text style={styles.statsTitle}>Top Keyphrases</Text>
          </View>
          <Divider style={styles.divider} />
          <View style={styles.topKeyphrases}>
            {topKeyphrases.map((kp, index) => (
              <Chip
                key={index}
                style={[styles.topKeyphraseChip]}
                mode="flat"
                icon={index === 0 ? 'key-variant' : index === 1 ? 'key-outline' : 'label-outline'}
              >
                {kp?.keyphrase || 'Unknown'}
              </Chip>
            ))}
          </View>
          {/* Average score removed as requested */}
        </Card.Content>
      </Card>

      <Text style={styles.resultsTitle}>All Keyphrases</Text>

      {/* Debug log for sortedKeyphrases */}
      {console.log('sortedKeyphrases:', JSON.stringify(sortedKeyphrases, null, 2))}

      <Animated.View style={[styles.listContainer, { opacity: fadeAnim }]}>
        {loading ? (
          <KeyphraseSkeleton count={7} />
        ) : sortedKeyphrases.length > 0 ? (
          <FlatList
            data={sortedKeyphrases}
            keyExtractor={(item, index) => `${item?.keyphrase || 'unknown'}-${index}`}
            renderItem={({ item, index }) => {
              console.log('Rendering item:', JSON.stringify(item, null, 2));
              // Use the simplified component
              return <SimpleKeyphraseItem keyphrase={item} index={index} />;
            }}
            ListEmptyComponent={<Text style={{padding: 20, textAlign: 'center'}}>No keyphrases found</Text>}
            style={styles.list}
            contentContainerStyle={styles.listContent}
            showsVerticalScrollIndicator={false}
            ListFooterComponent={<Footer />}
          />
        ) : (
          <View style={{padding: 20, alignItems: 'center'}}>
            <Text style={{fontSize: 16, marginBottom: 10}}>No keyphrases found</Text>
            <Text style={{textAlign: 'center', opacity: 0.7}}>Try extracting keyphrases again with different text</Text>
          </View>
        )}
      </Animated.View>

      <Surface style={styles.buttonContainer} elevation={4}>
        <Button
          mode="contained"
          onPress={() => navigation.goBack()}
          style={styles.button}
          icon="text-box-search"
          contentStyle={styles.buttonContent}
        >
          Extract More
        </Button>
      </Surface>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
    padding: 16,
  },
  headerCard: {
    marginBottom: 16,
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
  controlCard: {
    marginBottom: 16,
    borderRadius: 16,
    overflow: 'hidden',
    elevation: 2,
  },
  sortContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  sortLabelContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  sortIcon: {
    margin: 0,
  },
  sortLabel: {
    fontSize: 16,
    fontWeight: '600',
  },
  sortButton: {
    borderRadius: 20,
    borderWidth: 1,
  },
  sortButtonLabel: {
    fontSize: 12,
    marginHorizontal: 4,
  },
  statsCard: {
    marginBottom: 16,
    borderRadius: 16,
    overflow: 'hidden',
    elevation: 2,
  },
  statsTitleContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  statsIconBackground: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: 'rgba(123, 104, 238, 0.1)',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  statsIcon: {
    margin: 0,
  },
  statsTitle: {
    fontSize: 18,
    fontWeight: '600',
  },
  divider: {
    marginBottom: 16,
  },
  statsRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 8,
  },
  statsText: {
    fontSize: 14,
    opacity: 0.8,
  },
  topKeyphrases: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginBottom: 8,
  },
  topKeyphraseChip: {
    margin: 4,
    backgroundColor: 'rgba(98, 0, 238, 0.1)',
    borderRadius: 20,
    paddingVertical: 2,
    paddingHorizontal: 4,
    elevation: 1,
  },
  resultsTitle: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 8,
    marginTop: 8,
  },
  listContainer: {
    flex: 1,
    marginBottom: 16,
    minHeight: 200, // Ensure the container has a minimum height
    borderWidth: 0, // Add a border for debugging
    borderColor: 'red',
  },
  list: {
    flex: 1,
  },
  listContent: {
    paddingBottom: 16,
  },
  buttonContainer: {
    padding: 16,
    borderRadius: 16,
    backgroundColor: '#fff',
    marginTop: 8,
  },
  button: {
    borderRadius: 12,
    elevation: 2,
  },
  buttonContent: {
    paddingVertical: 10,
    height: 50,
  },
});

export default ResultsScreen;
