import React, { useState, useEffect, useCallback } from 'react';
import { View, StyleSheet, FlatList, Share, Platform, Linking, RefreshControl } from 'react-native';
import { Text, Card, Button, Divider, IconButton, Dialog, Portal, FAB, Chip, Menu, Surface, useTheme } from 'react-native-paper';
import { LinearGradient } from 'expo-linear-gradient';
import { NativeStackScreenProps } from '@react-navigation/native-stack/lib/typescript/src/types';
import { RootStackParamList } from '../../App';
import { getHistory, clearHistory, HistoryItem, exportHistory } from '../utils/storage';
import Footer from '../components/Footer';

type HistoryScreenProps = NativeStackScreenProps<RootStackParamList, 'History'>;

/**
 * History screen to display past keyphrase extractions
 */
const HistoryScreen: React.FC<HistoryScreenProps> = ({ navigation }) => {
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [clearDialogVisible, setClearDialogVisible] = useState(false);
  const [exportDialogVisible, setExportDialogVisible] = useState(false);
  const [menuVisible, setMenuVisible] = useState(false);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [exportUrl, setExportUrl] = useState<string | null>(null);
  const theme = useTheme();

  // Load history when the screen is focused
  useEffect(() => {
    const unsubscribe = navigation.addListener('focus', () => {
      loadHistory();
    });

    return unsubscribe;
  }, [navigation]);

  // Handle pull-to-refresh
  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await loadHistory();
    setRefreshing(false);
  }, []);

  // Load history from storage
  const loadHistory = async () => {
    setLoading(true);
    const historyItems = await getHistory();
    setHistory(historyItems);
    setLoading(false);
  };

  // Handle clearing history
  const handleClearHistory = async () => {
    await clearHistory();
    setHistory([]);
    setClearDialogVisible(false);
  };

  // Handle exporting history
  const handleExportHistory = async () => {
    const url = await exportHistory();
    setExportUrl(url);
    setExportDialogVisible(true);
  };

  // Handle sharing history
  const handleShareHistory = async () => {
    try {
      const historyText = history
        .map(item => {
          const date = formatDate(item.timestamp);
          const keyphrases = item.keyphrases
            .map(kp => `${kp.keyphrase} (${kp.score.toFixed(2)})`)
            .join(', ');
          return `Date: ${date}\nText: ${truncateText(item.text, 50)}\nKeyphrases: ${keyphrases}\n`;
        })
        .join('\n---\n\n');

      await Share.share({
        message: `Keyphrase Extraction History\n\n${historyText}`,
        title: 'Keyphrase Extraction History'
      });
    } catch (error) {
      console.error('Error sharing history:', error);
    }
  };

  // Format date for display
  const formatDate = (timestamp: number) => {
    return new Date(timestamp).toLocaleString();
  };

  // Truncate text for preview
  const truncateText = (text: string, maxLength = 100) => {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
  };

  // Render a history item
  const renderHistoryItem = ({ item, index }: { item: HistoryItem, index: number }) => {
    // Calculate a gradient color based on the index
    const gradientStart = index % 2 === 0 ? '#6A5ACD' : '#7B68EE';
    const gradientEnd = index % 2 === 0 ? '#9370DB' : '#9683EC';

    return (
      <Card
        style={styles.card}
        onPress={() => navigation.navigate('Results', {
          keyphrases: item.keyphrases,
          text: item.text
        })}
        mode="elevated"
      >
        <LinearGradient
          colors={[gradientStart, gradientEnd]}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 0 }}
          style={styles.cardGradient}
        >
          <View style={styles.cardHeader}>
            <View style={styles.dateContainer}>
              <IconButton icon="clock-outline" size={18} iconColor="#fff" style={styles.dateIcon} />
              <Text style={styles.timestamp}>{formatDate(item.timestamp)}</Text>
            </View>
            <Chip
              icon="key-chain-variant"
              compact
              mode="outlined"
              style={styles.countChip}
              textStyle={styles.countChipText}
            >
              {item.keyphrases.length}
            </Chip>
          </View>
        </LinearGradient>

        <Card.Content style={styles.cardContent}>
          <Text style={styles.textPreview}>{truncateText(item.text, 120)}</Text>

          <Divider style={styles.divider} />

          <View style={styles.keyPhraseTitleContainer}>
            <View style={styles.keyPhraseIconBackground}>
              <IconButton
                icon="key-star"
                size={16}
                iconColor={theme.colors.primary}
                style={styles.keyPhraseIcon}
              />
            </View>
            <Text style={styles.keyphraseTitle}>Top Keyphrases</Text>
          </View>

          <View style={styles.topKeyphrases}>
            {item.keyphrases.slice(0, 3).map((kp, idx) => (
              <Chip
                key={idx}
                style={[styles.keyphraseChip, { backgroundColor: `rgba(106, 90, 205, ${0.1 + (0.15 * (3-idx) / 3)})` }]}
                mode="flat"
                compact
                icon={idx === 0 ? 'key-variant' : idx === 1 ? 'key-outline' : 'label-outline'}
              >
                {kp.keyphrase}
              </Chip>
            ))}
          </View>

          <View style={styles.cardActions}>
            <Button
              mode="text"
              icon="text-search"
              compact
              style={styles.actionButton}
              onPress={() => navigation.navigate('Results', {
                keyphrases: item.keyphrases,
                text: item.text
              })}
            >
              View
            </Button>
            <Button
              mode="text"
              icon="share-variant"
              compact
              style={styles.actionButton}
              onPress={() => handleShareItem(item)}
            >
              Share
            </Button>
          </View>
        </Card.Content>
      </Card>
    );
  };

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
                  icon="history"
                  size={36}
                  iconColor="#fff"
                  style={styles.headerIcon}
                />
              </View>
              <Text style={styles.title}>Extraction History</Text>
              <Text style={styles.subtitle}>
                {history.length} {history.length === 1 ? 'entry' : 'entries'} saved
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

      <View style={styles.actionsContainer}>
        <Menu
          visible={menuVisible}
          onDismiss={() => setMenuVisible(false)}
          anchor={<Button icon="dots-vertical" mode="text" onPress={() => setMenuVisible(true)}>Actions</Button>}
        >
          <Menu.Item
            onPress={() => {
              setMenuVisible(false);
              handleShareHistory();
            }}
            title="Share History"
            leadingIcon="share"
            disabled={history.length === 0}
          />
          {Platform.OS === 'web' && (
            <Menu.Item
              onPress={() => {
                setMenuVisible(false);
                handleExportHistory();
              }}
              title="Export as JSON"
              leadingIcon="file-download"
              disabled={history.length === 0}
            />
          )}
          <Menu.Item
            onPress={() => {
              setMenuVisible(false);
              setClearDialogVisible(true);
            }}
            title="Clear History"
            leadingIcon="delete"
            disabled={history.length === 0}
          />
        </Menu>
      </View>

      {history.length === 0 ? (
        <View style={styles.emptyContainer}>
          <LinearGradient
            colors={['rgba(106, 90, 205, 0.1)', 'rgba(106, 90, 205, 0.05)']}
            style={styles.emptyGradient}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
          >
            <IconButton icon="text-box-search-outline" size={80} style={styles.emptyIcon} iconColor={theme.colors.primary} />
            <Text style={styles.emptyText}>
              {loading ? 'Loading history...' : 'No extraction history found'}
            </Text>
            <Text style={styles.emptySubtext}>
              {!loading && 'Start extracting keyphrases to build your history'}
            </Text>
            {!loading && (
              <Button
                mode="contained"
                onPress={() => navigation.navigate('Home')}
                style={styles.emptyButton}
                icon="key-plus"
                contentStyle={styles.emptyButtonContent}
              >
                Extract New Keyphrases
              </Button>
            )}
          </LinearGradient>
        </View>
      ) : (
        <FlatList
          data={history}
          keyExtractor={(item) => item.id}
          renderItem={({ item, index }) => renderHistoryItem({ item, index })}
          contentContainerStyle={styles.list}
          ItemSeparatorComponent={() => <View style={styles.separator} />}
          showsVerticalScrollIndicator={false}
          ListFooterComponent={<Footer />}
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={onRefresh}
              colors={[theme.colors.primary]}
              tintColor={theme.colors.primary}
              progressBackgroundColor="#ffffff"
            />
          }
        />
      )}

      <FAB
        icon="home"
        style={styles.fab}
        onPress={() => navigation.navigate('Home')}
        label="New Extraction"
      />

      <Portal>
        <Dialog
          visible={clearDialogVisible}
          onDismiss={() => setClearDialogVisible(false)}
        >
          <Dialog.Title>Clear History</Dialog.Title>
          <Dialog.Content>
            <Text>Are you sure you want to clear all history? This action cannot be undone.</Text>
          </Dialog.Content>
          <Dialog.Actions>
            <Button onPress={() => setClearDialogVisible(false)}>Cancel</Button>
            <Button onPress={handleClearHistory} textColor={theme.colors.error}>Clear</Button>
          </Dialog.Actions>
        </Dialog>

        <Dialog
          visible={exportDialogVisible}
          onDismiss={() => setExportDialogVisible(false)}
        >
          <Dialog.Title>Export History</Dialog.Title>
          <Dialog.Content>
            <Text>Your history has been exported as a JSON file.</Text>
            {exportUrl && (
              <Button
                mode="contained"
                onPress={() => {
                  Linking.openURL(exportUrl);
                  setExportDialogVisible(false);
                }}
                style={styles.exportButton}
                icon="download"
              >
                Download JSON File
              </Button>
            )}
          </Dialog.Content>
          <Dialog.Actions>
            <Button onPress={() => setExportDialogVisible(false)}>Close</Button>
          </Dialog.Actions>
        </Dialog>
      </Portal>
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
  actionsContainer: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    marginBottom: 16,
  },
  list: {
    paddingBottom: 80, // Space for FAB
  },
  card: {
    marginBottom: 16,
    borderRadius: 16,
    elevation: 3,
    overflow: 'hidden',
  },
  cardGradient: {
    padding: 16,
  },
  cardContent: {
    paddingTop: 16,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  dateContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  dateIcon: {
    margin: 0,
    marginRight: 4,
  },
  timestamp: {
    fontSize: 14,
    color: '#fff',
    fontWeight: '500',
  },
  countChip: {
    height: 28,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    borderColor: 'rgba(255, 255, 255, 0.3)',
  },
  countChipText: {
    color: '#fff',
  },
  textPreview: {
    fontSize: 15,
    marginBottom: 16,
    lineHeight: 22,
    color: '#343A40',
  },
  divider: {
    marginVertical: 16,
    height: 1.5,
    backgroundColor: 'rgba(106, 90, 205, 0.1)',
  },
  keyPhraseTitleContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  keyPhraseIconBackground: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: 'rgba(123, 104, 238, 0.1)',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 8,
  },
  keyPhraseIcon: {
    margin: 0,
  },
  keyphraseTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#343A40',
  },
  topKeyphrases: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginBottom: 16,
  },
  keyphraseChip: {
    margin: 4,
    borderRadius: 20,
    height: 32,
  },
  cardActions: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    marginTop: 8,
  },
  actionButton: {
    marginLeft: 8,
  },
  separator: {
    height: 8,
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  emptyGradient: {
    width: '100%',
    borderRadius: 24,
    padding: 32,
    alignItems: 'center',
    justifyContent: 'center',
  },
  emptyIcon: {
    marginBottom: 24,
  },
  emptyText: {
    fontSize: 20,
    fontWeight: '600',
    textAlign: 'center',
    marginBottom: 12,
    color: '#343A40',
  },
  emptySubtext: {
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 32,
    opacity: 0.7,
    color: '#495057',
  },
  emptyButton: {
    borderRadius: 12,
    paddingHorizontal: 8,
  },
  emptyButtonContent: {
    paddingVertical: 8,
    paddingHorizontal: 16,
  },
  button: {
    marginTop: 16,
    borderRadius: 8,
  },
  fab: {
    position: 'absolute',
    margin: 16,
    right: 0,
    bottom: 0,
  },
  exportButton: {
    marginTop: 16,
  },
});

export default HistoryScreen;
