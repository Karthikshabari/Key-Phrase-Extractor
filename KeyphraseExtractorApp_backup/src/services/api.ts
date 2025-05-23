import axios from 'axios';

// Default API URL - updated with your latest ngrok URL
const API_URL = 'https://b975-34-87-248-44.ngrok-free.app';

// Interface for the keyphrase extraction request
export interface KeyphraseExtractionRequest {
  text: string;
  domain?: string;
}

// Interface for a keyphrase with its score
export interface Keyphrase {
  keyphrase: string;
  score: number;
  expansions?: string[];
}

// Interface for the keyphrase extraction response
export interface KeyphraseExtractionResponse {
  original_keyphrases: [string, number][];
  expanded_keyphrases: {
    [keyphrase: string]: [string, number][];
  };
}

/**
 * Extract keyphrases from the provided text
 * @param text The text to extract keyphrases from
 * @param domain Optional domain for domain-specific extraction
 * @returns A promise that resolves to the extracted keyphrases with scores and expansions
 */
export const extractKeyphrases = async (text: string, domain?: string): Promise<Keyphrase[]> => {
  try {
    const response = await axios.post<KeyphraseExtractionResponse>(
      `${API_URL}/extract_keyphrases`,
      { text, domain }
    );

    // Debug log the raw response
    console.log('Raw API response:', JSON.stringify(response.data, null, 2));

    // Helper function to capitalize first letter of each word
    const capitalizeWords = (text: string): string => {
      return text.split(' ')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
        .join(' ');
    };

    // Check if we have the new response format with original_keyphrases and expanded_keyphrases
    if (response.data.original_keyphrases && response.data.expanded_keyphrases) {
      // Process the new format
      const result = response.data.original_keyphrases.map(item => {
        if (Array.isArray(item) && item.length === 2) {
          const [keyphrase, score] = item;
          const keyphraseStr = String(keyphrase);

          // Get expansions for this keyphrase if available
          const expansionsList = response.data.expanded_keyphrases[keyphraseStr] || [];

          // Extract just the expansion text (not the scores) for the UI
          const expansions = expansionsList.length > 0
            ? expansionsList.map(exp => capitalizeWords(String(exp[0])))
            : [];

          // Return the keyphrase with its score and expansions
          return {
            keyphrase: capitalizeWords(keyphraseStr),
            score,
            expansions
          };
        } else {
          console.log('Item is not in expected format:', item);
          return { keyphrase: capitalizeWords(String(item)), score: 0.5, expansions: [] };
        }
      });

      // Debug log the final result
      console.log('Processed keyphrases with expansions:', JSON.stringify(result, null, 2));
      return result;
    } else {
      // Fallback for old format if needed
      console.error('API response does not have the expected format');
      return [];
    }
  } catch (error) {
    console.error('Error extracting keyphrases:', error);
    throw error;
  }
};

/**
 * Update the API URL
 * @param url The new API URL
 */
export const updateApiUrl = (url: string) => {
  // This is a simple implementation. In a real app, you might want to store this in AsyncStorage
  (axios.defaults.baseURL as any) = url;
};
