const createExpoWebpackConfigAsync = require('@expo/webpack-config');
const path = require('path');

module.exports = async function (env, argv) {
  const config = await createExpoWebpackConfigAsync(env, argv);

  // Add polyfills for Node.js modules
  if (!config.resolve.fallback) {
    config.resolve.fallback = {};
  }

  // Add polyfills for Node.js modules
  config.resolve.fallback.crypto = require.resolve('crypto-browserify');
  config.resolve.fallback.stream = require.resolve('stream-browserify');
  config.resolve.fallback.buffer = require.resolve('buffer/');
  config.resolve.fallback.vm = require.resolve('vm-browserify');

  // Add a rule for font files
  config.module.rules.push({
    test: /\.(woff|woff2|eot|ttf|otf)$/i,
    type: 'asset/resource',
  });

  // Add resolve.alias for react-native-vector-icons
  if (!config.resolve.alias) {
    config.resolve.alias = {};
  }

  // Map react-native-vector-icons to our web implementations
  config.resolve.alias['react-native-vector-icons/MaterialCommunityIcons'] =
    path.resolve(__dirname, 'src/components/MaterialCommunityIcon.web.js');
  config.resolve.alias['react-native-vector-icons/MaterialIcons'] =
    path.resolve(__dirname, 'src/components/MaterialIcons.web.js');

  return config;
};
