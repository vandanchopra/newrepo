import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  TextInput,
  Switch,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

interface Settings {
  apiUrl: string;
  offlineMode: boolean;
  memoryEnabled: boolean;
  darkMode: boolean;
}

const DEFAULT_SETTINGS: Settings = {
  apiUrl: 'http://localhost:8000',
  offlineMode: false,
  memoryEnabled: true,
  darkMode: false,
};

export function SettingsScreen() {
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [serverStatus, setServerStatus] = useState<'checking' | 'connected' | 'disconnected'>('checking');

  useEffect(() => {
    checkServerConnection();
  }, [settings.apiUrl]);

  const checkServerConnection = async () => {
    setServerStatus('checking');
    try {
      const response = await fetch(`${settings.apiUrl}/health`, {
        method: 'GET',
        timeout: 5000,
      } as any);

      if (response.ok) {
        setServerStatus('connected');
      } else {
        setServerStatus('disconnected');
      }
    } catch {
      setServerStatus('disconnected');
    }
  };

  const handleSave = () => {
    // In a real app, save to AsyncStorage or another persistence layer
    Alert.alert('Settings Saved', 'Your settings have been saved successfully.');
  };

  const handleReset = () => {
    Alert.alert(
      'Reset Settings',
      'Are you sure you want to reset all settings to defaults?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Reset',
          style: 'destructive',
          onPress: () => setSettings(DEFAULT_SETTINGS),
        },
      ]
    );
  };

  return (
    <SafeAreaView style={styles.container} edges={['bottom']}>
      <ScrollView contentContainerStyle={styles.content}>
        {/* Server Configuration */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Server Configuration</Text>

          <View style={styles.field}>
            <Text style={styles.label}>API URL</Text>
            <TextInput
              style={styles.input}
              value={settings.apiUrl}
              onChangeText={(value) =>
                setSettings((prev) => ({ ...prev, apiUrl: value }))
              }
              placeholder="http://localhost:8000"
              autoCapitalize="none"
              autoCorrect={false}
              keyboardType="url"
            />
          </View>

          <View style={styles.statusRow}>
            <Text style={styles.label}>Server Status</Text>
            <View style={styles.statusContainer}>
              <View
                style={[
                  styles.statusDot,
                  serverStatus === 'connected' && styles.connected,
                  serverStatus === 'disconnected' && styles.disconnected,
                  serverStatus === 'checking' && styles.checking,
                ]}
              />
              <Text style={styles.statusText}>
                {serverStatus === 'checking'
                  ? 'Checking...'
                  : serverStatus === 'connected'
                  ? 'Connected'
                  : 'Disconnected'}
              </Text>
            </View>
          </View>

          <TouchableOpacity
            style={styles.checkButton}
            onPress={checkServerConnection}
          >
            <Text style={styles.checkButtonText}>Check Connection</Text>
          </TouchableOpacity>
        </View>

        {/* Agent Settings */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Agent Settings</Text>

          <View style={styles.switchRow}>
            <View>
              <Text style={styles.label}>Offline Mode</Text>
              <Text style={styles.description}>
                Use mock responses when server is unavailable
              </Text>
            </View>
            <Switch
              value={settings.offlineMode}
              onValueChange={(value) =>
                setSettings((prev) => ({ ...prev, offlineMode: value }))
              }
              trackColor={{ false: '#e5e7eb', true: '#0ea5e9' }}
            />
          </View>

          <View style={styles.switchRow}>
            <View>
              <Text style={styles.label}>Memory Enabled</Text>
              <Text style={styles.description}>
                Store conversation history for context
              </Text>
            </View>
            <Switch
              value={settings.memoryEnabled}
              onValueChange={(value) =>
                setSettings((prev) => ({ ...prev, memoryEnabled: value }))
              }
              trackColor={{ false: '#e5e7eb', true: '#0ea5e9' }}
            />
          </View>
        </View>

        {/* Appearance */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Appearance</Text>

          <View style={styles.switchRow}>
            <View>
              <Text style={styles.label}>Dark Mode</Text>
              <Text style={styles.description}>
                Use dark color scheme
              </Text>
            </View>
            <Switch
              value={settings.darkMode}
              onValueChange={(value) =>
                setSettings((prev) => ({ ...prev, darkMode: value }))
              }
              trackColor={{ false: '#e5e7eb', true: '#0ea5e9' }}
            />
          </View>
        </View>

        {/* Actions */}
        <View style={styles.actions}>
          <TouchableOpacity style={styles.saveButton} onPress={handleSave}>
            <Text style={styles.saveButtonText}>Save Settings</Text>
          </TouchableOpacity>

          <TouchableOpacity style={styles.resetButton} onPress={handleReset}>
            <Text style={styles.resetButtonText}>Reset to Defaults</Text>
          </TouchableOpacity>
        </View>

        {/* App Info */}
        <View style={styles.appInfo}>
          <Text style={styles.appInfoText}>DroidRun Mobile v1.0.0</Text>
          <Text style={styles.appInfoText}>Autonomous Mobile Agent</Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f3f4f6',
  },
  content: {
    padding: 16,
  },
  section: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 16,
  },
  field: {
    marginBottom: 16,
  },
  label: {
    fontSize: 14,
    fontWeight: '500',
    color: '#374151',
    marginBottom: 8,
  },
  description: {
    fontSize: 12,
    color: '#6b7280',
    maxWidth: '80%',
  },
  input: {
    backgroundColor: '#f3f4f6',
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
    color: '#1f2937',
  },
  statusRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  statusContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 6,
  },
  connected: {
    backgroundColor: '#22c55e',
  },
  disconnected: {
    backgroundColor: '#ef4444',
  },
  checking: {
    backgroundColor: '#f59e0b',
  },
  statusText: {
    fontSize: 14,
    color: '#6b7280',
  },
  checkButton: {
    backgroundColor: '#f3f4f6',
    borderRadius: 8,
    padding: 12,
    alignItems: 'center',
  },
  checkButtonText: {
    color: '#0ea5e9',
    fontWeight: '500',
  },
  switchRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#f3f4f6',
  },
  actions: {
    marginTop: 8,
  },
  saveButton: {
    backgroundColor: '#0ea5e9',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    marginBottom: 12,
  },
  saveButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  resetButton: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  resetButtonText: {
    color: '#ef4444',
    fontSize: 16,
    fontWeight: '500',
  },
  appInfo: {
    alignItems: 'center',
    paddingVertical: 24,
  },
  appInfoText: {
    fontSize: 12,
    color: '#9ca3af',
    marginBottom: 4,
  },
});
