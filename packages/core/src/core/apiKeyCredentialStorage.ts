/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { HybridTokenStorage } from '../mcp/token-storage/hybrid-token-storage.js';
import type { OAuthCredentials } from '../mcp/token-storage/types.js';
import { debugLogger } from '../utils/debugLogger.js';
import { AuthType } from './contentGenerator.js';

const KEYCHAIN_SERVICE_NAME = 'gemini-cli-api-key';

const storage = new HybridTokenStorage(KEYCHAIN_SERVICE_NAME);

function getEntryName(authType?: string): string {
  if (authType === AuthType.USE_DEEPSEEK) {
    return 'deepseek-api-key';
  }
  return 'default-api-key';
}

/**
 * Load cached API key
 */
export async function loadApiKey(authType?: string): Promise<string | null> {
  try {
    const entryName = getEntryName(authType);
    const credentials = await storage.getCredentials(entryName);

    if (credentials?.token?.accessToken) {
      return credentials.token.accessToken;
    }

    return null;
  } catch (error: unknown) {
    // Log other errors but don't crash, just return null so user can re-enter key
    debugLogger.error('Failed to load API key from storage:', error);
    return null;
  }
}

/**
 * Save API key
 */
export async function saveApiKey(
  apiKey: string | null | undefined,
  authType?: string,
): Promise<void> {
  const entryName = getEntryName(authType);
  if (!apiKey || apiKey.trim() === '') {
    try {
      await storage.deleteCredentials(entryName);
    } catch (error: unknown) {
      // Ignore errors when deleting, as it might not exist
      debugLogger.warn('Failed to delete API key from storage:', error);
    }
    return;
  }

  // Wrap API key in OAuthCredentials format as required by HybridTokenStorage
  const credentials: OAuthCredentials = {
    serverName: entryName,
    token: {
      accessToken: apiKey,
      tokenType: 'ApiKey',
    },
    updatedAt: Date.now(),
  };

  await storage.setCredentials(credentials);
}

/**
 * Clear cached API key
 */
export async function clearApiKey(authType?: string): Promise<void> {
  try {
    const entryName = getEntryName(authType);
    await storage.deleteCredentials(entryName);
  } catch (error: unknown) {
    debugLogger.error('Failed to clear API key from storage:', error);
  }
}
