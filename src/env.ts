/**
 * Environment initialization - loads .env BEFORE any other imports
 * This file must be imported first to ensure environment variables are available
 */

import { config as loadEnv } from 'dotenv';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

// Get the directory of this file
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Load .env file from project root (one level up from src/)
// This must happen before config.ts is loaded
loadEnv({ path: join(__dirname, '..', '.env') });

// Re-export for convenience
export { loadEnv };
