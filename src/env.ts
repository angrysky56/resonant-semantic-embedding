/**
 * Environment initialization - loads .env BEFORE any other imports
 * This file must be imported first to ensure environment variables are available
 */

import { config as loadEnv } from 'dotenv';

// Load .env file into process.env
// This must happen before config.ts is loaded
loadEnv();

// Re-export for convenience
export { loadEnv };
