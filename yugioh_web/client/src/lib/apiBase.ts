/** Base URL of the Python FastAPI backend, for every consumer in the client.
 *  Override at build time with VITE_API_BASE. */
export const API_BASE =
  import.meta.env.VITE_API_BASE || "http://localhost:8000";
