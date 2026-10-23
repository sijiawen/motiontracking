// Fill these in after deploying (see README.md, step 5).
// The anon key is designed to be public; never put the service_role key here.
window.APP_CONFIG = {
  APP_NAME: "Creatorgen",
  // e.g. "https://your-workspace--origin-video-compare-api.modal.run"
  API_URL: "https://sijiawen--origin-video-compare-api.us-east.modal.run/",
  // From Supabase -> Project Settings -> API
  SUPABASE_URL: "https://jfxvakeezsdrugbsqnea.supabase.co",
  SUPABASE_ANON_KEY: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImpmeHZha2VlenNkcnVnYnNxbmVhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3OTI0NDEyNDQsImV4cCI6MjEwODAxNzI0NH0.F800RweJULauFKD6BZrCfscdPbOFGfrb8xjzlXjYROo",
  // Google Cloud -> Google Auth Platform -> Clients -> your Web client -> Client ID.
  // Public (not a secret). Leave empty to use the older redirect sign-in.
  GOOGLE_CLIENT_ID: "",
  MAX_SECONDS: 60,
  MAX_MB: 50,
  DAILY_LIMIT: 0, // per visitor per day; 0 = unlimited (also change it in backend/modal_app.py)
};
// Leave API_URL empty (or add ?demo=1 to the address) to preview the site with sample results.
