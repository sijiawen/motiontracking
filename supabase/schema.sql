-- Run this once in Supabase: Dashboard -> SQL Editor -> New query -> paste -> Run.
-- It creates the table that logs every comparison and two private storage buckets.

create table if not exists public.comparisons (
  id               uuid primary key default gen_random_uuid(),
  created_at       timestamptz not null default now(),
  finished_at      timestamptz,
  status           text not null default 'awaiting_upload'
                   check (status in ('awaiting_upload', 'queued', 'processing', 'done', 'error')),
  stage            text,
  progress         int default 0,
  error            text,

  -- what was uploaded
  video_a_path     text, video_a_name text, video_a_size bigint, video_a_duration real,
  video_b_path     text, video_b_name text, video_b_size bigint, video_b_duration real,

  -- who uploaded (anonymous)
  ip_hash          text,     -- salted one-way hash, used for the daily limit
  user_agent       text,
  consent_at       timestamptz,

  -- results
  overall_score    real,     -- 0..1 final score = highest of the three tests
  raw_score        real,     -- 0..1, before calibration
  calibrated       boolean,
  verdict          text,
  mirrored         boolean,
  model_scores     jsonb,    -- {"pose": 0.9, "seg": ..., ...}
  model_scores_raw jsonb,
  tests            jsonb,    -- {"driver": "scene", "body": {...}, "scene": {...}, "script": {...}}
  transcripts      jsonb,    -- both transcripts + the shared phrases
  alignment        jsonb,    -- matched time ranges + per-model similarity timelines
  frames           jsonb,    -- paths of the example images per model
  timings          jsonb
);

-- (for databases created before the scene + script tests were added)
alter table public.comparisons add column if not exists tests jsonb;
alter table public.comparisons add column if not exists transcripts jsonb;
alter table public.comparisons add column if not exists video_a_url text;
alter table public.comparisons add column if not exists video_b_url text;

create index if not exists comparisons_created_at_idx on public.comparisons (created_at desc);
create index if not exists comparisons_ip_hash_idx on public.comparisons (ip_hash, created_at);

-- Lock the table down: with RLS on and no policies, the public "anon" key can't read
-- or write anything. Only the backend (service_role key) can.
alter table public.comparisons enable row level security;

-- Private buckets. Videos are capped at 50 MB (the Supabase free-plan maximum).
insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values
  ('videos', 'videos', false, 52428800,
   array['video/mp4', 'video/quicktime', 'video/webm', 'video/x-m4v', 'video/x-matroska',
         'video/x-msvideo', 'application/octet-stream']),
  ('results', 'results', false, 10485760, array['image/jpeg', 'image/png'])
on conflict (id) do update
  set public = excluded.public,
      file_size_limit = excluded.file_size_limit,
      allowed_mime_types = excluded.allowed_mime_types;

-- A friendly view for browsing in the Table Editor.
drop view if exists public.comparison_overview;
create view public.comparison_overview
with (security_invoker = true) as
select
  created_at,
  status,
  round((overall_score * 100)::numeric) as score_pct,
  verdict,
  tests->>'driver' as strongest_test,
  round(((tests->'body'->>'score')::numeric) * 100) as body_pct,
  round(((tests->'scene'->>'score')::numeric) * 100) as scene_pct,
  round(((tests->'script'->>'score')::numeric) * 100) as script_pct,
  mirrored,
  video_a_name,
  video_b_name,
  round(video_a_duration::numeric, 1) as a_seconds,
  round(video_b_duration::numeric, 1) as b_seconds,
  left(ip_hash, 8) as visitor,
  error,
  id
from public.comparisons
order by created_at desc;
