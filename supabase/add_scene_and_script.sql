-- One-time update for the scene + script tests.
-- Supabase -> SQL Editor -> New query -> paste all of this -> Run.

alter table public.comparisons add column if not exists tests jsonb;
alter table public.comparisons add column if not exists transcripts jsonb;

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
