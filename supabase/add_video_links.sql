-- One-time update so people can paste a link instead of uploading.
-- Supabase -> SQL Editor -> New query -> paste all of this -> Run.

alter table public.comparisons add column if not exists video_a_url text;
alter table public.comparisons add column if not exists video_b_url text;
