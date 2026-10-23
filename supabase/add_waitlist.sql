-- One-time update for the creator waitlist.
-- Supabase -> SQL Editor -> New query -> paste all of this -> Run.

create table if not exists public.waitlist (
  id            uuid primary key default gen_random_uuid(),
  created_at    timestamptz not null default now(),
  updated_at    timestamptz not null default now(),
  email         text not null unique,
  name          text,
  channels      jsonb not null default '{}'::jsonb,  -- {"tiktok": "@me", "youtube": "...", ...}
  library_size  text,                                -- under_50 | 50_200 | 200_1000 | over_1000
  seen_copies   text,                                -- yes | not_sure | no
  notes         text,
  manual_check  boolean not null default false,      -- wants a hand-checked scan
  status        text not null default 'new',         -- your own notes: new | contacted | scanned ...
  user_id       uuid,
  source        text,
  ip_hash       text,
  user_agent    text
);
-- Only the backend reads/writes the waitlist.
alter table public.waitlist enable row level security;
create index if not exists waitlist_created_idx on public.waitlist (created_at desc);

-- Handy view: Table Editor -> waitlist_overview
create or replace view public.waitlist_overview
with (security_invoker = true) as
select
  created_at as signed_up,
  email,
  name,
  channels->>'tiktok'    as tiktok,
  channels->>'instagram' as instagram,
  channels->>'youtube'   as youtube,
  channels->>'x'         as x,
  channels->>'other'     as other,
  library_size,
  seen_copies,
  manual_check,
  status,
  notes
from public.waitlist
order by created_at desc;
