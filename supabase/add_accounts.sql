-- One-time update for sign-in, free limits and paid plans.
-- Supabase -> SQL Editor -> New query -> paste all of this -> Run.

create table if not exists public.profiles (
  id                      uuid primary key references auth.users (id) on delete cascade,
  email                   text,
  plan                    text not null default 'free',   -- free | creator | pro
  status                  text,                           -- Stripe subscription status
  billing_interval        text,                           -- month | year
  stripe_customer_id      text unique,
  stripe_subscription_id  text,
  period_start            timestamptz,                    -- current billing period
  period_end              timestamptz,
  created_at              timestamptz not null default now()
);
-- Only the backend reads/writes profiles.
alter table public.profiles enable row level security;

alter table public.comparisons add column if not exists user_id uuid;
create index if not exists comparisons_user_idx on public.comparisons (user_id, created_at);

-- Handy view: who's signed up and on which plan.
create or replace view public.customer_overview
with (security_invoker = true) as
select
  p.email,
  p.plan,
  p.status,
  p.billing_interval,
  p.period_end,
  p.created_at as signed_up,
  (select count(*) from public.comparisons c
    where c.user_id = p.id and c.status = 'done') as comparisons_run
from public.profiles p
order by p.created_at desc;
