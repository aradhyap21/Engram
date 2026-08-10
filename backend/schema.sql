-- MemoryMesh Database Schema
-- Run this in your Supabase SQL editor

-- Enable UUID extension
create extension if not exists "uuid-ossp";

-- Nodes table
create table public.nodes (
    id uuid default uuid_generate_v4() primary key,
    content text not null,
    entity_type text,
    canonical_name text,
    aliases jsonb default '[]'::jsonb,
    embedding jsonb,
    pending_review boolean default false,
    source_mention text,
    model_used text,
    decided_at timestamp with time zone,
    strength float default 1.0,
    access_count int default 0,
    created_at timestamp with time zone default timezone('utc'::text, now())
);

-- Edges table
create table public.edges (
    id uuid default uuid_generate_v4() primary key,
    from_id uuid references public.nodes(id),
    to_id uuid references public.nodes(id),
    relationship text,
    fact_text text,
    valid_at timestamp with time zone,
    invalid_at timestamp with time zone,
    weight float default 1.0,
    created_at timestamp with time zone default timezone('utc'::text, now())
);

-- Optional: add indexes for performance
create index on public.nodes (content);
create index on public.edges (from_id);
create index on public.edges (to_id);