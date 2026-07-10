-- MemoryMesh Database Schema
-- Run this in your Supabase SQL editor

-- Enable UUID extension
create extension if not exists "uuid-ossp";

-- Nodes table
create table public.nodes (
    id uuid default uuid_generate_v4() primary key,
    content text not null,
    entity_type text,
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
    weight float default 1.0,
    created_at timestamp with time zone default timezone('utc'::text, now())
);

-- Optional: add indexes for performance
create index on public.nodes (content);
create index on public.edges (from_id);
create index on public.edges (to_id);