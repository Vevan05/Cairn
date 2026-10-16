-- Init schema for Cairn free-tier Postgres (Neon/Supabase)
-- Run: psql $DATABASE_URL -f scripts/init_db.sql

CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft',
    primary_metric TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS variants (
    experiment_id TEXT REFERENCES experiments(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    weight INTEGER NOT NULL,
    config_json JSONB,
    PRIMARY KEY (experiment_id, name)
);

CREATE TABLE IF NOT EXISTS exposures (
    event_id TEXT UNIQUE NOT NULL,
    experiment_id TEXT REFERENCES experiments(id) ON DELETE CASCADE,
    variant TEXT,
    user_id TEXT,
    ts TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS outcomes (
    event_id TEXT UNIQUE NOT NULL,
    experiment_id TEXT REFERENCES experiments(id) ON DELETE CASCADE,
    variant TEXT,
    user_id TEXT,
    metric TEXT,
    value DOUBLE PRECISION,
    ts TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS results (
    experiment_id TEXT REFERENCES experiments(id) ON DELETE CASCADE,
    metric TEXT,
    variant TEXT,
    n INTEGER,
    mean DOUBLE PRECISION,
    ci_low DOUBLE PRECISION,
    ci_high DOUBLE PRECISION,
    p_value DOUBLE PRECISION,
    computed_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_exposures_exp ON exposures(experiment_id);
CREATE INDEX IF NOT EXISTS idx_outcomes_exp ON outcomes(experiment_id);
