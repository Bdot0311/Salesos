# Cache-First Lead Generation Proxy

FastAPI application that caches People Data Labs API results to reduce costs.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create `.env` file:
```bash
cp .env.example .env
# Edit .env with your credentials
```

3. Ensure PostgreSQL is running and create a database:
```sql
CREATE DATABASE leads_db;
```

4. Run the server:
```bash
uvicorn main:app --reload
```

## API Endpoints

### POST /search
Search for leads (cache-first).

```json
{
  "job_title": "Software Engineer",
  "location": "San Francisco",
  "industry": "Technology",
  "company": "Google",
  "company_size": "10001+",
  "seniority": "senior",
  "limit": 10
}
```

Response includes `source: "cache"` or `source: "api"` to indicate data origin.

### GET /health
Health check.

### GET /cache/stats
View cache statistics.

### DELETE /cache
Clear all cached data.
